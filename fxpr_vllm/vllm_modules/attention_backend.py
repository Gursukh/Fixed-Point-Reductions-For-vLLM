import logging
from typing import List, Optional, Type

import torch

from vllm.v1.attention.backend import (
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
)
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

from ..fixed_point_kernels.attention import unified_attention_fxp
from ..fixed_point_kernels.fixed_point import fixed_tl_dtype
from .config import get_runtime_config

logger = logging.getLogger("fxpr_vllm")

_flash_meta_cls: Optional[Type[AttentionMetadata]] = None
_flash_builder_cls: Optional[Type[AttentionMetadataBuilder]] = None


def _lazy_import_flash_meta() -> None:
    """Import FlashAttention metadata/builder classes on first use.

    vLLM imports flash_attn lazily to avoid CUDA side effects at module
    load. We reuse its metadata/builder so our backend plugs into the existing
    scheduler path without reimplementing them.
    """
    global _flash_meta_cls, _flash_builder_cls
    if _flash_meta_cls is None:
        from vllm.v1.attention.backends.flash_attn import (
            FlashAttentionMetadata,
            FlashAttentionMetadataBuilder,
        )

        _flash_meta_cls = FlashAttentionMetadata
        _flash_builder_cls = FlashAttentionMetadataBuilder


class DeterministicAttentionBackend(TritonAttentionBackend):
    """Fixed-point deterministic attention backend."""

    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> Type["DeterministicAttentionImpl"]:
        return DeterministicAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type[AttentionMetadata]:
        _lazy_import_flash_meta()
        assert _flash_meta_cls is not None
        return _flash_meta_cls

    @staticmethod
    def get_builder_cls() -> Type[AttentionMetadataBuilder]:
        _lazy_import_flash_meta()
        assert _flash_builder_cls is not None
        return _flash_builder_cls


class DeterministicAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str = "auto",
        blocktable_size: int = 16,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        **kwargs,
    ) -> None:
        if sliding_window is not None:
            raise NotImplementedError(
                "DeterministicAttention does not yet support sliding window."
            )

        cfg = get_runtime_config()

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.blocktable_size = blocktable_size
        self.logits_soft_cap = logits_soft_cap
        self.attn_type = attn_type
        self.frac_bits = cfg.frac_bits
        self.fxp_dtype = fixed_tl_dtype(cfg.fxp_int_bits)

        if alibi_slopes is not None:
            slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
            assert slopes.shape == (num_heads,), (
                f"alibi_slopes shape {tuple(slopes.shape)} != (num_heads={num_heads},)"
            )
            self.alibi_slopes: Optional[torch.Tensor] = slopes
        else:
            self.alibi_slopes = None

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Optional[AttentionMetadata],
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run deterministic attention for one layer on a packed batch.

        Single unified kernel call over the full packed batch (prefill + decode).
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "DeterministicAttention does not support fp8/quantized output scales."
            )
        num_tokens = query.shape[0]

        query = query.view(num_tokens, self.num_heads, self.head_size)
        key = key.view(num_tokens, self.num_kv_heads, self.head_size)
        value = value.view(num_tokens, self.num_kv_heads, self.head_size)

        if output is None:
            output = torch.empty(
                num_tokens,
                self.num_heads,
                self.head_size,
                dtype=query.dtype,
                device=query.device,
            )
        else:
            output = output.view(num_tokens, self.num_heads, self.head_size)

        if attn_metadata is None:
            # vLLM profiling pass before metadata is built.
            output.zero_()
            return output.view(num_tokens, self.num_heads * self.head_size)

        # Lazily move alibi slopes to the query device on first forward.
        if self.alibi_slopes is not None and self.alibi_slopes.device != query.device:
            self.alibi_slopes = self.alibi_slopes.to(query.device)

        query_start_loc = attn_metadata.query_start_loc.to(torch.int32)
        seq_lens = attn_metadata.seq_lens.to(torch.int32)
        block_table = attn_metadata.block_table

        q32 = _to_fp32(query)
        o32 = torch.empty_like(q32)

        unified_attention_fxp(
            q=q32,
            kv_cache=kv_cache,
            o=o32,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table=block_table,
            alibi_slopes=self.alibi_slopes,
            is_causal=True,
            softmax_scale=self.scale,
            frac_bits=self.frac_bits,
            fxp_dtype=self.fxp_dtype,
        )
        _copy_from_fp32(output, o32)

        return output.view(num_tokens, self.num_heads * self.head_size)

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Write new K/V tokens into the paged KV cache.

        kv_cache layout is ``(num_blocks, 2, block_size, num_kv_heads, head_size)``
        so K/V are unbound along dim 1.
        """
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return

        from vllm._custom_ops import reshape_and_cache_flash

        key_cache, value_cache = kv_cache.unbind(1)

        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )


def _to_fp32(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.float32 and t.is_contiguous():
        return t
    return t.to(torch.float32).contiguous()


def _copy_from_fp32(dst: torch.Tensor, src_fp32: torch.Tensor) -> None:
    if dst.dtype == torch.float32:
        if dst.data_ptr() != src_fp32.data_ptr():
            dst.copy_(src_fp32)
        return
    dst.copy_(src_fp32.to(dst.dtype))
