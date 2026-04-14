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

from ..fixed_point_kernels.fixed_point import fixed_tl_dtype
from ..fixed_point_kernels.prefill import context_attention_fwd_fxp_paged
from ..fixed_point_kernels.decode import decode_attention_fwd_fp_kernel
from .config import get_runtime_config

logger = logging.getLogger("fxpr_vllm")

_flash_meta_cls: Optional[Type[AttentionMetadata]] = None
_flash_builder_cls: Optional[Type[AttentionMetadataBuilder]] = None


def _lazy_import_flash_meta() -> None:

    global _flash_meta_cls, _flash_builder_cls
    if _flash_meta_cls is None:
        from vllm.v1.attention.backends.flash_attn import (
            FlashAttentionMetadata,
            FlashAttentionMetadataBuilder,
        )

        _flash_meta_cls = FlashAttentionMetadata
        _flash_builder_cls = FlashAttentionMetadataBuilder


class DeterministicAttentionBackend(TritonAttentionBackend):
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

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

        return FlashAttentionBackend.get_kv_cache_stride_order(
            include_num_layers_dimension=include_num_layers_dimension
        )


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
        if alibi_slopes is not None:
            raise NotImplementedError("DeterministicAttention does not support ALiBi.")
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
        self.num_kv_splits = cfg.num_kv_splits
        self.fxp_dtype = fixed_tl_dtype(cfg.fxp_int_bits)

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

        num_prefill, num_decode, num_prefills = self._split_prefill_decode(
            attn_metadata, num_tokens
        )

        if num_prefill > 0:
            self._prefill(
                query[:num_prefill],
                kv_cache,
                output[:num_prefill],
                attn_metadata,
                num_prefills=num_prefills,
            )

        if num_decode > 0:
            self._decode(
                query[num_prefill : num_prefill + num_decode],
                kv_cache,
                output[num_prefill : num_prefill + num_decode],
                attn_metadata,
                num_prefills=num_prefills,
                num_decode=num_decode,
            )

        return output.view(num_tokens, self.num_heads * self.head_size)

    @staticmethod
    def _split_prefill_decode(
        attn_metadata: AttentionMetadata, num_tokens: int
    ) -> tuple[int, int, int]:

        max_query_len = int(attn_metadata.max_query_len)
        num_reqs = int(attn_metadata.query_start_loc.numel() - 1)
        if max_query_len > 1:
            return num_tokens, 0, num_reqs
        return 0, num_tokens, 0

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:

        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return

        from vllm._custom_ops import reshape_and_cache_flash

        key_cache = kv_cache[0]
        value_cache = kv_cache[1]

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

    def _prefill(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: AttentionMetadata,
        num_prefills: int,
    ) -> None:
        seq_lens = attn_metadata.seq_lens[:num_prefills].to(torch.int32)
        if seq_lens.numel() == 0:
            return
        # query_start_loc has length num_reqs+1; we need the [0, num_prefills] slice
        # so each request's start AND stop offset is available to the kernel.
        query_start_loc = attn_metadata.query_start_loc[: num_prefills + 1].to(
            torch.int32
        )
        block_table = attn_metadata.block_table[:num_prefills]
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        max_query_len = int(query_lens.max().item())

        key_cache = kv_cache[0]
        value_cache = kv_cache[1]

        q32 = _to_fp32(query)
        o32 = torch.empty_like(q32)

        context_attention_fwd_fxp_paged(
            q32,
            key_cache,
            value_cache,
            o32,
            query_start_loc,
            seq_lens,
            block_table,
            max_query_len=max_query_len,
            is_causal=True,
            softmax_scale=self.scale,
            frac_bits=self.frac_bits,
            fxp_dtype=self.fxp_dtype,
        )
        _copy_from_fp32(output, o32)

    def _decode(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: AttentionMetadata,
        num_prefills: int,
        num_decode: int,
    ) -> None:
        key_cache = kv_cache[0]
        value_cache = kv_cache[1]
        page_size = key_cache.shape[1]

        seq_lens = attn_metadata.seq_lens[num_prefills : num_prefills + num_decode].to(
            torch.int32
        )
        block_tables = attn_metadata.block_table[
            num_prefills : num_prefills + num_decode
        ]

        q32 = _to_fp32(query)
        o32 = torch.empty_like(q32)

        batch, num_heads, _ = q32.shape
        head_dim_v = value_cache.shape[-1]

        attn_logits = torch.empty(
            (batch, num_heads, self.num_kv_splits, head_dim_v + 1),
            dtype=torch.float32,
            device=q32.device,
        )
        lse = torch.empty((batch, num_heads), dtype=torch.float32, device=q32.device)

        decode_attention_fwd_fp_kernel(
            q32,
            key_cache,
            value_cache,
            o32,
            lse,
            block_tables,
            seq_lens,
            attn_logits,
            num_kv_splits=self.num_kv_splits,
            sm_scale=self.scale,
            page_size=page_size,
            frac_bits=self.frac_bits,
            fxp_dtype=self.fxp_dtype,
        )
        _copy_from_fp32(output, o32)


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
