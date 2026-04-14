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
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        """Return the registry key used by VLLM_ATTENTION_BACKEND."""
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> Type["DeterministicAttentionImpl"]:
        """Return the attention-impl class that runs deterministic kernels."""
        return DeterministicAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type[AttentionMetadata]:
        """Return vLLM's :class:`FlashAttentionMetadata` (reused unchanged)."""
        _lazy_import_flash_meta()
        assert _flash_meta_cls is not None
        return _flash_meta_cls

    @staticmethod
    def get_builder_cls() -> Type[AttentionMetadataBuilder]:
        """Return vLLM's flash-attn metadata builder (reused unchanged)."""
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
        """Return the on-device KV-cache tensor shape expected by this backend.

        Args:
            num_blocks:      Total paged blocks allocated to the cache.
            block_size:      Tokens per paged block.
            num_kv_heads:    KV head count after tensor-parallel sharding.
            head_size:       Per-head hidden dimension.
            cache_dtype_str: vLLM cache dtype spec (unused; shape is dtype-agnostic).

        Returns:
            (2, num_blocks, block_size, num_kv_heads, head_size) — the leading 2 splits keys (0) and values (1).
        """
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        """Return the KV-cache stride order, delegating to vLLM's flash-attn backend.

        Args:
            include_num_layers_dimension: If True, include the leading layer axis in the permutation.

        Returns:
            Tuple of axis indices specifying storage order.
        """
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
        """Configure a deterministic attention impl for one model layer.

        Args:
            num_heads:       Query heads after TP sharding.
            head_size:       Per-head hidden dimension.
            scale:           Softmax scale (typically 1/sqrt(head_size)).
            num_kv_heads:    KV heads after TP sharding (equal to num_heads for MHA, smaller for GQA/MQA).
            alibi_slopes:    Unsupported; must be None.
            sliding_window:  Unsupported; must be None.
            kv_cache_dtype:  vLLM cache dtype spec forwarded to the cache writer.
            blocktable_size: Max blocks per request (unused; kept for API parity).
            logits_soft_cap: Optional soft-cap, stored but not applied by the current kernels.
            attn_type:       Layer attention type (decoder / encoder / encoder-only).
            **kwargs:        Swallowed for forward-compatibility with vLLM.

        Raises:
            NotImplementedError: If ALiBi or sliding-window is requested.
        """
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
        """Run deterministic attention for one layer on a packed batch.

        Args:
            layer:              vLLM attention layer (supplies KV scales during caching).
            query:              (num_tokens, num_heads    * head_size) packed queries.
            key:                (num_tokens, num_kv_heads * head_size) packed keys.
            value:              (num_tokens, num_kv_heads * head_size) packed values.
            kv_cache:           (2, num_blocks, block_size, num_kv_heads, head_size) paged K/V cache.
            attn_metadata:      Scheduler metadata; None during the vLLM profiling pass (a zeroed output is returned).
            output:             (num_tokens, num_heads    * head_size) optional pre-allocated output buffer.
            output_scale:       Unsupported.
            output_block_scale: Unsupported.
            **kwargs:           Swallowed for forward-compatibility.

        Returns:
            (num_tokens, num_heads * head_size) attention output.

        Raises:
            NotImplementedError: If quantised output scales are requested.
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
        """Classify a packed batch as all-prefill or all-decode.

        Args:
            attn_metadata: Scheduler metadata for the current step.
            num_tokens:    Total tokens in the packed batch.

        Returns:
            (num_prefill_tokens, num_decode_tokens, num_prefill_requests) — exactly one of the first two is non-zero.
        """
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
        """Write new K/V tokens into the paged KV cache. No-op for encoder / encoder-only types.

        Args:
            layer:        Source of _k_scale / _v_scale for optional fp8 cache.
            key:          (num_tokens, num_kv_heads, head_size) new keys.
            value:        (num_tokens, num_kv_heads, head_size) new values.
            kv_cache:     (2, num_blocks, block_size, num_kv_heads, head_size) paged cache.
            slot_mapping: (num_tokens,) flat slot indices into the cache.
        """
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
        """Run the paged fixed-point prefill kernel and write into output.

        Args:
            query:         (num_prefill_tokens, num_heads,     head_size) prefill queries.
            kv_cache:      (2, num_blocks, block_size, num_kv_heads, head_size) paged cache.
            output:        (num_prefill_tokens, num_heads,     head_size) pre-allocated output.
            attn_metadata: Scheduler metadata providing seq_lens, query_start_loc and block_table.
            num_prefills:  Number of prefill requests (leading rows of per-request metadata).
        """
        seq_lens = attn_metadata.seq_lens[:num_prefills].to(torch.int32)
        if seq_lens.numel() == 0:
            return
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
        """Run the paged fixed-point decode kernel and write into output.

        Internally allocates scratch attn_logits of shape
        (num_decode, num_heads, num_kv_splits, head_dim_v + 1) and lse of shape
        (num_decode, num_heads) for the split-KV online-softmax reduction.

        Args:
            query:         (num_decode, num_heads, head_size) decode queries (one token per request).
            kv_cache:      (2, num_blocks, block_size, num_kv_heads, head_size) paged cache.
            output:        (num_decode, num_heads, head_size) pre-allocated output.
            attn_metadata: Scheduler metadata providing seq_lens and block_table.
            num_prefills:  Number of prefill requests preceding the decode rows in the packed metadata.
            num_decode:    Number of decode requests.
        """
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
    """Return t as a contiguous float32 tensor, aliasing when already so.

    Args:
        t: Any floating tensor.

    Returns:
        t if already contiguous float32, otherwise a new contiguous float32 copy of the same shape.
    """
    if t.dtype == torch.float32 and t.is_contiguous():
        return t
    return t.to(torch.float32).contiguous()


def _copy_from_fp32(dst: torch.Tensor, src_fp32: torch.Tensor) -> None:
    """Copy a float32 kernel result into a destination of arbitrary dtype.

    Args:
        dst:      Destination tensor, any dtype, same shape as src_fp32.
        src_fp32: Source float32 tensor; if dst already aliases it, this is a no-op.
    """
    if dst.dtype == torch.float32:
        if dst.data_ptr() != src_fp32.data_ptr():
            dst.copy_(src_fp32)
        return
    dst.copy_(src_fp32.to(dst.dtype))
