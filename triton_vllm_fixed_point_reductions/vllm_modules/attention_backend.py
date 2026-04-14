"""
Deterministic attention backend for vLLM.

Replaces FlashAttention / FlashInfer with fixed-point kernels for
Q·Kᵀ, softmax, and P·V. Reuses FlashAttention's metadata and cache
management — only the compute is replaced.

Activate via: VLLM_ATTENTION_BACKEND=DETERMINISTIC
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Type

import torch

from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
)

from ..fixed_point_kernels.prefill import context_attention_fwd_fxp_kernel
from ..fixed_point_kernels.decode import decode_attention_fwd_fp_kernel

DEFAULT_NUM_KV_SPLITS = 8
DEBUG_TIMING = os.getenv("VLLM_FXP_DEBUG_TIMING", "0") == "1"

# ---------------------------------------------------------------------------
# Re-use FlashAttention metadata — it already handles paged KV cache
# block tables, slot mappings, sequence lengths, etc.
# We only replace the compute, not the bookkeeping.
# ---------------------------------------------------------------------------

_flash_meta_cls: Optional[Type[AttentionMetadata]] = None
_flash_builder_cls: Optional[Type[AttentionMetadataBuilder]] = None


def _lazy_import_flash_meta():
    """Deferred import so the module loads even if flash_attn isn't installed."""
    global _flash_meta_cls, _flash_builder_cls
    if _flash_meta_cls is None:
        from vllm.v1.attention.backends.flash_attn import (
            FlashAttentionMetadata,
            FlashAttentionMetadataBuilder,
        )

        _flash_meta_cls = FlashAttentionMetadata
        _flash_builder_cls = FlashAttentionMetadataBuilder


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


class DeterministicAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"
 
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
        """
        Match FlashAttention's KV cache layout so existing cache ops work.
        Shape: (2, num_blocks, block_size, num_kv_heads, head_size)
        Layer 0 = keys, layer 1 = values.
        """
        return (2, num_blocks, block_size, num_kv_heads, head_size)

# ---------------------------------------------------------------------------
# Attention implementation
# ---------------------------------------------------------------------------

# Default fractional bits — should match the GEMM kernel setting.
DEFAULT_ATTN_FRAC_BITS = 16


class DeterministicAttentionImpl(AttentionImpl):
    """
    Deterministic attention using fixed-point accumulation.

    Covers:
        - S = scale * (Q · Kᵀ)   — fixed-point matmul
        - P = softmax(S)          — fixed-point reduction (max, sum-exp)
        - O = P · V               — fixed-point matmul

    Separate kernel paths for prefill (full causal) and decode (paged cache).
    """

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
        **kwargs: Any,
    ) -> None:
        if alibi_slopes is not None:
            raise NotImplementedError(
                "DeterministicAttention does not support ALiBi. "
                "Qwen3 uses RoPE — this should not be needed."
            )
        if sliding_window is not None:
            raise NotImplementedError(
                "DeterministicAttention does not yet support sliding window. "
                "Add causal + window masking to the prefill kernel if needed."
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads  # GQA group count
        self.kv_cache_dtype = kv_cache_dtype
        self.blocktable_size = blocktable_size
        self.logits_soft_cap = logits_soft_cap
        self.frac_bits = DEFAULT_ATTN_FRAC_BITS

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Optional[AttentionMetadata],
        output: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Main entry point called by vLLM's Attention module.

        Args:
            query:  (num_tokens, num_heads * head_size)
            key:    (num_tokens, num_kv_heads * head_size)
            value:  (num_tokens, num_kv_heads * head_size)
            kv_cache: paged KV cache tensor.
            attn_metadata: sequence lengths, block tables, slot mapping, etc.
            output: pre-allocated output buffer (num_tokens, num_heads * head_size).
        """
        t0 = time.perf_counter() if DEBUG_TIMING else 0.0
        num_tokens = query.shape[0]
        layer_name = getattr(layer, "layer_name", "unknown") if DEBUG_TIMING else ""

        # Reshape to (num_tokens, num_heads, head_size)
        query = query.view(num_tokens, self.num_heads, self.head_size)
        key = key.view(num_tokens, self.num_kv_heads, self.head_size)
        value = value.view(num_tokens, self.num_kv_heads, self.head_size)

        if output is None:
            output = torch.empty(
                num_tokens, self.num_heads, self.head_size,
                dtype=query.dtype, device=query.device,
            )
        else:
            output = output.view(num_tokens, self.num_heads, self.head_size)

        # vLLM may call attention during profiling with no metadata.
        if attn_metadata is None:
            output.zero_()
            if DEBUG_TIMING:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                print(
                    f"[fxp-attn] layer={layer_name} profile-no-metadata "
                    f"tokens={num_tokens} total_ms={dt_ms:.3f}"
                )
            return output.view(num_tokens, self.num_heads * self.head_size)

        # ---- Write K, V into the paged KV cache ----
        t_cache = time.perf_counter() if DEBUG_TIMING else 0.0
        if kv_cache.numel() > 0:
            self._write_to_cache(layer, key, value, kv_cache, attn_metadata)
        cache_ms = (time.perf_counter() - t_cache) * 1000.0 if DEBUG_TIMING else 0.0

        # ---- Prefill/decode split ----
        num_prefill = getattr(attn_metadata, "num_prefill_tokens", None)
        num_decode = getattr(attn_metadata, "num_decode_tokens", None)
        num_prefills = getattr(attn_metadata, "num_prefills", None)

        # vLLM 0.19 FlashAttentionMetadata uses query_start_loc/max_query_len
        # instead of num_prefill_tokens/num_decode_tokens.
        if num_prefill is None or num_decode is None or num_prefills is None:
            max_query_len = int(getattr(attn_metadata, "max_query_len", 0))
            num_reqs = int(attn_metadata.query_start_loc.numel() - 1)
            if max_query_len > 1:
                num_prefill = num_tokens
                num_decode = 0
                num_prefills = num_reqs
            else:
                num_prefill = 0
                num_decode = num_tokens
                num_prefills = 0

        # ---- Prefill phase ----
        t_prefill = time.perf_counter() if DEBUG_TIMING else 0.0
        if num_prefill > 0:
            self._prefill(
                query[:num_prefill],
                key[:num_prefill],
                value[:num_prefill],
                output[:num_prefill],
                attn_metadata,
                num_prefills=num_prefills,
            )
        prefill_ms = (time.perf_counter() - t_prefill) * 1000.0 if DEBUG_TIMING else 0.0

        # ---- Decode phase ----
        t_decode = time.perf_counter() if DEBUG_TIMING else 0.0
        if num_decode > 0:
            self._decode(
                query[num_prefill:num_prefill + num_decode],
                kv_cache,
                output[num_prefill:num_prefill + num_decode],
                attn_metadata,
                num_prefills=num_prefills,
                num_decode=num_decode,
            )
        decode_ms = (time.perf_counter() - t_decode) * 1000.0 if DEBUG_TIMING else 0.0

        if DEBUG_TIMING:
            total_ms = (time.perf_counter() - t0) * 1000.0
            print(
                f"[fxp-attn] layer={layer_name} tokens={num_tokens} "
                f"prefill={num_prefill} decode={num_decode} "
                f"cache_ms={cache_ms:.3f} prefill_ms={prefill_ms:.3f} "
                f"decode_ms={decode_ms:.3f} total_ms={total_ms:.3f}"
            )

        return output.view(num_tokens, self.num_heads * self.head_size)

    # ---------------------------------------------------------------
    # KV cache write
    # ---------------------------------------------------------------

    @staticmethod
    def _write_to_cache(
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> None:
        """
        Scatter-write K and V into the paged cache.

        This is a per-element write (no reduction) — already deterministic.
        We reuse vLLM's existing cache_ops to get the correct slot mapping.
        """
        from vllm._custom_ops import reshape_and_cache_flash

        key_cache = kv_cache[0]
        value_cache = kv_cache[1]

        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            attn_metadata.slot_mapping.flatten(),
            attn_metadata.kv_cache_dtype
            if hasattr(attn_metadata, "kv_cache_dtype")
            else "auto",
            layer._k_scale,
            layer._v_scale,
        )

    # ---------------------------------------------------------------
    # Prefill
    # ---------------------------------------------------------------

    def _prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: AttentionMetadata,
        num_prefills: int,
    ) -> None:
        """
        Prefill attention: each prompt's tokens attend causally
        to all preceding tokens in the same prompt.
        """
        seq_lens_all = getattr(
            attn_metadata,
            "seq_lens_tensor",
            getattr(attn_metadata, "seq_lens"),
        )
        seq_lens = seq_lens_all[:num_prefills].to(torch.int32)
        if seq_lens.numel() == 0:
            return
        # query_start_loc has length num_seqs + 1; the kernel indexes by
        # cur_batch in [0, batch), so the trailing total is unused.
        start_loc = attn_metadata.query_start_loc[:num_prefills].to(torch.int32)
        max_input_len = int(seq_lens.max().item())

        orig_dtype = query.dtype
        q32 = query.to(torch.float32).contiguous()
        k32 = key.to(torch.float32).contiguous()
        v32 = value.to(torch.float32).contiguous()
        o32 = torch.empty_like(q32)

        context_attention_fwd_fxp_kernel(
            q32,
            k32,
            v32,
            o32,
            start_loc,
            seq_lens,
            max_input_len=max_input_len,
            is_causal=True,
            softmax_scale=self.scale,
            frac_bits=self.frac_bits,
        )
        output.copy_(o32.to(orig_dtype))

    # ---------------------------------------------------------------
    # Decode
    # ---------------------------------------------------------------

    def _decode(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: AttentionMetadata,
        num_prefills: int,
        num_decode: int,
    ) -> None:
        """
        Decode attention: each token attends to its full KV history
        stored in the paged cache. Uses the two-stage split-KV kernel.
        """
        key_cache = kv_cache[0]     # (num_blocks, block_size, kv_heads, head_size)
        value_cache = kv_cache[1]
        page_size = key_cache.shape[1]

        seq_lens_all = getattr(
            attn_metadata,
            "seq_lens_tensor",
            getattr(attn_metadata, "seq_lens"),
        )
        block_tables_all = getattr(
            attn_metadata,
            "block_tables",
            getattr(attn_metadata, "block_table"),
        )
        seq_lens = seq_lens_all[
            num_prefills : num_prefills + num_decode
        ].to(torch.int32)
        block_tables = block_tables_all[
            num_prefills : num_prefills + num_decode
        ]

        orig_dtype = query.dtype
        q32 = query.to(torch.float32).contiguous()   # (batch, heads, dim)
        o32 = torch.empty_like(q32)

        batch = q32.shape[0]
        num_heads = q32.shape[1]
        head_dim_v = value_cache.shape[-1]
        num_kv_splits = DEFAULT_NUM_KV_SPLITS

        attn_logits = torch.empty(
            (batch, num_heads, num_kv_splits, head_dim_v + 1),
            dtype=torch.float32,
            device=q32.device,
        )
        lse = torch.empty(
            (batch, num_heads), dtype=torch.float32, device=q32.device
        )

        decode_attention_fwd_fp_kernel(
            q32,
            key_cache,
            value_cache,
            o32,
            lse,
            block_tables,
            seq_lens,
            attn_logits,
            num_kv_splits=num_kv_splits,
            sm_scale=self.scale,
            page_size=page_size,
            frac_bits=self.frac_bits,
        )
        output.copy_(o32.to(orig_dtype))