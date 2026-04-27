import logging

import torch

from vllm.v1.attention.backend import (
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
)
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

from .. import _cuda  # noqa: F401  (registers torch.ops.fxpr.*)
from .config import get_runtime_config

logger = logging.getLogger("fxpr_vllm")

# Pre-scale factor for ALiBi slopes: the kernel runs the softmax in
# log2 domain, so slopes are pre-multiplied by 1/ln(2) here once.
RCP_LN2 = 1.4426950408889634

_flash_meta_cls: type[AttentionMetadata] | None = None
_flash_builder_cls: type[AttentionMetadataBuilder] | None = None


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
    def get_impl_cls() -> type["DeterministicAttentionImpl"]:
        return DeterministicAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        _lazy_import_flash_meta()
        assert _flash_meta_cls is not None
        return _flash_meta_cls

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
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
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: AttentionType | None = AttentionType.DECODER,
        # vLLM v1 has been growing the signature (kv_sharing_target_layer_name,
        # sinks, use_irope, …). Accept any further positional args here so
        # we stay compatible with new vLLM revisions without caring about
        # the names we don't use.
        *_extra_positional,
        **kwargs,
    ) -> None:
        # vLLM passes attn_type=None to mean "default" (decoder). Reject only
        # explicit encoder / cross-attention types we don't support.
        if attn_type is not None and attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                f"DeterministicAttention only supports decoder attention; "
                f"got attn_type={attn_type!r}."
            )
        attn_type = attn_type or AttentionType.DECODER
        if kv_cache_dtype not in ("auto", "fp16", "bf16"):
            raise NotImplementedError(
                f"DeterministicAttention does not support kv_cache_dtype="
                f"{kv_cache_dtype!r}; supported: auto, fp16, bf16."
            )

        cfg = get_runtime_config()

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        # logits_soft_cap=None / 0 / negative all mean "off"; the kernel
        # treats LOGIT_SOFTCAP > 0.0 as the activation condition.
        self.logits_soft_cap = float(logits_soft_cap) if logits_soft_cap else 0.0
        self.window_size = int(sliding_window) if sliding_window else 0
        self.attn_type = attn_type
        self.frac_bits = cfg.frac_bits
        self.fxp_int_bits = cfg.fxp_int_bits

        if alibi_slopes is not None:
            slopes = torch.tensor(alibi_slopes, dtype=torch.float32) * RCP_LN2
            assert slopes.shape == (
                num_heads,
            ), f"alibi_slopes shape {tuple(slopes.shape)} != (num_heads={num_heads},)"
            # Pre-scaled by RCP_LN2 once at construction; the launcher will
            # not re-multiply on every forward.
            self.alibi_slopes: torch.Tensor | None = slopes
        else:
            self.alibi_slopes = None

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
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
        max_query_len = int(attn_metadata.max_query_len)

        # The kernel reads Q / KV in their native dtype (fp32/fp16/bf16)
        # and widens to fp32 element-wise inside, so we never copy the
        # KV cache. Q and KV must share dtype; if they don't (rare),
        # widen Q only — that tensor is small.
        if query.dtype != kv_cache.dtype:
            q_in = query.to(kv_cache.dtype)
        else:
            q_in = query
        out_fp32 = (
            output if output.dtype == torch.float32
            else torch.empty_like(output, dtype=torch.float32)
        )

        torch.ops.fxpr.unified_attention_fxp(
            q_in,
            kv_cache,
            out_fp32,
            query_start_loc,
            seq_lens,
            block_table,
            max_query_len,
            self.alibi_slopes,
            True,
            float(self.scale),
            int(self.frac_bits),
            int(self.fxp_int_bits),
            float(self.logits_soft_cap),
            int(self.window_size),
        )

        if out_fp32.data_ptr() != output.data_ptr():
            output.copy_(out_fp32.to(output.dtype))

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

        kv_cache layout is (num_blocks, 2, block_size, num_kv_heads, head_size)
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
