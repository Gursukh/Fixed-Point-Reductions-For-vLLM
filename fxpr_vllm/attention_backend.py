import logging
from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    MultipleOf,
)

from . import _lib  # noqa: F401
from .config import get_runtime_config

logger = logging.getLogger("fxpr_vllm")

# softmax runs in log2 space, so alibi slopes get scaled by 1/ln(2).
RCP_LN2 = 1.4426950408889634

_flash_meta_cls: type[AttentionMetadata] | None = None
_flash_builder_cls: type[AttentionMetadataBuilder] | None = None


def _lazy_import_flash_meta() -> None:
    global _flash_meta_cls, _flash_builder_cls
    if _flash_meta_cls is None:
        from vllm.v1.attention.backends.flash_attn import (
            FlashAttentionMetadata,
            FlashAttentionMetadataBuilder,
        )

        _flash_meta_cls = FlashAttentionMetadata
        _flash_builder_cls = FlashAttentionMetadataBuilder


class DeterministicAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (1, 0, 2, 3, 4, 5)
        return (0, 1, 2, 3, 4)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

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

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size >= 32


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
        *_extra_positional,
        **kwargs,
    ) -> None:
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
        self.logits_soft_cap = float(logits_soft_cap) if logits_soft_cap else 0.0
        self.window_size = int(sliding_window) if sliding_window else 0
        self.attn_type = attn_type
        self.fxp_int_bits = cfg.fxp_int_bits
        self.fxp_frac_bits = cfg.fxp_frac_bits

        if alibi_slopes is not None:
            slopes = torch.tensor(alibi_slopes, dtype=torch.float32) * RCP_LN2
            assert slopes.shape == (
                num_heads,
            ), f"alibi_slopes shape {tuple(slopes.shape)} != (num_heads={num_heads},)"
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
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "DeterministicAttention does not support fp8/quantized output scales."
            )

        # warm kernels on first forward so requests don't eat a JIT spike.
        from .warmup import warmup_attention

        warmup_attention(
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            kv_cache.dtype,
            query.device,
            self.window_size,
            self.logits_soft_cap,
            self.fxp_int_bits,
            self.fxp_frac_bits,
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
            output.zero_()
            return output.view(num_tokens, self.num_heads * self.head_size)

        if self.alibi_slopes is not None and self.alibi_slopes.device != query.device:
            self.alibi_slopes = self.alibi_slopes.to(query.device)

        # skip the .to() dispatch when already int32 (this runs per layer).
        query_start_loc = attn_metadata.query_start_loc
        if query_start_loc.dtype != torch.int32:
            query_start_loc = query_start_loc.to(torch.int32)
        seq_lens = attn_metadata.seq_lens
        if seq_lens.dtype != torch.int32:
            seq_lens = seq_lens.to(torch.int32)
        block_table = attn_metadata.block_table
        max_query_len = int(attn_metadata.max_query_len)

        # q, kv and output must share a dtype.
        if query.dtype != kv_cache.dtype:
            q_in = query.to(kv_cache.dtype)
        else:
            q_in = query
        if output.dtype != q_in.dtype:
            output = output.to(q_in.dtype)

        torch.ops.fxpr.unified_attention_fxp(
            q_in,
            kv_cache,
            output,
            query_start_loc,
            seq_lens,
            block_table,
            max_query_len,
            self.alibi_slopes,
            True,
            float(self.scale),
            int(self.fxp_int_bits),
            int(self.fxp_frac_bits),
            float(self.logits_soft_cap),
            int(self.window_size),
            0,  # num_kv_splits=0: auto-pick the split count
        )

        return output.view(num_tokens, self.num_heads * self.head_size)

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
