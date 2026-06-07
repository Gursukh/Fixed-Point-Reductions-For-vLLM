from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.parameter import ModelWeightParameter

from .library_ops import gemm_fxp
from .config import get_runtime_config
from .warmup import warmup_gemm, warmup_rms_norm


class FixedPointConfig(QuantizationConfig):
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "FixedPointConfig()"

    @classmethod
    def get_name(cls) -> str:
        return "fixedpoint"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FixedPointConfig":
        return cls()

    def get_quant_method(
        self,
        layer: nn.Module,
        prefix: str,
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase):
            return FixedPointLinearMethod(self)
        # ParallelLMHead subclasses this. cuBLAS picks GEMV at M=1 and GEMM at
        # M>=2, which flips tokens on Blackwell; gemm_fxp stays batch-invariant.
        # On by default; opt out via FXPR_DISABLE_LM_HEAD to use vLLM's default.
        if isinstance(layer, VocabParallelEmbedding):
            if get_runtime_config().enable_lm_head:
                return FixedPointEmbeddingMethod(self)
            return None
        return None

    def get_scaled_act_names(self) -> list[str]:
        return []


class FixedPointLinearMethod(QuantizeMethodBase):
    def __init__(self, config: FixedPointConfig) -> None:
        self.config = config
        cfg = get_runtime_config()
        self.fxp_int_bits = cfg.fxp_int_bits
        self.fxp_frac_bits = cfg.fxp_frac_bits

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: Sequence[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:

        weight_loader = extra_weight_attrs.pop("weight_loader")
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # weight is (out, in); GEMM wants (K, N) = (in, out).
        with torch.no_grad():
            w_t = layer.weight.data.t().contiguous()

        layer.weight_native = w_t
        del layer.weight
        layer.register_parameter("weight", None)

        # warm gemm_fxp for this shape now so we don't JIT on first request.
        warmup_gemm(layer.weight_native, self.fxp_int_bits, self.fxp_frac_bits)
        # warm rms_norm too; one binary per (hidden, dtype) covers all batches.
        K = layer.weight_native.shape[0]
        warmup_rms_norm(K, layer.weight_native.dtype, layer.weight_native.device)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return gemm_fxp(
            x, layer.weight_native, bias, self.fxp_int_bits, self.fxp_frac_bits
        )


class FixedPointEmbeddingMethod(QuantizeMethodBase):
    """Quant method for VocabParallelEmbedding / ParallelLMHead.

    Embedding lookups stay on F.embedding; only the lm_head matmul uses gemm_fxp.
    """

    def __init__(self, config: FixedPointConfig) -> None:
        self.config = config
        cfg = get_runtime_config()
        self.fxp_int_bits = cfg.fxp_int_bits
        self.fxp_frac_bits = cfg.fxp_frac_bits

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: Sequence[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        # plain Parameter is fine; the loader ignores ModelWeightParameter dims.
        weight = nn.Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # keep weight for F.embedding; weight_native is its transpose for the
        # lm_head matmul.
        with torch.no_grad():
            layer.weight_native = layer.weight.data.t().contiguous()

        # warm gemm_fxp for the lm_head shape now.
        warmup_gemm(layer.weight_native, self.fxp_int_bits, self.fxp_frac_bits)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return gemm_fxp(
            x, layer.weight_native, bias, self.fxp_int_bits, self.fxp_frac_bits
        )

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(input_, layer.weight)
