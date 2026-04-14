from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

import torch
import torch.nn as nn

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.layers.quantization import register_quantization_config

from ..fixed_point_kernels.fixed_point import fxp_tl_dtype
from ..fixed_point_kernels.gemm import launch_gemm_fxp
from ..register import DEFAULT_FRAC_BITS, get_runtime_config

logger = logging.getLogger("vllm_fixed_point_reductions")


@register_quantization_config("fixed_point_det")
class FixedPointConfig(QuantizationConfig):
    def __init__(self, frac_bits: int = DEFAULT_FRAC_BITS) -> None:
        self.frac_bits = frac_bits

    def __repr__(self) -> str:
        return f"FixedPointConfig(frac_bits={self.frac_bits})"

    @classmethod
    def get_name(cls) -> str:
        return "fixed_point_det"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FixedPointConfig":
        frac_bits = config.get("frac_bits", get_runtime_config().frac_bits)
        return cls(frac_bits=frac_bits)

    def get_quant_method(
        self,
        layer: nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            return FixedPointLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class FixedPointLinearMethod(QuantizeMethodBase):
    def __init__(self, config: FixedPointConfig) -> None:
        self.config = config

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
        total_output_size = sum(output_partition_sizes)

        weight = ModelWeightParameter(
            data=torch.empty(
                total_output_size,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=extra_weight_attrs.get("weight_loader"),
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: nn.Module) -> None:

        with torch.no_grad():
            w_fp32_t = layer.weight.data.to(torch.float32).t().contiguous()
        layer.weight_fp32_t = w_fp32_t

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x2d = x.reshape(-1, x.shape[-1]).to(torch.float32).contiguous()

        w_t = getattr(layer, "weight_fp32_t", None)
        if w_t is None:
            w_t = layer.weight.data.to(torch.float32).t().contiguous()

        fxp_dtype = fxp_tl_dtype(get_runtime_config().fxp_int_bits)
        out = launch_gemm_fxp(
            x2d, w_t, frac_bits=self.config.frac_bits, fxp_dtype=fxp_dtype
        )
        out = out.view(*x.shape[:-1], out.shape[-1])

        if bias is not None:
            out = out + bias.to(torch.float32)

        return out.to(orig_dtype)
