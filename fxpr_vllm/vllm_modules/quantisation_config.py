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

from ..fixed_point_kernels.fixed_point import fixed_tl_dtype
from ..fixed_point_kernels.gemm import launch_gemm_fxp
from .config import DEFAULT_FRAC_BITS, get_runtime_config

logger = logging.getLogger("fxpr_vllm")

_INT_BITS_FOR_TL = {
    "int16": 16,
    "int32": 32,
    "int64": 64,
}


def _int_bits_of(fxp_dtype) -> int:
    name = getattr(fxp_dtype, "name", str(fxp_dtype))
    if name not in _INT_BITS_FOR_TL:
        raise ValueError(f"Unsupported fxp dtype {fxp_dtype!r}")
    return _INT_BITS_FOR_TL[name]


@register_quantization_config("fixed_point_det")
class FixedPointConfig(QuantizationConfig):
    def __init__(self, frac_bits: int = DEFAULT_FRAC_BITS) -> None:
        """Create a fixed-point quantisation config.

        Args:
            frac_bits: Number of fractional bits used by the GEMM accumulator.
        """
        fxp_dtype = fixed_tl_dtype(get_runtime_config().fxp_int_bits)
        int_bits = _int_bits_of(fxp_dtype)
        if not isinstance(frac_bits, int) or not (0 <= frac_bits < int_bits):
            raise ValueError(
                f"frac_bits must be an int in [0, {int_bits}); got {frac_bits!r}"
            )
        self.frac_bits = frac_bits

    def __repr__(self) -> str:
        """Return a human-readable representation including frac_bits."""
        return f"FixedPointConfig(frac_bits={self.frac_bits})"

    @classmethod
    def get_name(cls) -> str:
        """Return the vLLM quantisation method name used in --quantization."""
        return "fixed_point_det"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        """Return the activation dtypes accepted by the fixed-point GEMM."""
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        """Return the minimum CUDA compute capability (major * 10 + minor)."""
        return 70

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        """Return the list of HF config filenames consumed by this method (none)."""
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FixedPointConfig":
        """Build a :class:`FixedPointConfig` from a serialised dict.

        Args:
            config: Raw dict (e.g. HF quantization_config); frac_bits overrides the runtime default when present.

        Returns:
            A configured :class:`FixedPointConfig`.
        """
        frac_bits = config.get("frac_bits", get_runtime_config().frac_bits)
        return cls(frac_bits=frac_bits)

    def get_quant_method(
        self,
        layer: nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        """Return a :class:`FixedPointLinearMethod` for linear layers, else None.

        Args:
            layer:  The module being quantised.
            prefix: Qualified name of the layer within the model (unused).

        Returns:
            A linear-method wrapper for :class:`LinearBase` layers, else None so vLLM uses its default.
        """
        if isinstance(layer, LinearBase):
            return FixedPointLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        """Return activation names that require scaling (none for this method)."""
        return []


class FixedPointLinearMethod(QuantizeMethodBase):
    def __init__(self, config: FixedPointConfig) -> None:
        """Bind this linear method to its parent quantisation config.

        Args:
            config: The :class:`FixedPointConfig` providing frac_bits.
        """
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
        """Allocate a weight parameter of shape (sum(output_partition_sizes), input_size_per_partition).

        Args:
            layer:                    Linear layer receiving the parameter.
            input_size_per_partition: Input features on this tensor-parallel shard.
            output_partition_sizes:   Output features per fused shard; summed to form the weight's first dim.
            input_size:               Full input feature count (unused; kept for the vLLM API).
            output_size:              Full output feature count (unused).
            params_dtype:             Dtype of the allocated weight.
            **extra_weight_attrs:     Extra loader hooks; weight_loader is forwarded to :class:`ModelWeightParameter`.
        """
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
        """Pre-transpose the weight once after checkpoint loading.

        Attaches layer.weight_t of shape (input_size_per_partition, total_output_size).
        Weights are kept in bfloat16 (matching checkpoint precision); the GEMM kernel
        widens to fp32 internally via tl.load(...).to(tl.float32).

        Args:
            layer: Linear layer whose weight has just been loaded.
        """
        with torch.no_grad():
            w_t = layer.weight.data.to(torch.bfloat16).t().contiguous()
        layer.weight_t = w_t

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the deterministic fixed-point GEMM for this linear layer.

        Args:
            layer: Linear layer holding weight_t (preferred) or the raw weight.
            x:     (..., in_features) input activations; leading dims are flattened for the matmul.
            bias:  (out_features,) optional bias.

        Returns:
            (..., out_features) tensor in the same dtype as x.
        """
        orig_dtype = x.dtype
        w_t = getattr(layer, "weight_t", None)
        if w_t is None:
            w_t = layer.weight.data.to(torch.bfloat16).t().contiguous()

        # Cast input to match weight dtype; the GEMM kernel widens both to fp32
        # internally, so no precision is lost relative to the original fp32 path.
        x2d = x.reshape(-1, x.shape[-1])
        if x2d.dtype != w_t.dtype:
            x2d = x2d.to(w_t.dtype)
        x2d = x2d.contiguous()

        fxp_dtype = fixed_tl_dtype(get_runtime_config().fxp_int_bits)
        out = launch_gemm_fxp(
            x2d, w_t, frac_bits=self.config.frac_bits, fxp_dtype=fxp_dtype
        )
        out = out.view(*x.shape[:-1], out.shape[-1])

        if bias is not None:
            out = out + bias.to(torch.float32)

        return out.to(orig_dtype)
