import logging

import torch
import triton

from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.layernorm import RMSNorm

from ..fixed_point_kernels.fixed_point import fixed_tl_dtype
from ..fixed_point_kernels.rms_norm import rms_norm_fxp_kernel
from .config import get_runtime_config

logger = logging.getLogger("vllm_deterministic")


def _launch_rms_norm_fxp(
    x: torch.Tensor,
    weight_fp32: torch.Tensor,
    eps: float,
    frac_bits: int,
    fxp_dtype,
) -> torch.Tensor:
    """Launch the Triton RMSNorm kernel with a fixed-point squared-sum reduction.

    Args:
        x: Input activations, shape (..., hidden). Any leading dims are
            flattened into the batch axis before launch. Must be on CUDA.
        weight_fp32: RMSNorm scale weights, shape (hidden,), dtype float32.
        eps: Variance epsilon added before the reciprocal square root.
        frac_bits: Number of fractional bits in the fixed-point Q-format used
            for the sum of squares.
        fxp_dtype: Triton integer dtype (tl.int16 / tl.int32 / tl.int64)
            used as the fixed-point accumulator.

    Returns:
        Normalised tensor with the same shape as x.
    """
    assert x.is_cuda and weight_fp32.is_cuda
    x2d = x.reshape(-1, x.shape[-1]).contiguous()
    batch, hidden = x2d.shape
    y = torch.empty_like(x2d)
    block = triton.next_power_of_2(max(hidden, 1))

    rms_norm_fxp_kernel[(batch,)](
        x2d,
        weight_fp32,
        y,
        x2d.stride(0),
        hidden,
        eps=eps,
        BLOCK=block,
        FRAC_BITS=frac_bits,
        FXP_DTYPE=fxp_dtype,
    )
    return y.view_as(x)


@CustomOp.register("rms_norm")
@CustomOp.register_oot(name="RMSNorm")
class DeterministicRMSNorm(RMSNorm):
    def _get_weight_fp32(self) -> torch.Tensor:
        """Return the layer weight upcast to float32, cached across calls.

        The cache is keyed by the underlying storage pointer so hot-swapping
        the weight (LoRA, adapter reload) invalidates the cached float32 copy.

        Returns:
            Tensor of shape (hidden,) and dtype float32.
        """
        weight = self.weight
        key = weight.data_ptr()
        cached = getattr(self, "_weight_fp32_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        w = weight.to(torch.float32)
        self._weight_fp32_cache = (key, w)
        return w

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Deterministic RMSNorm forward on CUDA, matching vLLM's fused-residual API.

        Args:
            x: Activations of shape (..., hidden).
            residual: Optional residual of shape (..., hidden). When provided,
                x + residual is computed in float32 and normalised, and the
                new residual is also returned (matching vLLM's fused form).

        Returns:
            If residual is None, the normalised tensor of shape
            (..., hidden) cast back to the input dtype. Otherwise a tuple
            (normalised, new_residual) with the same shapes.
        """
        orig_dtype = x.dtype
        weight = self._get_weight_fp32()
        cfg = get_runtime_config()
        frac_bits = cfg.frac_bits
        fxp_dtype = fixed_tl_dtype(cfg.fxp_int_bits)

        if residual is not None:
            new_residual = x.to(torch.float32) + residual.to(torch.float32)
            out = _launch_rms_norm_fxp(
                new_residual, weight, self.variance_epsilon, frac_bits, fxp_dtype
            )
            return out.to(orig_dtype), new_residual.to(residual.dtype)

        out = _launch_rms_norm_fxp(
            x.to(torch.float32), weight, self.variance_epsilon, frac_bits, fxp_dtype
        )
        return out.to(orig_dtype)
