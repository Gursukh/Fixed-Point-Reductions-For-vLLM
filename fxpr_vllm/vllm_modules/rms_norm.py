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
        cached = getattr(self, "_weight_fp32", None)
        if cached is not None:
            return cached
        w = self.weight.to(torch.float32)
        self._weight_fp32 = w
        return w

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
