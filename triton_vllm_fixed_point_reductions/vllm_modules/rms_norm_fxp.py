import logging
import os
import torch
import triton

from vllm.model_executor.custom_op import CustomOp, op_registry
from vllm.model_executor.layers.layernorm import RMSNorm

from ..fixed_point_kernels.rms_norm import rms_norm_fxp_kernel


logger = logging.getLogger("vllm_deterministic")
DEBUG_RMS = os.getenv("VLLM_FXP_DEBUG_RMS", "0") == "1"


def _launch_rms_norm_fxp(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda
    x2d = x.reshape(-1, x.shape[-1]).contiguous()
    batch, hidden = x2d.shape
    y = torch.empty_like(x2d)
    block = triton.next_power_of_2(max(hidden, 1))

    rms_norm_fxp_kernel[(batch,)](
        x2d,
        weight,
        y,
        x2d.stride(0),
        hidden,
        eps=eps,
        BLOCK=block,
    )
    return y.view_as(x) 

if "rms_norm" in op_registry:
    del op_registry["rms_norm"]

@CustomOp.register("rms_norm")
@CustomOp.register_oot(name="RMSNorm")
class DeterministicRMSNorm(RMSNorm):
    """
    Drop-in replacement for vLLM's RMSNorm.

    Inherits __init__ (weight param, epsilon, etc.) and forward_native
    from the parent. Only the CUDA path is overridden.

    Handles both shapes that appear in Qwen3:
        - (num_tokens, hidden_size)       — layer norms
        - (num_tokens, num_heads, head_size) — QK-norm in Qwen3Attention
    """

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        weight = self.weight.to(torch.float32)

        self._fxp_rms_debug_logged = True

        if residual is not None:
            new_residual = (x.to(torch.float32) + residual.to(torch.float32))
            out = _launch_rms_norm_fxp(new_residual, weight, self.variance_epsilon)
            return out.to(orig_dtype), new_residual.to(residual.dtype)

        out = _launch_rms_norm_fxp(
            x.to(torch.float32), weight, self.variance_epsilon
        )
        return out.to(orig_dtype)
