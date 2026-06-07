import torch

from vllm.model_executor.layers.layernorm import RMSNorm

from .library_ops import rms_norm_fxp as rms_norm_fxp_op
from .library_ops import rms_norm_fxp_residual as rms_norm_fxp_residual_op


class DeterministicRMSNorm(RMSNorm):
    def _forward_fxp(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # fused residual path: returns (normed, residual + x).
        if residual is not None:
            out = rms_norm_fxp_residual_op(
                x, residual, self.weight, self.variance_epsilon
            )
            return out, residual

        return rms_norm_fxp_op(x, self.weight, self.variance_epsilon)

    # vLLM uses forward_native under torch.compile and forward_cuda otherwise.
    # Bind both or it silently falls back to the batch-variant RMSNorm.
    forward_cuda = _forward_fxp
    forward_native = _forward_fxp
