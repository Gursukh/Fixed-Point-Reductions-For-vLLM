import logging

import torch

from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.layernorm import RMSNorm

from ..library_ops import rms_norm_fxp as rms_norm_fxp_op
from .config import get_runtime_config

logger = logging.getLogger("vllm_deterministic")


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
        fxp_int_bits = cfg.fxp_int_bits

        if residual is not None:
            new_residual = x.to(torch.float32) + residual.to(torch.float32)
            out = rms_norm_fxp_op(
                new_residual, weight, self.variance_epsilon, frac_bits, fxp_int_bits
            )
            return out.to(orig_dtype), new_residual.to(residual.dtype)

        out = rms_norm_fxp_op(
            x.to(torch.float32), weight, self.variance_epsilon, frac_bits, fxp_int_bits
        )
        return out.to(orig_dtype)
