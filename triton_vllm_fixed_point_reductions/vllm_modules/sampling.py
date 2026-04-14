"""
Deterministic log-softmax for the sampling stage.

LogitsProcessor applies temperature scaling and then the sampler may
compute log-softmax over the vocabulary dimension. That's a reduction
(max + sum-exp over ~150k elements) subject to floating-point
non-associativity.

This module wraps the logits processor to use a fixed-point log-softmax
when the sampler requests normalised log-probs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..fixed_point_kernels.softmax import log_softmax_fxp


def deterministic_log_softmax(
    logits: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """
    Log-softmax with deterministic reductions.

    On CUDA, dispatches to the fixed-point Triton kernel
    (``log_softmax_fxp``), which accumulates ``sum(exp(x - max))`` in
    Q-format fixed point so the reduction is order-invariant.

    For CPU tensors, falls back to an fp64 computation — not fully
    associative, but reproducible on a single device for a fixed shape.
    """
    if logits.is_cuda:
        return log_softmax_fxp(logits, dim=dim)

    orig_dtype = logits.dtype
    logits_f64 = logits.to(torch.float64)

    max_val = logits_f64.max(dim=dim, keepdim=True).values
    shifted = logits_f64 - max_val
    exp_shifted = shifted.exp()
    sum_exp = exp_shifted.sum(dim=dim, keepdim=True)
    log_sum_exp = sum_exp.log()

    result = shifted - log_sum_exp
    return result.to(orig_dtype)


def patch_sampler_log_softmax() -> None:
    """
    Replace ``Sampler.compute_logprobs`` in vLLM v1 with a version that
    routes through the deterministic fixed-point log-softmax kernel.
    """
    from vllm.v1.sample.sampler import Sampler

    def _det_compute_logprobs(logits: torch.Tensor) -> torch.Tensor:
        return deterministic_log_softmax(logits.to(torch.float32), dim=-1)

    Sampler.compute_logprobs = staticmethod(_det_compute_logprobs)