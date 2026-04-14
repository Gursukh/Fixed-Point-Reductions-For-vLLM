import torch

from ..fixed_point_kernels.fixed_point import fixed_tl_dtype
from ..fixed_point_kernels.softmax import log_softmax_fxp
from .config import get_runtime_config


def deterministic_log_softmax(
    logits: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """Compute log-softmax deterministically via the fixed-point softmax kernel.

    Args:
        logits: CUDA tensor of arbitrary shape (..., V) where V is the
            reduction axis when dim == -1. Any floating dtype is accepted;
            the kernel upcasts internally to float32.
        dim: Axis along which to normalise. Defaults to the last axis.

    Returns:
        Tensor of the same shape and dtype as logits, containing log-softmax
        values that are bitwise-reproducible across SM/warp schedules.
    """
    assert logits.is_cuda, "Input tensor must be on CUDA device"
    cfg = get_runtime_config()
    return log_softmax_fxp(
        logits,
        fxp_dtype=fixed_tl_dtype(cfg.fxp_int_bits),
        dim=dim,
        frac_bits=cfg.frac_bits,
    )
