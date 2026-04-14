import torch

from ..fixed_point_kernels.fixed_point import fixed_tl_dtype
from ..fixed_point_kernels.softmax import log_softmax_fxp
from .config import get_runtime_config


def deterministic_log_softmax(
    logits: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """Compute log softmax in a deterministic way, regardless of whether the input is on CPU or GPU."""
    assert logits.is_cuda, "Input tensor must be on CUDA device"

    cfg = get_runtime_config()
    return log_softmax_fxp(
        logits,
        fxp_dtype=fixed_tl_dtype(cfg.fxp_int_bits),
        dim=dim,
        frac_bits=cfg.frac_bits,
    )
