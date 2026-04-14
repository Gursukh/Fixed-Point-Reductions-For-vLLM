import torch

from ..fixed_point_kernels.fixed_point import fxp_tl_dtype
from ..fixed_point_kernels.softmax import log_softmax_fxp
from ..register import get_runtime_config


def deterministic_log_softmax(
    logits: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    if logits.is_cuda:
        cfg = get_runtime_config()
        return log_softmax_fxp(
            logits,
            fxp_dtype=fxp_tl_dtype(cfg.fxp_int_bits),
            dim=dim,
            frac_bits=cfg.frac_bits,
        )

    orig_dtype = logits.dtype
    logits_f64 = logits.to(torch.float64)

    max_val = logits_f64.max(dim=dim, keepdim=True).values
    shifted = logits_f64 - max_val
    sum_exp = shifted.exp().sum(dim=dim, keepdim=True)
    log_sum_exp = sum_exp.log()

    return (shifted - log_sum_exp).to(orig_dtype)


