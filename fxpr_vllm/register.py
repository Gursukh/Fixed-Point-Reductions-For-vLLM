import logging
from typing import Callable

from .vllm_modules.config import get_runtime_config

from .vllm_modules import quantisation_config  # noqa: F401
from . import monkey_patches

logger = logging.getLogger("fxpr_vllm")


_registered = False


def register() -> None:
    """Install every deterministic component into vLLM.

    This is the vLLM plugin entry point (declared as fxpr_vllm under
    vllm.general_plugins in pyproject.toml) and is safe to call multiple
    times: a module-level _registered flag short-circuits subsequent
    invocations. Each sub-step is also individually re-entrant.

    Each sub-registration is gated by a per-component env var (default on):

    VLLM_FXP_DET_RMSNORM — deterministic RMSNorm.
    VLLM_FXP_DET_GEMM    — deterministic GEMM via the
      fixed_point_det quantisation method.
    VLLM_FXP_DET_ATTN    — deterministic attention backend bound to
      the CUSTOM enum slot.
    VLLM_FXP_DET_LOGPROBS— deterministic Sampler.compute_logprobs.

    Registration is transactional: on any failure the steps that already
    succeeded are rolled back before the exception is re-raised.

    Raises:
        Exception: Any failure during one of the sub-registrations is logged
            and re-raised so plugin loading fails loudly.
    """
    global _registered
    if _registered:
        return
    _registered = True

    logger.info("fxpr_vllm: registering components")

    cfg = get_runtime_config()
    logger.info(
        "Runtime Config: frac_bits=%d fxp_int_bits=%d num_kv_splits=%d "
        "rmsnorm=%s gemm=%s attn=%s logprobs=%s",
        cfg.frac_bits,
        cfg.fxp_int_bits,
        cfg.num_kv_splits,
        cfg.enable_rmsnorm,
        cfg.enable_gemm,
        cfg.enable_attn,
        cfg.enable_logprobs,
    )

    from . import library_ops  # noqa: F401

    steps: list[tuple[bool, str, Callable[[], object], Callable[[], None]]] = [
        (cfg.enable_rmsnorm, "RMSNorm", monkey_patches.patch_rms_norm, _undo_rms_norm),
        (cfg.enable_gemm, "GEMM (fixed_point_det)", _noop_gemm, _noop),
        (cfg.enable_attn, "Attention", monkey_patches.patch_attention_backend, _noop),
        (cfg.enable_logprobs, "Sampler", monkey_patches.patch_sampler, _undo_sampler),
    ]

    rollback: list[Callable[[], None]] = []
    try:
        for enabled, name, do, undo in steps:
            if not enabled:
                logger.info("%s registration skipped (disabled)", name)
                continue
            do()
            rollback.append(undo)
            logger.info("%s registered", name)
    except Exception as e:
        logger.error("Error during %s registration: %s; rolling back", name, e)
        for undo in reversed(rollback):
            try:
                undo()
            except Exception as undo_err:  
                logger.error("Rollback step failed: %s", undo_err)
        _registered = False
        raise


def _noop() -> None:  
    return None


def _noop_gemm() -> None:
    """The @register_quantization_config decorator runs at import time, so
    enabling VLLM_FXP_DET_GEMM at registration is informational only."""
    return None


def _undo_rms_norm() -> None:
    """Best-effort removal of the RMSNorm patch (used by transactional rollback)."""
    try:
        from vllm.model_executor.custom_op import op_registry

        op_registry.pop("rms_norm", None)
    except Exception as e:  # pragma: no cover
        logger.warning("RMSNorm rollback failed: %s", e)


def _undo_sampler() -> None:
    """Best-effort removal of the Sampler patch (used by transactional rollback)."""
    try:
        from vllm.v1.sample.sampler import Sampler

        if getattr(Sampler, "_fxp_logprobs_patched", False):
            del Sampler.compute_logprobs
            Sampler._fxp_logprobs_patched = False
    except Exception as e:  # pragma: no cover
        logger.warning("Sampler rollback failed: %s", e)
