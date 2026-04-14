import logging
import os
import sys
from dataclasses import dataclass


DEFAULT_FRAC_BITS = 16
DEFAULT_NUM_KV_SPLITS = 8
DEFAULT_FXP_INT_BITS = 32

logger = logging.getLogger("vllm_fixed_point_reductions")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring malformed %s=%r; using default %d", name, raw, default)
        return default


@dataclass(frozen=True)
class FxpRuntimeConfig:
    frac_bits: int = DEFAULT_FRAC_BITS
    num_kv_splits: int = DEFAULT_NUM_KV_SPLITS
    fxp_int_bits: int = DEFAULT_FXP_INT_BITS


def load_runtime_config() -> FxpRuntimeConfig:
    int_bits = _env_int("VLLM_FXP_INT_BITS", DEFAULT_FXP_INT_BITS)
    if int_bits not in (16, 32, 64):
        logger.warning(
            "Invalid VLLM_FXP_INT_BITS=%d; must be 16/32/64. Using default %d.",
            int_bits,
            DEFAULT_FXP_INT_BITS,
        )
        int_bits = DEFAULT_FXP_INT_BITS
    return FxpRuntimeConfig(
        frac_bits=_env_int("VLLM_FXP_FRAC_BITS", DEFAULT_FRAC_BITS),
        num_kv_splits=_env_int("VLLM_FXP_NUM_KV_SPLITS", DEFAULT_NUM_KV_SPLITS),
        fxp_int_bits=int_bits,
    )


_runtime_config: FxpRuntimeConfig | None = None


def get_runtime_config() -> FxpRuntimeConfig:
    global _runtime_config
    if _runtime_config is None:
        _runtime_config = load_runtime_config()
    return _runtime_config


def set_runtime_config(cfg: FxpRuntimeConfig) -> None:
    global _runtime_config
    _runtime_config = cfg


_registered = False


def _register_rms_norm() -> None:

    from vllm.model_executor.custom_op import op_registry
    import vllm.model_executor.layers.layernorm as layernorm_mod

    if "rms_norm" in op_registry:
        del op_registry["rms_norm"]

    from .vllm_modules.rms_norm_fxp import DeterministicRMSNorm

    original_rms_norm = layernorm_mod.RMSNorm
    layernorm_mod.RMSNorm = DeterministicRMSNorm

    patched_modules = 0
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name.startswith("vllm.model_executor.models."):
            continue
        if getattr(mod, "RMSNorm", None) is original_rms_norm:
            setattr(mod, "RMSNorm", DeterministicRMSNorm)
            patched_modules += 1

    logger.info("RMSNorm patched (preloaded_model_modules=%d)", patched_modules)


def _register_quant_config() -> None:

    from .vllm_modules.quantisation_config import FixedPointConfig

    logger.info("Quant config registered: %s", FixedPointConfig.get_name())


def _register_attention_backend() -> None:
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    from .vllm_modules.attention_backend import DeterministicAttentionBackend

    backend_path = (
        f"{DeterministicAttentionBackend.__module__}."
        f"{DeterministicAttentionBackend.__qualname__}"
    )

    register_backend(
        AttentionBackendEnum.CUSTOM,
        class_path=backend_path,
    )
    logger.info(
        "Attention backend registered under CUSTOM (%s). "
        "Activate with VLLM_ATTENTION_BACKEND=CUSTOM.",
        backend_path,
    )


def _register_sampler() -> None:
    import torch
    from vllm.v1.sample.sampler import Sampler

    from .vllm_modules.sampling import deterministic_log_softmax

    if getattr(Sampler, "_fxp_logprobs_patched", False):
        return

    def _det_compute_logprobs(logits: torch.Tensor) -> torch.Tensor:
        return deterministic_log_softmax(logits.to(torch.float32), dim=-1)

    Sampler.compute_logprobs = staticmethod(_det_compute_logprobs)
    Sampler._fxp_logprobs_patched = True
    logger.info("Sampler log-softmax patched")


def register() -> None:

    global _registered
    if _registered:
        return
    _registered = True

    logger.info("vllm-deterministic: registering components")

    cfg = get_runtime_config()
    logger.info(
        "  frac_bits=%d num_kv_splits=%d fxp_int_bits=%d",
        cfg.frac_bits,
        cfg.num_kv_splits,
        cfg.fxp_int_bits,
    )

    _register_rms_norm()
    _register_quant_config()
    _register_attention_backend()
    _register_sampler()
