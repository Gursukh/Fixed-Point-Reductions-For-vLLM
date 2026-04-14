import logging
import os
from dataclasses import dataclass


DEFAULT_FRAC_BITS = 16
DEFAULT_NUM_KV_SPLITS = 8
DEFAULT_FXP_INT_BITS = 32

logger = logging.getLogger("fxpr_vllm")


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
