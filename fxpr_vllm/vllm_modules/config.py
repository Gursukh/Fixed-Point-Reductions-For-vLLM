import logging
import os
from dataclasses import dataclass

DEFAULT_FRAC_BITS = 16
DEFAULT_FXP_INT_BITS = 32
DEFAULT_NUM_KV_SPLITS = 8

logger = logging.getLogger("fxpr_vllm")


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable.

    Args:
        name: Environment variable name.
        default: Value returned when the variable is unset, empty, or malformed.

    Returns:
        The parsed integer, or default on any failure.
    """
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring malformed %s=%r; using default %d", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean env var; values 0/false/no/off are False, anything else (set) is True."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


@dataclass(frozen=True)
class FxpRuntimeConfig:
    frac_bits: int = DEFAULT_FRAC_BITS
    fxp_int_bits: int = DEFAULT_FXP_INT_BITS
    num_kv_splits: int = DEFAULT_NUM_KV_SPLITS
    enable_rmsnorm: bool = True
    enable_gemm: bool = True
    enable_attn: bool = True
    enable_logprobs: bool = True


def load_runtime_config() -> FxpRuntimeConfig:
    """Build an :class:`FxpRuntimeConfig` from VLLM_FXP_* environment variables.

    Invalid VLLM_FXP_INT_BITS values (not in {16, 32, 64}) are logged and
    replaced with the default.

    Returns:
        A frozen runtime config reflecting the current environment.
    """
    int_bits = _env_int("VLLM_FXP_INT_BITS", DEFAULT_FXP_INT_BITS)
    if int_bits not in (16, 32, 64):
        logger.warning(
            "Invalid VLLM_FXP_INT_BITS=%d; must be 16/32/64. Using default %d.",
            int_bits,
            DEFAULT_FXP_INT_BITS,
        )
        int_bits = DEFAULT_FXP_INT_BITS
    frac_bits = _env_int("VLLM_FXP_FRAC_BITS", DEFAULT_FRAC_BITS)
    if not (0 <= frac_bits < int_bits):
        logger.warning(
            "Invalid VLLM_FXP_FRAC_BITS=%d; must satisfy 0 <= frac_bits < %d. "
            "Using default %d.",
            frac_bits,
            int_bits,
            DEFAULT_FRAC_BITS,
        )
        frac_bits = DEFAULT_FRAC_BITS
    num_kv_splits = _env_int("VLLM_FXP_NUM_KV_SPLITS", DEFAULT_NUM_KV_SPLITS)
    if num_kv_splits < 1:
        logger.warning(
            "Invalid VLLM_FXP_NUM_KV_SPLITS=%d; must be >= 1. Using default %d.",
            num_kv_splits,
            DEFAULT_NUM_KV_SPLITS,
        )
        num_kv_splits = DEFAULT_NUM_KV_SPLITS

    return FxpRuntimeConfig(
        frac_bits=frac_bits,
        fxp_int_bits=int_bits,
        num_kv_splits=num_kv_splits,
        enable_rmsnorm=_env_bool("VLLM_FXP_DET_RMSNORM", True),
        enable_gemm=_env_bool("VLLM_FXP_DET_GEMM", True),
        enable_attn=_env_bool("VLLM_FXP_DET_ATTN", True),
        enable_logprobs=_env_bool("VLLM_FXP_DET_LOGPROBS", True),
    )


_runtime_config: FxpRuntimeConfig | None = None


def get_runtime_config() -> FxpRuntimeConfig:
    """Return the process-wide runtime config, loading it from env on first use.

    Returns:
        The cached :class:`FxpRuntimeConfig` singleton.
    """
    global _runtime_config
    if _runtime_config is None:
        _runtime_config = load_runtime_config()
    return _runtime_config


def set_runtime_config(cfg: FxpRuntimeConfig) -> None:
    """Override the process-wide runtime config.

    Args:
        cfg: New config to use for all subsequent :func:`get_runtime_config` calls.
    """
    global _runtime_config
    _runtime_config = cfg
