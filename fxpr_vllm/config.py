import logging
import os
from dataclasses import dataclass

DEFAULT_FXP_INT_BITS = 32
DEFAULT_FXP_FRAC_BITS = 16

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


def _env_flag(name: str) -> bool:
    return os.getenv(name) == "1"


@dataclass(frozen=True)
class FxpRuntimeConfig:
    fxp_int_bits: int = DEFAULT_FXP_INT_BITS
    fxp_frac_bits: int = DEFAULT_FXP_FRAC_BITS
    enable_rms_norm: bool = True
    enable_lm_head: bool = True
    disable_attention_warmup: bool = False
    disable_gemm_warmup: bool = False
    disable_rms_norm_warmup: bool = False


def load_runtime_config() -> FxpRuntimeConfig:
    int_bits = _env_int("FXPR_INT_BITS", DEFAULT_FXP_INT_BITS)
    if int_bits not in (16, 32, 64):
        logger.warning(
            "Invalid FXPR_INT_BITS=%d; must be 16/32/64. Using default %d.",
            int_bits,
            DEFAULT_FXP_INT_BITS,
        )
        int_bits = DEFAULT_FXP_INT_BITS
    # Storage width is 16/32/64, but frac_bits is free within that: any value in
    # [1, int_bits) works (need at least the sign bit). This is the accuracy vs
    # range knob.
    frac_bits = _env_int("FXPR_FRAC_BITS", DEFAULT_FXP_FRAC_BITS)
    if not (1 <= frac_bits < int_bits):
        logger.warning(
            "Invalid FXPR_FRAC_BITS=%d; must be an integer in [1, %d) for "
            "FXPR_INT_BITS=%d. Using default %d.",
            frac_bits,
            int_bits,
            int_bits,
            DEFAULT_FXP_FRAC_BITS,
        )
        frac_bits = DEFAULT_FXP_FRAC_BITS

    return FxpRuntimeConfig(
        fxp_int_bits=int_bits,
        fxp_frac_bits=frac_bits,
        enable_rms_norm=not _env_flag("FXPR_DISABLE_RMS_NORM"),
        enable_lm_head=not _env_flag("FXPR_DISABLE_LM_HEAD"),
        disable_attention_warmup=_env_flag("FXPR_DISABLE_ATTENTION_WARMUP"),
        disable_gemm_warmup=_env_flag("FXPR_DISABLE_GEMM_WARMUP"),
        disable_rms_norm_warmup=_env_flag("FXPR_DISABLE_RMS_NORM_WARMUP"),
    )


_runtime_config: FxpRuntimeConfig | None = None


def get_runtime_config() -> FxpRuntimeConfig:
    global _runtime_config
    if _runtime_config is None:
        _runtime_config = load_runtime_config()
    return _runtime_config
