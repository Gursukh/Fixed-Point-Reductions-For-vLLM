import logging
import sys

from .vllm_modules.config import get_runtime_config

logger = logging.getLogger("fxpr_vllm")


_registered = False


def _register_rms_norm() -> None:
    """Register the deterministic RMSNorm implementation, replacing the default one."""

    from vllm.model_executor.custom_op import op_registry
    import vllm.model_executor.layers.layernorm as layernorm_mod

    if "rms_norm" in op_registry:
        del op_registry["rms_norm"]

    from .vllm_modules.rms_norm import DeterministicRMSNorm

    original_rms_norm = layernorm_mod.RMSNorm
    layernorm_mod.RMSNorm = DeterministicRMSNorm

    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name.startswith("vllm.model_executor.models."):
            continue

        if getattr(mod, "RMSNorm", None) is original_rms_norm:
            setattr(mod, "RMSNorm", DeterministicRMSNorm)


def _register_quant_config() -> None:
    """Register the quantisation config for the fixed-point reductions in gemms."""

    from .vllm_modules.quantisation_config import FixedPointConfig

    # Force registration of the config class by calling a method on it.
    FixedPointConfig.get_name()


def _register_attention_backend() -> None:
    """Register the deterministic attention backend, replacing the "CUSTOM" backend enum."""

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


def _register_sampler() -> None:
    """Patch the Sampler class to use the deterministic log-softmax implementation."""

    from vllm.v1.sample.sampler import Sampler
    from .vllm_modules.sampling import deterministic_log_softmax

    if getattr(Sampler, "_fxp_logprobs_patched", False):
        return

    Sampler.compute_logprobs = staticmethod(deterministic_log_softmax)
    Sampler._fxp_logprobs_patched = True
    logger.info("Sampler log-softmax patched")


def register() -> None:
    """Install every deterministic component into vLLM.

    This is the vLLM plugin entry point (declared as
    fixed_point_reductions under vllm.general_plugins in
    pyproject.toml) and is safe to call multiple times: a module-level
    _registered flag short-circuits subsequent invocations.

    In order, this:

    1. Replaces vllm.model_executor.layers.layernorm.RMSNorm with
       :class:`DeterministicRMSNorm`, patching both the global custom-op
       registry and any already-imported model modules.
    2. Forces FixedPointConfig to register itself under the
       fixed_point_det quantisation method.
    3. Binds :class:`DeterministicAttentionBackend` to vLLM's CUSTOM
       attention-backend slot.
    4. Patches Sampler.compute_logprobs to use the deterministic
       fixed-point log-softmax.

    Raises:
        Exception: Any failure during one of the sub-registrations is logged
            and re-raised so plugin loading fails loudly.
    """
    global _registered
    if _registered:
        return
    _registered = True

    logger.info("vllm-deterministic: registering components")

    cfg = get_runtime_config()
    logger.info(
        "Runtime Config: frac_bits=%d num_kv_splits=%d fxp_int_bits=%d",
        cfg.frac_bits,
        cfg.num_kv_splits,
        cfg.fxp_int_bits,
    )

    try:
        _register_rms_norm()
        logger.info("RMSNorm registered")

        _register_quant_config()
        logger.info("Quant config registered")

        _register_attention_backend()
        logger.info("Attention backend registered")

        _register_sampler()
        logger.info("Sampler registered")
    except Exception as e:
        logger.error("Error during registration: %s", e)
        raise
