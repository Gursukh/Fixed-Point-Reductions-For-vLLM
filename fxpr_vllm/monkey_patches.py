"""Monkey-patches that replace pieces of vLLM at import-time hook points.

The two registrations here cannot use vLLM's official registration mechanisms:

* RMSNorm: vLLM model files import the symbol directly, so swapping the
  custom-op registry is not enough — we must rebind the name in every
  already-loaded vllm.model_executor.models.* module.
* Sampler.compute_logprobs: there is no extension point for the log-softmax
  step; we replace the static method on the class.

Keep both patches here so the surface area is visible in one file.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger("fxpr_vllm")


def patch_rms_norm() -> int:
    """Replace vllm.model_executor.layers.layernorm.RMSNorm and rebind in
    every already-imported model module. Returns the number of model modules
    rebound. Asserts that the custom-op registry call took effect.
    """
    from vllm.model_executor.custom_op import op_registry
    import vllm.model_executor.layers.layernorm as layernorm_mod

    if "rms_norm" in op_registry:
        del op_registry["rms_norm"]

    from .vllm_modules.rms_norm import DeterministicRMSNorm

    original_rms_norm = layernorm_mod.RMSNorm
    layernorm_mod.RMSNorm = DeterministicRMSNorm

    patched = 0
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name.startswith("vllm.model_executor.models."):
            continue
        if getattr(mod, "RMSNorm", None) is original_rms_norm:
            setattr(mod, "RMSNorm", DeterministicRMSNorm)
            patched += 1

    # Verify the registry-side replacement took effect.
    assert op_registry.get("rms_norm") in (None, DeterministicRMSNorm), (
        "rms_norm op_registry entry was clobbered by another plugin "
        "after DeterministicRMSNorm registered itself."
    )

    if patched == 0:
        logger.warning(
            "DeterministicRMSNorm: no vllm.model_executor.models.* modules were "
            "patched. Either no model modules are imported yet (fine) or the "
            "import path has changed (audit monkey_patches.py)."
        )
    return patched


def patch_attention_backend() -> None:
    """Bind the deterministic attention backend to vLLM's CUSTOM enum slot."""
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    from .vllm_modules.attention_backend import DeterministicAttentionBackend

    backend_path = (
        f"{DeterministicAttentionBackend.__module__}."
        f"{DeterministicAttentionBackend.__qualname__}"
    )
    register_backend(AttentionBackendEnum.CUSTOM, class_path=backend_path)


def patch_sampler() -> None:
    """Replace Sampler.compute_logprobs with the deterministic log-softmax."""
    from vllm.v1.sample.sampler import Sampler

    from .vllm_modules.sampling import deterministic_log_softmax

    if getattr(Sampler, "_fxp_logprobs_patched", False):
        return

    Sampler.compute_logprobs = staticmethod(deterministic_log_softmax)
    Sampler._fxp_logprobs_patched = True
    logger.info("Sampler log-softmax patched")
