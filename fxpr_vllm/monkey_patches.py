from __future__ import annotations

import logging
import sys

logger = logging.getLogger("fxpr_vllm")


def patch_rms_norm() -> None:
    from vllm.model_executor.custom_op import op_registry, op_registry_oot
    import vllm.model_executor.layers.layernorm as layernorm_mod

    from .rms_norm import DeterministicRMSNorm

    DeterministicRMSNorm.name = "rms_norm"
    op_registry["rms_norm"] = DeterministicRMSNorm
    op_registry_oot["RMSNorm"] = DeterministicRMSNorm
    layernorm_mod.RMSNorm = DeterministicRMSNorm

    # Models that already did `from ... import RMSNorm` hold the old class and
    # won't see this patch.
    early = [m for m in sys.modules if m.startswith("vllm.model_executor.models.")]
    if early:
        logger.warning(
            "patched RMSNorm but %d model module(s) already imported; "
            "those still use the original",
            len(early),
        )


def patch_attention_backend() -> None:
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    from .attention_backend import DeterministicAttentionBackend

    backend_path = (
        f"{DeterministicAttentionBackend.__module__}."
        f"{DeterministicAttentionBackend.__qualname__}"
    )
    register_backend(AttentionBackendEnum.CUSTOM, class_path=backend_path)
