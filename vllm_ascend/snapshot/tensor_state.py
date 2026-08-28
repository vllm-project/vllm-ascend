# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Iterator

import torch
import torch.nn as nn
from vllm.logger import logger


def persist_tensor_attributes(module: nn.Module, names: Iterable[str]) -> None:
    for name in names:
        tensor = getattr(module, name)
        delattr(module, name)
        module.register_buffer(name, tensor)


def persist_tensor_lists(module: nn.Module, names: Iterable[str]) -> None:
    for name in names:
        for index, tensor in enumerate(getattr(module, name)):
            module.register_buffer(f"_snapshot_{name}_{index}", tensor)


def set_persistent_tensor(module: nn.Module, name: str, tensor: torch.Tensor) -> torch.Tensor:
    if name in module._buffers:
        module._buffers[name] = tensor
    else:
        module.register_buffer(name, tensor)
    return module._buffers[name]


def _iter_derived_state_owners(model: nn.Module) -> Iterator[tuple[str, object]]:
    seen_ids: set[int] = set()
    for name, module in model.named_modules():
        for suffix, owner in (("", module), (".impl", getattr(module, "impl", None))):
            if owner is None or id(owner) in seen_ids:
                continue
            seen_ids.add(id(owner))
            yield f"{name}{suffix}", owner


def reset_runtime_tensor_state(owners: Iterable[object]) -> int:
    reset = 0
    seen_ids: set[int] = set()
    for owner in owners:
        if id(owner) in seen_ids:
            continue
        seen_ids.add(id(owner))
        reset_state = getattr(owner, "reset_snapshot_runtime_state", None)
        if callable(reset_state):
            reset_state()
            reset += 1
    return reset


def reset_model_runtime_tensor_state(models: Iterable[nn.Module | None]) -> int:
    owners = (owner for model in models if model is not None for _, owner in _iter_derived_state_owners(model))
    return reset_runtime_tensor_state(owners)


def restore_derived_tensor_state(model: nn.Module, act_dtype: torch.dtype, label: str) -> None:
    restored = 0

    for _, owner in _iter_derived_state_owners(model):
        restore = getattr(owner, "restore_snapshot_derived_state", None)
        if not callable(restore):
            continue
        restore(act_dtype)
        restored += 1

    logger.info(
        "[restore model] [%s] reloaded non-persistent derived weights for %d modules",
        label,
        restored,
    )
    if restored == 0:
        logger.warning(
            "[restore model] [%s] no non-persistent derived-weight reload targets found; "
            "attention decode may still use stale derived weights",
            label,
        )


def restore_global_tensor_state(
    model: nn.Module,
    hf_config: object,
    device: torch.device,
) -> None:
    from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPMetadataBuilder
    from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
    from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
    from vllm_ascend.ops.rotary_embedding import reload_cos_and_sin_after_restore

    restored: list[str] = []
    if AscendDSAMetadataBuilder.reload_hadamard_after_restore(hf_config, device):
        restored.append("dsa.hadamard")
    if AscendDSACPMetadataBuilder.reload_hadamard_after_restore(hf_config, device):
        restored.append("dsa_cp.hadamard")
    if AscendSFAImpl.reload_hadamard_after_restore(device):
        restored.append("sfa.hadamard")
    if reload_cos_and_sin_after_restore(model):
        restored.append("mla_rope.cos_sin")
    logger.info(
        "[restore model] rebuilt global non-persistent state: %s",
        restored if restored else "none",
    )
