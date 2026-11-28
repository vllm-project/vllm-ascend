# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Initialization-time GDN RecoverSSM providers with a common cache ABI."""

import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from importlib import import_module, util
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class GDNRecoverSSMOps:
    name: str
    verify: Callable[..., torch.Tensor]
    commit: Callable[..., None] | None = None
    compact: Callable[..., None] | None = None
    commit_step: Callable[..., None] | None = None


def _ascendc_verify(A_log, a, b, dt_bias, q, k, v, *, spec_query_len, native_op=None, **kwargs):
    # Optional native imports stay worker-local. The installed FLA overlay
    # must expose these public interfaces; a missing provider must not silently
    # turn an AscendC measurement into a Triton measurement.
    if native_op is None:
        from fla_npu.ops.ascendc import gdn_recoverssm_verify as native_op

    checkpoint = kwargs.pop("checkpoint_state")
    replay = kwargs.pop("replay_cache")
    offsets = kwargs.pop("query_start_loc")
    indices = kwargs.pop("state_indices")
    if replay.shape[2] != spec_query_len:
        raise ValueError("RecoverSSM replay capacity differs from the speculative window")
    # Qwen4Exp prepares these model-owned constants after loading/reloading.
    # Other GDN models retain the functional dtype conversion fallback.
    return native_op(A_log.float(), a, b, dt_bias.float(), q, k, v, checkpoint, replay, offsets, indices, **kwargs)


def _ascendc_commit(state, replay, table, indices, lens, final, boundary, recovery, native_op=None, **kwargs):
    if native_op is None:
        from fla_npu.ops.ascendc import gdn_recoverssm_commit as native_op

    batch = indices.numel()
    return native_op(
        state, replay, table, indices, lens[:batch], final[:batch], boundary[:batch], recovery[:batch], **kwargs
    )


def _ascendc_compact(conv, *args, native_op=None, **kwargs):
    if native_op is None:
        from fla_npu.ops.ascendc import compact_conv_state as native_op

    # Four address/stride table rows, then active indices and plan vectors.
    tables, indices, vectors = args[:4], args[4], args[5:]
    batch = indices.numel()
    return native_op(conv, *tables, indices, *(vector[:batch] for vector in vectors), **kwargs)


def _load_native_package(root: str | None):
    if root is None:
        return import_module("fla_npu.ops.ascendc")
    # FLA delta wheels may contain only the recovery operators. Keep their
    # stable-ABI namespace and OPP separate from the base prefill/conv package.
    path = Path(root).resolve()
    library = path / "opp/vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so"
    if not (path / "__init__.py").is_file() or not library.is_file():
        raise ValueError(f"Incomplete GDN RecoverSSM native package: {path}")
    namespace = "_vllm_ascend_gdn_fla_" + sha256(str(path).encode()).hexdigest()[:12]
    if namespace not in sys.modules:
        spec = util.spec_from_file_location(namespace, path / "__init__.py", submodule_search_locations=[str(path)])
        assert spec is not None
        module = util.module_from_spec(spec)
        # This is FLA's wheel-layer adapter context: importing the selected
        # child adapter uses these libraries without executing the base
        # package controller or replacing its installed Python modules.
        module._WHEEL_LAYER_CONTEXT = {"namespace": namespace, "libraries": [str(library)]}
        sys.modules[namespace] = module
    return import_module(namespace + ".ops.ascendc")


def load_gdn_recoverssm_backend(name: str, *, native_root: str | None = None) -> GDNRecoverSSMOps:
    """Resolve once during worker/context construction, not in every layer."""
    if name == "triton":
        from vllm_ascend.ops.triton.gdn.gdn_recoverssm_commit import gdn_recoverssm_commit
        from vllm_ascend.ops.triton.gdn.gdn_recoverssm_verify import gdn_recoverssm_verify
        from vllm_ascend.ops.triton.mamba.compact_conv_state import compact_conv_state

        return GDNRecoverSSMOps(name, gdn_recoverssm_verify, gdn_recoverssm_commit, compact_conv_state)
    if name not in ("tilelang", "ascendc"):
        raise ValueError(f"Unknown GDN RecoverSSM backend: {name}")
    from vllm_ascend.device.device_config import is_950

    if not is_950():
        raise ValueError(f"GDN RecoverSSM backend {name!r} requires Ascend A5")
    if name == "tilelang":
        try:
            import tilelang  # noqa: F401 - fail before model execution if the optional runtime is absent

            from vllm_ascend.ops.tilelang_recoverssm.full_backend import commit_step, verify_step
        except ImportError as exc:
            raise ImportError(
                "TileLang GDN RecoverSSM requires the optional operators from PR #22552 and TileLang"
            ) from exc
        return GDNRecoverSSMOps(name, verify_step, commit_step=commit_step)
    try:
        native = _load_native_package(native_root)
        verify, commit, compact = native.gdn_recoverssm_verify, native.gdn_recoverssm_commit, native.compact_conv_state
    except (ImportError, AttributeError) as exc:
        raise ImportError("AscendC GDN RecoverSSM requires FLA PR #1319 Verify, Commit and Compact exports") from exc
    return GDNRecoverSSMOps(
        name,
        partial(_ascendc_verify, native_op=verify),
        partial(_ascendc_commit, native_op=commit),
        partial(_ascendc_compact, native_op=compact),
    )


def initialize_gdn_recoverssm_backend(layer: Any) -> None:
    if not layer.use_gdn_recoverssm:
        return
    from vllm_ascend.ascend_config import get_ascend_config

    config = get_ascend_config()
    name = config.gdn_recoverssm_backend
    if name != "triton" and (layer.head_k_dim, layer.head_v_dim) != (128, 128):
        raise ValueError(f"GDN RecoverSSM backend {name!r} requires K=V=128")
    layer._gdn_recoverssm_verify = load_gdn_recoverssm_backend(
        name, native_root=config.gdn_recoverssm_native_root
    ).verify
    layer._gdn_recoverssm_backend = name


def prepare_gdn_recoverssm_constants(model: torch.nn.Module) -> None:
    """Refresh model-owned constants after loading; no global Tensor cache."""
    for layer in model.modules():
        if getattr(layer, "_gdn_recoverssm_backend", None) != "ascendc":
            continue
        for name in ("A_log", "dt_bias"):
            value = getattr(layer, name).detach().float()
            layer.register_buffer(f"_recoverssm_{name}_fp32", value, persistent=False)
