# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression cover for the host-offload aware layerwise reload.

The reload's materialization helper is replaced by a recording stand-in, so "an
offloaded parameter got a device allocation" is an assertion a CPU-only runner can
make.  The patch is loaded from its file: importing ``vllm_ascend.patch.worker``
would apply every worker patch in the process.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

PATCH_FILE = Path(__file__).resolve().parents[3] / "vllm_ascend" / "patch" / "worker" / "patch_offload_reload.py"

# Producer/consumer contract: ``AscendParallelEngramEmbedding`` publishes the host
# range under this attribute and the reload reads it back.
HOST_ATTR = "_host_offload_storage"


@pytest.fixture(scope="module")
def patch():
    """Load ``patch_offload_reload`` on its own."""
    spec = importlib.util.spec_from_file_location("_patch_offload_reload_under_test", PATCH_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


class _Upstream:
    """Stand-in for vLLM's reload helpers, recording device materializations."""

    def __init__(self, meta_module: str, layerwise_module: str) -> None:
        self.allocated: list[str] = []
        self.upstream_copies = 0
        self.place_calls = 0
        self.meta: Any = ModuleType(meta_module)
        self.layerwise: Any = ModuleType(layerwise_module)
        self.meta.materialize_layer = self._materialize_layer
        self.meta.get_layer_params_buffers = self._get_layer_params_buffers
        # ``layerwise`` relays the helper, which is why the reload resolves it
        # through that namespace rather than through ``meta``.
        self.layerwise.materialize_layer = self._relay_materialize_layer
        self.layerwise.get_layer_params_buffers = self._get_layer_params_buffers
        self.layerwise._copy_and_restore_kernel_tensors = self._upstream_copy_and_restore
        self.layerwise._place_kernel_tensors = self._place_kernel_tensors

    def _get_layer_params_buffers(self, layer):
        return dict(layer.named_parameters(recurse=False)), dict(layer.named_buffers(recurse=False))

    def _materialize_layer(self, layer, info) -> None:
        parameters, buffers = self._get_layer_params_buffers(layer)
        for name, tensor in {**parameters, **buffers}.items():
            if not tensor.is_meta:
                continue
            self.allocated.append(name)
            layer.register_parameter(
                name,
                nn.Parameter(torch.empty(tuple(tensor.size()), dtype=tensor.dtype), requires_grad=False),
            )

    def _relay_materialize_layer(self, layer, info) -> None:
        self.meta.materialize_layer(layer, info)

    def _upstream_copy_and_restore(self, layer, info) -> None:
        self.upstream_copies += 1
        parameters, _ = info.kernel_tensors
        for name, param in parameters.items():
            source = getattr(layer, name)
            if not source.is_meta:
                param.data.copy_(source)
        self._place_kernel_tensors(layer, info)

    def _place_kernel_tensors(self, layer, info) -> None:
        self.place_calls += 1


@pytest.fixture
def reload_env(patch, monkeypatch):
    """Install the stand-in upstream modules and make ``install()`` runnable again."""
    env = _Upstream(patch._META_MODULE, patch._LAYERWISE_MODULE)
    monkeypatch.setitem(sys.modules, patch._META_MODULE, env.meta)
    monkeypatch.setitem(sys.modules, patch._LAYERWISE_MODULE, env.layerwise)
    monkeypatch.setattr(patch, "_installed", False)
    # The attribute name is the contract; the producer is covered by its own test,
    # so the reload tests do not need to import the model package.
    monkeypatch.setattr(patch, "_host_offload_storage_attr", lambda: HOST_ATTR)
    return env


def _info(*, parameters=None) -> SimpleNamespace:
    return SimpleNamespace(
        kernel_tensors=(parameters if parameters is not None else {}, {}),
        kernel_non_persistent_buffers=set(),
        loaded_weights=[],
    )


def _layer(*, host: torch.Tensor | None = None, with_plain: bool = True, shape: tuple[int, int] = (4, 8)):
    layer = nn.Module()
    if with_plain:
        layer.register_parameter("plain", nn.Parameter(torch.empty(2, 3, device="meta")))
    offloaded = nn.Parameter(torch.empty(shape, device="meta"))
    if host is not None:
        offloaded.__dict__[HOST_ATTR] = host
    layer.register_parameter("offloaded", offloaded)
    return layer


def test_without_the_patch_an_offloaded_parameter_is_materialized_on_the_device(reload_env):
    layer = _layer(host=torch.zeros(4, 8))

    reload_env.meta.materialize_layer(layer, _info())

    assert reload_env.allocated == ["plain", "offloaded"]


def test_install_rebuilds_an_offloaded_parameter_on_its_host_storage(patch, reload_env):
    host = torch.full((4, 8), 7.0)
    layer = _layer(host=host)
    info = _info()

    patch.install()
    # The reload calls the relay; rebinding only ``meta`` would leave this one upstream.
    reload_env.layerwise.materialize_layer(layer, info)

    assert reload_env.allocated == ["plain"], "the offloaded table must not get a device allocation"
    assert layer.offloaded.data_ptr() == host.data_ptr(), "the parameter must alias the range the kernel reads"
    assert layer.offloaded.requires_grad is False
    assert info.host_offload_params == {"offloaded": host}
    assert float(layer.offloaded[0, 0]) == 7.0


def test_a_layer_of_only_offloaded_parameters_skips_the_upstream_helper(patch, reload_env):
    layer = _layer(host=torch.zeros(4, 8), with_plain=False)

    patch.install()
    reload_env.layerwise.materialize_layer(layer, _info())

    assert reload_env.allocated == []


def test_a_layout_change_falls_back_to_device_materialization(patch, reload_env):
    # A host range that no longer matches the parameter cannot be aliased.
    layer = _layer(host=torch.zeros(1, 1))
    info = _info()

    patch.install()
    reload_env.layerwise.materialize_layer(layer, info)

    assert "offloaded" in reload_env.allocated
    assert not hasattr(info, "host_offload_params")


def test_the_update_is_not_copied_back_out_of_a_device_copy(patch, reload_env):
    host = torch.zeros(4, 8)
    layer = _layer(host=host)
    kernel_plain = torch.zeros(2, 3)
    info = _info(parameters={"plain": kernel_plain})

    patch.install()
    reload_env.layerwise.materialize_layer(layer, info)
    layer.plain.data.fill_(3.0)
    reload_env.layerwise._copy_and_restore_kernel_tensors(layer, info)

    assert reload_env.upstream_copies == 0, "an offloaded layer must not take the device round trip"
    assert reload_env.place_calls == 1, "the kernel tensors are still restored"
    assert not hasattr(info, "host_offload_params"), "the per-layer state must not leak into the next layer"
    assert torch.equal(kernel_plain, torch.full((2, 3), 3.0))


def test_the_copy_hook_delegates_when_no_parameter_was_offloaded(patch, reload_env):
    layer = _layer()
    info = _info(parameters={"plain": torch.zeros(2, 3)})

    patch.install()
    reload_env.layerwise._copy_and_restore_kernel_tensors(layer, info)

    assert reload_env.upstream_copies == 1


def test_a_parameter_that_stopped_aliasing_its_host_range_is_rejected(patch, reload_env):
    host = torch.zeros(4, 8)
    layer = _layer(host=host)
    info = _info()

    patch.install()
    reload_env.layerwise.materialize_layer(layer, info)
    # Something replaced the parameter after materialization: the update written
    # into the host range would no longer be the one the kernel reads.
    layer.register_parameter("offloaded", nn.Parameter(torch.zeros(4, 8)))

    with pytest.raises(AssertionError, match="no longer aliases"):
        reload_env.layerwise._copy_and_restore_kernel_tensors(layer, info)


def test_install_is_idempotent(patch, reload_env):
    patch.install()
    first = reload_env.layerwise.materialize_layer

    patch.install()

    assert reload_env.layerwise.materialize_layer is first


def test_install_degrades_when_upstream_has_no_reload_package(patch, reload_env, monkeypatch):
    monkeypatch.setitem(sys.modules, patch._LAYERWISE_MODULE, None)
    monkeypatch.setattr(patch, "_installed", False)
    before = reload_env.meta.materialize_layer

    patch.install()

    assert patch._installed is True
    assert reload_env.meta.materialize_layer is before, "an unsupported upstream keeps its behaviour"


def test_the_producer_publishes_the_attribute_the_patch_reads():
    pytest.importorskip("vllm_ascend.models.deepseek_v41.engram.embedding")
    from vllm_ascend.models.deepseek_v41.engram import embedding

    assert embedding.HOST_OFFLOAD_STORAGE_ATTR == HOST_ATTR
