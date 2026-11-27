# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for the Ascend mamba_attn_hybrid (H-Spec) speculator dispatch
and its Ascend adaptation contracts."""

import importlib.util
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.worker.v2 import spec_decode as dispatch
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    AscendDFlashSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import (
    AscendDSparkSpeculator,
)


def _hspec_vllm_available() -> bool:
    try:
        return importlib.util.find_spec("vllm.v1.worker.gpu.spec_decode.mamba_attn_hybrid.speculator") is not None
    except ModuleNotFoundError:
        # find_spec imports parent packages; a vLLM without the H-Spec patch
        # has no mamba_attn_hybrid package at all.
        return False


def _make_spec_config(method="dspark", with_mamba=False):
    config = SimpleNamespace(
        method=method,
        use_dspark=lambda: method == "dspark",
        use_dflash=lambda: method == "dflash",
        use_eagle=lambda: False,
    )
    if with_mamba:
        config.use_mamba_attn_hybrid = lambda: method == "mamba_attn_hybrid"
    return config


def _make_vllm_config(method="dspark", with_mamba=False, **parallel):
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        **parallel,
    )
    return SimpleNamespace(
        speculative_config=_make_spec_config(method, with_mamba),
        parallel_config=parallel_config,
    )


@pytest.fixture
def stub_speculator_ctors(monkeypatch):
    """Avoid the heavy real speculator __init__ during dispatch tests."""
    created: dict[str, object] = {}

    def record(name):
        def _init(self, vllm_config, device):
            created[name] = vllm_config

        return _init

    monkeypatch.setattr(AscendDSparkSpeculator, "__init__", record("dspark"))
    return created


def test_dispatch_dspark_without_mamba_capability(monkeypatch, stub_speculator_ctors):
    """A vLLM without use_mamba_attn_hybrid must not break other methods."""

    def fail(*args, **kwargs):
        raise AssertionError("mamba_attn_hybrid dispatch must not be entered")

    monkeypatch.setattr(dispatch, "AscendMambaAttnHybridSpeculator", fail, raising=False)
    vllm_config = _make_vllm_config("dspark", with_mamba=False)
    speculator = dispatch.init_speculator(vllm_config, device=None)
    assert isinstance(speculator, AscendDSparkSpeculator)
    assert "dspark" in stub_speculator_ctors


def test_dispatch_dflash_without_mamba_capability(monkeypatch):
    """Same guard for dflash, which is dispatched after dspark."""
    monkeypatch.setattr(AscendDFlashSpeculator, "__init__", lambda self, vllm_config, device: None)
    vllm_config = _make_vllm_config("dflash", with_mamba=False)
    speculator = dispatch.init_speculator(vllm_config, device=None)
    assert isinstance(speculator, AscendDFlashSpeculator)


def test_dispatch_mamba_hybrid(monkeypatch, stub_speculator_ctors):
    """When the vLLM carries the H-Spec patch, dispatch to the Ascend class."""
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    fake_mamba = type("MambaAttnHybridSpeculator", (DSparkSpeculator,), {"__init__": lambda self, v, d: None})
    fake_utils = type("M", (), {})
    fake_upstream = SimpleNamespace(MambaAttnHybridSpeculator=fake_mamba)
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.gpu.spec_decode.mamba_attn_hybrid.speculator",
        fake_upstream,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.gpu.spec_decode.mamba_attn_hybrid.utils",
        fake_utils,
    )
    vllm_config = _make_vllm_config("mamba_attn_hybrid", with_mamba=True)
    speculator = dispatch.init_speculator(vllm_config, device=None)
    from vllm_ascend.worker.v2.spec_decode.mamba_attn_hybrid.speculator import (
        AscendMambaAttnHybridSpeculator,
    )

    assert isinstance(speculator, AscendMambaAttnHybridSpeculator)
    assert "dspark" not in stub_speculator_ctors


def test_mamba_dispatch_missing_upstream_fails_fast(monkeypatch):
    """A vLLM without the H-Spec patch yields a clear NotImplementedError."""
    vllm_config = _make_vllm_config("mamba_attn_hybrid", with_mamba=True)
    # Block both the ascend speculator and its upstream base from importing.
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.gpu.spec_decode.mamba_attn_hybrid.speculator",
        None,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.worker.v2.spec_decode.mamba_attn_hybrid.speculator",
        None,
    )
    with pytest.raises(NotImplementedError, match="H-Spec support"):
        dispatch.init_speculator(vllm_config, device=None)


def test_mamba_rejects_dcp_and_pcp():
    from vllm_ascend.worker.v2.spec_decode.mamba_attn_hybrid.speculator import (
        AscendMambaAttnHybridSpeculator,
    )

    for dcp, pcp in ((2, 1), (1, 2)):
        vllm_config = _make_vllm_config(
            "mamba_attn_hybrid",
            with_mamba=True,
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
        )
        with pytest.raises(NotImplementedError, match="DCP/PCP"):
            AscendMambaAttnHybridSpeculator(vllm_config, device=None)


def test_enforce_eager_disables_acl_graph(monkeypatch):
    """enforce_eager must force CUDAGraphMode.NONE before the graph manager
    is created (the DSpark contract this speculator inherits)."""
    from vllm.config.compilation import CUDAGraphMode

    requested_modes = []
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.dspark.speculator.DSparkSpeculator.init_cudagraph_manager",
        lambda self, mode: requested_modes.append(mode),
    )

    from vllm_ascend.worker.v2.spec_decode.mamba_attn_hybrid.speculator import (
        AscendMambaAttnHybridSpeculator,
    )

    speculator = AscendMambaAttnHybridSpeculator.__new__(AscendMambaAttnHybridSpeculator)
    speculator.speculative_config = SimpleNamespace(enforce_eager=True)
    speculator.update_stream = None
    speculator.query_cudagraph_manager = MagicMock()
    speculator.init_cudagraph_manager(CUDAGraphMode.FULL)
    assert requested_modes == [CUDAGraphMode.NONE]


def test_set_attn_mirrors_group_causal_and_preserves_group_order(monkeypatch):
    """After set_attn, _group_causal follows the drafter's dflash_causal, and
    attn_backends keeps the cache-group's original layer order."""
    from vllm.config import set_current_vllm_config

    from vllm_ascend.worker.v2.spec_decode.mamba_attn_hybrid.speculator import (
        AscendMambaAttnHybridSpeculator,
    )

    group_layers = ["model.layers.0.self_attn.attn", "model.layers.1.attn"]
    kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=group_layers)])
    layer_map = {name: MagicMock(get_attn_backend=MagicMock(return_value=name)) for name in group_layers}
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.dspark.speculator.get_layers_from_vllm_config",
        lambda *a, **k: layer_map,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.dspark.speculator._get_graph_update_backend",
        lambda groups: type("B", (), {}),
        raising=False,
    )

    speculator = AscendMambaAttnHybridSpeculator.__new__(AscendMambaAttnHybridSpeculator)
    speculator.vllm_config = SimpleNamespace()
    speculator.attn_vllm_config = SimpleNamespace()
    speculator.draft_attn_layer_names = set(group_layers)
    speculator.model = SimpleNamespace(sliding_attention_layer_names=set())
    speculator.attn_groups = []
    speculator.dflash_causal = False
    speculator._context_slot_mappings = torch.zeros(4)

    def fake_upstream_set_attn(self, *args):
        # AscendDSpark walks groups first, then MambaAttnHybrid validates the
        # recipe and decides dflash_causal; emulate that final assignment.
        self.dflash_causal = True

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.dspark.speculator.DSparkSpeculator.set_attn",
        fake_upstream_set_attn,
    )

    with set_current_vllm_config(SimpleNamespace()):
        speculator.set_attn(None, kv_cache_config, None, None, None)

    assert list(speculator.attn_backends) == group_layers
    assert speculator._group_causal is True


def test_propose_is_not_overridden_on_ascend_class():
    """The latent-seed prefill lives in the upstream speculator (dp_sync
    contract); the Ascend class must not shadow it."""
    from vllm_ascend.worker.v2.spec_decode.mamba_attn_hybrid.speculator import (
        AscendMambaAttnHybridSpeculator,
    )

    assert "propose" not in AscendMambaAttnHybridSpeculator.__dict__


@pytest.mark.skipif(not _hspec_vllm_available(), reason="requires a vLLM build with H-Spec support")
def test_propose_latent_seed_bridge(monkeypatch):
    """The upstream mamba propose fills the latent seed from fusion hidden
    states and forwards dp_sync to the DSpark proposal path."""
    from vllm.v1.worker.gpu.spec_decode.mamba_attn_hybrid.speculator import (  # type: ignore[import-not-found]
        MambaAttnHybridSpeculator,
    )

    captured = {}

    def fake_dspark_propose(self, *args, **kwargs):
        captured["last_hidden_states"] = args[3]
        captured["dp_sync"] = kwargs.get("dp_sync")
        return "ok"

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.dspark.speculator.DSparkSpeculator.propose",
        fake_dspark_propose,
    )

    speculator = MambaAttnHybridSpeculator.__new__(MambaAttnHybridSpeculator)
    speculator.latent_seed = torch.zeros(2, 4)
    input_batch = SimpleNamespace(num_reqs=2, query_start_loc=torch.tensor([0, 3, 6]))
    aux = [torch.arange(12, dtype=torch.float32).reshape(6, 2)]
    num_rejected = torch.tensor([0, 1])

    result = MambaAttnHybridSpeculator.propose(
        speculator,
        input_batch,
        attn_metadata={},
        slot_mappings={},
        last_hidden_states=torch.zeros(6, 4),
        aux_hidden_states=aux,
        num_sampled=torch.tensor([1, 1]),
        num_rejected=num_rejected,
        last_sampled=torch.tensor([1, 1]),
        next_prefill_tokens=torch.zeros(2),
        temperature=torch.ones(2),
        seeds=torch.zeros(2),
        dp_sync="sync-state",
    )

    assert result == "ok"
    # anchor[0] = 2, anchor[1] = 3 - 1 - 1 = 1
    assert torch.equal(speculator.latent_seed[0], aux[0][2])
    assert torch.equal(speculator.latent_seed[1], aux[0][1])
    assert captured["dp_sync"] == "sync-state"


@pytest.mark.skipif(not _hspec_vllm_available(), reason="requires a vLLM build with H-Spec support")
def test_propose_zero_hidden_states_without_aux(monkeypatch):
    from vllm.v1.worker.gpu.spec_decode.mamba_attn_hybrid.speculator import (  # type: ignore[import-not-found]
        MambaAttnHybridSpeculator,
    )

    captured = {}

    def fake_dspark_propose(self, *args, **kwargs):
        captured["last_hidden_states"] = args[3]
        return "ok"

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.dspark.speculator.DSparkSpeculator.propose",
        fake_dspark_propose,
    )

    speculator = MambaAttnHybridSpeculator.__new__(MambaAttnHybridSpeculator)
    speculator.latent_seed = torch.zeros(1, 4)
    speculator.hidden_states = torch.zeros(8, 4)
    MambaAttnHybridSpeculator.propose(
        speculator,
        input_batch=SimpleNamespace(num_reqs=1, query_start_loc=torch.tensor([0, 3])),
        attn_metadata={},
        slot_mappings={},
        last_hidden_states=torch.ones(6, 4),
        aux_hidden_states=None,
        num_sampled=torch.tensor([1]),
        num_rejected=torch.tensor([0]),
        last_sampled=torch.tensor([1]),
        next_prefill_tokens=torch.zeros(1),
        temperature=torch.ones(1),
        seeds=torch.zeros(1),
    )
    assert torch.equal(captured["last_hidden_states"], torch.zeros(6, 4))
