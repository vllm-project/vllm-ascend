# SPDX-License-Identifier: Apache-2.0
"""Exercise PP negotiation against the installed upstream layout resolver."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def resolver(monkeypatch):
    utils = pytest.importorskip("vllm.v1.attention.backends.utils")
    original = utils.resolve_kv_cache_layout
    # Load only this patch, without importing all platform initialization code.
    path = Path(__file__).parents[4] / "vllm_ascend/patch/platform/patch_kv_cache_layout.py"
    core = SimpleNamespace(resolve_kv_cache_layout=original)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", core)
    monkeypatch.setattr(utils, "resolve_kv_cache_layout", original)
    monkeypatch.setattr(utils.envs, "VLLM_KV_CACHE_LAYOUT", None)
    monkeypatch.setattr(utils, "get_kv_connector_cache_layout", lambda _: None)
    spec = importlib.util.spec_from_file_location("test_layout_patch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert core.resolve_kv_cache_layout is module.resolve_kv_cache_layout
    return module.resolve_kv_cache_layout, utils


def config():
    return SimpleNamespace(cache_config=SimpleNamespace(kv_cache_layout=None))


def test_pp_dspark_stages_choose_common_layout(resolver):
    resolve, _ = resolver
    target = ["LBNHC", "LBHNC", "BLNHC", "BLHNC", "BHLNC", "LHBNC"]
    draft = ["BLHNC", "LBHNC"]
    workers = [target] * 24 + [draft] * 8
    cfg = config()
    assert resolve(cfg, workers).name == "LBHNC"
    assert cfg.cache_config.kv_cache_layout == "LBHNC"


def test_worker_preference_orders_may_differ(resolver):
    resolve, _ = resolver
    assert resolve(config(), [["BLHNC", "LBHNC"], ["LBHNC", "BLHNC"]]).name == "BLHNC"


@pytest.mark.parametrize("workers", [[], [[]], [["BLHNC"], []], [["BLHNC"], ["LBHNC"]]])
def test_empty_worker_intersection_rejected(resolver, workers):
    resolve, _ = resolver
    with pytest.raises(ValueError, match="No .*KV cache layout"):
        resolve(config(), workers)


@pytest.mark.parametrize("requested", ["BLHNC", "LBHNC", "HND"])
def test_explicit_layout_and_alias_remain_supported(resolver, monkeypatch, requested):
    resolve, utils = resolver
    monkeypatch.setattr(utils.envs, "VLLM_KV_CACHE_LAYOUT", requested)
    actual = resolve(config(), [["LBNHC", "LBHNC", "BLHNC"], ["BLHNC", "LBHNC"]])
    assert actual.name == ("LBHNC" if requested == "HND" else requested)


def test_explicit_layout_must_be_supported_on_every_worker(resolver, monkeypatch):
    resolve, utils = resolver
    monkeypatch.setattr(utils.envs, "VLLM_KV_CACHE_LAYOUT", "LBNHC")
    with pytest.raises(ValueError, match="does not satisfy every supported set"):
        resolve(config(), [["LBNHC", "BLHNC"], ["BLHNC"]])


@pytest.mark.parametrize("requested", [None, "LHBNC"])
def test_mixed_cache_shapes_still_require_block_compact_layout(resolver, monkeypatch, requested):
    resolve, utils = resolver
    specs = [
        SimpleNamespace(num_heads=1, num_states=1, page_size_bytes=4096),
        SimpleNamespace(num_heads=8, num_states=2, page_size_bytes=4096),
    ]
    monkeypatch.setattr(utils.envs, "VLLM_KV_CACHE_LAYOUT", requested)
    workers = [["LHBNC", "BLHNC"], ["BLHNC", "LHBNC"]]
    if requested is None:
        assert resolve(config(), workers, specs).name == "BLHNC"
    else:
        with pytest.raises(ValueError, match="does not satisfy every supported set"):
            resolve(config(), workers, specs)


def test_no_block_compact_candidate_rejected(resolver):
    resolve, _ = resolver
    specs = [
        SimpleNamespace(num_heads=1, num_states=1, page_size_bytes=4096),
        SimpleNamespace(num_heads=8, num_states=2, page_size_bytes=4096),
    ]
    with pytest.raises(ValueError, match="mixed HNC shapes"):
        resolve(config(), [["LHBNC", "BHLNC"], ["LHBNC"]], specs)


def test_incompatible_connector_preference_falls_back(resolver, monkeypatch):
    resolve, utils = resolver
    monkeypatch.setattr(utils, "get_kv_connector_cache_layout", lambda _: "LBNHC")
    assert resolve(config(), [["LBNHC", "BLHNC"], ["BLHNC"]]).name == "BLHNC"
