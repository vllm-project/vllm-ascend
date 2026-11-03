# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerWeightTransferEngine

MULTI_SOURCE = hasattr(NCCLTrainerWeightTransferEngine, "_broadcast_plan")


@pytest.fixture
def example():
    path = Path(__file__).resolve().parents[4] / "examples/rl/rlhf_hccl_multi_source.py"
    spec = importlib.util.spec_from_file_location("rlhf_hccl_multi_source_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not MULTI_SOURCE, reason="requires ownership metadata")
def test_stage_sources_share_global_metadata_and_iterate_only_owned_weights(example):
    pairs = [
        (name, torch.ones(2))
        for name in (
            "model.embed_tokens.weight",
            "model.layers.0.weight",
            "model.layers.1.weight",
            "model.layers.2.weight",
            "model.layers.3.weight",
            "model.norm.weight",
            "lm_head.weight",
        )
    ]
    model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=4), named_parameters=lambda: iter(pairs))
    sources = [example.StageSource(model, rank) for rank in (0, 1)]
    assert sources[0].metadata == sources[1].metadata
    for rank, source in enumerate(sources):
        assert source.metadata is source.metadata
        expected = [m.name for m in source.metadata if m.owners == (rank,)]
        assert expected
        assert [name for name, _ in source.iter_full()] == expected
    assert set(sources[0].owned_names(0)).isdisjoint(sources[1].owned_names(1))


@pytest.mark.skipif(not MULTI_SOURCE, reason="requires upstream multi-source send loop")
@pytest.mark.parametrize("failure", [False, True])
def test_example_resumes_only_when_every_source_completed(example, monkeypatch, failure):
    events = []
    monkeypatch.setattr(sys, "argv", ["example", "--rounds", "1"])
    monkeypatch.setattr(example.dist, "init_process_group", lambda *_: None)
    monkeypatch.setattr(example.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(example.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(example.dist, "broadcast_object_list", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(example.dist, "barrier", lambda: events.append("barrier"))
    monkeypatch.setattr(example.dist, "destroy_process_group", lambda: events.append("destroy"))
    monkeypatch.setattr(example.torch.npu, "set_device", lambda _: None)
    monkeypatch.setattr(example.AutoModelForCausalLM, "from_pretrained", MagicMock())
    monkeypatch.setattr(example, "get_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(example, "get_open_port", lambda: 1234)
    response = MagicMock()
    response.json.return_value = {"world_size": 2}
    monkeypatch.setattr(example.requests, "get", lambda *_args, **_kwargs: response)

    def post(url, **kwargs):
        events.append(url.rsplit("/", 1)[1])
        return response

    def send():
        events.append("send")
        if failure:
            raise RuntimeError("source failed")

    engine = SimpleNamespace(send_weights=send, shutdown=lambda: events.append("close"))
    monkeypatch.setattr(example.requests, "post", post)
    monkeypatch.setattr(example.HCCLTrainerWeightTransferEngine, "trainer_init", MagicMock(return_value=engine))
    if failure:
        with pytest.raises(RuntimeError, match="source failed"):
            example.main()
        assert events == ["pause", "barrier", "send", "close", "destroy"]
    else:
        example.main()
        assert events == ["pause", "barrier", "send", "barrier", "resume", "barrier", "close", "destroy"]


@pytest.mark.skipif(MULTI_SOURCE, reason="legacy capability rejection")
def test_example_rejects_unsupported_vllm_before_opening_a_group(example, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["example"])
    init = MagicMock()
    monkeypatch.setattr(example.dist, "init_process_group", init)
    with pytest.raises(RuntimeError, match="requires vLLM ownership"):
        example.main()
    init.assert_not_called()
