# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.deepseek_v41.engram import npu
from vllm_ascend.models.hy_v4_flash import engram


@pytest.fixture
def shared_embedding(monkeypatch):
    group = SimpleNamespace(rank_in_group=0, world_size=2, cpu_group=object())
    buffers = {}

    class SharedBuffer:
        def __init__(self, shape, dtype, device, peers):
            key = (tuple(shape), dtype)
            self.tensor = buffers.setdefault(key, torch.empty(shape, dtype=dtype, device="cpu"))
            self.shm = SimpleNamespace(name="test-shared-table")
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(engram, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(engram, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(engram, "get_dp_group", lambda: group)
    monkeypatch.setattr(engram, "in_the_same_node_as", lambda _: [True, True])
    monkeypatch.setattr(npu, "SharedUvaBuffer", SharedBuffer)
    monkeypatch.setattr(torch.npu, "current_device", lambda: 0)
    original = engram.FlashEmbedding._allocate_storage

    def allocate(self, shape, dtype, leaf):
        if not shape[0]:
            return torch.empty(shape, dtype=dtype, device="cpu")
        return original(self, shape, dtype, leaf)

    monkeypatch.setattr(engram.FlashEmbedding, "_allocate_storage", allocate)

    def gather(output, error, group):
        output[:] = [error, None]

    monkeypatch.setattr(engram.dist, "all_gather_object", gather)
    return group


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("fp8", [False, True])
def test_shared_table_preserves_shards_and_lookup_without_dp_collectives(shared_embedding, monkeypatch, tp_rank, fp8):
    group = shared_embedding
    monkeypatch.setattr(engram, "get_tensor_model_parallel_rank", lambda: tp_rank)
    weight = (torch.arange(36).reshape(18, 2).float() / 8).to(torch.float8_e4m3fn if fp8 else torch.bfloat16)
    scales = torch.arange(1, 19).bfloat16() / 4
    leader = engram.FlashEmbedding([3, 4, 5, 6], 2, None, True, True)
    leader.load_weight("weight", weight)
    if fp8:
        leader.load_weight("weight_scale_inv", scales)
    group.rank_in_group = 1
    follower = engram.FlashEmbedding([3, 4, 5, 6], 2, None, True, True)
    # A peer must attach to the writer's values instead of overwriting them.
    follower.load_weight("weight", (weight.float() + 1).to(weight.dtype))
    if fp8:
        follower.load_weight("weight_scale_inv", scales + 1)
    assert leader.embedding.weight.data_ptr() == follower.embedding.weight.data_ptr()
    expected = weight[leader.vocab_start : leader.vocab_end]
    stored = expected.view(torch.uint8) if fp8 else expected
    torch.testing.assert_close(leader.embedding.weight, stored, atol=0, rtol=0)
    if fp8:
        torch.testing.assert_close(
            follower.embedding.weight_scale_inv,
            scales[leader.vocab_start : leader.vocab_end],
            atol=0,
            rtol=0,
        )

    def forbid_collective(*args, **kwargs):
        pytest.fail("DP-shared lookup must not communicate across replicas")

    monkeypatch.setattr(engram.dist, "all_gather_object", forbid_collective)
    hashes = torch.tensor([[0, 3, 7, 12], [2, 6, 11, 17]])
    ids = hashes[:, leader.head_start : leader.head_end]
    expected_rows = weight.float()[ids]
    if fp8:
        expected_rows *= scales.float()[ids, None]
    torch.testing.assert_close(leader(hashes), expected_rows, atol=0, rtol=0)
    torch.testing.assert_close(follower(hashes), expected_rows, atol=0, rtol=0)


def test_shared_table_propagates_writer_load_failure(shared_embedding, monkeypatch):
    embedding = engram.FlashEmbedding([3, 4, 5, 6], 2, None, True, True)

    def fail(*args):
        raise ValueError("writer copy failed")

    monkeypatch.setattr(engram, "default_weight_loader", fail)
    with pytest.raises(RuntimeError, match="writer copy failed"):
        embedding.load_weight("weight", torch.zeros(18, 2, dtype=torch.bfloat16))


def test_shared_table_rejects_unsupported_topologies(shared_embedding, monkeypatch):
    with pytest.raises(ValueError, match="CPU offload"):
        engram.FlashEmbedding([3, 4], 2, None, False, True)
    shared_embedding.world_size = 1
    with pytest.raises(ValueError, match="DP > 1"):
        engram.FlashEmbedding([3, 4], 2, None, True, True)
    shared_embedding.world_size = 2
    monkeypatch.setattr(engram, "in_the_same_node_as", lambda _: [True, False])
    with pytest.raises(ValueError, match="same node"):
        engram.FlashEmbedding([3, 4], 2, None, True, True)


def test_shared_table_cleanup_keeps_alias_on_unregister_failure(shared_embedding, monkeypatch):
    embedding = engram.FlashEmbedding([3, 4], 2, None, True, True)
    embedding.load_weight("weight", torch.zeros(7, 2, dtype=torch.bfloat16))
    buffer = embedding._shared_buffers["weight"]
    pointer = embedding.embedding.weight.data_ptr()

    def fail():
        raise RuntimeError("unregister failed")

    with monkeypatch.context() as patcher:
        patcher.setattr(buffer, "close", fail)
        with pytest.raises(RuntimeError, match="unregister failed"):
            embedding.close_host_offload()
    assert embedding.embedding.weight.data_ptr() == pointer
    assert embedding._shared_buffers["weight"] is buffer
    embedding.close_host_offload()
    embedding.close_host_offload()
    assert buffer.closed
    assert embedding.embedding.weight.numel() == 0
    assert not embedding._shared_buffers


def test_private_table_load_preserves_parameter_device(shared_embedding):
    embedding = engram.FlashEmbedding([3, 4], 2, None, False, False)
    weight = torch.arange(14).reshape(7, 2).bfloat16()
    with torch.device("meta"):
        embedding.load_weight("weight", weight)
    assert embedding.embedding.weight.device.type == "cpu"
    torch.testing.assert_close(embedding.embedding.weight, weight[:3], atol=0, rtol=0)
