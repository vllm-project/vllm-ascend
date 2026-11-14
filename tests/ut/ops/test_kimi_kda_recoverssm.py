# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for request-indexed RecoverSSM records."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.triton.kda import recoverssm


class RecordingKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


def make_layer(*, pages=11, rows=4, heads=2, dim=4, capacity=4):
    conv = torch.zeros(pages, 3 * heads * dim, capacity + 2, dtype=torch.bfloat16)
    state = torch.zeros(pages, heads, dim, dim)
    correction = torch.zeros(rows, heads, capacity, dim)
    key_gate = torch.zeros(rows, heads, capacity, 2 * dim, dtype=torch.bfloat16)
    records = SimpleNamespace(correction=correction, key_gate=key_gate)
    return SimpleNamespace(
        kv_cache=(conv, state),
        recoverssm_records=records,
        A_log=torch.zeros(heads),
        dt_bias=torch.zeros(heads * dim),
        gate_lower_bound=-5.0,
        local_num_heads=heads,
        head_dim=dim,
    )


@pytest.fixture(autouse=True)
def device_helpers(monkeypatch):
    monkeypatch.setattr(
        recoverssm,
        "triton",
        SimpleNamespace(next_power_of_2=lambda x: 1 << (x - 1).bit_length(), cdiv=lambda x, y: (x + y - 1) // y),
    )
    monkeypatch.setattr(recoverssm, "is_conv_state_dim_first", lambda: True)
    monkeypatch.setattr(recoverssm, "init_device_properties_triton", lambda: None)
    monkeypatch.setattr(recoverssm, "is_950", lambda: False)
    monkeypatch.setattr(recoverssm, "get_ub_size_bytes", lambda: 192 * 1024)


def test_context_binds_request_records_and_detects_rebinding():
    layers = [make_layer() for _ in range(2)]
    context = recoverssm.KDARecoverSSMCommitContext.create(layers, spec_query_len=4, max_num_reqs=4)
    assert context.is_bound_to(layers)
    assert context.correction_caches[0] is layers[0].recoverssm_records.correction
    old = layers[0].kv_cache
    layers[0].kv_cache = (old[0].clone(), *old[1:])
    assert not context.is_bound_to(layers)
    layers[0].kv_cache = old
    layers[0].recoverssm_records.key_gate = layers[0].recoverssm_records.key_gate.clone()
    assert not context.is_bound_to(layers)


def test_context_rejects_mixed_layout_and_insufficient_scratch():
    layer = make_layer()
    layer.kv_cache = (*layer.kv_cache, torch.zeros(1))
    with pytest.raises(ValueError, match="convolution and checkpoint caches"):
        recoverssm.KDARecoverSSMCommitContext.create([make_layer(), layer], spec_query_len=4, max_num_reqs=4)
    with pytest.raises(ValueError, match="record capacity"):
        recoverssm.KDARecoverSSMCommitContext.create([make_layer(rows=2)], spec_query_len=4, max_num_reqs=4)


def test_verify_accepts_raw_beta_and_request_records(monkeypatch):
    layer = make_layer()
    records = layer.recoverssm_records
    q = torch.randn(1, 5, 2, 4, dtype=torch.bfloat16)
    beta = torch.randn(1, 5, 2, dtype=q.dtype)
    offsets = torch.tensor([0, 3, 5], dtype=torch.int32)
    indices = torch.tensor([7, 3], dtype=torch.int32)
    kernel = RecordingKernel()
    monkeypatch.setattr(recoverssm, "_kda_recoverssm_verify_kernel", kernel)
    out = recoverssm.kda_recoverssm_verify(
        q,
        q,
        q,
        q,
        beta,
        layer.A_log,
        layer.dt_bias,
        -5.0,
        layer.kv_cache[1],
        records.correction,
        records.key_gate,
        offsets,
        indices,
        4,
    )
    assert out.shape == q.shape
    assert len(kernel.calls) == 1
    assert kernel.calls[0][1][4] is beta


def test_commit_launches_for_request_records(monkeypatch):
    context = recoverssm.KDARecoverSSMCommitContext.create([make_layer()], spec_query_len=4, max_num_reqs=4)
    plan, compact, commit = RecordingKernel(), RecordingKernel(), RecordingKernel()
    monkeypatch.setattr(recoverssm, "_prepare_commit_plan_kernel", plan)
    monkeypatch.setattr(recoverssm, "_compact_conv_state_kernel", compact)
    monkeypatch.setattr(recoverssm, "_commit_kda_state_kernel", commit)
    context.commit(
        torch.tensor([2, 1], dtype=torch.int32),
        torch.tensor([7, 3], dtype=torch.int32),
        torch.tensor([0, 3, 5], dtype=torch.int32),
    )
    assert len(plan.calls) == len(compact.calls) == len(commit.calls) == 1
