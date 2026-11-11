#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Host metadata regression tests for the opt-in anchor-union experiment.

Run `pytest tests/ut/test_dflash_anchor_state.py`.
"""

import json
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.dflash_topm_state import DFlashTopMState

TOKENS_PER_REQUEST = 8
EXPERT_COUNT = 256
ROUTE_TOP_K = 6
PROTECTED_ROWS = 3


def make_state(tmp_path, **config):
    path = tmp_path / "anchor_config.json"
    values = {
        "verify_block_size": TOKENS_PER_REQUEST,
        "num_experts": EXPERT_COUNT,
        "route_top_k": ROUTE_TOP_K,
        "scoring_func": "sqrtsoftplus",
        "expected_router_layers": 40,
        "protected_rows": PROTECTED_ROWS,
        "suffix_pool_top_k": 2,
        "fused_rows": [8, 512],
    }
    values.update(config)
    path.write_text(json.dumps(values))
    return DFlashTopMState(str(path))


@pytest.mark.parametrize("backend", ["unknown", "anchor_union_topm"])
def test_reject_unvalidated_backend(tmp_path, backend):
    with pytest.raises(ValueError):
        make_state(tmp_path, backend=backend)


def test_legacy_m_setting_is_ignored(tmp_path):
    state = make_state(tmp_path, backend="anchor_union_fused_native", m=17)
    assert "m" not in vars(state)


def test_baseline_allocates_no_role_buffers(tmp_path):
    state = make_state(tmp_path, backend="baseline")
    state.prepare(TOKENS_PER_REQUEST, [(0, TOKENS_PER_REQUEST)], "cpu")
    assert state.current_roles is None
    assert not state.roles_gpu and not state.workspaces


@pytest.fixture
def cpu_role_state(tmp_path, monkeypatch):
    original_zeros = torch.zeros

    def unpinned_zeros(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_zeros(*args, **kwargs)

    # Isolate host layout semantics; pinned NPU transfers are tested on device.
    monkeypatch.setattr(torch, "zeros", unpinned_zeros)
    return make_state(tmp_path, backend="anchor_union_fused_native")


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
def test_roles_and_pointer_reuse(cpu_role_state, batch):
    state = cpu_role_state
    size = batch * TOKENS_PER_REQUEST
    segments = [(start, TOKENS_PER_REQUEST) for start in range(0, size, TOKENS_PER_REQUEST)]
    state.prepare(size, segments, "cpu")
    expected = torch.full((size,), 2, dtype=torch.int32)
    for start, _ in segments:
        expected[start : start + PROTECTED_ROWS] = 1
    assert torch.equal(state.current_roles, expected)
    pointer = state.current_roles.data_ptr()
    state.prepare(size, [(0, 8)], "cpu")
    expected.zero_()
    expected[:PROTECTED_ROWS] = 1
    expected[PROTECTED_ROWS:8] = 2
    assert torch.equal(state.current_roles, expected)
    assert state.current_roles.data_ptr() == pointer
    state.prepare(size, [], "cpu")
    context = SimpleNamespace(dflash_graph_layout=None, dflash_verify_rows=())
    state.begin(context)
    assert context.dflash_topm_roles is None


@pytest.mark.parametrize("segment", [(-1, 8), (0, 0), (8, 16)])
def test_invalid_segment_rejected(cpu_role_state, segment):
    with pytest.raises(ValueError, match="outside padded token buffer"):
        cpu_role_state.prepare(TOKENS_PER_REQUEST, [segment], "cpu")


def test_reference_requires_matching_roles(cpu_role_state):
    with pytest.raises(ValueError, match="must match"):
        cpu_role_state.route_reference(
            torch.zeros(TOKENS_PER_REQUEST, EXPERT_COUNT),
            torch.zeros(TOKENS_PER_REQUEST // 2),
            top_k=ROUTE_TOP_K,
            renormalize=True,
            routed_scaling_factor=1.5,
            e_score_correction_bias=None,
        )


def test_dspark_config_is_exposed(cpu_role_state):
    assert cpu_role_state.verify_block_size == 8
    assert cpu_role_state.num_experts == 256
    assert cpu_role_state.route_top_k == 6
    assert cpu_role_state.scoring_func == "sqrtsoftplus"
    assert cpu_role_state.expected_router_layers == 40
    logits = torch.zeros(8, 256, dtype=torch.float32)
    assert cpu_role_state.supports_fused_route(
        logits,
        top_k=6,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        e_score_correction_bias=torch.zeros(256),
    )


def test_dspark_512_contract_and_padding_roles(cpu_role_state):
    state = cpu_role_state
    state.prepare(512, [(16, TOKENS_PER_REQUEST)], "cpu")
    assert torch.count_nonzero(state.current_roles).item() == TOKENS_PER_REQUEST
    assert torch.count_nonzero(state.current_roles == 1).item() == PROTECTED_ROWS
    assert torch.count_nonzero(state.current_roles == 2).item() == (TOKENS_PER_REQUEST - PROTECTED_ROWS)
    assert torch.count_nonzero(state.current_roles[:16]).item() == 0
    assert torch.count_nonzero(state.current_roles[24:]).item() == 0
    assert state.supports_fused_route(
        torch.zeros(512, 256, dtype=torch.float32),
        top_k=6,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        e_score_correction_bias=torch.zeros(256),
    )


def test_original_qwen_fused_contract_remains_available(tmp_path):
    state = make_state(
        tmp_path,
        backend="anchor_union_fused_native",
        num_experts=128,
        route_top_k=8,
        scoring_func="softmax",
        expected_router_layers=48,
        fused_rows=[16],
    )
    logits = torch.zeros(16, 128, dtype=torch.bfloat16)
    assert state.supports_fused_route(
        logits,
        top_k=8,
        scoring_func="softmax",
        routed_scaling_factor=1.0,
        e_score_correction_bias=None,
    )


def test_dspark_reference_uses_bias_for_selection_and_raw_scores_for_weights(
    cpu_role_state,
):
    state = cpu_role_state
    state.prepare(TOKENS_PER_REQUEST, [(0, TOKENS_PER_REQUEST)], "cpu")
    logits = torch.linspace(-2, 2, EXPERT_COUNT).repeat(TOKENS_PER_REQUEST, 1)
    logits[3:] = logits[3:].flip(1)
    bias = torch.linspace(0.25, -0.25, EXPERT_COUNT)

    weights, ids = state.route_reference(
        logits,
        state.current_roles,
        top_k=ROUTE_TOP_K,
        renormalize=True,
        routed_scaling_factor=1.5,
        e_score_correction_bias=bias,
    )

    raw_scores = torch.nn.functional.softplus(logits).sqrt()
    selection_scores = raw_scores + bias
    initial_ids = selection_scores.topk(ROUTE_TOP_K, dim=-1).indices
    pool = torch.zeros(EXPERT_COUNT, dtype=torch.bool)
    pool[initial_ids[:PROTECTED_ROWS].flatten()] = True
    pool[initial_ids[PROTECTED_ROWS:, :2].flatten()] = True
    restricted = selection_scores.masked_fill(
        (state.current_roles[:, None] == 2) & ~pool[None, :],
        torch.finfo(selection_scores.dtype).min,
    )
    expected_ids = restricted.topk(ROUTE_TOP_K, dim=-1).indices
    expected_weights = raw_scores.gather(1, expected_ids)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
    expected_weights *= 1.5

    assert torch.equal(ids, expected_ids.to(torch.int32))
    assert torch.equal(weights, expected_weights)
