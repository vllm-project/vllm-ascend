# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Identical replay visibility is independent of physical cache groups."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.ut.attention.test_dsa_v41_cp import _batch
from tests.ut.attention.test_dsa_v41_cp import builders as pcp_builders
from vllm_ascend.attention import dsa_v41

builders = pcp_builders


@pytest.mark.parametrize("different_floor", [False, True])
@pytest.mark.parametrize("isolated_override", [False, True])
@pytest.mark.parametrize("plan_ratios", [(0, 2, 1), (2, 1)])
@pytest.mark.parametrize("sparse_windows", [False, True])
def test_replay_plans_follow_visibility_not_physical_cache_group(
    builders, monkeypatch, different_floor, isolated_override, plan_ratios, sparse_windows
):
    template = builders(world=1, max_seqs=4)["swa"]
    template.vllm_config.model_config.hf_text_config.compress_ratios = [0, 2, 1]
    owners = [
        dsa_v41.AscendDSAV41SWAMetadataBuilder(
            template.kv_cache_spec, template.layer_names, template.vllm_config, torch.device("cpu")
        )
        for _ in range(2)
    ]
    visible = Mock(wraps=dsa_v41.build_replay_swa_visible_lens)
    monkeypatch.setattr(dsa_v41, "build_replay_swa_visible_lens", visible)
    plan = Mock(return_value=torch.zeros(dsa_v41.V41_METADATA_BUFFER_SIZE, dtype=torch.int32))
    window_calls = []

    def build_window(positions, window_size, *, visible_lengths, indices_output, lengths_output):
        # Model the sparse backend's separate, padded persistent length output.
        # Sharing must use the visibility contract rather than this buffer's identity.
        window_calls.append(visible_lengths)
        lengths_output.zero_()
        lengths_output[: visible_lengths.shape[0]].copy_(visible_lengths)
        indices_output.fill_(-1)
        return indices_output, lengths_output

    for owner in owners:
        owner._uses_a5_packed_cache = True
        owner._a5_smla_metadata = torch.zeros(dsa_v41.V41_METADATA_BUFFER_SIZE, dtype=torch.int32)
        owner._a5_smla_length_rows = torch.empty(64, 1, dtype=torch.int32)
        if sparse_windows:
            owner._a5_causal_swa_indices = torch.empty(64, 1, owner._window_size, dtype=torch.int32)
            owner._a5_causal_swa_lengths = torch.empty(64, 1, dtype=torch.int32)
        owner._device_backend = SimpleNamespace(
            mixed_quant_sparse_flash_mla_metadata=plan, build_window_indices=build_window
        )
        owner.enable_device_metadata()
    common = _batch(2, 0, "prefill").local
    floor = torch.zeros_like(common.seq_lens)
    batch_cache: dict[str, object] = {}
    results, tasks = [], []
    for index, owner in enumerate(owners):
        view = floor[:]
        if index and different_floor:
            view = common.seq_lens - (common.query_start_loc[1:] - common.query_start_loc[:-1])
        results.append(
            owner.build(
                0,
                common.replace(replay_start=view, pcp_context=None),
                common_v41_metadata={},
                common_v41_batch_metadata=batch_cache,
                v41_smla_override=isolated_override,
                smla_plan_ratios=plan_ratios,
            )
        )
        tasks.extend(owner.take_device_metadata_tasks())
    for task in tasks:
        task.run()
    expected = 2 if different_floor else 1
    assert visible.call_count == expected
    if sparse_windows:
        assert len(window_calls) == 2
        for value, lengths in zip(results, window_calls):
            assert value.ori_topk_length is not lengths
            torch.testing.assert_close(value.ori_topk_length[: lengths.shape[0]], lengths)
            assert torch.count_nonzero(value.ori_topk_length[lengths.shape[0] :]) == 0
    share_plan = not different_floor and not isolated_override
    assert plan.call_count == len(plan_ratios) * (1 if share_plan else 2)
    for value in results:
        assert tuple(value.a5_smla_plans) == plan_ratios
        assert value.smla_metadata is value.a5_smla_plans.get(0)
    for ratio in plan_ratios:
        assert (results[0].a5_smla_plans[ratio] is results[1].a5_smla_plans[ratio]) == share_plan
    if not different_floor:
        torch.testing.assert_close(results[0].ori_topk_length, results[1].ori_topk_length)
