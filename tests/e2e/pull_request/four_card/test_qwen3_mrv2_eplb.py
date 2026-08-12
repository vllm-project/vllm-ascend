# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import hashlib
import os
from typing import Any
from unittest.mock import patch

import pytest

from tests.e2e.conftest import DPVllmRunner, wait_until_npu_memory_free

MODEL = os.environ.get("QWEN3_MRV2_EPLB_MODEL_PATH", "vllm-ascend/Qwen3-30B-A3B-W8A8")
PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "Water freezes at a temperature of",
    "The author of Pride and Prejudice is",
    "The chemical symbol for gold is",
    "The square root of 144 is",
    "The opposite of hot is",
    "The first month of the year is",
]
EXPECTED_ANSWER_PREFIXES = [
    ("Paris",),
    ("Jupiter",),
    ("32 degrees Fahrenheit", "0 degrees Celsius"),
    ("Jane Austen",),
    ("Au",),
    ("12",),
    ("cold",),
    ("January",),
]


class EplbSnapshotWorkerExtension:
    model_runner: Any

    def get_eplb_snapshot(self) -> dict[str, Any]:
        state = self.model_runner.eplb.state
        assert state is not None
        assert len(state.model_states) == 1
        model_state = next(iter(state.model_states.values()))
        mapping = model_state.physical_to_logical_map.detach().cpu().contiguous()
        policy = state.policy
        policy_impl = getattr(policy, "policy", None)
        history = getattr(policy_impl, "average_to_peak_history", {})
        load_windows = [model_state.expert_load_window.detach().cpu() for model_state in state.model_states.values()]
        global_load = None
        if model_state.eplb_stats is not None:
            global_load = model_state.eplb_stats.global_expert_load_window.detach().cpu()
        return {
            "policy": type(policy).__name__,
            "history_size": len(history),
            "is_async": state.is_async,
            "policy_cycles": state.async_policy_cycles,
            "completed_cycles": state.async_completed_cycles,
            "committed_layers": state.async_committed_layers,
            "load_window_min": min(int(window.min().item()) for window in load_windows),
            "negative_load_count": sum(int((window < 0).sum().item()) for window in load_windows),
            "global_load_min": None if global_load is None else int(global_load.min().item()),
            "global_load_max": None if global_load is None else int(global_load.max().item()),
            "global_negative_load_count": None if global_load is None else int((global_load < 0).sum().item()),
            "layer_fingerprints": [hashlib.sha256(layer.numpy().tobytes()).hexdigest() for layer in mapping],
        }


def _flatten_snapshots(snapshots: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [snapshot for dp_snapshots in snapshots for snapshot in dp_snapshots]


def _assert_expected_answers(outputs, name: str) -> None:
    assert len(outputs) == len(PROMPTS) == len(EXPECTED_ANSWER_PREFIXES)
    for prompt_idx, (prompt, (_, output_text), expected_prefixes) in enumerate(
        zip(PROMPTS, outputs, EXPECTED_ANSWER_PREFIXES)
    ):
        assert output_text.startswith(prompt), (
            f"{name} returned text that does not start with its prompt for "
            f"prompt {prompt_idx}: expected prefix {prompt!r}, got {output_text!r}"
        )
        completion = output_text[len(prompt) :].lstrip()
        matching_prefix = next(
            (prefix for prefix in expected_prefixes if completion.startswith(prefix)),
            None,
        )
        assert matching_prefix is not None, (
            f"{name} produced an incorrect answer for prompt {prompt_idx}: "
            f"expected the completion to start with one of {expected_prefixes!r}, "
            f"got {completion!r}"
        )
        suffix = completion[len(matching_prefix) :]
        assert not suffix or not (suffix[0].isalnum() or suffix[0] == "_"), (
            f"{name} only matched an answer as part of a longer word for "
            f"prompt {prompt_idx}: matched {matching_prefix!r}, got {completion!r}"
        )


def _run_dp2_tp2():
    runner_kwargs: dict[str, Any] = {
        "data_parallel_size": 2,
        "tensor_parallel_size": 2,
        "enable_expert_parallel": True,
        "max_model_len": 2048,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 2048,
        "compilation_config": {"cudagraph_mode": "FULL_AND_PIECEWISE"},
        "quantization": "ascend",
        "distributed_executor_backend": "mp",
        "async_scheduling": True,
        "gpu_memory_utilization": 0.7,
        "block_size": 128,
        "enable_prefix_caching": False,
        "worker_extension_cls": ("tests.e2e.pull_request.four_card.test_qwen3_mrv2_eplb.EplbSnapshotWorkerExtension"),
        "dp_start_timeout": 1800,
        "dp_request_timeout": 1800,
    }
    runner_kwargs.update(
        {
            "enable_eplb": True,
            "eplb_config": {
                "window_size": 2,
                "step_interval": 2,
                "num_redundant_experts": 4,
                "log_balancedness": False,
                "use_async": True,
            },
            "additional_config": {
                "eplb_config": {
                    "load_collection_phase": "prefill",
                }
            },
        }
    )

    with DPVllmRunner(MODEL, **runner_kwargs) as runner:
        before = _flatten_snapshots(runner.collective_rpc("get_eplb_snapshot"))
        outputs = runner.generate_greedy(PROMPTS, max_tokens=16)
        after = _flatten_snapshots(runner.collective_rpc("get_eplb_snapshot"))
        return outputs, before, after


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="eplb",
    parallel="DP,TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="full_and_piecewise",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_BUFFSIZE": "1024",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.7, max_wait_seconds=180)
def test_qwen3_moe_w8a8_dp2_tp2_async_stair_eplb_accuracy():
    eplb_outputs, before, after = _run_dp2_tp2()

    assert before is not None and after is not None
    assert len(before) == len(after) == 4
    assert all(snapshot["policy"] == "StairEplbPolicyAdapter" for snapshot in before)
    assert all(snapshot["is_async"] for snapshot in before)
    assert all(snapshot["history_size"] > 0 for snapshot in after), (
        f"STAIR did not commit a policy decision; final EPLB snapshots: {after}"
    )
    assert all(snapshot["policy_cycles"] > 0 for snapshot in after)
    assert all(snapshot["completed_cycles"] > 0 for snapshot in after), (
        f"No asynchronous EPLB cycle completed; final snapshots: {after}"
    )
    assert all(snapshot["committed_layers"] > 0 for snapshot in after)
    assert all(snapshot["negative_load_count"] == 0 for snapshot in after), (
        f"Rank-local load collection contains invalid counts: {after}"
    )
    assert all(snapshot["load_window_min"] >= 0 for snapshot in after)
    assert all(snapshot["global_load_min"] is not None for snapshot in after), (
        f"STAIR did not consume a global load window: {after}"
    )
    assert all(snapshot["global_load_min"] >= 0 for snapshot in after)
    assert all(snapshot["global_load_max"] > 0 for snapshot in after), (
        f"STAIR consumed an empty global load window: {after}"
    )
    assert all(snapshot["global_negative_load_count"] == 0 for snapshot in after)

    changed_layer_counts = [
        sum(before_fp != after_fp for before_fp, after_fp in zip(old["layer_fingerprints"], new["layer_fingerprints"]))
        for old, new in zip(before, after)
    ]
    assert all(changed_layer_count > 0 for changed_layer_count in changed_layer_counts), (
        "No async EPLB transfer was committed on every EP rank"
    )
    assert len({tuple(snapshot["layer_fingerprints"]) for snapshot in after}) == 1, (
        "Committed EPLB mappings diverged across EP ranks"
    )

    _assert_expected_answers(eplb_outputs, "MRV2 asynchronous STAIR EPLB")
