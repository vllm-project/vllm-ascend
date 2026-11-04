"""Checks for the reproducible DSpark benchmark command builder."""

import json
from types import SimpleNamespace

from experiments.dspark_tp.run_30b_k_matrix import planned_cases
from experiments.dspark_tp.run_case import serve_command, speculative_config


def test_speculative_config_uses_requested_width():
    assert speculative_config("target", "/draft", 8) is None
    assert speculative_config("adaptive", "/draft", 8) == {
        "method": "dspark",
        "model": "/draft",
        "num_speculative_tokens": 8,
        "enable_adaptive_verification": True,
    }


def test_physical_case_preserves_requested_width_and_minimum():
    args = SimpleNamespace(
        case="physical",
        model="/target",
        draft_model="/draft",
        num_speculative_tokens=8,
        port=8006,
        max_model_len=8192,
        gpu_memory_utilization=0.93,
        tp=4,
        max_num_seqs=256,
        physical_min_k=3,
    )
    command = serve_command(args)
    spec = json.loads(command[command.index("--speculative-config") + 1])
    additional = json.loads(command[command.index("--additional-config") + 1])
    assert spec["num_speculative_tokens"] == 8
    assert spec["enable_adaptive_verification"] is True
    assert additional["dynamic_spec_config"]["physical_k"] == {"min_k": 3}
    assert command[command.index("--tensor-parallel-size") + 1] == "4"


def test_matrix_pairs_adaptive_and_physical_at_each_tp_and_k():
    assert planned_cases([4, 2], [3, 8]) == [
        (4, "target", None),
        (4, "adaptive", 3),
        (4, "physical", 3),
        (4, "adaptive", 8),
        (4, "physical", 8),
        (2, "target", None),
        (2, "adaptive", 3),
        (2, "physical", 3),
        (2, "adaptive", 8),
        (2, "physical", 8),
    ]
