from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch


def _load_update_seq_lens_cpu():
    source_path = Path(__file__).parents[3] / "vllm_ascend" / "worker" / "v2" / "model_runner.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    runner_cls = next(
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner"
    )
    update_method = next(
        node for node in runner_cls.body if isinstance(node, ast.FunctionDef) and node.name == "_update_seq_lens_cpu"
    )
    method_source = ast.get_source_segment(source_path.read_text(encoding="utf-8"), update_method)
    namespace: dict[str, object] = {}
    exec(
        "from __future__ import annotations\n" + textwrap.dedent(method_source),
        {"np": np},
        namespace,
    )
    return namespace["_update_seq_lens_cpu"]


def test_update_seq_lens_cpu_uses_batch_arrays_for_active_prefix():
    update_seq_lens_cpu = _load_update_seq_lens_cpu()
    runner = SimpleNamespace()
    runner.speculator = None
    num_computed_tokens_cpu = torch.tensor([100, 11, 22, 33, 44, 55], dtype=torch.int32)
    runner.req_states = SimpleNamespace(
        req_id_to_index={},
        num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_computed_tokens_np=num_computed_tokens_cpu.numpy(),
    )
    seq_lens_cpu = torch.full((6,), 888, dtype=torch.int32)
    runner.input_buffers = SimpleNamespace(
        seq_lens_cpu=seq_lens_cpu,
        seq_lens_np=seq_lens_cpu.numpy(),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={},
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
    )

    update_seq_lens_cpu(
        runner,
        scheduler_output,
        ["logical_0", "logical_1", "logical_2"],
        np.array([4, 0, 2], dtype=np.int32),
        np.array([3, 5, 7], dtype=np.int32),
    )

    np.testing.assert_array_equal(
        runner.input_buffers.seq_lens_np,
        np.array([47, 105, 29, 888, 888, 888], dtype=np.int32),
    )
    torch.testing.assert_close(
        runner.input_buffers.seq_lens_cpu,
        torch.tensor([47, 105, 29, 888, 888, 888], dtype=torch.int32),
    )
    assert runner.input_buffers.seq_lens_np.dtype == np.int32
    assert np.shares_memory(runner.input_buffers.seq_lens_np, runner.input_buffers.seq_lens_cpu.numpy())


def test_update_seq_lens_cpu_consumes_spec_corrected_host_state():
    update_seq_lens_cpu = _load_update_seq_lens_cpu()
    runner = SimpleNamespace()
    runner.speculator = object()
    runner.num_computed_tokens_event = Mock()
    num_computed_tokens_cpu = torch.tensor([0, 11, 22, 33, 44, 55], dtype=torch.int32)
    runner.req_states = SimpleNamespace(
        req_id_to_index={"a": 3, "c": 4},
        num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_computed_tokens_np=num_computed_tokens_cpu.numpy(),
    )
    runner.num_computed_tokens_cpu = torch.tensor([0, 101, 202, 303, 404, 505], dtype=torch.int32)
    seq_lens_cpu = torch.full((6,), -1, dtype=torch.int32)
    runner.input_buffers = SimpleNamespace(
        seq_lens_cpu=seq_lens_cpu,
        seq_lens_np=seq_lens_cpu.numpy(),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={},
        scheduled_cached_reqs=SimpleNamespace(req_ids=["a", "c"]),
    )

    update_seq_lens_cpu(
        runner,
        scheduler_output,
        ["a", "b", "c"],
        np.array([3, 1, 4], dtype=np.int32),
        np.array([7, 5, 2], dtype=np.int32),
    )

    runner.num_computed_tokens_event.synchronize.assert_called_once_with()
    np.testing.assert_array_equal(
        runner.req_states.num_computed_tokens_np,
        np.array([0, 11, 22, 303, 404, 55], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        runner.input_buffers.seq_lens_np,
        np.array([310, 16, 406, -1, -1, -1], dtype=np.int32),
    )
    torch.testing.assert_close(
        runner.input_buffers.seq_lens_cpu,
        torch.tensor([310, 16, 406, -1, -1, -1], dtype=torch.int32),
    )
