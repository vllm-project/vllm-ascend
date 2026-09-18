# SPDX-License-Identifier: Apache-2.0
"""K3 device metadata must not serialize MRv2 on the rejected-length D2H copy."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def runner_type():
    path = ROOT / "vllm_ascend/worker/v2/model_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    capability = next(
        node
        for node in init.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr == "_use_device_seq_lens"
    )
    methods = {
        "postprocess_sampled",
        "postprocess_num_computed_tokens",
        "_copy_num_computed_tokens_to_cpu",
        "_update_seq_lens_cpu",
    }
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    cls.bases = [ast.Name(id="Parent", ctx=ast.Load())]

    class Parent:
        def postprocess_sampled(self, idx_mapping, sampled_tokens, num_sampled, num_rejected, query_start_loc):
            self.req_states.num_computed_tokens.gpu[idx_mapping] -= num_rejected

        def postprocess_num_computed_tokens(self, input_batch):
            self.req_states.num_computed_tokens.gpu += input_batch

    utils = ast.parse((ROOT / "vllm_ascend/utils.py").read_text(encoding="utf-8"))
    is_k3 = next(node for node in utils.body if getattr(node, "name", None) == "_is_kimi_k3_target")
    scope = {"Parent": Parent, "torch": torch, "np": np}
    exec(
        compile(
            "from __future__ import annotations\n" + ast.unparse(is_k3) + "\n" + ast.unparse(cls), str(path), "exec"
        ),
        scope,
    )
    runner = scope[cls.name]

    def configure(self, *, flash=True, pcp=1, arch="KimiK3ForConditionalGeneration", pp=1, dcp=1):
        self.device_metadata_executor = object() if flash else None
        parallel = SimpleNamespace(
            prefill_context_parallel_size=pcp, pipeline_parallel_size=pp, decode_context_parallel_size=dcp
        )
        config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[arch])))
        exec(
            compile(ast.Module(body=[capability], type_ignores=[]), str(path), "exec"),
            scope | {"self": self, "vllm_config": config, "parallel_config": parallel},
        )

    runner.configure = configure
    return runner


def make_runner(runner_type, *, device_lengths=True, last_stage=True):
    runner = runner_type()
    runner.configure(flash=device_lengths)
    runner.speculator = object() if last_stage else None
    runner.use_spec_pp = not last_stage
    # Slot 1 belongs to a cancelled request; slot 2 was reused for a new prompt.
    host = np.array([104, 999, 0, 2048], dtype=np.int32)
    runner.req_states = SimpleNamespace(
        num_computed_tokens_np=host,
        num_computed_tokens_cpu=torch.from_numpy(host),
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([104, 999, 0, 2048], dtype=torch.int32)),
        req_id_to_index={"decode": 0, "new": 2, "prefill": 3},
    )
    runner.num_computed_tokens_cpu = torch.tensor([101, 888, 777, 2048], dtype=torch.int32)
    runner.num_computed_tokens_event = SimpleNamespace(synchronize=Mock())
    runner.input_buffers = SimpleNamespace(seq_lens_cpu=torch.zeros(3, dtype=torch.int32))
    runner.input_buffers.seq_lens_np = runner.input_buffers.seq_lens_cpu.numpy()
    return runner


@pytest.mark.parametrize("pp,dcp", [(1, 1), (1, 8), (4, 1), (4, 8)])
def test_flash_k3_keeps_lengths_on_device_for_tp_dcp_and_pp(runner_type, pp, dcp):
    runner = runner_type()
    runner.configure(pp=pp, dcp=dcp)
    assert runner._use_device_seq_lens


@pytest.mark.parametrize("config", [{"flash": False}, {"pcp": 2}, {"arch": "DeepseekV3ForCausalLM"}])
def test_cpu_length_consumers_keep_the_existing_sync(runner_type, config):
    runner = runner_type()
    runner.configure(**config)
    assert not runner._use_device_seq_lens


@pytest.mark.parametrize("last_stage", [False, True])
@pytest.mark.parametrize("device_lengths", [False, True])
def test_rejection_mixed_prefill_and_reordered_requests(runner_type, last_stage, device_lengths):
    runner = make_runner(runner_type, device_lengths=device_lengths, last_stage=last_stage)
    if device_lengths:
        # No stream/event/copy may be touched by postprocessing in this path.
        runner.num_computed_tokens_event.synchronize.side_effect = AssertionError("CPU waited for device lengths")
        runner.postprocess_sampled(torch.tensor([0]), None, None, torch.tensor([3]), None)
        assert runner.req_states.num_computed_tokens.gpu[0] == 101
    scheduler = SimpleNamespace(
        num_scheduled_tokens={"new": 12, "prefill": 128, "decode": 4},
        scheduled_cached_reqs=SimpleNamespace(req_ids=["decode", "prefill"]),
    )
    runner._update_seq_lens_cpu(scheduler, ["new", "prefill", "decode"])
    assert runner.input_buffers.seq_lens_cpu.tolist() == [12, 2176, 108 if device_lengths else 105]
    assert runner.req_states.num_computed_tokens_np.tolist() == [104 if device_lengths else 101, 999, 0, 2048]
    assert runner.num_computed_tokens_event.synchronize.call_count == (0 if device_lengths else 1)


def test_nonlast_pp_prefill_advances_device_state_without_readback(runner_type):
    runner = make_runner(runner_type, last_stage=False)
    runner.postprocess_num_computed_tokens(torch.tensor([0, 0, 0, 128]))
    assert runner.req_states.num_computed_tokens.gpu[3] == 2176
    runner.num_computed_tokens_event.synchronize.assert_not_called()


@pytest.mark.parametrize("req_ids", [[], ["decode"], ["prefill", "new"]])
def test_cpu_length_update_preserves_unused_buffer_rows(runner_type, req_ids):
    runner = make_runner(runner_type)
    runner.input_buffers.seq_lens_cpu.fill_(-1)
    scheduled = {"decode": 4, "new": 12, "prefill": 128}
    scheduler = SimpleNamespace(num_scheduled_tokens=scheduled)
    runner._update_seq_lens_cpu(scheduler, req_ids)
    expected = {"decode": 108, "new": 12, "prefill": 2176}
    assert runner.input_buffers.seq_lens_cpu.tolist() == [expected[r] for r in req_ids] + [-1] * (3 - len(req_ids))
