# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only coverage of deferred PP postprocessing, without NPU imports."""

import __future__

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def mamba_state_cls():
    return type("MambaHybridModelState", (), {})


@pytest.fixture
def runner_cls(mamba_state_cls):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/model_runner.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    assignment = next(
        node
        for node in init.body
        if isinstance(node, ast.Assign)
        and any(getattr(target, "attr", None) == "sync_spec_pp_cpu_counts" for target in node.targets)
    )
    # Execute the real production methods with a CPU parent instead of importing NPU workers.
    init.body = [assignment]
    cls.body = [
        node
        for node in cls.body
        if getattr(node, "name", None)
        in {"__init__", "postprocess_sampled", "postprocess_num_computed_tokens", "_update_seq_lens_cpu"}
    ]
    cls.bases = [ast.Name(id="Parent", ctx=ast.Load())]

    class Parent:
        device_count: int
        events: list[str]
        num_computed_tokens_cpu: torch.Tensor

        def postprocess_sampled(self, idx_mapping, sampled_tokens, num_sampled, num_rejected, query_start_loc=None):
            self.last_idx_mapping = idx_mapping
            self.device_count -= num_rejected
            self.events.append("reject")

        def postprocess_num_computed_tokens(self, batch):
            self.device_count += batch.num_scheduled_tokens
            self.events.append("advance")

        def _copy_num_computed_tokens_to_cpu(self):
            self.num_computed_tokens_cpu[0] = self.device_count
            self.events.append("copy")

    namespace = {"Parent": Parent, "MambaHybridModelState": mamba_state_cls}
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(module, str(source), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace["NPUModelRunner"]


@pytest.mark.parametrize(
    "architecture,enabled",
    [
        ("KimiLinearForCausalLM", True),
        ("KimiK3ForCausalLM", True),
        ("KimiK3ForConditionalGeneration", True),
        ("Qwen3_5ForConditionalGeneration", True),
        ("DeepseekV4ForCausalLM", False),
        ("GlmMoeDsaForCausalLM", False),
    ],
)
@pytest.mark.parametrize("use_pp,steps", [(False, 3), (True, 0), (True, 3)])
@pytest.mark.parametrize("owns_speculator", [False, True])
@pytest.mark.parametrize("prefill_chunk", [False, True])
def test_exact_counts_after_rejection_or_chunk(
    runner_cls, architecture, enabled, use_pp, steps, owns_speculator, prefill_chunk
):
    runner = runner_cls.__new__(runner_cls)
    runner.use_pp = use_pp
    runner.is_last_pp_rank = False
    runner.model_state = object()
    runner.num_speculative_steps = steps
    runner.model_config = SimpleNamespace(architecture=architecture)
    runner.__init__(None, None)
    assert runner.sync_spec_pp_cpu_counts == (enabled and use_pp and steps > 0)
    runner.speculator = object() if owns_speculator else None
    runner.events = []
    runner.device_count = 19
    runner.num_computed_tokens_cpu = torch.tensor([35, -1])
    runner.num_computed_tokens_event = SimpleNamespace(synchronize=lambda: runner.events.append("wait"))
    # Slot 1 is newly allocated: its initialized count must not be overwritten by an old snapshot.
    runner.req_states = SimpleNamespace(
        req_id_to_index={"cached": 0, "new": 1}, num_computed_tokens_cpu=torch.tensor([35, 128])
    )
    runner.input_buffers = SimpleNamespace(seq_lens_cpu=torch.zeros(2, dtype=torch.int32))
    if prefill_chunk:
        runner.postprocess_num_computed_tokens(SimpleNamespace(num_scheduled_tokens=8))
    else:
        runner.postprocess_sampled(None, None, 2, 2)
    scheduler = SimpleNamespace(
        num_scheduled_tokens={"cached": 4, "new": 8},
        scheduled_cached_reqs=SimpleNamespace(req_ids=["cached"]),
    )
    runner._update_seq_lens_cpu(scheduler, ["cached", "new"])
    needs_sync = owns_speculator or runner.sync_spec_pp_cpu_counts
    copies = runner.sync_spec_pp_cpu_counts if prefill_chunk else needs_sync
    expected_events = ["advance" if prefill_chunk else "reject"]
    if copies:
        expected_events.append("copy")
    if needs_sync:
        expected_events.append("wait")
    assert runner.events == expected_events
    count = runner.device_count if copies and needs_sync else 35
    assert runner.req_states.num_computed_tokens_cpu.tolist() == [count, 128]
    assert runner.input_buffers.seq_lens_cpu.tolist() == [count + 4, 136]


@pytest.mark.parametrize(
    "use_pp,last_rank,hybrid,cache_mode,refresh",
    [
        (True, False, True, "align", True),
        (False, True, True, "align", False),
        (True, True, True, "align", False),
        (True, False, False, "align", False),
        (True, False, True, "none", False),
    ],
)
@pytest.mark.parametrize("indices", [[3, 1], [3, -1, 1], [-1, -1], []])
def test_deferred_mamba_postprocess_restores_batch_rows(
    runner_cls, mamba_state_cls, use_pp, last_rank, hybrid, cache_mode, refresh, indices
):
    runner = runner_cls.__new__(runner_cls)
    runner.use_pp = use_pp
    runner.is_last_pp_rank = last_rank
    runner.model_state = mamba_state_cls() if hybrid else object()
    runner.cache_config = SimpleNamespace(mamba_cache_mode=cache_mode)
    runner.speculator = None
    runner.sync_spec_pp_cpu_counts = False
    runner.events = []
    runner.device_count = 19
    idx_mapping = torch.tensor(indices, dtype=torch.int32)
    request_tables = torch.arange(16, dtype=torch.int32).reshape(4, 4)
    # A later PP batch has overwritten the persistent input block-table rows.
    input_tables = torch.full((len(indices), 4), -1, dtype=torch.int32)

    def gather(mapping, num_reqs):
        assert num_reqs == len(indices)
        assert mapping.tolist() == [max(index, 0) for index in indices]
        input_tables.copy_(request_tables[mapping.long()])
        runner.events.append("gather")

    runner.block_tables = SimpleNamespace(gather_block_tables=gather)
    runner.postprocess_sampled(idx_mapping, None, 2, 2)

    assert runner.events == (["gather", "reject"] if refresh else ["reject"])
    assert runner.last_idx_mapping is idx_mapping
    assert idx_mapping.tolist() == indices  # Preserve -1 so postprocess skips filtered rows.
    if refresh:
        torch.testing.assert_close(input_tables, request_tables[idx_mapping.clamp_min(0).long()])
    else:
        assert torch.all(input_tables == -1)
