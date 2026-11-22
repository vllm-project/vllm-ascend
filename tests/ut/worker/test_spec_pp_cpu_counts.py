# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only coverage of deferred PP postprocessing, without NPU imports."""

import __future__

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

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
        in {
            "__init__",
            "_can_skip_spec_cpu_length_sync",
            "initialize_spec_cpu_length_sync",
            "postprocess_sampled",
            "postprocess_num_computed_tokens",
            "_update_seq_lens_cpu",
        }
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
    utils = ast.parse((source.parents[2] / "utils.py").read_text())
    predicate = next(
        node for node in utils.body if isinstance(node, ast.FunctionDef) and node.name == "is_deepseek_v41"
    )
    module = ast.fix_missing_locations(ast.Module(body=[predicate, cls], type_ignores=[]))
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
    runner.model_config = SimpleNamespace(
        architecture=architecture, hf_config=SimpleNamespace(architectures=[architecture])
    )
    runner.speculative_config = None
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


@pytest.mark.parametrize("architecture", ["MiniMaxM31FlashA3DSparkModel", "DSparkMiniMaxDraftModel"])
@pytest.mark.parametrize("cpu_consumer", [False, True])
@pytest.mark.parametrize("sync_pp", [False, True])
def test_minimax_dspark_skips_sync_only_without_cpu_consumers(
    runner_cls, monkeypatch, architecture, cpu_consumer, sync_pp
):
    flash_backend = type("FlashBackend", (), {})
    sparse_backend = type("SparseBackend", (), {"get_name": staticmethod(lambda: "ASCEND_MINIMAX_M3_SPARSE")})
    dense_backend = type("DenseBackend", (), {"get_name": staticmethod(lambda: "ASCEND")})
    monkeypatch.setitem(
        runner_cls._can_skip_spec_cpu_length_sync.__globals__, "AscendFlashAttentionBackend", flash_backend
    )
    runner = runner_cls.__new__(runner_cls)
    runner.sync_spec_pp_cpu_counts = sync_pp
    runner.speculator = object()
    runner.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["MiniMaxM3SparseForCausalLM"]),
        hf_text_config=SimpleNamespace(model_type="minimax_m3_text"),
    )
    runner.speculative_config = SimpleNamespace(
        method="dspark", draft_model_config=SimpleNamespace(architectures=[architecture])
    )
    runner.attn_groups = [[SimpleNamespace(backend=flash_backend), SimpleNamespace(backend=sparse_backend)]]
    if cpu_consumer:
        runner.attn_groups[0].append(SimpleNamespace(backend=dense_backend))
    assert runner._can_skip_spec_cpu_length_sync() is (not cpu_consumer and not sync_pp)


def test_dflash_cpu_sync_policy_is_retained(runner_cls):
    runner = runner_cls.__new__(runner_cls)
    runner.sync_spec_pp_cpu_counts = False
    runner.speculator = object()
    runner.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"]),
        hf_text_config=SimpleNamespace(model_type="glm_moe_dsa"),
    )
    runner.speculative_config = SimpleNamespace(
        method="dflash", draft_model_config=SimpleNamespace(architectures=["DFlash2DraftModel"])
    )
    assert runner._can_skip_spec_cpu_length_sync()
    runner.sync_spec_pp_cpu_counts = True
    assert not runner._can_skip_spec_cpu_length_sync()


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("blocked", [None, "sparse_offload", "adaptive", "pp", "mixed_backends"])
def test_original_m3_dspark_cpu_sync_guards_are_retained(runner_cls, monkeypatch, eager, blocked):
    flash = type("FlashBackend", (), {})
    monkeypatch.setitem(runner_cls._can_skip_spec_cpu_length_sync.__globals__, "AscendFlashAttentionBackend", flash)
    runner = runner_cls.__new__(runner_cls)
    runner.sync_spec_pp_cpu_counts = False
    runner.model_config = SimpleNamespace(
        architecture="MiniMaxM3SparseForConditionalGeneration",
        hf_config=SimpleNamespace(architectures=["MiniMaxM3SparseForConditionalGeneration"]),
        hf_text_config=SimpleNamespace(model_type="minimax_m3"),
    )
    runner.speculative_config = SimpleNamespace(
        method="dspark",
        enforce_eager=eager,
        enable_adaptive_verification=blocked == "adaptive",
        draft_model_config=SimpleNamespace(architectures=["Qwen3DSparkModel"]),
    )
    runner.speculator = SimpleNamespace(attn_backends={"draft.0": flash})
    if blocked == "mixed_backends":
        runner.speculator.attn_backends["draft.1"] = object()
    runner.parallel_config = SimpleNamespace(
        pipeline_parallel_size=2 if blocked == "pp" else 1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    runner.ascend_config = SimpleNamespace(
        sparse_kv_offload_config=SimpleNamespace(enabled=blocked == "sparse_offload")
    )
    assert runner._can_skip_spec_cpu_length_sync() is (blocked is None)


@pytest.fixture
def fa3_runner(runner_cls):
    fa3 = type(
        "AscendFlashAttentionBackend",
        (),
        {
            "__module__": "vllm_ascend.attention.flash_attention_v3",
            "full_cls_name": Mock(
                return_value=("vllm_ascend.attention.flash_attention_v3", "AscendFlashAttentionBackend")
            ),
        },
    )
    runner = runner_cls.__new__(runner_cls)
    runner.sync_spec_pp_cpu_counts = False
    runner.model_config = SimpleNamespace(hf_config=SimpleNamespace(architectures=["OtherModel"]))
    runner.speculative_config = SimpleNamespace(method="mtp")
    runner.model_state = SimpleNamespace(initialize_spec_cpu_length_sync=Mock())
    # set_attn initializes nested draft groups, not an attn_backends mapping.
    runner.speculator = SimpleNamespace(attn_groups=[[SimpleNamespace(backend=fa3)], [SimpleNamespace(backend=fa3)]])
    return runner


@pytest.mark.parametrize(
    "case", ["fa3", "empty", "empty_group", "missing", "cann", "mixed", "no_speculator", "sync_pp"]
)
def test_fa3_draft_cpu_sync_policy(fa3_runner, case):
    runner = fa3_runner
    cann = type(
        "AscendFlashAttentionBackend",
        (),
        {
            "__module__": "vllm_ascend.attention.fa_v1",
            "full_cls_name": Mock(return_value=("vllm_ascend.attention.fa_v1", "AscendFlashAttentionBackend")),
        },
    )
    if case == "empty":
        runner.speculator.attn_groups = []
    elif case == "empty_group":
        runner.speculator.attn_groups = [[]]
    elif case == "missing":
        del runner.speculator.attn_groups
    elif case == "cann":
        runner.speculator.attn_groups = [[SimpleNamespace(backend=cann)]]
    elif case == "mixed":
        runner.speculator.attn_groups[1][0].backend = cann
    elif case == "no_speculator":
        runner.speculator = None
    elif case == "sync_pp":
        runner.sync_spec_pp_cpu_counts = True
    # No target groups or model-specific draft config: only draft backends matter.
    assert not runner._can_skip_spec_cpu_length_sync()
    runner.initialize_spec_cpu_length_sync()
    runner.model_state.initialize_spec_cpu_length_sync.assert_called_once_with(runner)
    assert runner._can_skip_spec_cpu_length_sync() is (case == "fa3")


def test_fa3_skips_cpu_snapshot_and_wait(fa3_runner):
    runner = fa3_runner
    runner.initialize_spec_cpu_length_sync()
    backend = runner.speculator.attn_groups[0][0].backend
    backend.full_cls_name.reset_mock()
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/model_runner.py"
    cls = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner"
    )
    copy = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_copy_num_computed_tokens_to_cpu"
    )
    namespace: dict[str, Any] = {}
    exec(compile(ast.Module(body=[copy], type_ignores=[]), str(source), "exec"), namespace)
    # No stream/buffer dependencies are supplied: taking the D2H path fails.
    namespace[copy.name](runner)
    runner.num_computed_tokens_event = SimpleNamespace(synchronize=lambda: pytest.fail("unexpected CPU wait"))
    runner.req_states = SimpleNamespace(req_id_to_index={"cached": 0}, num_computed_tokens_cpu=torch.tensor([20]))
    runner.input_buffers = SimpleNamespace(seq_lens_cpu=torch.zeros(1, dtype=torch.int32))
    scheduler = SimpleNamespace(
        num_scheduled_tokens={"cached": 4}, scheduled_cached_reqs=SimpleNamespace(req_ids=["cached"])
    )
    runner._update_seq_lens_cpu(scheduler, ["cached"])
    assert runner.input_buffers.seq_lens_cpu.tolist() == [24]

    backend.full_cls_name.assert_not_called()
