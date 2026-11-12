import importlib
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

import vllm_ascend.worker.v2.spec_decode.dflash.speculator as dflash_module
import vllm_ascend.worker.v2.spec_decode.dspark.speculator as dspark_module
from vllm_ascend.ascend_forward_context import get_mrv2_in_profile_run
from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
from vllm_ascend.attention.indexer import AscendSFAIndexerMetadataBuilder
from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator


@pytest.mark.parametrize(
    "builder_cls",
    [AscendDSAMetadataBuilder, AscendSFAIndexerMetadataBuilder, AscendSFAMetadataBuilder],
)
@pytest.mark.parametrize(
    "speculative_config,expected",
    [
        (None, AttentionCGSupport.UNIFORM_BATCH),
        (SimpleNamespace(method="dspark", enable_adaptive_verification=False), AttentionCGSupport.UNIFORM_BATCH),
        (SimpleNamespace(method="eagle", enable_adaptive_verification=True), AttentionCGSupport.UNIFORM_BATCH),
        (SimpleNamespace(method="dspark", enable_adaptive_verification=True), AttentionCGSupport.ALWAYS),
    ],
)
def test_adaptive_verification_cudagraph_support(builder_cls, speculative_config, expected):
    config = SimpleNamespace(speculative_config=speculative_config)
    assert builder_cls.get_cudagraph_support(config, Mock()) is expected


@pytest.mark.parametrize(
    ("adaptive_verification", "context_len", "expected"),
    [
        (object(), 8192, True),
        (object(), 0, False),
        (None, 8192, False),
    ],
)
def test_adaptive_tail_dummy_run_balances_moe_routing(adaptive_verification, context_len, expected):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.input_buffers = SimpleNamespace(dummy_num_tokens=None)  # type: ignore[assignment]
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)
    runner.debugger = None
    runner.adaptive_verification = adaptive_verification
    runner.ascend_config = SimpleNamespace(xlite_graph_config=SimpleNamespace(enabled=False))
    runner.eplb = Mock()
    observed = []

    def fake_dummy_run(*args, **kwargs):
        observed.append(get_mrv2_in_profile_run())
        return None, None

    with (
        patch.object(GPUModelRunner, "_dummy_run", side_effect=fake_dummy_run),
        patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", return_value=False),
    ):
        runner._dummy_run(256, context_len=context_len)

    assert observed == [expected]
    runner.eplb.step.assert_called_once_with(is_dummy=True, is_profile=False)
    assert get_mrv2_in_profile_run() is False


def test_adaptive_verification_patch_uses_uncompiled_budget_assignment(monkeypatch):
    import vllm.v1.worker.gpu.spec_decode.adaptive_verification as adaptive

    allocator = Mock()
    monkeypatch.setattr(adaptive, "_assign_draft_token_budget", allocator, raising=False)
    monkeypatch.setattr(adaptive, "_assign_draft_token_budget_compiled", object(), raising=False)
    module = importlib.import_module("vllm_ascend.patch.worker.patch_adaptive_verification")
    importlib.reload(module)

    assert adaptive._assign_draft_token_budget_compiled is allocator


@pytest.mark.parametrize(
    "local_reqs,dp_reqs,target_capacity,expected_reqs",
    [(1, 3, 16, 1), (0, 3, 16, 3), (0, 1, 8, 1)],
    ids=["real-rank", "idle-rank", "idle-rank-single-request"],
)
def test_adaptive_draft_uses_its_own_request_count(monkeypatch, local_reqs, dp_reqs, target_capacity, expected_reqs):
    dummy = local_reqs == 0
    batch_reqs = target_capacity if dummy else local_reqs
    batch = SimpleNamespace(
        num_reqs=batch_reqs,
        num_tokens=target_capacity,
        is_prefilling_np=np.zeros(batch_reqs, dtype=bool),
        seq_lens_cpu_upper_bound=torch.ones(batch_reqs, dtype=torch.int32),
    )
    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.pcp_manager = None
    spec.use_dcp = False
    spec.enable_adaptive_verification = True
    spec.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(), model_config=SimpleNamespace(max_model_len=128)
    )
    spec.input_buffers = SimpleNamespace(positions=torch.arange(80))
    spec.max_num_reqs, spec.max_num_tokens = 16, 80
    spec.num_query_per_req = spec.num_speculative_steps = 5
    spec.dp_size, spec.dp_rank, spec.dcp_size = 3, 0 if local_reqs else 2, 1
    spec.draft_kv_cache_group_id, spec.draft_kv_cache_group_ids = 0, [0]
    spec.block_tables = SimpleNamespace(
        slot_mappings=[None], input_block_tables=[None], block_sizes=[32], cp_rank=0, cp_interleave=1
    )
    spec._context_slot_mappings = [None]
    spec.parallel_drafting_token_id, spec.sample_from_anchor = 0, False
    spec.query_cudagraph_manager = None
    for name in ("context_positions", "sample_indices", "sample_pos", "sample_idx_mapping", "temperature", "seeds"):
        setattr(spec, name, None)
    monkeypatch.setattr(AscendDSparkSpeculator, "attn_vllm_config", property(lambda self: self.vllm_config))
    monkeypatch.setattr(dspark_module, "build_attn_metadata_wrapper", nullcontext)
    monkeypatch.setattr(dspark_module, "build_attn_metadata_factory", lambda *args, **kwargs: nullcontext())
    kernel = Mock()
    monkeypatch.setattr(dflash_module, "prepare_dflash_inputs_triton", kernel)
    upstream = dspark_module.dflash_speculator
    monkeypatch.setattr(upstream, "prepare_dflash_inputs", dflash_module.prepare_dflash_inputs_factory(32))

    class InputsPrepared(Exception):
        pass

    # Stop after real propose() prepares inputs and constructs the draft DP
    # agreement, before dispatch can enter graph/device/collective execution.
    dispatch = Mock(side_effect=InputsPrepared)
    monkeypatch.setattr(upstream, "dispatch_cg_and_sync_dp", dispatch)
    sync_kwargs = {} if vllm_version_is("0.33.0") else {"max_query_len": None}
    target_sync = DPSyncState(
        num_tokens_across_dp=torch.full((3,), target_capacity),
        uniform_token_count=None,
        eager=False,
        num_reqs=dp_reqs,
        **sync_kwargs,
    )
    with pytest.raises(InputsPrepared):
        spec.propose(
            batch,
            {},
            {},
            torch.empty(0),
            None,
            *[torch.zeros(16) for _ in range(6)],
            dp_sync_state=target_sync,
            dummy_run=dummy,
        )

    kernel.assert_called_once()
    assert kernel.call_args.kwargs["num_reqs_override"] == expected_reqs
    assert kernel.call_args.args[22] == 5  # Query tokens per draft request.
    assert dispatch.call_args.args[1:3] == (expected_reqs, dp_reqs * 5)
    draft_sync = dispatch.call_args.kwargs["dp_sync_state"]
    assert draft_sync is not None
    torch.testing.assert_close(draft_sync.num_tokens_across_dp, torch.full((3,), dp_reqs * 5))
    assert batch.num_reqs == batch_reqs
