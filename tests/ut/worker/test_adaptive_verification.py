import importlib
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker.gpu import attn_utils
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode import acceptance_estimator as estimator_module
from vllm.v1.worker.gpu.spec_decode import adaptive_verification as adaptive
from vllm.v1.worker.gpu.spec_decode import speculator as upstream_speculator

import vllm_ascend.worker.v2.spec_decode.dflash.speculator as dflash_module
import vllm_ascend.worker.v2.spec_decode.dspark.speculator as dspark_module
from vllm_ascend.ascend_forward_context import get_mrv2_in_profile_run
from vllm_ascend.attention.dsa_v1 import AscendDSABackend, AscendDSAMetadataBuilder
from vllm_ascend.attention.indexer import AscendSFAIndexerBackend, AscendSFAIndexerMetadataBuilder
from vllm_ascend.attention.sfa_v1 import AscendSFABackend, AscendSFAMetadataBuilder
from vllm_ascend.patch.worker import patch_model_runner
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator
from vllm_ascend.worker.v2.spec_decode.mtp.speculator import AscendMTPSpeculator


@pytest.mark.parametrize(
    "builder_cls",
    [AscendDSAMetadataBuilder, AscendSFAIndexerMetadataBuilder, AscendSFAMetadataBuilder],
)
@pytest.mark.parametrize(
    "method,adaptive,num_speculative_tokens,expected_bound",
    [
        (None, False, None, 1),
        ("mtp", True, None, 1),
        ("mtp", True, 1, 2),
        ("mtp", False, 5, 6),
        ("mtp", True, 5, 6),
        ("dflash2", True, 5, 6),
        ("dspark", True, 5, 6),
    ],
)
def test_adaptive_verification_backend_capabilities(
    builder_cls, method, adaptive, num_speculative_tokens, expected_bound
):
    speculative_config = (
        None
        if method is None
        else SimpleNamespace(
            method=method,
            enable_adaptive_verification=adaptive,
            num_speculative_tokens=num_speculative_tokens,
            parallel_drafting=method in ("dflash2", "dspark"),
        )
    )
    config = SimpleNamespace(speculative_config=speculative_config, use_v2_model_runner=True)
    spec = Mock()
    assert builder_cls.get_cudagraph_support(config, spec) is AttentionCGSupport.UNIFORM_BATCH
    assert builder_cls.get_varlen_cudagraph_max_query_len(config, spec) == expected_bound


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


@pytest.fixture
def cpu_manager_resources(monkeypatch):
    # Only replace CUDA resources; manager construction and budget selection run unchanged.
    monkeypatch.setattr(torch.cuda, "Stream", Mock())
    monkeypatch.setattr(torch.cuda, "Event", Mock())
    monkeypatch.setattr(torch.cuda, "current_stream", Mock())
    monkeypatch.setattr(adaptive, "stream", lambda *args: nullcontext())
    monkeypatch.setattr(adaptive, "CpuGpuBuffer", partial(adaptive.CpuGpuBuffer, pin_memory=False))


def _attn_group(backend, builder):
    return SimpleNamespace(
        backend=backend,
        get_metadata_builder=lambda _: builder,
        kv_cache_spec=Mock(),
        layer_names=[backend.__name__],
    )


def _adaptive_manager_kwargs(groups, enabled=True, steps=3):
    req_states = SimpleNamespace(
        num_speculative_steps=steps,
        device=torch.device("cpu"),
        max_num_reqs=2,
        max_num_batched_tokens=16,
        req_id_to_index={"a": 0, "b": 1},
        num_computed_tokens_np=np.zeros(2, dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.zeros(2, dtype=np.int32)),
    )
    config = SimpleNamespace(
        use_v2_model_runner=True,
        speculative_config=SimpleNamespace(num_speculative_tokens=3, parallel_drafting=False),
    )
    return dict(
        enable_adaptive_verification=enabled,
        attn_groups=[groups],
        req_states=req_states,
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        num_bonus_tokens=1,
        max_total_logits=16,
        vllm_config=config,
    )


@pytest.mark.parametrize(
    "enabled,steps,unsupported,error",
    [
        (False, 3, "host", None),
        (True, 3, None, None),
        (True, 4, None, "vary up to 5"),
        (True, 3, "host", "does not support"),
        (True, 3, AscendDSAMetadataBuilder, "AscendDSABackend allows none"),
        (True, 3, AscendSFAMetadataBuilder, "AscendSFABackend allows none"),
        (True, 3, AscendSFAIndexerMetadataBuilder, "AscendSFAIndexerBackend allows none"),
    ],
    ids=["off", "supported", "over-bound", "host-lengths", "dsa", "sfa", "indexer"],
)
def test_adaptive_verification_admission(cpu_manager_resources, monkeypatch, enabled, steps, unsupported, error):
    groups = [
        _attn_group(AscendDSABackend, AscendDSAMetadataBuilder),
        _attn_group(AscendSFABackend, AscendSFAMetadataBuilder),
        _attn_group(AscendSFAIndexerBackend, AscendSFAIndexerMetadataBuilder),
    ]
    if unsupported == "host":
        monkeypatch.setattr(AscendSFABackend, "supports_device_cpu_query_lens_mismatch", classmethod(lambda cls: False))
    elif unsupported is not None:
        monkeypatch.setattr(unsupported, "get_varlen_cudagraph_max_query_len", classmethod(lambda *args: None))
    kwargs = _adaptive_manager_kwargs(groups, enabled=enabled, steps=steps)
    with pytest.raises(ValueError, match=error) if error else nullcontext():
        manager = adaptive.maybe_create_adaptive_verification_manager(**kwargs)
        if enabled:
            assert isinstance(manager, adaptive.AdaptiveVerificationManager)
        else:
            assert manager is None


@pytest.mark.parametrize("enabled", [False, True])
def test_adaptive_verification_manager_initialization(cpu_manager_resources, monkeypatch, enabled):
    kwargs = _adaptive_manager_kwargs([_attn_group(AscendSFABackend, AscendSFAMetadataBuilder)], enabled=enabled)
    group = kwargs["attn_groups"][0][0]
    group.layer_names = ["target"]
    speculator = AscendMTPSpeculator.__new__(AscendMTPSpeculator)
    speculator.draft_attn_layer_names = {"draft"}
    speculator.enable_adaptive_verification = enabled
    cache_spec = SimpleNamespace(
        block_size=128,
        uses_slot_mapping=True,
        dcp_sharded=False,
        block_table_token_alignment=1,
        max_num_blocks_per_req=lambda *args: 1,
    )
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        is_encoder_decoder=False,
        max_model_len=128,
        speculator=speculator,
        vllm_config=kwargs["vllm_config"],
        device=torch.device("cpu"),
        jit_warmup_registry=SimpleNamespace(activate=nullcontext),
        model_state=SimpleNamespace(
            get_additional_cg_support=lambda: (AttentionCGSupport.ALWAYS, None),
            num_new_sampled_tokens_per_step=1,
        ),
        req_states=kwargs["req_states"],
        input_buffers=SimpleNamespace(query_start_loc=kwargs["query_start_loc"]),
        vocab_size=128,
        max_num_reqs=2,
        max_num_tokens=16,
        dcp_size=1,
        dcp_rank=0,
    )
    upstream = patch_model_runner.upstream
    monkeypatch.setattr(
        upstream,
        "init_attn_backend",
        lambda *args, **kw: (kwargs["attn_groups"], attn_utils.AttentionCGSupportInfo(), [128]),
    )
    manager_call = Mock(wraps=upstream.maybe_create_adaptive_verification_manager)
    monkeypatch.setattr(upstream, "maybe_create_adaptive_verification_manager", manager_call)

    class BeforeCacheAllocation(Exception):
        pass

    # Stop after the real manager call, before device cache allocation or graph setup.
    monkeypatch.setattr(upstream, "BlockTables", Mock(side_effect=BeforeCacheAllocation))
    cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=cache_spec, layer_names=["target", "draft"])]
    )
    with pytest.raises(BeforeCacheAllocation):
        patch_model_runner.initialize_kv_cache(runner, cache_config)
    manager_call.assert_called_once()
    assert manager_call.call_args.kwargs["enable_adaptive_verification"] is enabled
    assert manager_call.call_args.kwargs["target_layer_names"] == {"target"}
    assert (runner.adaptive_verification is not None) is enabled
    if enabled:
        assert isinstance(runner.adaptive_verification, adaptive.AdaptiveVerificationManager)
        assert runner.adaptive_verification.req_states is runner.req_states


@pytest.mark.parametrize("speculator_cls", [AscendMTPSpeculator, AscendDFlash2Speculator])
@pytest.mark.parametrize("enabled", [False, True])
def test_adaptive_verification_estimator_lifecycle(monkeypatch, speculator_cls, enabled):
    # CPU-only vLLM exposes a Triton placeholder without these host utilities.
    monkeypatch.setattr(estimator_module.triton, "next_power_of_2", lambda n: 1 << (n - 1).bit_length(), raising=False)
    monkeypatch.setattr(estimator_module.triton, "cdiv", lambda a, b: (a + b - 1) // b, raising=False)
    speculator = speculator_cls.__new__(speculator_cls)
    draft_config = SimpleNamespace(get_hidden_size=lambda: 8, get_vocab_size=lambda: 8)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="mtp" if speculator_cls is AscendMTPSpeculator else "dflash2",
            enable_adaptive_verification=enabled,
            num_speculative_tokens=3,
            draft_model_config=draft_config,
            use_local_argmax_reduction=False,
            draft_sample_method="greedy",
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=8),
        model_config=SimpleNamespace(
            supports_multimodal_inputs=False, max_model_len=128, dtype=torch.float32, use_fp64_gumbel=False
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            data_parallel_size=1,
            data_parallel_rank=0,
            decode_context_parallel_size=1,
            use_ubatching=False,
        ),
    )
    monkeypatch.setattr(upstream_speculator, "_target_feeds_hc_residual", lambda config: False)
    # Exercise the shared constructor's config -> enable -> estimator policy,
    # without method-specific attention/graph buffers or model discovery.
    upstream_speculator.DraftModelSpeculator.__init__(speculator, config, torch.device("cpu"))
    assert speculator.enable_adaptive_verification is enabled
    assert speculator.use_acceptance_estimator is enabled
    # Skip only checkpoint loading. Inherited load_model must create the real estimator.
    monkeypatch.setattr(speculator, "load_draft_model", lambda *args: torch.nn.Identity())
    monkeypatch.setattr(upstream_speculator, "get_layers_from_vllm_config", lambda *args: {})
    speculator.load_model(torch.nn.Identity())
    assert isinstance(speculator.acceptance_estimator, estimator_module.OnlineAcceptanceEstimator) is enabled
    kernels = {}
    for name in ("_local_max_sumexp_kernel", "_predict_kernel", "_accumulate_kernel"):
        kernel = Mock()
        kernel.__getitem__ = Mock(return_value=Mock())
        monkeypatch.setattr(estimator_module, name, kernel)
        kernels[name] = kernel.__getitem__.return_value
    logits, mapping, step = torch.zeros(2, 8), torch.tensor([1, 0]), torch.tensor([0, 1])
    sampled, rejected = torch.tensor([2, 1]), torch.tensor([1, 2])
    if speculator_cls is AscendMTPSpeculator:
        speculator.model.compute_logits = Mock(return_value=logits)
        speculator.sample_draft(
            torch.zeros(2, 8), torch.zeros(2), mapping, speculator.temperature, speculator.seeds, step, None
        )
        prediction_mapping, prediction_step = mapping, step
    else:
        # Model forward and candidate walk are device boundaries. Keep the real
        # DFlash2 generation method that hands realized candidate scores to AV.
        speculator._run_model = Mock(return_value=torch.zeros(6, 8))
        speculator.sample_indices = torch.arange(6)
        speculator.sample_pos = torch.arange(6)
        speculator.sample_idx_mapping = mapping.repeat_interleave(3)
        speculator.sample_col = torch.arange(3).repeat(2)
        speculator.top_k = 2
        candidate_scores = torch.zeros(2, 3, 2)
        speculator.model.compute_candidates = Mock(
            return_value=(torch.zeros(6, 2, dtype=torch.int64), torch.zeros(6, 2))
        )
        speculator._score_candidates = Mock(return_value=candidate_scores)
        speculator.candidate_sampler = SimpleNamespace(sample=Mock(), scores=candidate_scores)
        speculator._generate_draft(2, 6, {}, {}, None)
        speculator.candidate_sampler.sample.assert_called_once()
        prediction_mapping, prediction_step = speculator.sample_idx_mapping, speculator.sample_col
    speculator.observe_verification(mapping, sampled, rejected)
    for kernel in kernels.values():
        assert kernel.call_count == int(enabled)
    if enabled:
        prediction_args = kernels["_predict_kernel"].call_args.args
        assert prediction_args[4] is speculator.draft_token_confidence_probs
        torch.testing.assert_close(prediction_args[12], prediction_mapping)
        torch.testing.assert_close(prediction_args[14], prediction_step)
        feedback_args = kernels["_accumulate_kernel"].call_args.args
        assert feedback_args[2] is mapping
        assert feedback_args[3] is sampled
        assert feedback_args[4] is rejected
        assert speculator.acceptance_estimator._steps_since_refit == 1


def test_adaptive_verification_budget(cpu_manager_resources, monkeypatch):
    kwargs = _adaptive_manager_kwargs([_attn_group(AscendSFABackend, AscendSFAMetadataBuilder)])
    manager = adaptive.maybe_create_adaptive_verification_manager(**kwargs)
    tp_group = SimpleNamespace(broadcast_object=lambda value, src: value)
    monkeypatch.setattr(adaptive, "get_tp_group", lambda: tp_group)
    # Supply a deterministic profile through the real cost-table builder.
    manager.set_cost_curves([(1, 1.0), (2, 1.0)], [(n, float(n)) for n in range(1, 17)])
    scheduled, drafts = {"a": 4, "b": 4}, {"a": [1, 2, 3], "b": [4, 5, 6]}
    batch = SimpleNamespace(num_reqs=2, idx_mapping=torch.tensor([1, 0]))
    high = torch.full((2, 3), 0.99)
    low = torch.tensor([[0.01, 0.02, 0.03], [0.02, 0.01, 0.03]])
    manager.record_confidences(high, batch)
    assert manager.get_num_tokens(scheduled, drafts) == 8
    manager.record_confidences(low, batch)
    torch.testing.assert_close(manager._confidence_probs, low.flip(0))
    # The double buffer reads the preceding step, not the just-recorded values.
    assert manager.get_num_tokens(scheduled, drafts) == 8
    manager.record_confidences(low, batch)
    assert manager.get_num_tokens(scheduled, drafts) == 2
    assert manager._batch_budget == ({"a": 3, "b": 3}, {"a": 1, "b": 1}, 0)
