# mypy: disable-error-code="annotation-unchecked, arg-type, assignment, attr-defined, misc"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused CPU unit tests for prefill_context_kv_only across Ascend MRV2."""

import importlib
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.lilicorr.speculator import LiLiCorrSpeculator

from vllm_ascend import utils
from vllm_ascend.attention.flashmla import target_attn_groups_without_draft_flashmla
from vllm_ascend.attention.mla_v1 import AscendMLABackend
from vllm_ascend.models.deepseek_v4 import dspark as dsv4
from vllm_ascend.models.deepseek_v41 import dspark as dsv41
from vllm_ascend.models.kimi_k3 import kimi_k3_dspark as k3_dspark
from vllm_ascend.models.qwen4_exp.cpu_length_sync import can_skip_qwen_cpu_length_sync
from vllm_ascend.ops.triton.spec_decode import prepare_dflash_inputs as dflash_ops
from vllm_ascend.patch.worker import patch_qwen3_dflash  # noqa: F401
from vllm_ascend.worker.device_metadata import TargetDeviceMetadata
from vllm_ascend.worker.v2 import attn_utils, model_runner
from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState, _partition_gdn_groups
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    AscendDFlashSpeculator,
    detect_draft_attn_architecture,
)
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator
from vllm_ascend.worker.v2.spec_decode.lilicorr.speculator import AscendLiLiCorrSpeculator

kernel_warmup = importlib.import_module("vllm_ascend.model_executor.warmup.kernel_warmup")


def _lin(o: int = 4, i: int = 4) -> nn.Linear:
    return nn.Linear(i, o, bias=False)


def _mod(**kw: object) -> nn.Module:
    m = nn.Module()
    for k, v in kw.items():
        setattr(m, k, v)
    return m


def _assert_retained(wrap: nn.Module, kept: tuple[torch.Tensor, ...], freed: tuple[torch.Tensor, ...]) -> None:
    if not utils.vllm_version_is("0.34.0"):
        spec = DFlashSpeculator.__new__(DFlashSpeculator)
        spec.context_kv_only, spec.model = True, wrap
        spec.free_unused_weights(nn.Module())
        assert all(t.numel() > 0 for t in kept) and all(t.numel() == 0 for t in freed)
        return
    kept_ids = {id(p) for mod in wrap.context_kv_modules() for p in mod.parameters()}
    assert all(id(t) in kept_ids for t in kept) and all(id(t) not in kept_ids for t in freed)


def test_context_kv_only_helpers_and_v0340_gating(monkeypatch):
    monkeypatch.setattr("vllm_ascend.worker.v2.spec_decode.lmhead_tp_utils.lmhead_tp_enable", lambda: True)
    spec = AscendLiLiCorrSpeculator.__new__(AscendLiLiCorrSpeculator)
    spec.use_local_argmax_reduction, spec.context_kv_only = True, True
    monkeypatch.setattr("vllm_ascend.utils.vllm_version_is", lambda ver: ver == "0.34.0")
    cfg = NS(prefill_context_kv_only=True)
    assert not (utils.is_prefill_context_kv_only(cfg) or utils.is_context_kv_only_speculator(spec))
    assert utils.get_num_scheduled_spec_tokens(NS(num_scheduled_spec_tokens=0), 3) == 3
    with pytest.raises(NotImplementedError):
        spec._lmhead_tp_validate_draft_sampling()
    monkeypatch.setattr("vllm_ascend.utils.vllm_version_is", lambda _ver: False)
    assert utils.is_prefill_context_kv_only(cfg) and utils.is_context_kv_only_speculator(spec)
    assert not utils.is_context_kv_only_speculator(MagicMock())
    assert utils.get_num_scheduled_spec_tokens(NS(num_scheduled_spec_tokens=0), 3) == 0
    assert utils.get_num_scheduled_spec_tokens(NS(num_scheduled_spec_tokens=True), 3) == 3
    spec._lmhead_tp_validate_draft_sampling()


@pytest.mark.parametrize("fused", [False, True])
def test_qwen3_dflash_context_kv_modules_and_unannotated_fallbacks(fused: bool):
    def ln() -> nn.LayerNorm:
        return nn.LayerNorm(4)

    sa = _mod(k_norm=ln(), qkv_proj=_lin(12), o_proj=_lin(), attn=_lin())
    lay = _mod(self_attn=sa, mlp=_lin())
    qm = DFlashQwen3Model.__new__(DFlashQwen3Model)
    nn.Module.__init__(qm)
    qm.hidden_norm, qm.norm, qm.fc, qm._num_attn_layers = ln(), ln(), _lin(), 1
    qm._fused_kv_weight = torch.zeros(8, 4) if fused else None
    qm.use_aux_hidden_state, qm._attn_layers, qm._context_qkv_projs = True, [sa.attn], [sa.qkv_proj]
    qm.layers = nn.ModuleList([lay])
    lm = DFlashQwen3ForCausalLM.__new__(DFlashQwen3ForCausalLM)
    nn.Module.__init__(lm)
    lm.model = qm
    kept = (qm.hidden_norm.weight, qm.fc.weight, sa.k_norm.weight, sa.attn.weight)
    freed = (qm.norm.weight, sa.o_proj.weight, lay.mlp.weight)
    if fused:
        freed = (*freed, sa.qkv_proj.weight)
    else:
        kept = (*kept, sa.qkv_proj.weight)
    _assert_retained(lm, kept, freed)

    # Wrapper inheriting DFlashQwen3ForCausalLM around a model without context_kv_modules
    # (e.g. Glm5DSparkForCausalLM) keeps all weights without raising AttributeError.
    unannotated_lm = DFlashQwen3ForCausalLM.__new__(DFlashQwen3ForCausalLM)
    nn.Module.__init__(unannotated_lm)
    unannotated_lm.model = _mod(context_proj=_lin(), mlp=_lin())
    _assert_retained(unannotated_lm, (unannotated_lm.model.context_proj.weight, unannotated_lm.model.mlp.weight), ())

    # Subclass bypassing _build_fused_kv_buffers (e.g. quantized adapter without _attn_layers)
    # keeps all weights safely.
    class _NoFusedBuffersModel(DFlashQwen3Model):
        def _build_fused_kv_buffers(self) -> None:
            return None

    q_sub = _NoFusedBuffersModel.__new__(_NoFusedBuffersModel)
    nn.Module.__init__(q_sub)
    q_sub.hidden_norm, q_sub.mlp = ln(), _lin()
    _assert_retained(q_sub, (q_sub.hidden_norm.weight, q_sub.mlp.weight), ())


@pytest.mark.parametrize("context_kv_only", [False, True])
def test_speculators_and_dspark_propose_context_kv_only(monkeypatch, context_kv_only: bool):
    on_v034 = utils.vllm_version_is("0.34.0")
    monkeypatch.setattr("vllm_ascend.utils.vllm_version_is", lambda _: False)
    seen: list[CUDAGraphMode] = []

    def _cap_cg(self, mode: CUDAGraphMode) -> None:
        seen.append(mode)
        self.query_cudagraph_manager = NS()

    pairs = ((AscendDFlashSpeculator, DFlashSpeculator), (AscendDSparkSpeculator, DSparkSpeculator))
    for cls, base in (*pairs, (AscendLiLiCorrSpeculator, DFlashSpeculator)):
        sp = cls.__new__(cls)
        sp.speculative_config, sp.context_kv_only, sp.update_stream = NS(enforce_eager=False), True, None
        with patch.object(base, "init_cudagraph_manager", _cap_cg):
            sp.init_cudagraph_manager(CUDAGraphMode.FULL)
    assert seen == [CUDAGraphMode.NONE, CUDAGraphMode.NONE, CUDAGraphMode.NONE]

    # AscendDSparkSpeculator bypasses draft LM-head TP wrapping when context_kv_only is enabled.
    ds_wrap = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    ds_wrap.context_kv_only, ds_wrap.max_num_reqs, ds_wrap.num_speculative_steps = context_kv_only, 2, 3
    orig_fn = MagicMock(name="compute_draft_logits")
    draft_mod = NS(compute_draft_logits=orig_fn)
    with patch("vllm_ascend.worker.v2.spec_decode.dspark.speculator.lmhead_tp_enable", return_value=True):
        ds_wrap._lmhead_tp_wrap_draft_logits(draft_mod)
    assert (draft_mod.compute_draft_logits is orig_fn) is context_kv_only

    # AscendLiLiCorrSpeculator skips vocab-sharding validation when context_kv_only is enabled.
    lc = AscendLiLiCorrSpeculator.__new__(AscendLiLiCorrSpeculator)
    lc.context_kv_only = True
    dummy_model = nn.Module()
    with patch.object(LiLiCorrSpeculator, "load_draft_model", return_value=dummy_model):
        assert lc.load_draft_model(nn.Module(), set()) is dummy_model

    if on_v034:
        return
    ds = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    ds.context_kv_only, ds.use_dcp, ds.attn_architecture, ds.use_flashmla = context_kv_only, True, "MLA", False
    ds.pcp_manager, ds.dp_size, ds.dp_rank, ds.max_num_reqs, ds.max_num_tokens = None, 1, 0, 2, 8
    ds.num_query_per_req, ds.num_speculative_steps, ds.arange_np = 4, 3, np.arange(8, dtype=np.int32)
    ds.draft_kv_cache_group_id, ds.draft_kv_cache_group_ids, ds._group_causal = 0, [], True
    ds.supports_decode_metadata_update, ds._proposal_host_lengths = False, None
    ds.vllm_config = NS(parallel_config=NS(), model_config=NS(max_model_len=128))
    ds.kv_cache_config, ds.block_tables = NS(kv_cache_groups=[]), NS(slot_mappings=torch.zeros(1, 8, dtype=torch.int32))
    ds.input_buffers = NS(positions=torch.arange(8), seq_lens=torch.tensor([5, 0], dtype=torch.int32))
    ds.draft_tokens = torch.zeros(2, 3, dtype=torch.int32)
    ds.query_cudagraph_manager = NS(dispatch=lambda r, t, *a, **kw: BatchExecutionDescriptor(CUDAGraphMode.NONE, t, r))
    ds._store_hidden_states, ds._precompute_context_kv = MagicMock(), MagicMock()
    ds.prepare_context_anchor, ds._prepare_eplb_forward, ds._generate_draft = MagicMock(), MagicMock(), MagicMock()
    ds._prepare_draft_dcp_metadata_inputs = MagicMock(return_value=(torch.tensor([5]), torch.zeros(1, dtype=bool)))
    monkeypatch.setattr(AscendDSparkSpeculator, "attn_vllm_config", property(lambda s: s.vllm_config))
    b = NS(num_reqs=1, num_tokens=4, is_prefilling_np=np.array([True], dtype=np.bool_))
    b.seq_lens_cpu_upper_bound = torch.tensor([5], dtype=torch.int32)
    b.query_start_loc = torch.tensor([0, 4], dtype=torch.int32)
    z1, p = torch.zeros(1, dtype=torch.int32), "vllm_ascend.worker.v2.spec_decode.dspark.speculator"
    empty: dict[str, object] = {}
    args = (b, empty, empty, torch.zeros(4, 4), None, z1, z1, z1, z1, torch.ones(1), torch.zeros(1, dtype=torch.int64))
    with (
        patch(f"{p}.build_attn_metadata_wrapper", nullcontext),
        patch(f"{p}.build_attn_metadata_factory", lambda *_a, **_kw: nullcontext()),
        patch.object(DSparkSpeculator, "_build_attn_metadata", return_value={}),
    ):
        out = ds.propose(*args)
    assert out.shape == (1, 3) and (ds._generate_draft.called is (not context_kv_only))
    assert ds._prepare_draft_dcp_metadata_inputs.called is (not context_kv_only)


def test_empty_builders_warmup_sync_and_triton_clamp(monkeypatch):
    on_v034 = utils.vllm_version_is("0.34.0")
    monkeypatch.setattr("vllm_ascend.utils.vllm_version_is", lambda _: False)
    eg = NS(metadata_builders=[], backend=AscendMLABackend, layer_names=["draft.0"], get_metadata_builder=MagicMock())
    t1, z11 = torch.tensor([0, 1], dtype=torch.int32), torch.zeros(1, 1, dtype=torch.int32)
    p0, o1 = torch.zeros(1, dtype=torch.int64), torch.ones(1, dtype=torch.int32)
    kw = {"attn_groups": [[eg]], "num_reqs": 1, "num_tokens": 1, "query_start_loc_gpu": t1, "query_start_loc_cpu": t1}
    kw |= {"max_query_len": 1, "seq_lens": o1, "seq_lens_np": np.ones(1, dtype=np.int32), "max_seq_len": 1}
    kw |= {"positions": p0, "block_tables": (z11,), "slot_mappings": z11, "kv_cache_config": NS(kv_cache_groups=[NS()])}
    with (
        patch("vllm_ascend.worker.v2.attn_utils.uses_turboquant_groups", return_value=False),
        patch.object(AscendMLABackend, "get_impl_cls", return_value=object),
    ):
        assert attn_utils.build_attn_metadata(**kw) == {}
        assert detect_draft_attn_architecture([[eg]]) == ("MLA", False, False)
    assert target_attn_groups_without_draft_flashmla([[eg]], {"draft.0"}) == [[eg]]
    assert _partition_gdn_groups([[eg]]) == ([[eg]], [])
    dm = TargetDeviceMetadata.__new__(TargetDeviceMetadata)
    dm._failed, dm._tasks, dm.executor = False, (), NS(submission_in_flight=False)
    with dm.build([[eg]], full_graph=True):
        pass
    assert dm._tasks == ()
    runner = model_runner.NPUModelRunner.__new__(model_runner.NPUModelRunner)
    runner.speculator, runner.attn_groups = NS(draft_attn_layer_names=set()), [[eg]]
    assert runner._has_target_flashmla_attention() is False
    # Kernel warmup skips both when prefill_context_kv_only=True and when draft group has empty builders.
    mr_ckv = NS(jit_warmup_registry=NS(), speculative_config=NS(prefill_context_kv_only=True))
    assert kernel_warmup._register_dspark_swa_indices_warmup(NS(model_runner=mr_ckv)) == 0
    mr_empty = NS(jit_warmup_registry=NS(activate=nullcontext), speculative_config=NS(prefill_context_kv_only=False))
    with patch.object(kernel_warmup, "_iter_dspark_draft_groups", return_value=[(NS(), eg)]):
        assert kernel_warmup._register_dspark_swa_indices_warmup(NS(model_runner=mr_empty)) == 0
    # Qwen4Exp CPU length sync skips empty-builder groups alongside a valid QSA group.
    if not on_v034:
        from vllm_ascend.models.qwen4_exp.model_state import AscendQwen4ExpModelState
        from vllm_ascend.models.qwen4_exp.qsa import AscendQSABackend, AscendQSAMetadataBuilder
        from vllm_ascend.worker.v2.spec_decode.mtp.speculator import AscendMTPSpeculator

        qsa_group = NS(
            backend=AscendQSABackend,
            get_metadata_builder=lambda _: object.__new__(AscendQSAMetadataBuilder),
        )
        draft = object.__new__(AscendMTPSpeculator)
        draft.attn_groups, draft.attn_architecture, draft.attn_backend = [[qsa_group, eg]], "QSA", AscendQSABackend
        q_runner = NS(
            model_state=object.__new__(AscendQwen4ExpModelState),
            ascend_config=NS(sparse_kv_offload_config=NS(enabled=False)),
            parallel_config=NS(decode_context_parallel_size=1, prefill_context_parallel_size=1),
            supports_mm_inputs=False,
            model_config=NS(architecture="Qwen4ExpForCausalLM"),
            attn_groups=[[qsa_group, eg]],
            speculator=draft,
        )
        assert can_skip_qwen_cpu_length_sync(q_runner) is True
    assert not eg.get_metadata_builder.called

    monkeypatch.setattr(dflash_ops, "init_device_properties_triton", lambda: None)
    monkeypatch.setattr(dflash_ops, "get_vectorcore_num", lambda: 16)
    monkeypatch.setattr(dflash_ops, "_prepare_dflash_inputs_kernel", MagicMock())
    monkeypatch.setattr(dflash_ops.triton, "next_power_of_2", lambda n: 1 << (n - 1).bit_length(), raising=False)
    zt, zr, f4 = torch.zeros(10, dtype=torch.int32), torch.zeros(4, dtype=torch.int32), torch.zeros(4)
    q5 = torch.zeros(5, dtype=torch.int32)
    bufs = NS(input_ids=zt, positions=zt, query_start_loc=q5, seq_lens=zr, is_padding=torch.ones(10, dtype=torch.bool))
    b = NS(num_reqs=4, num_scheduled_tokens=np.full(4, 2, dtype=np.int32), positions=zt, query_start_loc=q5)
    b.idx_mapping = zr
    args = (bufs, zt, zt, zt, zr, zr, zr, f4, zr, b, zr, zr, zr, zr, f4, zr, torch.zeros(4, 2, dtype=torch.int32))
    dflash_ops.prepare_dflash_inputs_triton(*args, 16, 0, 1, 1, 0, 4, 3, 4, 10, 64)
    assert torch.all(~bufs.is_padding)


@pytest.mark.parametrize(("scheduled", "fallback", "expect_spec"), [(0, 3, False), (True, 0, False), (False, 3, True)])
def test_mamba_hybrid_and_runner_sync_guards(monkeypatch, scheduled: object, fallback: int, expect_spec: bool):
    monkeypatch.setattr("vllm_ascend.utils.vllm_version_is", lambda _: False)
    st = AscendMambaHybridModelState.__new__(AscendMambaHybridModelState)
    st._align_mode = False
    st.vllm_config = NS(num_speculative_tokens=fallback, num_scheduled_spec_tokens=scheduled)
    st.vllm_config.speculative_config, st.vllm_config.parallel_config = None, NS()
    st.model_config, st.num_accepted_tokens_gpu = NS(max_model_len=128), torch.zeros(4, dtype=torch.int32)
    t2, a2 = torch.tensor([0, 4], dtype=torch.int32), np.array([0, 4], dtype=np.int32)
    b = NS(num_reqs=1, num_tokens=4, is_prefilling_np=np.array([True], dtype=np.bool_), prefill_runs_as_decode_np=None)
    b.num_draft_tokens_per_req, b.num_scheduled_tokens, b.max_query_len = None, np.array([4], dtype=np.int32), 4
    b.query_start_loc, b.query_start_loc_np, b.seq_lens, b.seq_lens_np = t2, a2, t2[1:], a2[1:]
    b.dcp_local_seq_lens, b.positions, b.attn_state, b.idx_mapping = None, torch.arange(4), None, t2[:1]
    args = (b, CUDAGraphMode.NONE, (torch.zeros(1, 1, dtype=torch.int32),), torch.zeros(1, 4, dtype=torch.int32))
    with patch("vllm_ascend.worker.v2.model_states.mamba_hybrid.build_attn_metadata", return_value={"ok": True}) as mk:
        res = st._build_attn_metadata_for_batch(
            *args, [[NS(layer_names=[])]], NS(kv_cache_groups=[]), False, use_pcp_padding=False
        )
    spec_m = mk.call_args.kwargs["model_specific_attn_metadata"]
    assert res == {"ok": True} and (spec_m.num_accepted_tokens is not None) is expect_spec


@pytest.mark.parametrize(
    ("config_kind", "scheduled", "fallback", "expected"),
    [
        pytest.param("attr", 0, 3, False, id="explicit-zero-overrides-fallback"),
        pytest.param("attr", 2, 0, True, id="positive-live-count-overrides-zero-fallback"),
        pytest.param("absent", None, 3, True, id="absent-takes-positive-fallback"),
        pytest.param("absent", None, 0, False, id="absent-takes-zero-fallback"),
        pytest.param("attr", None, 3, True, id="none-takes-fallback"),
        pytest.param("magic_mock", None, 3, True, id="magicmock-takes-fallback"),
        pytest.param("attr", True, 0, False, id="bool-true-ignored"),
        pytest.param("attr", False, 3, True, id="bool-false-ignored"),
    ],
)
def test_npu_model_runner_init_sync_spec_pp_cpu_counts(
    monkeypatch, config_kind: str, scheduled: object, fallback: int, expected: bool
):
    monkeypatch.setattr("vllm_ascend.utils.vllm_version_is", lambda _: False)
    if config_kind == "magic_mock":
        vllm_config: object = MagicMock()
    else:
        vllm_config = NS(
            parallel_config=NS(enable_eplb=False, pipeline_parallel_size=2, data_parallel_size=1),
            offload_config=NS(offload_backend="auto"),
        )
        if config_kind == "attr":
            vllm_config.num_scheduled_spec_tokens = scheduled

    ascend_config = NS(
        eplb_config=None,
        dump_config_path=None,
        finegrained_tp_config=NS(requires_graph=lambda _cfg: False),
    )

    def _parent_init(self, cfg, dev):
        self.vllm_config, self.device = cfg, dev
        self.compilation_config = NS(cudagraph_mode=CUDAGraphMode.NONE, mode=NS(), has_full_cudagraphs=lambda: False)
        self.model_config = NS(enforce_eager=False, architecture="KimiK3ForCausalLM", hf_config=NS())
        self.speculative_config = NS(method="dspark")
        self.use_pp, self.is_last_pp_rank = True, True
        self.parallel_config = NS(enable_eplb=False, pipeline_parallel_size=2, data_parallel_size=1)
        self.pp_handler = NS(broadcast_group=object())
        self.max_num_reqs, self.max_model_len, self.max_num_tokens = 2, 32, 8
        self.num_speculative_steps, self.vocab_size, self.dtype = fallback, 16, torch.float16
        self.req_states, self.input_buffers, self.speculator = object(), object(), object()

    mr_mod = "vllm_ascend.worker.v2.model_runner"
    with (
        patch(f"{mr_mod}.get_ascend_config", return_value=ascend_config),
        patch(f"{mr_mod}.set_potential_max_tokens"),
        patch(f"{mr_mod}.resolve_spec_pp_support", return_value=None),
        patch(f"{mr_mod}.torch_cuda_wrapper", return_value=nullcontext()),
        patch.object(GPUModelRunner, "__init__", _parent_init),
        patch("torch.distributed.get_backend", return_value="gloo"),
        patch(f"{mr_mod}.AscendEPLBController", return_value="eplb"),
        patch(f"{mr_mod}.init_speculator", return_value=NS(context_kv_only=True)),
        patch(f"{mr_mod}.AscendRequestState", return_value="req"),
        patch(f"{mr_mod}.AscendInputBuffers", return_value="buf"),
        patch(f"{mr_mod}.set_cos_and_sin"),
        patch(f"{mr_mod}.set_mc2_tokens_capacity"),
        patch(f"{mr_mod}.set_mc2_mask"),
        patch(f"{mr_mod}.breakable_cudagraph.is_breakable_cudagraph_enabled", return_value=False),
        patch("torch.npu.Event", return_value="event"),
        patch("torch.npu.Stream", return_value="stream"),
        patch("torch.empty", return_value=torch.zeros(2, dtype=torch.int32)),
        patch(
            "vllm_ascend.observability.runtime_guard.processor.RuntimeGuardProcessor.bind",
            return_value=MagicMock(name="runtime_guard"),
        ),
        patch(f"{mr_mod}.install_trace", return_value=None),
    ):
        runner = model_runner.NPUModelRunner(vllm_config, torch.device("cpu"))

    assert runner.sync_spec_pp_cpu_counts is expected
    with patch(f"{mr_mod}.is_deepseek_v41", return_value=False):
        assert runner._can_skip_spec_cpu_length_sync() is (not expected)
    with (
        patch.object(GPUModelRunner, "postprocess_num_computed_tokens"),
        patch.object(runner, "_copy_num_computed_tokens_to_cpu") as mock_copy,
    ):
        runner.postprocess_num_computed_tokens(NS())
    assert mock_copy.called is expected


@pytest.mark.parametrize(
    ("mod", "model_cls", "lm_cls", "layer_attr"),
    [
        pytest.param(
            dsv4,
            dsv4.DeepseekV4DSparkModel,
            dsv4.DSparkDeepseekV4ForCausalLM,
            "DeepseekV4DecoderLayer",
            id="v4",
        ),
        pytest.param(
            dsv41,
            dsv41.DeepseekV41DSparkModel,
            dsv41.DSparkDeepseekV41ForCausalLM,
            "DeepseekV41DSparkDecoderLayer",
            id="v41",
        ),
    ],
)
@pytest.mark.parametrize("flag", [False, True])
def test_deepseek_v4_and_v41_dspark_context_kv_and_oproj_gating(
    monkeypatch, mod, model_cls, lm_cls, layer_attr: str, flag: bool
):
    def ln() -> nn.LayerNorm:
        return nn.LayerNorm(4)

    cfg = NS(hc_mult=2, hidden_size=4, vocab_size=8, draft_vocab_size=None, dspark_block_size=4)
    cfg.num_hidden_layers, cfg.dspark_target_layer_ids, cfg.dspark_markov_rank, cfg.n_mtp_layers = 2, [0], 2, 1
    cfg.head_dim, cfg.rms_norm_eps, cfg.hc_eps, cfg.quantization_config = 4, 1e-6, 1e-5, None
    sc = NS(prefill_context_kv_only=flag, draft_model_config=NS(hf_config=cfg, hf_text_config=cfg))
    vc = NS(speculative_config=sc, parallel_config=NS(use_sequence_parallel_moe=False))
    vc.model_config, vc.quant_config = NS(dtype=torch.float16), None
    impls: list[MagicMock] = []

    def _mk(*_a, **_kw):
        m_im = MagicMock(spec=mod.AscendDSAImpl)
        impls.append(m_im)
        sa_d = _mod(kv_norm=ln(), wq_a=_lin(), dsa_attn=_mod(swa_cache_layer=_lin(), dsa_attn=_mod(impl=m_im)))
        return _mod(use_sequence_parallel_moe=False, self_attn=sa_d, mlp=_lin())

    if hasattr(mod, "normalize_deepseek_v41_config"):
        monkeypatch.setattr(mod, "normalize_deepseek_v41_config", lambda c: c)
    n_a = "VocabParallelEmbedding MergedColumnParallelLinear ColumnParallelLinear RMSNorm"
    for a in (*n_a.split(), "DSparkMarkovHead", "DSparkConfidenceHead"):
        monkeypatch.setattr(mod, a, lambda *a, **kw: _lin())
    monkeypatch.setattr(mod, layer_attr, _mk)
    with patch("vllm_ascend.utils.vllm_version_is", return_value=False):
        dm = model_cls(vllm_config=vc, prefix="model")
    assert len(impls) == 1 and (impls[0].reserve_oproj_buffers.called is (not flag))
    lm = lm_cls.__new__(lm_cls)
    nn.Module.__init__(lm)
    lm.model = dm
    sa = dm.layers["2"].self_attn
    kept = (dm.main_proj.weight, dm.main_norm.weight, dm.context_wkv_proj.weight, sa.kv_norm.weight)
    freed = (dm.embed_tokens.weight, sa.wq_a.weight, dm.layers["2"].mlp.weight)
    _assert_retained(lm, kept, freed)


@pytest.mark.parametrize("prolog_v3", [False, True])
def test_kimi_k3_dspark_context_kv_modules(prolog_v3: bool):
    def ln() -> nn.LayerNorm:
        return nn.LayerNorm(4)

    k3 = k3_dspark.AscendK3DSparkModel.__new__(k3_dspark.AscendK3DSparkModel)
    nn.Module.__init__(k3)
    k3.context_proj, k3.context_norm, k3.embed_tokens = _lin(), ln(), _lin()
    kv_ln, q_proj, q_ln, kv_b = ln(), _lin(), ln(), _lin()
    k3_sa = _mod(
        fused_qkv_a_proj=_lin(),
        mla_attn=_mod(mla_attn=_lin()),
        kv_a_layernorm=kv_ln,
        q_proj=q_proj,
        q_a_layernorm=q_ln,
        kv_b_proj=kv_b,
    )
    k3_sa.impl = NS(
        kv_a_layernorm=kv_ln,
        q_proj=q_proj,
        q_a_layernorm=q_ln,
        kv_b_proj=kv_b,
        supports_prolog_v3_writer=lambda: prolog_v3,
    )
    k3.layers = nn.ModuleList([_mod(self_attn=k3_sa, mlp=_lin())])
    k3_lm = k3_dspark.AscendK3DSparkForCausalLM.__new__(k3_dspark.AscendK3DSparkForCausalLM)
    nn.Module.__init__(k3_lm)
    k3_lm.model = k3
    kept = (k3.context_proj.weight, k3.context_norm.weight, k3_sa.fused_qkv_a_proj.weight, kv_ln.weight)
    freed = (k3.embed_tokens.weight, k3.layers[0].mlp.weight)
    prolog_weights = (q_proj.weight, q_ln.weight, kv_b.weight)
    if prolog_v3:
        kept = (*kept, *prolog_weights)
    else:
        freed = (*freed, *prolog_weights)
    _assert_retained(k3_lm, kept, freed)
