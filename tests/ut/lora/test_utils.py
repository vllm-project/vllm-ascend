# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import vllm.lora.layers as ll
import vllm.lora.utils as lora_utils
import vllm.model_executor.layers.linear as lin
from vllm.config import LoRAConfig
from vllm.lora.layers import LoRAMapping
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import LoRALayerWeights
from vllm.lora.model_manager import LRUCacheLoRAModelManager
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor.custom_op import maybe_get_oot_by_class
from vllm.model_executor.layers.fused_moe import MoERunner
from vllm.model_executor.models import SupportsLoRA

import vllm_ascend.lora.utils as au
from vllm_ascend.lora.fused_moe import AscendFusedMoE3DWithLoRA, AscendFusedMoEWithLoRA
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU
from vllm_ascend.ops.linear import AscendQKVParallelLinear

_AU = "vllm_ascend.lora.utils"


def _lcfg(max_loras=1, **kw):
    return LoRAConfig(max_loras=max_loras, max_lora_rank=8, lora_dtype=torch.bfloat16, **kw)


def _stub(cls, oot=True, **attrs):
    obj = object.__new__(maybe_get_oot_by_class(cls) if oot else cls)
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


def _base_lin(base_cls, prefix: str, in_dim: int = 4, out_dim: int = 6, **extra) -> torch.nn.Module:
    layer = _stub(base_cls)
    torch.nn.Module.__init__(layer)
    attrs = dict(
        input_size=in_dim,
        output_size=out_dim,
        input_size_per_partition=in_dim,
        output_partition_sizes=[out_dim],
        tp_size=1,
        tp_rank=0,
        skip_bias_add=False,
        return_bias=False,
        bias=None,
        prefix=prefix,
        quant_method=None,
        weight=torch.nn.Parameter(torch.zeros(out_dim, in_dim, dtype=torch.bfloat16)),
    )
    if base_cls is not lin.ReplicatedLinear:
        attrs.update(output_size_per_partition=out_dim, output_sizes=[out_dim])
        attrs["gather_output" if base_cls is lin.ColumnParallelLinear else "input_is_parallel"] = False
        if base_cls is lin.RowParallelLinear:
            attrs["reduce_results"] = True
    for k, v in {**attrs, **extra}.items():
        setattr(layer, k, v)
    return layer


def _lin(w_cls, b_cls, prefix, in_dim, out_dim, *, tp_size=1, max_loras=1):
    layer = _stub(w_cls, oot=False)
    torch.nn.Module.__init__(layer)
    layer.base_layer = _stub(b_cls, oot=False, prefix=prefix)
    layer.n_slices = len(out_dim) if isinstance(out_dim, list) else 1
    layer.input_size, layer.tp_size, layer.tp_rank, layer.device = in_dim, tp_size, 0, torch.device("cpu")
    if isinstance(out_dim, list):
        layer.output_sizes, layer.output_slices, layer.output_ids = out_dim, tuple(out_dim), (0,) * len(out_dim)
    else:
        layer.output_size = out_dim
    layer.create_lora_weights(max_loras, _lcfg(max_loras))
    return layer


def _fake_model(*, dp_size=1, n_local_groups=1, multimodal=False, include_moe=False):
    class _FakeModel(torch.nn.Module, SupportsLoRA):
        config, is_3d_moe_weight, is_non_gated_moe = None, False, False
        embedding_modules, packed_modules_mapping, lora_skip_prefixes = {}, {}, []
        get_mm_mapping = lambda self: SimpleNamespace(language_model=("language_model.",), tower_model=(), connector=())
        get_expert_mapping = lambda self: [("experts.0.w1", "experts.0.gate_proj.", 0, "w1")]

    model, pfx = _FakeModel(), ("language_model.model.layers.0" if multimodal else "model.layers.0")
    l0 = torch.nn.Module()
    l0.self_attn = torch.nn.Module()
    l0.self_attn.wq_a = _base_lin(lin.ReplicatedLinear, f"{pfx}.self_attn.wq_a", 8, 8)
    if include_moe:
        l0.mlp = torch.nn.Module()
        l0.mlp.shared_experts = sh = torch.nn.Module()
        sh.down_proj = _base_lin(lin.ReplicatedLinear, f"{pfx}.mlp.shared_experts.down_proj")
        moe = _stub(MoERunner)
        torch.nn.Module.__init__(moe)
        p_cfg = SimpleNamespace(tp_size=1, tp_rank=0, ep_rank=0, use_ep=False)
        moe.moe_config = SimpleNamespace(
            dp_size=dp_size,
            hidden_dim=4,
            num_local_experts=2,
            num_experts=2,
            intermediate_size_per_partition=4,
            experts_per_token=1,
            is_act_and_mul=True,
            moe_parallel_config=p_cfg,
        )
        moe.routed_experts = SimpleNamespace(quant_method=SimpleNamespace(is_monolithic=False))
        moe._shared_experts, moe.set_lora_context = sh, MagicMock()
        l0.mlp.experts = moe
    else:
        l0.self_attn.wo_a = _base_lin(
            lin.ColumnParallelLinear, f"{pfx}.self_attn.wo_a", _ascend_dsa_n_local_groups=n_local_groups
        )
        l0.self_attn.k_proj = _base_lin(lin.ColumnParallelLinear, f"{pfx}.self_attn.k_proj")
        l0.self_attn.wo_b = _base_lin(lin.RowParallelLinear, f"{pfx}.self_attn.wo_b", 6, 4)
    inner = torch.nn.Module()
    inner.layers = torch.nn.ModuleList([l0])
    setattr(model, "language_model" if multimodal else "model", torch.nn.Module() if multimodal else inner)
    if multimodal:
        model.language_model.model = inner
    return model


def _punica() -> PunicaWrapperNPU:
    p = object.__new__(PunicaWrapperNPU)
    PunicaWrapperBase.__init__(p, 16, 2, "cpu")
    p._single_lora_mask = None
    return p


def test_refresh_all_lora_classes_prepends_wrappers_and_guards_upstream_fallbacks() -> None:
    """Verify Ascend LoRA wrappers prepend registry and guard unsupported upstream fallbacks."""
    sentinel, original = object(), lora_utils._all_lora_classes
    upstream = (
        ll.ColumnParallelLinearWithLoRA,
        ll.ColumnParallelLinearWithShardedLoRA,
        ll.MergedColumnParallelLinearWithLoRA,
        ll.MergedColumnParallelLinearWithShardedLoRA,
        ll.MergedColumnParallelLinearVariableSliceWithLoRA,
        ll.QKVParallelLinearWithLoRA,
        ll.QKVParallelLinearWithShardedLoRA,
        ll.MergedQKVParallelLinearWithLoRA,
        ll.MergedQKVParallelLinearWithShardedLoRA,
        ll.RowParallelLinearWithLoRA,
        ll.RowParallelLinearWithShardedLoRA,
        ll.ReplicatedLinearWithLoRA,
    )
    try:
        lora_utils._all_lora_classes = (*upstream, FusedMoEWithLoRA, sentinel, FusedMoE3DWithLoRA)
        au.refresh_all_lora_classes()
        classes = list(lora_utils._all_lora_classes)
        au.refresh_all_lora_classes()
        assert list(lora_utils._all_lora_classes) == classes and sentinel in classes
        assert classes[:7] == [
            au.AscendReplicatedLinearWithLoRA,
            au.AscendRowParallelLinearWithLoRA,
            au.AscendColumnParallelLinearWithLoRA,
            au.AscendMergedColumnParallelLinearWithLoRA,
            au.AscendMergedQKVParallelLinearWithLoRA,
            AscendFusedMoEWithLoRA,
            AscendFusedMoE3DWithLoRA,
        ]
        assert all(c not in classes for c in (*upstream, FusedMoEWithLoRA, FusedMoE3DWithLoRA))
        assert AscendFusedMoEWithLoRA.__name__ == "FusedMoEWithLoRA"
        assert AscendFusedMoE3DWithLoRA.__name__ == "FusedMoE3DWithLoRA"

        g = {
            u: next(c for c in classes if c.__bases__ == (u,) and getattr(c, "_ascend_unsupported_lora_guarded", 0))
            for u in upstream
        }
        c0, c1, bad = _lcfg(), _lcfg(fully_sharded_loras=True), {"_ascend_lora_unsupported_reason": "no"}
        gu, qkv = ["gate_proj", "up_proj"], ["q_proj", "k_proj", "v_proj"]
        c_ok, c_2g, c_1g = (
            _stub(lin.ColumnParallelLinear),
            _stub(lin.ColumnParallelLinear, _ascend_dsa_n_local_groups=2),
            _stub(lin.ColumnParallelLinear, _ascend_dsa_n_local_groups=1),
        )
        m_ok, m_bad = (
            _stub(lin.MergedColumnParallelLinear, output_sizes=[8, 8, 8]),
            _stub(lin.MergedColumnParallelLinear, output_sizes=[8, 8, 8], **bad),
        )
        q_bad, aq_bad = _stub(lin.QKVParallelLinear, **bad), _stub(AscendQKVParallelLinear, oot=False, **bad)
        r_ok, r_bad, rep_bad = (
            _stub(lin.RowParallelLinear),
            _stub(lin.RowParallelLinear, **bad),
            _stub(lin.ReplicatedLinear, **bad),
        )
        for cls, src, cfg, pk, exp in (
            (au.AscendColumnParallelLinearWithLoRA, c_ok, c0, ["wo_a"], True),
            (au.AscendColumnParallelLinearWithLoRA, c_2g, c0, ["wo_a"], False),
            (au.AscendMergedColumnParallelLinearWithLoRA, m_bad, c0, gu, False),
            (au.AscendMergedQKVParallelLinearWithLoRA, aq_bad, c0, qkv, False),
            (au.AscendReplicatedLinearWithLoRA, rep_bad, c0, ["wkv"], False),
            (g[ll.ColumnParallelLinearWithLoRA], c_2g, c0, ["wo_a"], False),
            (g[ll.ColumnParallelLinearWithShardedLoRA], c_ok, c1, ["k_proj"], True),
            (g[ll.ColumnParallelLinearWithShardedLoRA], c_1g, c1, ["wo_a"], False),
            (g[ll.MergedColumnParallelLinearWithLoRA], m_ok, c0, gu, True),
            (g[ll.MergedColumnParallelLinearWithLoRA], m_bad, c0, gu, False),
            (g[ll.MergedColumnParallelLinearWithShardedLoRA], m_bad, c1, gu, False),
            (g[ll.MergedColumnParallelLinearVariableSliceWithLoRA], m_ok, c0, ["q", "k", "v"], True),
            (g[ll.MergedColumnParallelLinearVariableSliceWithLoRA], m_bad, c0, ["q", "k", "v"], False),
            (g[ll.QKVParallelLinearWithLoRA], q_bad, c0, ["qkv"], False),
            (g[ll.QKVParallelLinearWithShardedLoRA], q_bad, c1, ["qkv"], False),
            (g[ll.MergedQKVParallelLinearWithLoRA], q_bad, c0, qkv, False),
            (g[ll.MergedQKVParallelLinearWithShardedLoRA], q_bad, c1, qkv, False),
            (g[ll.RowParallelLinearWithLoRA], r_ok, c0, [], True),
            (g[ll.RowParallelLinearWithLoRA], r_bad, c0, [], False),
            (g[ll.RowParallelLinearWithShardedLoRA], r_ok, c1, [], True),
            (g[ll.RowParallelLinearWithShardedLoRA], r_bad, c1, [], False),
            (g[ll.ReplicatedLinearWithLoRA], rep_bad, c0, ["wkv"], False),
        ):
            assert (
                bool(
                    cls.can_replace_layer(source_layer=src, lora_config=cfg, packed_modules_list=pk, model_config=None)
                )
                is exp
            )
    finally:
        lora_utils._all_lora_classes = original


def test_ascend_dense_wrappers_slot_tracking_and_dim_validation() -> None:
    cfg = _lcfg(2)
    for w_cls, b_cls, packed in (
        (au.AscendReplicatedLinearWithLoRA, maybe_get_oot_by_class(lin.ReplicatedLinear), []),
        (au.AscendRowParallelLinearWithLoRA, maybe_get_oot_by_class(lin.RowParallelLinear), []),
        (au.AscendColumnParallelLinearWithLoRA, maybe_get_oot_by_class(lin.ColumnParallelLinear), []),
        (
            au.AscendMergedColumnParallelLinearWithLoRA,
            maybe_get_oot_by_class(lin.MergedColumnParallelLinear),
            ["g", "u"],
        ),
        (au.AscendMergedQKVParallelLinearWithLoRA, AscendQKVParallelLinear, ["q", "k", "v"]),
    ):
        assert w_cls.can_replace_layer(
            source_layer=object.__new__(b_cls), lora_config=cfg, packed_modules_list=packed, model_config=None
        )

    s = _lin(au.AscendReplicatedLinearWithLoRA, lin.ReplicatedLinear, "l.0.wq_a", 4, 6, max_loras=2)
    m = _lin(
        au.AscendMergedColumnParallelLinearWithLoRA, lin.MergedColumnParallelLinear, "l.0.gu", 4, [6, 6], max_loras=2
    )
    assert s.lora_b_packed is None and m.lora_a_packed is None and m.lora_b_packed is None
    s.set_lora(1, torch.ones(8, 4, dtype=torch.bfloat16), torch.ones(6, 8, dtype=torch.bfloat16))
    m.set_lora(1, [torch.ones(8, 4, dtype=torch.bfloat16)] * 2, [torch.ones(6, 8, dtype=torch.bfloat16)] * 2)
    assert s._lora_slots == {1} and m._lora_slots == {1}
    with patch.object(BaseLinearLayerWithLoRA, "_apply_lora_to_output", return_value="up") as up:
        assert s._apply_lora_to_output(torch.randn(2, 4, dtype=torch.bfloat16), torch.zeros(2, 6)) == "up"
        assert m._apply_lora_to_output(torch.randn(2, 4, dtype=torch.bfloat16), torch.zeros(2, 12)) == "up"
    assert up.call_count == 2
    s.reset_lora(1)
    assert s._lora_slots == set()

    rep = _lin(au.AscendReplicatedLinearWithLoRA, lin.ReplicatedLinear, "m.0.wq_a", 4096, 1024)
    row = _lin(au.AscendRowParallelLinearWithLoRA, lin.RowParallelLinear, "m.0.wo_b", 512, 4096, tp_size=8)
    col = _lin(au.AscendColumnParallelLinearWithLoRA, lin.ColumnParallelLinear, "m.0.wo_a", 1024, 512, tp_size=8)
    col.is_merged_col_linear = False
    for layer, adim, bdim, msg in (
        (rep, 5120, 1280, "in=4096, out=1024"),
        (rep, 2048, 1024, "in=4096, out=1024"),
        (row, 5120, 4096, r"wo_b.*slot 0.*\(8, 5120\).*in=4096, out=4096"),
        (row, 2048, 4096, "in=4096, out=4096"),
        (col, 1024, 5120, "in=1024, out=4096"),
        (col, 1024, 2048, "in=1024, out=4096"),
    ):
        with pytest.raises(ValueError, match=msg):
            layer.set_lora(0, torch.ones(8, adim, dtype=torch.bfloat16), torch.ones(bdim, 8, dtype=torch.bfloat16))
    for layer, fa, fb, sa, sb in ((row, 4096, 4096, 512, 4096), (col, 1024, 4096, 1024, 512)):
        layer.set_lora(0, torch.ones(8, fa, dtype=torch.bfloat16), torch.ones(fb, 8, dtype=torch.bfloat16))
        layer.set_lora(0, torch.zeros(8, sa, dtype=torch.bfloat16), torch.zeros(sb, 8, dtype=torch.bfloat16))


def test_warn_unmatched_lora_modules_and_skipped_name_normalization(caplog) -> None:
    def _mgr(modules, loras, packed=None):
        return SimpleNamespace(
            modules=modules,
            packed_modules=packed or {},
            _registered_adapters={1: SimpleNamespace(loras=loras)},
            _get_lora_layer_weights=lambda lm, n: lm.loras.get(n),
            _get_text_lora_name=lambda _lm, _n: None,
        )

    req, p = SimpleNamespace(lora_name="moe", lora_int_id=1), "language_model.model.layers.0"
    c3d = _mgr(
        {f"{p}.mlp.experts": object()},
        {"model.layers.0.ffn.experts.base_layer": object(), "model.layers.0.ffn.experts": object()},
        {f"{p}.mlp.experts": [f"{p}.mlp.experts.0.w1"]},
    )
    c2d = _mgr(
        {f"{p}.mlp.experts": object()},
        {f"{p}.mlp.experts": object(), f"{p}.mlp.experts.1.gate_proj": object()},
        {f"{p}.mlp.experts": [f"{p}.mlp.experts.1.gate_proj"]},
    )
    for mgr, exp_un, exp_log in (
        (c3d, ["model.layers.0.ffn.experts", "model.layers.0.ffn.experts.base_layer"], "is_3d_lora_weight=true"),
        (c2d, [], ""),
        (_mgr({f"{p}.self_attn.wq_a": object()}, {}), [], "matched 0 target modules"),
    ):
        caplog.clear()
        with caplog.at_level("WARNING"):
            assert au.warn_unmatched_lora_modules(SimpleNamespace(_adapter_manager=mgr), req) == exp_un
        assert (caplog.text == "") if not exp_log else (exp_log in caplog.text)
    mm = SimpleNamespace(get_mm_mapping=lambda: SimpleNamespace(language_model=["language_model"]))
    cands = au._expand_skipped_module_names(mm, SimpleNamespace(), f"{p}.self_attn.compressor.wkv")
    assert "model.layers.0.self_attn.compressor.wkv" in cands and "layers.0.self_attn.compressor.wkv" in cands
    assert ".model.layers.0.self_attn.compressor.wkv" not in cands


def test_install_dp_moe_lora_load_guard_warns_offline_lru_reload_and_online_once(caplog) -> None:
    """Verify load guard warns on unmatched keys across offline LRU evict/reload and online add_lora."""
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner

    def _lw(name: str) -> LoRALayerWeights:
        return LoRALayerWeights(
            name, 8, 8, torch.ones(8, 8, dtype=torch.bfloat16), torch.ones(8, 8, dtype=torch.bfloat16)
        )

    original = lora_utils._all_lora_classes
    try:
        au.refresh_all_lora_classes()
        hf_cfg = SimpleNamespace(get_text_config=lambda: SimpleNamespace(max_position_embeddings=16))
        vllm_cfg = SimpleNamespace(
            lora_config=_lcfg(max_cpu_loras=1),
            parallel_config=SimpleNamespace(enable_expert_parallel=False),
            scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=16),
            model_config=SimpleNamespace(hf_config=hf_cfg, get_vocab_size=lambda: 32, is_encoder_decoder=False),
        )
        wm = LRUCacheWorkerLoRAManager(vllm_cfg, torch.device("cpu"), embedding_modules={})
        with patch("vllm.lora.model_manager.get_punica_wrapper", return_value=_punica()):
            wm.create_lora_manager(_fake_model(), vllm_cfg)
        au.install_dp_moe_lora_load_guard(wm)
        assert au.is_lora_load_guard_installed(wm)

        wqa_key = "model.layers.0.self_attn.wq_a"
        adapters = {
            1: LoRAModel(1, 8, {wqa_key: _lw(wqa_key), "model.layers.0.attn.missing_proj": _lw("x")}),
            2: LoRAModel(2, 8, {wqa_key: _lw(wqa_key)}),
        }
        req1, req2 = LoRARequest("a1", 1, "/u/1"), LoRARequest("a2", 2, "/u/2")
        with patch.object(
            wm._load_adapter, "_orig_load_adapter", side_effect=lambda r: adapters[r.lora_int_id].clone(r.lora_int_id)
        ):
            steps = (
                (lambda: wm.set_active_adapters({req1}, LoRAMapping((1,), (1,))), 1),
                (lambda: wm.set_active_adapters({req1}, LoRAMapping((1,), (1,))), 0),
                (lambda: wm.set_active_adapters({req2}, LoRAMapping((2,), (2,))), 0),
                (lambda: wm.set_active_adapters({req1}, LoRAMapping((1,), (1,))), 1),
                (lambda: (wm.remove_all_adapters(), wm.add_dummy_lora(LoRARequest("d", 3, "/u/3"), 8)), 0),
            )
            for idx, (fn, exp) in enumerate(steps):
                caplog.clear()
                with caplog.at_level("WARNING"):
                    fn()
                assert len([r for r in caplog.records if "missing_proj" in r.message]) == exp
                assert idx != 2 or 1 not in wm.list_adapters()
            wm.remove_all_adapters()
            runner = object.__new__(NPUModelRunner)
            for mgr in (wm, SimpleNamespace(_adapter_manager=wm._adapter_manager, add_adapter=lambda _r: True)):
                runner.lora_manager = mgr
                caplog.clear()
                with caplog.at_level("WARNING"):
                    assert runner.add_lora(req1) is True
                assert len([r for r in caplog.records if "missing_proj" in r.message]) == 1
    finally:
        lora_utils._all_lora_classes = original


def test_dp_moe_and_wo_a_wrapping_and_load_guards() -> None:
    """Verify DP>1 expert skip/rejection, wo_a skip/rejection, and runner hook installation."""
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from vllm_ascend.worker.v2.model_runner import NPUModelRunner

    def _mgr(model, cfg, punica):
        vcfg = SimpleNamespace(
            lora_config=cfg, parallel_config=SimpleNamespace(enable_expert_parallel=False), model_config=None
        )
        with patch("vllm.lora.model_manager.get_punica_wrapper", return_value=punica):
            return LRUCacheLoRAModelManager(model, 2, 16, 32, cfg, torch.device("cpu"), vllm_config=vcfg)

    def _req(i, *ks):
        return SimpleNamespace(loaded_model=SimpleNamespace(id=i, loras=dict.fromkeys(ks, object())))

    original = lora_utils._all_lora_classes
    try:
        au.refresh_all_lora_classes()
        punica, model = _punica(), _fake_model(dp_size=2, include_moe=True)
        mgr = _mgr(model, _lcfg(max_cpu_loras=2), punica)
        assert isinstance(mgr.modules["model.layers.0.self_attn.wq_a"], au.AscendReplicatedLinearWithLoRA)
        assert isinstance(mgr.modules["model.layers.0.mlp.shared_experts.down_proj"], au.AscendReplicatedLinearWithLoRA)
        assert "model.layers.0.mlp.experts" not in mgr.modules
        model.model.layers[0].mlp.experts.set_lora_context.assert_not_called()
        dummy = mgr.create_dummy_lora(1, 4, model.embedding_modules)
        assert set(dummy.loras) == {"model.layers.0.self_attn.wq_a", "model.layers.0.mlp.shared_experts.down_proj"}
        assert mgr.add_adapter(dummy) and mgr.activate_adapter(1)

        reg, wm = {}, SimpleNamespace(_adapter_manager=mgr, _load_adapter=lambda req: req.loaded_model)
        au.install_dp_moe_lora_load_guard(wm)
        m_ok = wm._load_adapter(_req(1, "model.layers.0.self_attn.wq_a", "model.layers.0.mlp.shared_experts.down_proj"))
        reg[m_ok.id] = m_ok
        for bid, k in (
            (2, "model.layers.0.mlp.experts.0.gate_proj"),
            (3, "model.layers.0.mlp.experts"),
            (4, "model.layers.0.mlp.experts.base_layer"),
        ):
            with pytest.raises(NotImplementedError, match="data_parallel_size > 1"):
                wm._load_adapter(_req(bid, "model.layers.0.self_attn.wq_a", k))
        assert set(reg) == {1}

        mm_wm = SimpleNamespace(
            _adapter_manager=SimpleNamespace(
                model=_fake_model(dp_size=2, multimodal=True, include_moe=True),
                modules={"language_model.model.layers.0.self_attn.wq_a": object()},
                _get_text_lora_names=lambda n: ["model.layers.0.mlp.experts"] if n.endswith("experts") else [],
            ),
            _load_adapter=lambda _r: SimpleNamespace(id=5, loras={"model.layers.0.mlp.experts.0.gate_proj": object()}),
        )
        au.install_dp_moe_lora_load_guard(mm_wm)
        with pytest.raises(NotImplementedError, match="data_parallel_size > 1"):
            mm_wm._load_adapter(SimpleNamespace())
        with pytest.raises(ValueError, match="experts"):
            _mgr(_fake_model(dp_size=2, include_moe=True), _lcfg(max_cpu_loras=2, target_modules=["experts"]), punica)

        for n_g, otp, sharded, exp_wrap in (
            (2, False, False, False),
            (1, True, False, False),
            (1, False, True, False),
            (1, False, False, True),
        ):
            with patch(f"{_AU}._safe_oproj_tp_enable", return_value=otp):
                w_mgr = _mgr(
                    _fake_model(n_local_groups=n_g, multimodal=not exp_wrap),
                    _lcfg(max_cpu_loras=2, fully_sharded_loras=sharded),
                    punica,
                )
            pfx = "model.layers.0.self_attn" if exp_wrap else "language_model.model.layers.0.self_attn"
            assert (f"{pfx}.wo_a" in w_mgr.modules) is exp_wrap
            assert (f"{pfx}.wo_a" in w_mgr.create_dummy_lora(1, 4, {}).loras) is exp_wrap
            if not sharded:
                assert isinstance(w_mgr.modules[f"{pfx}.wq_a"], au.AscendReplicatedLinearWithLoRA)
                assert isinstance(w_mgr.modules[f"{pfx}.wo_b"], au.AscendRowParallelLinearWithLoRA)
                assert not exp_wrap or isinstance(w_mgr.modules[f"{pfx}.wo_a"], au.AscendColumnParallelLinearWithLoRA)
            else:
                assert isinstance(w_mgr.modules[f"{pfx}.wq_a"], BaseLayerWithLoRA)
            w_wm = SimpleNamespace(_adapter_manager=w_mgr, _load_adapter=lambda req: req.loaded_model)
            with patch(f"{_AU}._safe_oproj_tp_enable", return_value=otp):
                au.install_dp_moe_lora_load_guard(w_wm)
            assert w_wm._load_adapter(_req(1, f"{pfx}.wq_a")).id == 1
            for bad_k in (f"{pfx}.wo_a", "model.layers.0.self_attn.wo_a"):
                if exp_wrap:
                    assert w_wm._load_adapter(_req(2, f"{pfx}.wq_a", bad_k)).id == 2
                else:
                    with pytest.raises(NotImplementedError, match="wo_a"):
                        w_wm._load_adapter(_req(2, f"{pfx}.wq_a", bad_k))

        runner = object.__new__(NPUModelRunner)
        runner.lora_manager = SimpleNamespace(marker="wm")
        with (
            patch.object(GPUModelRunner, "load_lora_model", return_value="m") as sup,
            patch(f"{_AU}.install_dp_moe_lora_load_guard") as guard,
        ):
            assert runner.load_lora_model("m", SimpleNamespace(), torch.device("cpu")) == "m"
        sup.assert_called_once()
        guard.assert_called_once_with(runner.lora_manager)
    finally:
        lora_utils._all_lora_classes = original
