# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter keys must resolve to the modules the model actually builds.

DSV4 HF checkpoints spell the attention block ``.attn.`` while the vLLM model
tree keeps ``.self_attn.``. Adapter keys that miss the module are dropped
without an error, so the rename rule is covered directly here.
"""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
import vllm.lora.utils as upstream_lora_utils
from vllm.config import LoRAConfig
from vllm.lora.lora_model import LoRAModel
from vllm.lora.model_manager import LRUCacheLoRAModelManager
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase
from vllm.lora.utils import parse_fine_tuned_lora_name
from vllm.model_executor.custom_op import maybe_get_oot_by_class
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models import SupportsLoRA
from vllm.model_executor.models.interfaces import supports_lora

from vllm_ascend.lora.punica_npu import PunicaWrapperNPU
from vllm_ascend.lora.utils import (
    AscendReplicatedLinearWithLoRA,
    install_dp_moe_lora_load_guard,
    refresh_all_lora_classes,
)
from vllm_ascend.models.deepseek_v4.compressor import Compressor
from vllm_ascend.models.deepseek_v4.model import AscendDeepseekV4ForCausalLM
from vllm_ascend.models.deepseek_v4.vl_model import AscendDeepseekV4ForConditionalGeneration


def _rename_mapper():
    # ``WorkerLoRAManager`` forwards ``hf_to_vllm_mapper.get_rename_mapper()``.
    return AscendDeepseekV4ForCausalLM.build_lora_weights_mapper().get_rename_mapper()


def test_hf_attn_adapter_key_reaches_the_self_attn_module():
    module_name, is_lora_a = parse_fine_tuned_lora_name(
        "base_model.model.model.layers.7.attn.wq_a.lora_A.weight",
        _rename_mapper(),
    )
    assert module_name == "model.layers.7.self_attn.wq_a"
    assert is_lora_a


def test_all_projected_linears_are_re_rooted():
    for projection in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
        module_name, is_lora_a = parse_fine_tuned_lora_name(
            f"base_model.model.model.layers.3.attn.{projection}.lora_B.weight",
            _rename_mapper(),
        )
        assert module_name == f"model.layers.3.self_attn.{projection}"
        assert not is_lora_a


def test_keys_already_using_the_tree_spelling_are_unchanged():
    module_name, _ = parse_fine_tuned_lora_name(
        "model.layers.1.self_attn.wkv.lora_A.weight",
        _rename_mapper(),
    )
    assert module_name == "model.layers.1.self_attn.wkv"


def test_hf_ffn_moe_and_shared_expert_keys_reach_mlp_modules():
    expected = [
        (
            "base_model.model.model.layers.5.ffn.experts.12.w1.lora_A.weight",
            "model.layers.5.mlp.experts.12.gate_proj",
            True,
        ),
        (
            "base_model.model.model.layers.5.ffn.experts.12.w2.lora_B.weight",
            "model.layers.5.mlp.experts.12.down_proj",
            False,
        ),
        (
            "base_model.model.model.layers.5.ffn.experts.12.w3.lora_A.weight",
            "model.layers.5.mlp.experts.12.up_proj",
            True,
        ),
        (
            "base_model.model.model.layers.5.ffn.shared_experts.w1.lora_A.weight",
            "model.layers.5.mlp.shared_experts.gate_proj",
            True,
        ),
    ]
    for raw_key, expected_module, expected_is_a in expected:
        module_name, is_lora_a = parse_fine_tuned_lora_name(raw_key, _rename_mapper())
        assert module_name == expected_module
        assert is_lora_a is expected_is_a


def _vl_rename_mapper():
    return AscendDeepseekV4ForConditionalGeneration.build_lora_weights_mapper().get_rename_mapper()


def test_vl_wrapper_exposes_lora_contract_with_independent_mapping() -> None:
    assert supports_lora(AscendDeepseekV4ForConditionalGeneration)
    assert AscendDeepseekV4ForConditionalGeneration.packed_modules_mapping == {"gate_up_proj": ["gate_proj", "up_proj"]}
    assert (
        AscendDeepseekV4ForConditionalGeneration.packed_modules_mapping
        is not AscendDeepseekV4ForCausalLM.packed_modules_mapping
    )
    assert AscendDeepseekV4ForConditionalGeneration.embedding_modules == AscendDeepseekV4ForCausalLM.embedding_modules
    assert AscendDeepseekV4ForConditionalGeneration.is_3d_moe_weight == AscendDeepseekV4ForCausalLM.is_3d_moe_weight
    assert AscendDeepseekV4ForConditionalGeneration.is_non_gated_moe == AscendDeepseekV4ForCausalLM.is_non_gated_moe
    assert "vision." in AscendDeepseekV4ForConditionalGeneration.lora_skip_prefixes
    assert "aligner." in AscendDeepseekV4ForConditionalGeneration.lora_skip_prefixes
    assert not hasattr(AscendDeepseekV4ForConditionalGeneration, "hf_to_vllm_mapper")


def test_vl_hf_adapter_keys_re_root_onto_language_model() -> None:
    mapper = _vl_rename_mapper()
    cases = [
        (
            "base_model.model.model.layers.7.attn.wq_a.lora_A.weight",
            "language_model.model.layers.7.self_attn.wq_a",
            True,
        ),
        (
            "base_model.model.layers.3.attn.wo_b.lora_B.weight",
            "language_model.model.layers.3.self_attn.wo_b",
            False,
        ),
        (
            "base_model.model.model.layers.5.ffn.experts.12.w1.lora_A.weight",
            "language_model.model.layers.5.mlp.experts.12.gate_proj",
            True,
        ),
        (
            "base_model.model.model.layers.5.ffn.experts.12.w2.lora_B.weight",
            "language_model.model.layers.5.mlp.experts.12.down_proj",
            False,
        ),
        (
            "base_model.model.model.layers.5.ffn.shared_experts.w3.lora_B.weight",
            "language_model.model.layers.5.mlp.shared_experts.up_proj",
            False,
        ),
        (
            "base_model.model.lm_head.lora_A.weight",
            "language_model.lm_head",
            True,
        ),
        (
            "language_model.model.layers.1.self_attn.wkv.lora_A.weight",
            "language_model.model.layers.1.self_attn.wkv",
            True,
        ),
    ]
    for raw_key, expected_module, expected_is_a in cases:
        module_name, is_lora_a = parse_fine_tuned_lora_name(raw_key, mapper)
        assert module_name == expected_module
        assert is_lora_a is expected_is_a


def _build_real_vl_compressor_model() -> AscendDeepseekV4ForConditionalGeneration:
    model = AscendDeepseekV4ForConditionalGeneration.__new__(AscendDeepseekV4ForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = None
    model.hf_to_vllm_mapper = AscendDeepseekV4ForConditionalGeneration.build_lora_weights_mapper()
    model.vision = torch.nn.Module()
    model.vision.proj = torch.nn.Linear(8, 8)
    model.aligner = torch.nn.Module()
    model.aligner.w1 = torch.nn.Linear(8, 8)

    layer0 = torch.nn.Module()
    layer0.self_attn = torch.nn.Module()
    layer0.self_attn.wkv = _make_cpu_replicated_linear(8, 8, prefix="language_model.model.layers.0.self_attn.wkv")
    layer0.self_attn.compressor = _build_real_dsv4_compressor(
        prefix="language_model.model.layers.0.self_attn.compressor"
    )
    layer0.self_attn.indexer = torch.nn.Module()
    layer0.self_attn.indexer.compressor = _build_real_dsv4_compressor(
        prefix="language_model.model.layers.0.self_attn.indexer.compressor"
    )
    model.language_model = torch.nn.Module()
    model.language_model.model = torch.nn.Module()
    model.language_model.model.layers = torch.nn.ModuleList([layer0])
    return model


def test_vl_compressor_projections_are_unwrapped_and_rejected_on_load() -> None:
    original_classes = upstream_lora_utils._all_lora_classes
    try:
        refresh_all_lora_classes()
        model = _build_real_vl_compressor_model()
        lora_config = LoRAConfig(max_loras=1, max_lora_rank=8, lora_dtype=torch.bfloat16, max_cpu_loras=2)
        vllm_config = SimpleNamespace(
            lora_config=lora_config,
            parallel_config=SimpleNamespace(enable_expert_parallel=False),
            model_config=None,
        )
        punica_wrapper = object.__new__(PunicaWrapperNPU)
        PunicaWrapperBase.__init__(punica_wrapper, 16, 2, "cpu")

        with patch("vllm.lora.model_manager.get_punica_wrapper", return_value=punica_wrapper):
            manager = LRUCacheLoRAModelManager(
                model=model,
                max_num_seqs=2,
                max_num_batched_tokens=16,
                vocab_size=32,
                lora_config=lora_config,
                device=torch.device("cpu"),
                vllm_config=vllm_config,
            )

        assert "language_model.model.layers.0.self_attn.wkv" in manager.modules
        assert isinstance(
            manager.modules["language_model.model.layers.0.self_attn.wkv"],
            AscendReplicatedLinearWithLoRA,
        )
        assert "vision.proj" not in manager.modules
        assert "aligner.w1" not in manager.modules
        assert LoRAModel._should_skip_module("vision.proj.lora_A.weight", model.lora_skip_prefixes)
        assert LoRAModel._should_skip_module("aligner.w1.lora_A.weight", model.lora_skip_prefixes)
        assert not LoRAModel._should_skip_module("model.layers.0.attn.wkv.lora_A.weight", model.lora_skip_prefixes)
        for excluded in (
            "language_model.model.layers.0.self_attn.compressor.wkv",
            "language_model.model.layers.0.self_attn.compressor.wgate",
            "language_model.model.layers.0.self_attn.indexer.compressor.wkv",
            "language_model.model.layers.0.self_attn.indexer.compressor.wgate",
        ):
            assert excluded not in manager.modules

        worker_manager = SimpleNamespace(
            _adapter_manager=manager,
            _load_adapter=lambda req: req.loaded_model,
        )
        install_dp_moe_lora_load_guard(worker_manager)

        rename_mapper = model.hf_to_vllm_mapper.get_rename_mapper()
        mapped_ok, _ = parse_fine_tuned_lora_name(
            "base_model.model.model.layers.0.attn.wkv.lora_A.weight",
            rename_mapper,
        )
        assert mapped_ok == "language_model.model.layers.0.self_attn.wkv"
        assert worker_manager._load_adapter(
            SimpleNamespace(loaded_model=SimpleNamespace(id=1, loras={mapped_ok: object()}))
        )

        for idx, raw_key in enumerate(
            (
                "base_model.model.model.layers.0.attn.compressor.wkv.lora_A.weight",
                "base_model.model.model.layers.0.attn.compressor.wgate.lora_A.weight",
                "base_model.model.model.layers.0.attn.indexer.compressor.wkv.lora_A.weight",
                "base_model.model.model.layers.0.attn.indexer.compressor.wgate.lora_A.weight",
            ),
            start=2,
        ):
            mapped_bad, _ = parse_fine_tuned_lora_name(raw_key, rename_mapper)
            bad_adapter = SimpleNamespace(id=idx, loras={mapped_ok: None, mapped_bad: object()})
            with pytest.raises(NotImplementedError, match="Compressor"):
                worker_manager._load_adapter(SimpleNamespace(loaded_model=bad_adapter))
    finally:
        upstream_lora_utils._all_lora_classes = original_classes


def _make_cpu_replicated_linear(in_dim: int, out_dim: int, *, prefix: str = "", **_kwargs) -> torch.nn.Module:
    replicated_cls = maybe_get_oot_by_class(ReplicatedLinear)
    layer = object.__new__(replicated_cls)
    torch.nn.Module.__init__(layer)
    layer.input_size = in_dim
    layer.output_size = out_dim
    layer.output_partition_sizes = [out_dim]
    layer.input_size_per_partition = in_dim
    layer.tp_size = 1
    layer.tp_rank = 0
    layer.skip_bias_add = False
    layer.return_bias = False
    layer.bias = None
    layer.prefix = prefix
    layer.quant_method = None
    layer.weight = torch.nn.Parameter(torch.zeros(out_dim, in_dim, dtype=torch.bfloat16))
    return layer


def _build_real_dsv4_compressor(*, prefix: str, hidden_size: int = 8, head_dim: int = 4) -> Compressor:
    fake_vllm_cfg = SimpleNamespace()
    fake_model_cfg = SimpleNamespace(
        hidden_size=hidden_size,
        qk_rope_head_dim=2,
        rms_norm_eps=1e-6,
    )
    fake_cache_cfg = SimpleNamespace(block_size=16)
    fake_hw = SimpleNamespace(supports=lambda _cap: False)
    fake_block_sizes = {16: [(0, 0, 16, 16), (0, 0)]}

    with (
        patch("vllm_ascend.models.deepseek_v4.compressor.ReplicatedLinear", side_effect=_make_cpu_replicated_linear),
        patch("vllm_ascend.models.deepseek_v4.compressor.RMSNorm", side_effect=lambda *_a, **_kw: torch.nn.Identity()),
        patch(
            "vllm_ascend.models.deepseek_v4.compressor.AscendCompressorStateCache",
            side_effect=lambda *_a, **_kw: torch.nn.Module(),
        ),
        patch("vllm_ascend.models.deepseek_v4.compressor.get_current_hardware_profile", return_value=fake_hw),
        patch.dict(
            "vllm_ascend.models.layer.attention.layer.DSV4_BLOCK_SIZES",
            fake_block_sizes,
            clear=False,
        ),
    ):
        return Compressor(
            vllm_config=fake_vllm_cfg,
            config=fake_model_cfg,
            compress_ratio=4,
            head_dim=head_dim,
            cache_config=fake_cache_cfg,
            prefix=prefix,
        )


class _FakeDSV4CompressorTree(torch.nn.Module):
    supports_lora = SupportsLoRA.supports_lora
    config = None
    is_3d_moe_weight = False
    is_non_gated_moe = False
    embedding_modules: dict[str, str] = {}
    packed_modules_mapping: dict[str, list[str]] = {}
    lora_skip_prefixes: list[str] = []

    def __init__(self) -> None:
        super().__init__()
        layer0 = torch.nn.Module()
        layer0.self_attn = torch.nn.Module()
        layer0.self_attn.wkv = _make_cpu_replicated_linear(8, 8, prefix="model.layers.0.self_attn.wkv")
        layer0.self_attn.compressor = _build_real_dsv4_compressor(prefix="model.layers.0.self_attn.compressor")
        layer0.self_attn.indexer = torch.nn.Module()
        layer0.self_attn.indexer.compressor = _build_real_dsv4_compressor(
            prefix="model.layers.0.self_attn.indexer.compressor"
        )
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([layer0])


def test_dsv4_compressor_projections_are_unwrapped_and_rejected_on_load() -> None:
    original_classes = upstream_lora_utils._all_lora_classes
    try:
        refresh_all_lora_classes()
        model = _FakeDSV4CompressorTree()
        assert "Compressor" in model.model.layers[0].self_attn.compressor.wkv._ascend_lora_unsupported_reason
        assert "Compressor" in model.model.layers[0].self_attn.compressor.wgate._ascend_lora_unsupported_reason
        assert "Compressor" in model.model.layers[0].self_attn.indexer.compressor.wkv._ascend_lora_unsupported_reason

        lora_config = LoRAConfig(max_loras=1, max_lora_rank=8, lora_dtype=torch.bfloat16, max_cpu_loras=2)
        vllm_config = SimpleNamespace(
            lora_config=lora_config,
            parallel_config=SimpleNamespace(enable_expert_parallel=False),
            model_config=None,
        )
        punica_wrapper = object.__new__(PunicaWrapperNPU)
        PunicaWrapperBase.__init__(punica_wrapper, 16, 2, "cpu")

        with patch("vllm.lora.model_manager.get_punica_wrapper", return_value=punica_wrapper):
            manager = LRUCacheLoRAModelManager(
                model=model,
                max_num_seqs=2,
                max_num_batched_tokens=16,
                vocab_size=32,
                lora_config=lora_config,
                device=torch.device("cpu"),
                vllm_config=vllm_config,
            )

        assert "model.layers.0.self_attn.wkv" in manager.modules
        assert isinstance(manager.modules["model.layers.0.self_attn.wkv"], AscendReplicatedLinearWithLoRA)
        for excluded in (
            "model.layers.0.self_attn.compressor.wkv",
            "model.layers.0.self_attn.compressor.wgate",
            "model.layers.0.self_attn.indexer.compressor.wkv",
            "model.layers.0.self_attn.indexer.compressor.wgate",
        ):
            assert excluded not in manager.modules

        dummy = manager.create_dummy_lora(lora_id=1, rank=2, embedding_modules=model.embedding_modules)
        assert set(dummy.loras.keys()) == {"model.layers.0.self_attn.wkv"}
        assert manager.add_adapter(dummy) is True
        assert manager.activate_adapter(dummy.id) is True

        registered: dict[int, object] = {}

        def _fake_load(lora_request):
            return lora_request.loaded_model

        def _fake_add_adapter(lora_request):
            lora_model = worker_manager._load_adapter(lora_request)
            registered[lora_model.id] = lora_model
            return True

        worker_manager = SimpleNamespace(_adapter_manager=manager, _load_adapter=_fake_load)
        install_dp_moe_lora_load_guard(worker_manager)

        valid_adapter = SimpleNamespace(id=1, loras={"model.layers.0.self_attn.wkv": object()})
        assert _fake_add_adapter(SimpleNamespace(loaded_model=valid_adapter)) is True
        assert set(registered.keys()) == {1}

        for bad_id, bad_key in (
            (2, "model.layers.0.self_attn.compressor.wkv"),
            (3, "model.layers.0.self_attn.compressor.wgate"),
            (4, "model.layers.0.self_attn.indexer.compressor.wkv"),
            (5, "model.layers.0.self_attn.indexer.compressor.wgate"),
        ):
            bad_adapter = SimpleNamespace(
                id=bad_id,
                loras={"model.layers.0.self_attn.wkv": object(), bad_key: object()},
            )
            with pytest.raises(NotImplementedError, match="Compressor"):
                _fake_add_adapter(SimpleNamespace(loaded_model=bad_adapter))
        assert set(registered.keys()) == {1}
    finally:
        upstream_lora_utils._all_lora_classes = original_classes


def test_causallm_bare_layers_and_embed_prefixes_map_to_model_tree() -> None:
    mapper = _rename_mapper()
    cases = [
        (
            "base_model.model.layers.5.attn.wq_a.lora_A.weight",
            "model.layers.5.self_attn.wq_a",
            True,
        ),
        (
            "layers.5.ffn.experts.3.w1.lora_B.weight",
            "model.layers.5.mlp.experts.3.gate_proj",
            False,
        ),
        (
            "embed_tokens.lora_A.weight",
            "model.embed_tokens",
            True,
        ),
        (
            "model.embed_tokens.lora_B.weight",
            "model.embed_tokens",
            False,
        ),
    ]
    for raw_key, expected_module, expected_is_a in cases:
        module_name, is_lora_a = parse_fine_tuned_lora_name(raw_key, mapper)
        assert module_name == expected_module
        assert is_lora_a is expected_is_a


def test_dsv4_indexer_wq_b_lora_bypasses_multistream_and_w8a8_direct_matmul() -> None:
    from vllm.lora.layers.base import BaseLayerWithLoRA

    from vllm_ascend.models.deepseek_v4.indexer import (
        AscendIndexerMetadata,
        DeepseekV4Indexer,
        IndexerOverlapPlan,
    )
    from vllm_ascend.quantization.methods import AscendW8A8DynamicLinearMethod

    class _StubIndexerWqBLoRA(BaseLayerWithLoRA):
        calls: int = 0

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return x + 2.0

    class _StubCompressor:
        rotate: bool = False

        def __call__(self, **_: object) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.zeros((0, 4), dtype=torch.bfloat16), torch.zeros((0,), dtype=torch.int32)

    wq_b_wrapper = object.__new__(_StubIndexerWqBLoRA)
    torch.nn.Module.__init__(wq_b_wrapper)
    wq_b_wrapper.calls = 0
    cast(Any, wq_b_wrapper).quant_method = SimpleNamespace(
        quant_method=object.__new__(AscendW8A8DynamicLinearMethod),
    )

    indexer = object.__new__(DeepseekV4Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.wq_b = wq_b_wrapper
    indexer_any = cast(Any, indexer)
    indexer_any.cv_wq_b = SimpleNamespace(
        matmul=lambda *_: (_ for _ in ()).throw(AssertionError("must not call cv_wq_b")),
    )
    indexer.n_heads = 2
    indexer.head_dim = 4
    indexer.rope_head_dim = 2
    indexer.softmax_scale = 0.5
    indexer.skip_topk = False
    indexer.use_index_cache = False
    indexer.topk_indices_buffer = None
    indexer_any.compressor = SimpleNamespace()
    indexer_any.weights_proj = lambda hs: torch.ones((hs.shape[0], 2), dtype=hs.dtype)

    seen_calls: list[str] = []

    def _select_topk(*_: Any) -> torch.Tensor:
        seen_calls.append("select_topk")
        return torch.zeros((2, 1), dtype=torch.int32)

    def _quant_select(*_: Any) -> torch.Tensor:
        seen_calls.append("quant_select")
        return torch.ones((2, 1), dtype=torch.int32)

    indexer_any.ops = SimpleNamespace(
        unpack_dsa_indexer_kv_cache=lambda kv: ("k", "s", "sc", "f"),
        quantize_query=lambda q: (q, "qs"),
        select_topk=_select_topk,
    )

    cos = torch.ones((2, 2), dtype=torch.bfloat16)
    sin = torch.zeros((2, 2), dtype=torch.bfloat16)
    metadata = AscendIndexerMetadata(
        compressor=cast(
            Any,
            SimpleNamespace(
                cache=SimpleNamespace(
                    req_metadata=SimpleNamespace(cos={"layer0": cos}, sin={"layer0": sin}),
                    hadamard=torch.eye(4, dtype=torch.bfloat16),
                ),
            ),
        ),
    )
    hidden_states = torch.ones((2, 8), dtype=torch.bfloat16)
    qr = torch.ones((2, 8), dtype=torch.bfloat16)

    fake_c_ascend = SimpleNamespace(inplace_partial_rotary_mul=lambda *_, **__: None)
    with (
        patch.object(torch.ops, "_C_ascend", fake_c_ascend, create=True),
        patch(
            "vllm_ascend.models.deepseek_v4.indexer.torch_npu.npu_quant_matmul",
            side_effect=AssertionError("must not bypass wq_b LoRA wrapper via npu_quant_matmul"),
        ),
        patch("vllm_ascend.models.deepseek_v4.indexer.rotate_activation", side_effect=lambda q, *_: q),
        patch.object(
            indexer,
            "_cv_compute_query_and_update_cache_multistream",
            side_effect=AssertionError("must bypass multistream when indexer.wq_b is LoRA-wrapped"),
        ),
    ):
        q, *_ = indexer._indexer_qkv_prepare(
            hidden_states,
            qr,
            kv_cache=(),
            metadata=metadata,
            cos=cos,
            sin=sin,
            qr_pertoken_scale=torch.ones((2, 1), dtype=torch.float32),
            write_cache=False,
        )
        assert wq_b_wrapper.calls == 1
        assert torch.all(q == 3.0)

        overlap_calls: list[str] = []

        def _compute_ckv() -> tuple[str, str]:
            overlap_calls.append("compute")
            return ("ckv", "slot")

        def _scatter_ckv(ckv: str, slot: str) -> None:
            overlap_calls.append(f"scatter:{ckv}:{slot}")

        plan = IndexerOverlapPlan(
            compute_attention_compressed_kv=_compute_ckv,
            scatter_attention_compressed_kv=cast(Any, _scatter_ckv),
            aux_stream=object(),
        )
        out = indexer.forward(
            "layer0",
            hidden_states,
            qr,
            kv_cache=(),
            metadata=metadata,
            overlap_plan=plan,
            qr_pertoken_scale=torch.ones((2, 1), dtype=torch.float32),
            write_cache=False,
        )
        assert wq_b_wrapper.calls == 2
        assert seen_calls == ["select_topk"]
        assert torch.equal(out, torch.zeros((2, 1), dtype=torch.int32))

        plan_write = IndexerOverlapPlan(
            compute_attention_compressed_kv=_compute_ckv,
            scatter_attention_compressed_kv=cast(Any, _scatter_ckv),
            aux_stream=object(),
        )
        indexer_any.compressor = _StubCompressor()
        indexer_any.ops.quantize_update_cache_and_select_topk = _quant_select
        out_write = indexer.forward(
            "layer0",
            hidden_states,
            qr,
            kv_cache=(),
            metadata=metadata,
            overlap_plan=plan_write,
            qr_pertoken_scale=None,
            write_cache=True,
        )
        assert wq_b_wrapper.calls == 3
        assert overlap_calls == ["compute", "scatter:ckv:slot"]
        assert torch.equal(out_write, torch.ones((2, 1), dtype=torch.int32))
