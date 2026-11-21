# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter keys must resolve to the modules the model actually builds.

DSV4 HF checkpoints spell the attention block ``.attn.`` while the vLLM model
tree keeps ``.self_attn.``. Adapter keys that miss the module are dropped
without an error, so the rename rule is covered directly here.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import vllm.lora.utils as upstream_lora_utils
from vllm.config import LoRAConfig
from vllm.lora.model_manager import LRUCacheLoRAModelManager
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase
from vllm.lora.utils import parse_fine_tuned_lora_name
from vllm.model_executor.custom_op import maybe_get_oot_by_class
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models import SupportsLoRA

from vllm_ascend.lora.punica_npu import PunicaWrapperNPU
from vllm_ascend.lora.utils import (
    AscendReplicatedLinearWithLoRA,
    install_dp_moe_lora_load_guard,
    refresh_all_lora_classes,
)
from vllm_ascend.models.deepseek_v4.compressor import Compressor
from vllm_ascend.models.deepseek_v4.model import AscendDeepseekV4ForCausalLM


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
