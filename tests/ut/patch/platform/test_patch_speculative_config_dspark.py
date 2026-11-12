import json
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from transformers import PretrainedConfig, Qwen3Config
from vllm.config import ParallelConfig
from vllm.config.model import ModelConfig
from vllm.config.model_arch import ModelArchitectureConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.transformers_utils.configs.speculators import SpeculatorsConfig

from vllm_ascend.patch.platform import patch_speculative_config
from vllm_ascend.patch.platform.patch_speculative_config import (
    _ascend_verify_with_parallel_config,
    _normalize_deepseek_dspark_draft,
)


def test_legacy_qwen3_dspark_config_uses_qwen3_loader():
    config = Qwen3Config(
        architectures=["DSparkDraftModel"],
        block_size=7,
        dflash_config={
            "mask_token_id": 163824,
            "target_layer_ids": [7, 23, 51, 67, 83],
        },
    )

    normalized = SpeculativeConfig.hf_config_override(config)

    assert normalized is config
    assert normalized.architectures == ["Qwen3DSparkModel"]
    assert normalized.mask_token_id == 163824
    assert normalized.target_layer_ids == [7, 23, 51, 67, 83]
    assert getattr(normalized, "dspark_aux_hidden_state_format", None) is None
    assert normalized.block_size == 7


def test_deepseek_v4_vision_dspark_restores_draft_architecture():
    hf_config = SimpleNamespace(
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForConditionalGeneration"],
        dspark_target_layer_ids=[40, 41, 42],
    )
    hf_config.update = lambda values: hf_config.__dict__.update(values)
    model_arch_config = ModelArchitectureConfig(
        architectures=["DeepseekV4ForConditionalGeneration"],
        model_type="deepseek_v4",
        text_model_type=None,
        hidden_size=128,
        total_num_hidden_layers=43,
        total_num_attention_heads=8,
        head_size=16,
        vocab_size=1024,
        total_num_kv_heads=8,
        num_experts=256,
        num_experts_per_token=8,
        quantization_config=None,
        is_deepseek_mla=True,
        is_mm_prefix_lm=True,
        rswa_window=128,
        derived_max_model_len_and_key=(8192, "max_position_embeddings"),
    )
    registry = MagicMock()
    registry.inspect_model_cls.return_value = ("model-info", "DSparkDraftModel")

    class DraftModelConfig(SimpleNamespace):
        @property
        def architectures(self):
            return self.model_arch_config.architectures

    draft_model_config = DraftModelConfig(
        hf_config=hf_config,
        model_arch_config=model_arch_config,
        registry=registry,
    )

    _normalize_deepseek_dspark_draft(draft_model_config)

    assert hf_config.architectures == ["DSparkDraftModel"]
    assert draft_model_config.architectures == ["DSparkDraftModel"]
    assert draft_model_config.model_arch_config.is_mm_prefix_lm is False
    assert draft_model_config._architecture == "DSparkDraftModel"
    registry.inspect_model_cls.assert_called_once_with(["DSparkDraftModel"], draft_model_config)


class _DraftModelConfig(SimpleNamespace):
    @property
    def architectures(self):
        return self.model_arch_config.architectures

    @property
    def use_mla(self):
        return self.model_arch_config.is_deepseek_mla

    def get_total_num_kv_heads(self):
        return self.model_arch_config.total_num_kv_heads

    def get_model_arch_config(self):
        return replace(
            self.model_arch_config,
            architectures=self.hf_config.architectures,
            is_deepseek_mla="K3DSparkModel" in self.hf_config.architectures,
        )


@pytest.fixture
def initialize_dspark(monkeypatch):
    def initialize(hf_config, *, dcp_size=8, model="/test/local-draft", num_heads=8):
        arch = ModelArchitectureConfig(
            architectures=hf_config.architectures,
            model_type=hf_config.model_type,
            text_model_type=None,
            hidden_size=128,
            total_num_hidden_layers=2,
            total_num_attention_heads=num_heads,
            head_size=16,
            vocab_size=1024,
            total_num_kv_heads=1,
            num_experts=0,
            num_experts_per_token=0,
            quantization_config=None,
            is_deepseek_mla="K3DSparkModel" in hf_config.architectures,
            is_mm_prefix_lm=False,
            rswa_window=None,
            derived_max_model_len_and_key=(128, "max_position_embeddings"),
        )
        draft = _DraftModelConfig(
            model=model,
            hf_config=hf_config,
            hf_text_config=hf_config,
            model_arch_config=arch,
            registry=MagicMock(),
            max_model_len=128,
            quantization=None,
            is_moe=False,
            runner_type="draft",
            multimodal_config=None,
        )
        draft.registry.inspect_model_cls.side_effect = lambda architectures, _: ("model-info", architectures[0])
        target_pc = ParallelConfig(
            tensor_parallel_size=8,
            decode_context_parallel_size=dcp_size,
            distributed_executor_backend="mp",
        )
        original_pc = asdict(target_pc)
        target = MagicMock(max_model_len=128, quantization=None, hf_overrides={}, use_mla=True)
        target.hf_config = PretrainedConfig(model_type="deepseek_v3", architectures=["DeepseekV3ForCausalLM"])
        validated = []

        def verify(draft_pc):
            assert not draft_pc.enable_expert_parallel
            validated.append((list(draft.architectures), draft.use_mla, draft_pc.decode_context_parallel_size))
            ModelConfig.verify_with_parallel_config(draft, draft_pc)

        draft.verify_with_parallel_config = verify
        # Mock only checkpoint loading; run the real post-init, architecture
        # normalization, draft parallel-config creation, and after-validator.
        monkeypatch.setattr(patch_speculative_config.speculative_config, "ModelConfig", lambda **_: draft)
        create_draft_pc = SpeculativeConfig.create_draft_parallel_config
        created = []

        def create(target_parallel_config, *args, **kwargs):
            assert target_parallel_config is target_pc
            assert asdict(target_parallel_config) == original_pc
            created.append(target_parallel_config)
            return create_draft_pc(target_parallel_config, *args, **kwargs)

        monkeypatch.setattr(SpeculativeConfig, "create_draft_parallel_config", staticmethod(create))
        try:
            config = SpeculativeConfig(
                method="dspark",
                model=model,
                num_speculative_tokens=3,
                target_model_config=target,
                target_parallel_config=target_pc,
            )
        finally:
            assert asdict(target_pc) == original_pc
        assert created == [target_pc]
        assert config.target_parallel_config is target_pc
        assert config.draft_parallel_config is not target_pc
        assert config.draft_parallel_config.tensor_parallel_size == 8
        assert config.draft_parallel_config.decode_context_parallel_size == 1
        assert validated == [(list(draft.architectures), draft.use_mla, 1)]
        return config

    return initialize


@pytest.mark.parametrize("dcp_size", [1, 8])
def test_k3_dspark_preserves_target_dcp_during_real_initialization(initialize_dspark, dcp_size):
    config = initialize_dspark(
        PretrainedConfig(architectures=["K3DSparkModel"], model_type="k3_dspark", dspark_noise_token_id=1023),
        dcp_size=dcp_size,
    )
    assert config.draft_model_config.use_mla
    assert config.draft_model_config.hf_config.ptd_token_id == 1023
    assert config.target_parallel_config.decode_context_parallel_size == dcp_size


def test_dspark_real_after_validator_rejects_invalid_attention_heads(initialize_dspark):
    with pytest.raises(ValueError, match="must be divisible by tensor parallel size"):
        initialize_dspark(PretrainedConfig(architectures=["K3DSparkModel"], model_type="k3_dspark"), num_heads=7)


@pytest.fixture
def glm_dspark_checkpoint(tmp_path):
    source = {
        "architectures": ["Glm5DSparkForCausalLM"],
        "speculators_model_type": "dspark",
        "aux_hidden_state_layer_ids": [1, 2],
        "mask_token_id": 1023,
        "block_size": 3,
        "transformer_layer_config": {
            "hidden_size": 128,
            "num_attention_heads": 8,
            "max_position_embeddings": 128,
            "q_lora_rank": 16,
            "kv_lora_rank": 8,
            "qk_nope_head_dim": 8,
            "qk_rope_head_dim": 8,
            "v_head_dim": 16,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(source))
    return tmp_path


def test_glm_speculators_restores_mla_before_real_after_validator(initialize_dspark, glm_dspark_checkpoint):
    hf_config = SpeculatorsConfig.from_pretrained(glm_dspark_checkpoint, local_files_only=True)
    assert hf_config.architectures == ["Qwen3DSparkModel"]
    config = initialize_dspark(hf_config, model=str(glm_dspark_checkpoint))
    draft = config.draft_model_config
    assert draft.architectures == ["Glm5DSparkForCausalLM"]
    assert draft.use_mla
    assert draft._architecture == "Glm5DSparkForCausalLM"
    assert draft.hf_config.ptd_token_id == 1023
    draft.registry.inspect_model_cls.assert_called_once_with(["Glm5DSparkForCausalLM"], draft)


@pytest.mark.parametrize("q_lora_rank", [None, 0, "16"])
def test_glm_speculators_rejects_malformed_mla_dimensions(initialize_dspark, glm_dspark_checkpoint, q_lora_rank):
    hf_config = SpeculatorsConfig.from_pretrained(glm_dspark_checkpoint, local_files_only=True)
    hf_config.q_lora_rank = q_lora_rank
    with pytest.raises(ValueError, match="cannot use GQA"):
        initialize_dspark(hf_config, model=str(glm_dspark_checkpoint))


def test_gqa_pcp_dcp_preserves_other_upstream_checks(monkeypatch):
    model_config = SimpleNamespace(use_mla=False, is_moe=True, runner_type="generate")
    parallel_config = SimpleNamespace(
        enable_expert_parallel=True,
        tensor_parallel_size=8,
        prefill_context_parallel_size=2,
        decode_context_parallel_size=2,
    )
    validated = []

    def validate_config(model, config):
        validated.append((model, config))
        assert config is not parallel_config
        assert config.enable_expert_parallel is True
        assert config.decode_context_parallel_size == 1

    monkeypatch.setattr(patch_speculative_config, "_orig_verify_with_parallel_config", validate_config)
    _ascend_verify_with_parallel_config(model_config, parallel_config)
    assert len(validated) == 1
    assert validated[0][0] is model_config
    assert parallel_config.decode_context_parallel_size == 2


def test_non_moe_draft_keeps_original_parallel_validation_bypass(monkeypatch):
    model_config = SimpleNamespace(use_mla=False, is_moe=False, runner_type="draft")
    parallel_config = SimpleNamespace(enable_expert_parallel=True)
    validated = []
    monkeypatch.setattr(
        patch_speculative_config, "_orig_verify_with_parallel_config", lambda *args: validated.append(args)
    )
    assert _ascend_verify_with_parallel_config(model_config, parallel_config) is None
    assert not validated


def test_ascend_parallel_validation_preserves_original_config_on_error(monkeypatch):
    model_config = SimpleNamespace(use_mla=False, is_moe=True, runner_type="generate")
    parallel_config = SimpleNamespace(
        enable_expert_parallel=True,
        tensor_parallel_size=8,
        prefill_context_parallel_size=2,
        decode_context_parallel_size=2,
    )

    def fail(model, config):
        assert config.enable_expert_parallel is True
        assert config.decode_context_parallel_size == 1
        raise ValueError("other upstream validation failed")

    monkeypatch.setattr(patch_speculative_config, "_orig_verify_with_parallel_config", fail)
    with pytest.raises(ValueError, match="other upstream validation failed"):
        _ascend_verify_with_parallel_config(model_config, parallel_config)

    assert parallel_config.enable_expert_parallel is True
    assert parallel_config.decode_context_parallel_size == 2


def test_gqa_tp_only_dcp_keeps_upstream_validation(monkeypatch):
    model_config = SimpleNamespace(use_mla=False, is_moe=True, runner_type="generate")
    parallel_config = SimpleNamespace(
        enable_expert_parallel=True,
        tensor_parallel_size=8,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=2,
    )
    validated = []

    def validate_config(model, config):
        validated.append((model, config))

    monkeypatch.setattr(patch_speculative_config, "_orig_verify_with_parallel_config", validate_config)

    _ascend_verify_with_parallel_config(model_config, parallel_config)
    assert validated == [(model_config, parallel_config)]


@pytest.mark.parametrize("flattened", [False, True])
def test_deepseek_v41_dspark_selects_v41_drafter_and_expert_shape(flattened):
    text_config = SimpleNamespace(
        model_type="deepseek_v41_text",
        dspark_target_layer_ids=[37, 38, 39],
        dspark_n_routed_experts=128,
        num_nextn_predict_layers=3,
        dspark_num_experts_per_tok=3,
    )
    text_config.update = lambda values: text_config.__dict__.update(values)
    hf_config = SimpleNamespace(
        model_type="deepseek_v41",
        architectures=["DSparkDraftModel"],
        text_config=text_config,
    )
    hf_config.update = lambda values: hf_config.__dict__.update(values)
    if flattened:
        from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

        hf_config = DeepseekV41Config(
            text_config={
                "dspark_target_layer_ids": [37, 38, 39],
                "dspark_n_routed_experts": 128,
                "num_nextn_predict_layers": 3,
                "dspark_num_experts_per_tok": 3,
            }
        )
        hf_config.model_type = "deepseek_v4"
        hf_config.architectures = ["DSparkDraftModel"]
        text_config = hf_config
    model_arch_config = ModelArchitectureConfig(
        architectures=["DeepseekV41ForCausalLM"],
        model_type="deepseek_v41",
        text_model_type=text_config.model_type,
        hidden_size=5120,
        total_num_hidden_layers=43,
        total_num_attention_heads=64,
        head_size=512,
        vocab_size=129280,
        total_num_kv_heads=1,
        num_experts=384,
        num_experts_per_token=6,
        quantization_config=None,
        is_deepseek_mla=True,
        is_mm_prefix_lm=True,
        rswa_window=128,
        derived_max_model_len_and_key=(1048576, "max_position_embeddings"),
    )
    registry = MagicMock()
    registry.inspect_model_cls.return_value = ("model-info", "DeepseekV41DSparkModel")
    draft_model_config = SimpleNamespace(
        hf_config=hf_config,
        model_arch_config=model_arch_config,
        registry=registry,
    )

    _normalize_deepseek_dspark_draft(draft_model_config)

    assert hf_config.architectures == ["DeepseekV41DSparkModel"]
    assert hf_config.model_type == "deepseek_v41"
    assert text_config.n_routed_experts == 128
    assert text_config.num_experts_per_tok == 3
    assert text_config.n_mtp_layers == 3
    assert draft_model_config.model_arch_config.num_experts == 128
    assert draft_model_config.model_arch_config.num_experts_per_token == 3
    assert draft_model_config.model_arch_config.text_model_type == text_config.model_type
    registry.inspect_model_cls.assert_called_once_with(["DeepseekV41DSparkModel"], draft_model_config)


def test_deepseek_v41_dspark_keeps_expert_count_when_optional_field_is_absent():
    text_config = SimpleNamespace(
        model_type="deepseek_v41_text",
        dspark_target_layer_ids=[1],
        dspark_n_routed_experts=128,
        num_nextn_predict_layers=3,
    )
    text_config.update = lambda values: text_config.__dict__.update(values)
    hf_config = SimpleNamespace(model_type="deepseek_v41", text_config=text_config, architectures=["DSparkDraftModel"])
    hf_config.update = lambda values: hf_config.__dict__.update(values)
    model_arch_config = ModelArchitectureConfig(
        architectures=["DeepseekV41ForCausalLM"],
        model_type="deepseek_v41",
        text_model_type="deepseek_v41_text",
        hidden_size=5120,
        total_num_hidden_layers=43,
        total_num_attention_heads=64,
        head_size=512,
        vocab_size=129280,
        total_num_kv_heads=1,
        num_experts=384,
        num_experts_per_token=6,
        quantization_config=None,
        is_deepseek_mla=True,
        is_mm_prefix_lm=True,
        rswa_window=128,
        derived_max_model_len_and_key=(1048576, "max_position_embeddings"),
    )
    registry = MagicMock()
    registry.inspect_model_cls.return_value = ("model-info", "DeepseekV41DSparkModel")
    draft_model_config = SimpleNamespace(hf_config=hf_config, model_arch_config=model_arch_config, registry=registry)

    _normalize_deepseek_dspark_draft(draft_model_config)

    assert not hasattr(text_config, "num_experts_per_tok")
    assert draft_model_config.model_arch_config.num_experts_per_token == 6
