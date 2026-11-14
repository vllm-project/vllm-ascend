# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter keys must resolve to the modules the model actually builds.

DSV4 HF checkpoints spell the attention block ``.attn.`` while the vLLM model
tree keeps ``.self_attn.``. Adapter keys that miss the module are dropped
without an error, so the rename rule is covered directly here.
"""

from vllm.lora.utils import parse_fine_tuned_lora_name

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
