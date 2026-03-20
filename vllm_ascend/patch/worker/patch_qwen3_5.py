#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
# mypy: ignore-errors

from collections.abc import Iterable
from copy import deepcopy

import torch
import torch.nn.functional as F
from einops import rearrange
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLMBase,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5GatedDeltaNet,
    Qwen3_5Model,
    logger,
)
from vllm.model_executor.models.utils import is_pp_missing_parameter
from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

import vllm_ascend.envs as envs_ascend
from vllm_ascend.attention.utils import maybe_save_kv_layer_to_connector
from vllm_ascend.ops.triton.fla.sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)
from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch
from vllm_ascend.utils import enable_sp


def _qwen35_fused_in_proj_enabled() -> bool:
    return envs_ascend.VLLM_ASCEND_ENABLE_QWEN35_FUSED_IN_PROJ


def _qwen35_debug_fused_in_proj_enabled() -> bool:
    return envs_ascend.VLLM_ASCEND_DEBUG_QWEN35_FUSED_IN_PROJ


def _store_original_method_once(cls, attr_name: str):
    original_attr_name = f"_vllm_ascend_original_{attr_name}"
    if not hasattr(cls, original_attr_name):
        setattr(cls, original_attr_name, getattr(cls, attr_name))
    return getattr(cls, original_attr_name)


def _store_original_mapping_once(cls):
    original_attr_name = "_vllm_ascend_original_packed_modules_mapping"
    if not hasattr(cls, original_attr_name):
        setattr(cls, original_attr_name, deepcopy(cls.packed_modules_mapping))
    return deepcopy(getattr(cls, original_attr_name))


def _get_qwen35_local_in_proj_sizes(layer) -> list[int]:
    return [
        layer.key_dim // layer.tp_size,
        layer.key_dim // layer.tp_size,
        layer.value_dim // layer.tp_size,
        layer.value_dim // layer.tp_size,
        layer.num_v_heads // layer.tp_size,
        layer.num_v_heads // layer.tp_size,
    ]


def _split_qwen35_fused_in_proj_outputs(layer, projected_states: torch.Tensor):
    q_size, k_size, v_size, z_size, b_size, a_size = _get_qwen35_local_in_proj_sizes(layer)
    mixed_qkv, z, b, a = projected_states.split(
        [q_size + k_size + v_size, z_size, b_size, a_size],
        dim=-1,
    )
    z = z.reshape(z.size(0), -1, layer.head_v_dim)
    return mixed_qkv, z, b.contiguous(), a.contiguous()


def _compute_qwen35_legacy_in_proj_outputs(layer, hidden_states: torch.Tensor):
    q_size, k_size, v_size, z_size, b_size, a_size = _get_qwen35_local_in_proj_sizes(layer)
    q_weight, k_weight, v_weight, z_weight, b_weight, a_weight = layer.in_proj.weight.split(
        [q_size, k_size, v_size, z_size, b_size, a_size],
        dim=0,
    )
    query = F.linear(hidden_states, q_weight)
    key = F.linear(hidden_states, k_weight)
    value = F.linear(hidden_states, v_weight)
    z = F.linear(hidden_states, z_weight).reshape(hidden_states.size(0), -1, layer.head_v_dim)
    b = F.linear(hidden_states, b_weight).contiguous()
    a = F.linear(hidden_states, a_weight).contiguous()
    return torch.cat((query, key, value), dim=-1), z, b, a


def _validate_qwen35_fused_in_proj_outputs(
    layer,
    hidden_states: torch.Tensor,
    mixed_qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
):
    expected_outputs = _compute_qwen35_legacy_in_proj_outputs(layer, hidden_states)
    actual_outputs = (mixed_qkv, z, b, a)
    output_names = ("mixed_qkv", "z", "b", "a")
    for name, actual, expected in zip(output_names, actual_outputs, expected_outputs):
        if actual.shape != expected.shape:
            raise RuntimeError(
                f"{layer.prefix}: fused in_proj {name} shape mismatch, "
                f"actual={tuple(actual.shape)}, expected={tuple(expected.shape)}"
            )
        if not torch.allclose(actual, expected, atol=1e-5, rtol=1e-5):
            max_diff = (actual - expected).abs().max().item()
            raise RuntimeError(
                f"{layer.prefix}: fused in_proj {name} mismatch, max_diff={max_diff}"
            )


def _build_qwen35_fused_packed_modules_mapping(
    base_mapping: dict[str, list[str]],
) -> dict[str, list[str]]:
    mapping = deepcopy(base_mapping)
    mapping.pop("in_proj_qkvz", None)
    mapping.pop("in_proj_ba", None)
    mapping["in_proj"] = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"]
    return mapping


def _apply_qwen35_packed_modules_mapping():
    for cls in (Qwen3_5ForCausalLMBase, Qwen3_5ForConditionalGeneration):
        original_mapping = _store_original_mapping_once(cls)
        if _qwen35_fused_in_proj_enabled():
            cls.packed_modules_mapping = _build_qwen35_fused_packed_modules_mapping(original_mapping)
        else:
            cls.packed_modules_mapping = original_mapping


def _get_qwen35_stacked_params_mapping():
    stacked_params_mapping = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    if _qwen35_fused_in_proj_enabled():
        stacked_params_mapping.extend(
            [
                ("in_proj", "in_proj_qkv", (0, 1, 2)),
                ("in_proj", "in_proj_z", 3),
                ("in_proj", "in_proj_b", 4),
                ("in_proj", "in_proj_a", 5),
            ]
        )
    else:
        stacked_params_mapping.extend(
            [
                ("in_proj_qkvz", "in_proj_qkv", (0, 1, 2)),
                ("in_proj_qkvz", "in_proj_z", 3),
                ("in_proj_ba", "in_proj_b", 0),
                ("in_proj_ba", "in_proj_a", 1),
            ]
        )
    return stacked_params_mapping


_ORIGINAL_QWEN35_GDN_INIT = _store_original_method_once(Qwen3_5GatedDeltaNet, "__init__")
_ORIGINAL_QWEN35_GDN_FORWARD = _store_original_method_once(Qwen3_5GatedDeltaNet, "forward")
_ORIGINAL_QWEN35_MODEL_LOAD_WEIGHTS = _store_original_method_once(Qwen3_5Model, "load_weights")


class AscendQwen3_5GatedDeltaNet(Qwen3_5GatedDeltaNet):
    def __init__(self, *args, **kwargs):
        _ORIGINAL_QWEN35_GDN_INIT(self, *args, **kwargs)
        if not _qwen35_fused_in_proj_enabled():
            return

        self.in_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[
                self.key_dim,
                self.key_dim,
                self.value_dim,
                self.value_dim,
                self.num_v_heads,
                self.num_v_heads,
            ],
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{self.prefix}.in_proj",
        )
        del self.in_proj_qkvz
        del self.in_proj_ba

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        if not _qwen35_fused_in_proj_enabled():
            return _ORIGINAL_QWEN35_GDN_FORWARD(self, hidden_states, output)

        projected_states, _ = self.in_proj(hidden_states)
        num_tokens = projected_states.size(0)
        mixed_qkv, z, b, a = _split_qwen35_fused_in_proj_outputs(self, projected_states)
        if _qwen35_debug_fused_in_proj_enabled():
            _validate_qwen35_fused_in_proj_outputs(self, hidden_states, mixed_qkv, z, b, a)

        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            self.prefix,
        )

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        # Core attention computation (called by custom op).

        # NOTE: The processing logic of Qwen3_5GatedDeltaNet is the same as Qwen3NextGatedDeltaNet.
        # However, because the ops `torch_npu.npu_recurrent_gated_delta_rule`
        # currently does not support `ssm_state` inputs in float32 format,
        # we temporarily retain the current _forward_core implementation.
        # Once the ops supports float32 `ssm_state`, this patch should be removed.

        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            # V1 profile run
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache[forward_context.virtual_engine]
        conv_state = self_kv_cache[0].transpose(-1, -2)
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        if not enable_sp():
            mixed_qkv = mixed_qkv[:num_actual_tokens]
            b = b[:num_actual_tokens]
            a = a[:num_actual_tokens]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None:
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][: attn_metadata.num_spec_decodes],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )

        # 1.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            if mixed_qkv_non_spec is not None:
                conv_weights_T = conv_weights.transpose(0, 1)
                mixed_qkv_non_spec = torch.ops._C_ascend.causal_conv1d_fn(
                    mixed_qkv_non_spec,
                    conv_weights_T,
                    self.conv1d.bias,
                    activation=self.activation,
                    conv_state=self_kv_cache[0],
                    has_initial_state=has_initial_state,
                    non_spec_state_indices_tensor=non_spec_state_indices_tensor,
                    non_spec_query_start_loc=non_spec_query_start_loc,
                    pad_slot_id=PAD_SLOT_ID,
                )
        elif attn_metadata.num_decodes > 0:
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[: attn_metadata.num_actual_tokens],
                validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None
        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(mixed_qkv_non_spec)

        if attn_metadata.num_prefills > 0 or spec_sequence_masks is not None:
            g, beta = fused_gdn_gating_patch(self.A_log, a, b, self.dt_bias)
            if spec_sequence_masks is not None:
                if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                    g_spec = g
                    beta_spec = beta
                    g_non_spec = None
                    beta_non_spec = None
                else:
                    g_spec = g.index_select(1, spec_token_indx)
                    beta_spec = beta.index_select(1, spec_token_indx)
                    g_non_spec = g.index_select(1, non_spec_token_indx)
                    beta_non_spec = beta.index_select(1, non_spec_token_indx)
            else:
                g_spec = None
                beta_spec = None
                g_non_spec = g
                beta_non_spec = beta

            # 2. Recurrent attention
            if spec_sequence_masks is not None:
                core_attn_out_spec, last_recurrent_state = fused_recurrent_gated_delta_rule(
                    q=query_spec,
                    k=key_spec,
                    v=value_spec,
                    g=g_spec,
                    beta=beta_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=spec_query_start_loc[: attn_metadata.num_spec_decodes + 1],
                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                core_attn_out_spec, last_recurrent_state = None, None

            if attn_metadata.num_prefills > 0:
                initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
                initial_state[~has_initial_state, ...] = 0
                non_spec_chunked_prefill_meta = getattr(
                    attn_metadata,
                    "non_spec_chunked_prefill_meta",
                    None,
                )
                core_attn_out_non_spec, last_recurrent_state = chunk_gated_delta_rule(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=non_spec_query_start_loc,
                    prebuilt_meta=non_spec_chunked_prefill_meta,
                    head_first=False,
                    use_qk_l2norm_in_kernel=True,
                )
                ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(ssm_state.dtype)
            elif attn_metadata.num_decodes > 0:
                core_attn_out_non_spec, last_recurrent_state = fused_recurrent_gated_delta_rule(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                core_attn_out_non_spec, last_recurrent_state = None, None

        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec = fused_sigmoid_gating_delta_rule_update(
                A_log=self.A_log.contiguous(),
                dt_bias=self.dt_bias.contiguous(),
                q=query_non_spec.contiguous(),
                k=key_non_spec.contiguous(),
                v=value_non_spec.contiguous(),
                a=a.contiguous(),
                b=b.contiguous(),
                initial_state_source=ssm_state,
                initial_state_indices=non_spec_state_indices_tensor,
                cu_seqlens=non_spec_query_start_loc,
                use_qk_l2norm_in_kernel=True,
                softplus_beta=1.0,
                softplus_threshold=20.0,
            )

        # 3. Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            if not enable_sp():
                core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
            else:
                core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)[:num_actual_tokens]
        elif spec_sequence_masks is not None:
            if not enable_sp():
                core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
            else:
                core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)[:num_actual_tokens]
        else:
            if not enable_sp():
                core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)
            else:
                core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)[:num_actual_tokens]
        maybe_save_kv_layer_to_connector("", [])


def _patched_qwen35_model_load_weights(
    self,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    if not _qwen35_fused_in_proj_enabled():
        return _ORIGINAL_QWEN35_MODEL_LOAD_WEIGHTS(self, weights)

    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()
    expert_params_mapping = self.get_expert_mapping()
    is_fused_expert = False
    fused_expert_params_mapping = [
        ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
        ("experts.w2_weight", "experts.down_proj", 0, "w2"),
    ]
    num_experts = self.config.num_experts if hasattr(self.config, "num_experts") else 0
    for name, loaded_weight in weights:
        if "rotary_emb.inv_freq" in name:
            continue

        if name.startswith("mtp."):
            continue

        if name.endswith("scale"):
            name = maybe_remap_kv_scale_name(name, params_dict)
            if name is None:
                continue

        for param_name, weight_name, shard_id in _get_qwen35_stacked_params_mapping():
            if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                is_fused_expert = True
                expert_params_mapping = fused_expert_params_mapping

            if weight_name not in name:
                continue

            if "mlp.experts" in name:
                continue

            name = name.replace(weight_name, param_name)
            if name.endswith(".bias") and name not in params_dict:
                continue
            if is_pp_missing_parameter(name, self):
                continue
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            is_expert_weight = False
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue
                is_expert_weight = True
                name_mapped = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(name_mapped, self):
                    continue
                if is_fused_expert:
                    if "experts.gate_up_proj" in name:
                        loaded_weight = loaded_weight.chunk(2, dim=-2)
                        success_w1 = self.load_fused_expert_weights(
                            name_mapped,
                            params_dict,
                            loaded_weight[0],
                            "w1",
                            num_experts,
                        )
                        success_w3 = self.load_fused_expert_weights(
                            name_mapped,
                            params_dict,
                            loaded_weight[1],
                            "w3",
                            num_experts,
                        )
                        success = success_w1 and success_w3
                    else:
                        success = self.load_fused_expert_weights(
                            name_mapped,
                            params_dict,
                            loaded_weight,
                            shard_id,
                            num_experts,
                        )
                    if success:
                        name = name_mapped
                        break
                else:
                    if (
                        name_mapped.endswith(".bias")
                        or name_mapped.endswith("_bias")
                    ) and name_mapped not in params_dict:
                        continue
                    param = params_dict[name_mapped]
                    weight_loader = param.weight_loader
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                if success:
                    name = name_mapped
                    break
            else:
                if is_expert_weight:
                    continue
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    logger.warning_once(
                        f"Parameter {name} not found in params_dict, skip loading"
                    )
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
        loaded_params.add(name)
    return loaded_params


_apply_qwen35_packed_modules_mapping()
Qwen3_5GatedDeltaNet.__init__ = AscendQwen3_5GatedDeltaNet.__init__
Qwen3_5GatedDeltaNet.forward = AscendQwen3_5GatedDeltaNet.forward
Qwen3_5GatedDeltaNet._forward_core = AscendQwen3_5GatedDeltaNet._forward_core
Qwen3_5Model.load_weights = _patched_qwen35_model_load_weights
