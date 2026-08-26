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
"""Enable sequence-parallel MoE together with pipeline parallelism.

Upstream vLLM disables SP-MoE when PP > 1 because the PP boundary could not
carry sequence-sharded hidden states (see the TODO in deepseek_v2.py). This
opt-in patch (VLLM_ASCEND_ENABLE_SP_WITH_PP=1) re-enables the combination
for qwen3_next / qwen3_5 / deepseek_v2 by:

1. overriding the per-layer SP flag after model construction (dropping the
   pipeline_parallel_size == 1 condition),
2. replacing the decoder-layer forwards with versions that downgrade SP to
   an explicit all-reduce on steps whose token count is below the TP size
   (decode steps; SP padding rows otherwise produce NaN that spreads
   through every subsequent gather),
3. wrapping the model forwards so IntermediateTensors handed across PP
   stages are gathered back to full-token tensors first. The trigger is an
   explicit SP flag rather than a shape check: after SP padding a shard can
   equal full_num_tokens (T=1, tp=2), which would silently skip the gather
   and leak a pad row across the boundary.

Verified on Ascend 910 with Qwen3.5-35B-A3B (PP2 x DP2 x TP2 + EP +
allgather_reducescatter): prefill runs SP (per-layer reduce-scatter
kernels), decode runs the downgraded all-reduce path, outputs match the
PP=1-SP and PP>1-SP-off baselines.
"""

import functools
import inspect
from typing import TYPE_CHECKING

import torch
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.sequence import IntermediateTensors

from vllm_ascend.envs import VLLM_ASCEND_ENABLE_SP_WITH_PP

if TYPE_CHECKING:
    # runtime-injected in _install_patches; imported here only for type
    # checkers (never at module import time, see apply())
    from vllm.model_executor.models.deepseek_v2 import (
        DeepseekAttention,
        DeepseekV2MLP,
    )


def _qwen3next_layer_forward(
    self,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor = None,
    **kwargs: object,
):
    full_num_tokens = positions.shape[-1]
    # A shard is only ever >= tp_size (the downgrade path never chunks
    # smaller batches); sizes below tp_size that differ from full_num_tokens
    # are not SP shards, so do not treat them as such even though a padded
    # shard can equal full_num_tokens (T=1, tp=2).
    input_is_sequence_parallel = (
        self.use_attn_reduce_scatter_for_moe
        and residual is not None
        and hidden_states.shape[0] != full_num_tokens
        and hidden_states.shape[0] >= get_tensor_model_parallel_world_size()
    )

    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

    if input_is_sequence_parallel:
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        hidden_states = hidden_states[:full_num_tokens]

    if self.layer_type == "linear_attention":
        hidden_states = self.linear_attn(hidden_states=hidden_states)
    elif self.layer_type == "full_attention":
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
        )
    else:
        raise ValueError("Invalid layer_type")

    if self.layer_scale:
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states * (self.attn_layer_scale.to(hidden_states.dtype)[0] + 1)
        else:
            hidden_states = hidden_states * (self.attn_layer_scale.to(hidden_states.dtype) + 1)

    # Runtime downgrade: with fewer tokens than TP ranks (decode
    # steps), SP pads rows to the TP size; the pad rows go through
    # norm/experts and can produce NaN. Fall back to the allreduce path.
    sp_this_step = (
        self.use_attn_reduce_scatter_for_moe and hidden_states.shape[0] >= get_tensor_model_parallel_world_size()
    )
    if sp_this_step:
        tp_world_size = get_tensor_model_parallel_world_size()
        # small trick using minus, eg. -17 % 8 = 7
        sp_pad = (-hidden_states.shape[0]) % tp_world_size
        # pad if not divisible by world size
        hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, sp_pad))
        hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)
        if not input_is_sequence_parallel:
            residual = sequence_parallel_chunk(residual)
    elif self.use_attn_reduce_scatter_for_moe:
        # downgraded step: o_proj was built with reduce_results=False, so
        # the partial sums must still be reduced explicitly.
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)

    # Fully Connected
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    if sp_this_step:
        hidden_states = self.mlp(
            hidden_states,
            already_sequence_parallel=True,
        )
    else:
        hidden_states = self.mlp(hidden_states)

    if self.layer_scale:
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states * (self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1)
        else:
            assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                f"shape must be the same {len(hidden_states.shape)}, {len(self.ffn_layer_scale.shape)}"
            )
            hidden_states = hidden_states * (self.ffn_layer_scale.to(hidden_states.dtype) + 1)

    return hidden_states, residual


def _deepseek_layer_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    llama_4_scaling: torch.Tensor | None = None,
) -> torch.Tensor:
    full_num_tokens = positions.shape[0]
    input_is_sequence_parallel = (
        self.use_sequence_parallel_moe
        and residual is not None
        and hidden_states.shape[0] != full_num_tokens
        and hidden_states.shape[0] >= get_tensor_model_parallel_world_size()
    )

    # Self Attention
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

    if input_is_sequence_parallel:
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        hidden_states = hidden_states[:full_num_tokens]

    if self.use_mha:
        hidden_states = self.self_attn(positions, hidden_states)
    else:
        hidden_states = self.self_attn(positions, hidden_states, llama_4_scaling)

    if not isinstance(self.self_attn, DeepseekAttention) and hidden_states.dtype == torch.float16:  # noqa: F821
        # Fix FP16 overflow
        # We scale both hidden_states and residual before
        # rmsnorm, and rmsnorm result would not affect by scale.
        hidden_states *= 1.0 / self.routed_scaling_factor
        if self.layer_idx == 0:
            # The residual is shared by all layers, we only scale it on
            # first layer.
            residual *= 1.0 / self.routed_scaling_factor

    # Runtime downgrade: with fewer tokens than TP ranks (decode
    # steps), SP pads rows to the TP size; the pad rows go through
    # norm/experts and can produce NaN. Fall back to the allreduce path.
    sp_this_step = self.use_sequence_parallel_moe and hidden_states.shape[0] >= get_tensor_model_parallel_world_size()
    if sp_this_step:
        tp_world_size = get_tensor_model_parallel_world_size()
        # small trick using minus, eg. -17 % 8 = 7
        sp_pad = (-hidden_states.shape[0]) % tp_world_size
        # pad if not divisible by world size
        hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, sp_pad))
        hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)
        if not input_is_sequence_parallel:
            residual = sequence_parallel_chunk(residual)
    elif self.use_sequence_parallel_moe:
        # downgraded step: o_proj was built with reduce_results=False, so
        # the partial sums must still be reduced explicitly.
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)

    # Fully Connected
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    if sp_this_step:
        hidden_states = self.mlp(
            hidden_states,
            already_sequence_parallel=True,
        )
    else:
        hidden_states = self.mlp(hidden_states)

    if isinstance(self.mlp, DeepseekV2MLP) and hidden_states.dtype == torch.float16:  # noqa: F821
        # Fix FP16 overflow
        # Scaling the DeepseekV2MLP output, it is the input of
        # input_layernorm of next decoder layer.
        # The scaling of DeepseekV2MOE output would be done in the forward
        # of DeepseekV2MOE
        hidden_states *= 1.0 / self.routed_scaling_factor

    return hidden_states, residual


def sp_with_pp_enabled(parallel_config) -> bool:
    # Active for any PP size when SP-MoE is on. For PP > 1 it enables the
    # combination (upstream hard-disables it); for PP = 1 it routes the
    # layer forwards through the patched copies, whose small-batch
    # downgrade avoids the NaN produced by SP padding rows on decode steps
    # in the upstream forwards.
    return bool(
        VLLM_ASCEND_ENABLE_SP_WITH_PP and parallel_config is not None and parallel_config.use_sequence_parallel_moe
    )


def _gather_intermediate_tensors(tensors, full_num_tokens: int) -> IntermediateTensors:
    """Gather sequence-sharded hidden_states/residual to full-token tensors."""
    out = dict(tensors.tensors)
    for name in ("hidden_states", "residual"):
        t = out.get(name)
        if t is None or t.shape[0] == full_num_tokens:
            continue
        out[name] = tensor_model_parallel_all_gather(t, 0)[:full_num_tokens]
    return IntermediateTensors(out)


def _qwen3next_model_forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
):
    """Copy of vllm Qwen3NextModel.forward adapted for SP-with-PP.

    Differences from upstream (cdc4824a2):
    - the entry `sequence_parallel_chunk(hidden) + assert residual is None`
      block is dropped: it only holds on the first stage, and under PP the
      boundary carries full-token tensors (the patched decoder layers keep
      the residual stream full-token on every path);
    - the non-last-rank IntermediateTensors packing gathers sequence-sharded
      tensors back to full tokens (defensive: the patched layers emit
      full-token outputs, so this is normally a no-op).
    """
    from itertools import islice

    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    full_num_tokens = positions.shape[-1]
    aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
    for layer_idx, layer in enumerate(
        islice(self.layers, self.start_layer, self.end_layer),
        start=self.start_layer,
    ):
        hidden_states, residual = layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )
        self._maybe_add_hidden_state(aux_hidden_states, layer_idx + 1, hidden_states, residual)

    if not get_pp_group().is_last_rank:
        if hidden_states.shape[0] != full_num_tokens:
            hidden_states, residual = globals()["_all_gather_hidden_and_residual"](
                hidden_states,
                residual,
                full_num_tokens,
                self.config.hidden_size,
            )
        return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
    hidden_states, _ = self.norm(hidden_states, residual)
    if hidden_states.shape[0] != full_num_tokens:
        if aux_hidden_states:
            hidden_size = hidden_states.shape[-1]
            hidden_states = torch.cat([hidden_states, *aux_hidden_states], dim=-1)
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states[:full_num_tokens]
            hidden_states, *aux_hidden_states = hidden_states.split(hidden_size, dim=-1)
        else:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states[:full_num_tokens]
    if aux_hidden_states:
        return hidden_states, aux_hidden_states
    return hidden_states


def _wrap_model_forward(model_cls):
    orig_forward = model_cls.forward
    param_names = list(inspect.signature(orig_forward).parameters)

    @functools.wraps(orig_forward)
    def sp_pp_forward(self, *args, **kwargs):
        out = orig_forward(self, *args, **kwargs)
        if not isinstance(out, IntermediateTensors):
            return out
        vllm_config = getattr(self, "vllm_config", None)
        if not sp_with_pp_enabled(vllm_config.parallel_config if vllm_config is not None else None):
            return out
        if not any(
            getattr(layer, "use_attn_reduce_scatter_for_moe", False)
            or getattr(layer, "use_sequence_parallel_moe", False)
            for layer in getattr(self, "layers", [])
        ):
            return out
        positions = kwargs.get("positions")
        if positions is None and "positions" in param_names:
            idx = param_names.index("positions") - 1
            if 0 <= idx < len(args):
                positions = args[idx]
        if positions is None:
            return out
        return _gather_intermediate_tensors(out, positions.shape[-1])

    model_cls.forward = sp_pp_forward


def _wrap_layer_init(layer_cls):
    """Build the layer with SP semantics under PP.

    The SP flag is consumed at construction time: the attention output
    projection is built with reduce_results=not flag. Overriding the flag
    after construction cannot rebuild the projection, so instead a
    pipeline_parallel_size=1 view of the config is handed to the original
    __init__; the flag it computes under this view is then already correct
    (including the per-model MoE-layer conditions).
    """
    orig_init = layer_cls.__init__

    @functools.wraps(orig_init)
    def sp_pp_layer_init(self, vllm_config, *args, **kwargs):
        pc = vllm_config.parallel_config
        if not sp_with_pp_enabled(pc):
            orig_init(self, vllm_config, *args, **kwargs)
            return
        import copy

        cfg_view = copy.copy(vllm_config)
        pc_view = copy.copy(pc)
        pc_view.pipeline_parallel_size = 1
        cfg_view.parallel_config = pc_view
        orig_init(self, cfg_view, *args, **kwargs)
        # per-instance gate: only PP>1 instances take the patched forward;
        # PP=1 instances must stay byte-identical to unpatched vllm
        self._sp_with_pp_active = True

    layer_cls.__init__ = sp_pp_layer_init


def _install_patches():
    from vllm.model_executor.models import deepseek_v2, qwen3_5, qwen3_next

    # the copied layer forwards reference symbols from the model modules;
    # resolve them against this module's globals
    globals()["DeepseekV2MLP"] = deepseek_v2.DeepseekV2MLP
    globals()["DeepseekV2MoE"] = deepseek_v2.DeepseekV2MoE
    globals()["DeepseekAttention"] = deepseek_v2.DeepseekAttention
    if hasattr(qwen3_next, "_all_gather_hidden_and_residual"):
        globals()["_all_gather_hidden_and_residual"] = qwen3_next._all_gather_hidden_and_residual

    # upstream-drift sentinels: fail loudly instead of patching wrong code
    q_src = inspect.getsource(qwen3_next.Qwen3NextDecoderLayer.forward)
    assert "use_attn_reduce_scatter_for_moe:" in q_src, (
        "vllm Qwen3NextDecoderLayer.forward drifted; update patch_sp_with_pp.py"
    )
    d_src = inspect.getsource(deepseek_v2.DeepseekV2DecoderLayer.forward)
    assert "use_sequence_parallel_moe:" in d_src, (
        "vllm DeepseekV2DecoderLayer.forward drifted; update patch_sp_with_pp.py"
    )
    qm_src = inspect.getsource(qwen3_next.Qwen3NextModel.forward)
    assert "assert residual is None" in qm_src, (
        "vllm Qwen3NextModel.forward no longer contains the SP entry block "
        "this copy was adapted from; update patch_sp_with_pp.py"
    )

    _orig_qnext_forward = qwen3_next.Qwen3NextDecoderLayer.forward
    _orig_dsv2_forward = deepseek_v2.DeepseekV2DecoderLayer.forward

    @functools.wraps(_orig_qnext_forward)
    def _qnext_forward_gate(self, *args, **kwargs):
        if not getattr(self, "_sp_with_pp_active", False):
            return _orig_qnext_forward(self, *args, **kwargs)
        return _qwen3next_layer_forward(self, *args, **kwargs)

    @functools.wraps(_orig_dsv2_forward)
    def _dsv2_forward_gate(self, *args, **kwargs):
        if not getattr(self, "_sp_with_pp_active", False):
            return _orig_dsv2_forward(self, *args, **kwargs)
        return _deepseek_layer_forward(self, *args, **kwargs)

    qwen3_next.Qwen3NextDecoderLayer.forward = _qnext_forward_gate
    deepseek_v2.DeepseekV2DecoderLayer.forward = _dsv2_forward_gate

    # wrap the base first so the subclass's super().__init__ call goes
    # through the wrapped base
    _wrap_layer_init(qwen3_next.Qwen3NextDecoderLayer)
    _wrap_layer_init(qwen3_5.Qwen3_5DecoderLayer)
    _wrap_layer_init(deepseek_v2.DeepseekV2DecoderLayer)
    _qwen3next_model_forward_wrapped = functools.wraps(qwen3_next.Qwen3NextModel.forward)(
        _qwen3next_model_forward.__get__(qwen3_next.Qwen3NextModel)
    )
    qwen3_next.Qwen3NextModel.forward = _qwen3next_model_forward_wrapped
    _wrap_model_forward(deepseek_v2.DeepseekV2Model)


def apply():
    if not VLLM_ASCEND_ENABLE_SP_WITH_PP:
        return
    # Installing the patches requires importing the model modules, which
    # must NOT happen at platform-plugin registration time: importing
    # vllm.model_executor.layers.fused_moe before the NPU stack initializes
    # its MoE kernels leaves the router on the CUDA-only dispatch path.
    # Defer to the first model-class resolution instead.
    from vllm.model_executor.models.registry import ModelRegistry

    orig_resolve = ModelRegistry.resolve_model_cls
    state = {"installed": False}

    @functools.wraps(orig_resolve)
    def resolve_then_patch(self, *args, **kwargs):
        out = orig_resolve(self, *args, **kwargs)
        if not state["installed"]:
            state["installed"] = True
            _install_patches()
        return out

    ModelRegistry.resolve_model_cls = resolve_then_patch
