# SPDX-License-Identifier: Apache-2.0
"""Text-only loop-depth profiling with a GLM-5.3 checkpoint."""

from copy import copy

import torch
from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.utils import extract_layer_index
from vllm.sequence import IntermediateTensors

from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.models.deepseek_mtp import AscendGlmMoeDsaForCausalLM

# Seven complete IndexShare groups in the middle of the 78-layer checkpoint.
LOOP_START = 26
LOOP_END = LOOP_START + 28
PROFILE_WINDOW = 128


def _window_indices(positions, offsets):
    starts = (positions - offsets.numel() + 1).clamp_min(0)
    indices = starts[:, None] + offsets
    return indices.masked_fill(indices > positions[:, None], -1).to(torch.int32).unsqueeze(1)


def _merge_attention(global_result, local_result):
    global_out, global_max, global_sum = global_result
    local_out, local_max, local_sum = local_result
    global_lse = global_max.float() + global_sum.float().log()
    local_lse = local_max.float() + local_sum.float().log()
    local_weight = (local_lse - global_lse).sigmoid().nan_to_num()
    local_weight = local_weight.permute(1, 0, 2).reshape(*local_out.shape[:2], 1)
    return torch.lerp(global_out.float(), local_out.float(), local_weight).to(local_out.dtype)


class _ReplaySFA(AscendSFAImpl):
    """Reuse the prepared GLM projections and read two independent KV representations."""

    def _execute_sparse_flash_attention_process(
        self,
        ql_nope,
        q_pe,
        kv_cache,
        topk_indices,
        attn_metadata,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table=None,
    ):
        global_layer = self.global_layer
        global_metadata = get_forward_context().attn_metadata[global_layer.layer_name]
        attend = DeviceOperator.execute_sparse_flash_attention_process
        global_result = attend(
            self,
            ql_nope,
            q_pe,
            global_layer.kv_cache,
            topk_indices,
            global_metadata,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            return_lse=True,
        )
        local_result = attend(
            self,
            ql_nope,
            q_pe,
            kv_cache,
            self.window_indices[: ql_nope.shape[0]],
            attn_metadata,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            return_lse=True,
        )
        return _merge_attention(global_result, local_result)


class Glm54LoopProfileForCausalLM(AscendGlmMoeDsaForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        parallel = vllm_config.parallel_config
        config = vllm_config.model_config
        if not config.enforce_eager or parallel.pipeline_parallel_size != 1 or vllm_config.speculative_config:
            raise ValueError("GLM loop profiling requires eager execution, PP=1 and ordinary decoding")
        if parallel.decode_context_parallel_size != 1 or parallel.prefill_context_parallel_size != 1:
            raise ValueError("GLM loop profiling requires colocated, unsharded attention")
        if vllm_config.kv_transfer_config or vllm_config.attention_config.hisparse_config:
            raise ValueError("GLM loop profiling requires resident KV caches")
        hf = config.hf_text_config
        if hf.model_type != "glm_moe_dsa" or hf.num_hidden_layers != 78:
            raise ValueError("GLM loop profiling expects the 78-layer GLM-5.3 text checkpoint")
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # The fused MoE runner resolves each call by its ordinal in this
        # table. Replay must select the same experts before entering the suffix.
        moe_calls = vllm_config.compilation_config.static_all_moe_layers
        repeated = [name for name in moe_calls if LOOP_START <= extract_layer_index(name) < LOOP_END]
        if repeated:
            after_loop = moe_calls.index(repeated[-1]) + 1
            moe_calls[after_loop:after_loop] = repeated

        # Each group keeps its first-pass retrieval result until replay completes.
        # Full layers write this buffer and their shared layers read the same one.
        group_indices = None
        self.replay_caches = []
        for idx in range(LOOP_START, LOOP_END):
            global_layer = self.model.layers[idx].self_attn.mla_attn.mla_attn
            impl = global_layer.impl
            if not impl.skip_topk:
                if impl.topk_indices_buffer is None or not impl.use_index_cache:
                    raise ValueError("GLM loop profiling requires IndexShare buffers")
                group_indices = torch.empty_like(impl.topk_indices_buffer)
            if group_indices is None:
                raise ValueError("The loop must begin with a full indexer")
            impl.topk_indices_buffer = group_indices
            # A distinct registry entry lets the scheduler own and reclaim local
            # KV pages, including when the batch changes or a request is preempted.
            local_layer = copy(global_layer)
            local_layer.layer_name = global_layer.layer_name + ".loop_window"
            vllm_config.compilation_config.static_forward_context[local_layer.layer_name] = local_layer
            self.replay_caches.append(local_layer)
        if self.model.layers[LOOP_END].self_attn.mla_attn.mla_attn.impl.skip_topk:
            raise ValueError("The suffix must begin with a full indexer")

        # GLM-5.3 has no trained loop gates. Zero logits give a half-step per token.
        self.register_buffer("gate_weights", torch.zeros(2, hf.hidden_size, dtype=config.dtype), persistent=False)
        self.register_buffer("window_offsets", torch.arange(PROFILE_WINDOW), persistent=False)

    def _replay(self, idx, layer, positions, hidden, residual, window_indices):
        wrapper = layer.self_attn.mla_attn
        global_layer = wrapper.mla_attn
        local_layer = self.replay_caches[idx - LOOP_START]
        if not isinstance(local_layer.impl, _ReplaySFA):
            # Construct this view after ModelSlim has packed the checkpoint.
            # The view references the same weights, but owns its execution flags.
            impl = _ReplaySFA.__new__(_ReplaySFA)
            impl.__dict__ = vars(global_layer.impl).copy()
            impl.global_layer = global_layer
            impl.layer_name = local_layer.layer_name
            impl.has_indexer = False
            impl.skip_topk = True
            local_layer.impl = impl
        local_layer.impl.window_indices = window_indices
        wrapper.mla_attn = local_layer
        try:
            return layer(positions, hidden, residual)
        finally:
            wrapper.mla_attn = global_layer
            local_layer.impl.window_indices = None

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layers = self.model.layers
        hidden = self.model.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        residual = None
        for idx in range(LOOP_START):
            hidden, residual = layers[idx](positions, hidden, residual)

        window_indices = _window_indices(positions, self.window_offsets)
        for iteration in range(2):
            block_input = hidden + residual
            for idx in range(LOOP_START, LOOP_END):
                layer = layers[idx]
                if iteration == 1:
                    hidden, residual = self._replay(idx, layer, positions, hidden, residual, window_indices)
                else:
                    hidden, residual = layer(positions, hidden, residual)
            candidate = hidden + residual
            gate = torch.sigmoid(torch.nn.functional.linear(candidate, self.gate_weights[iteration : iteration + 1]))
            hidden, residual = gate * (candidate - block_input), block_input

        for idx in range(LOOP_END, len(layers)):
            hidden, residual = layers[idx](positions, hidden, residual)
        if hidden.shape[0] != positions.shape[0]:
            hidden = tensor_model_parallel_all_gather(hidden + residual, 0)[: positions.shape[0]]
            return self.model.norm(hidden)
        hidden, _ = self.model.norm(hidden, residual)
        return hidden
