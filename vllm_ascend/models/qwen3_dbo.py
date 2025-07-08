from collections.abc import Iterable
from typing import Any, Optional, Union, List
from types import SimpleNamespace

import torch
import torch_npu
from torch import nn
from transformers import PretrainedConfig

from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeModel
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.attention import AttentionMetadata
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.distributed import tensor_model_parallel_all_reduce, get_tensor_model_parallel_world_size, get_tp_group, \
    get_pp_group
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.utils import (make_empty_intermediate_tensors_factory, make_layers, maybe_prefix)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.compilation.decorators import support_torch_compile

from vllm_ascend.multistream.context import (
    advance_step_multistream_layer_context, get_multistream_comm_context,
    get_multistream_layer_context, set_multistream_context)
from vllm_ascend.multistream.base import MSEventKey
from vllm_ascend.multistream.layers import (MultiStreamPostTransformerLayer,
                                            MultiStreamPreTransformerLayer)
from vllm_ascend.multistream.metadata import (MultiStreamConfig,
                                              MultiStreamStepMetadata,
                                              make_multistream_metadata_ds)
from vllm_ascend.ops.fused_moe import AscendFusedMoE, select_experts, apply_mlp
from vllm_ascend.distributed.tensor_parallel import gather_from_sequence_parallel_region
import vllm_ascend.envs as envs_ascend
from vllm_ascend.models.qwen3_moe import CustomQwen3MoeForCausalLM

VLLM_ASCEND_ENABLE_DBO: bool = envs_ascend.VLLM_ASCEND_ENABLE_DBO


class Qwen3MoeDecoderLayerDBO(Qwen3MoeDecoderLayer):
    def __init__(
            self,
            config: PretrainedConfig,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
    ) -> None:
        super(Qwen3MoeDecoderLayerDBO, self).__init__(config, cache_config, quant_config, prefix)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tp_group().rank_in_group
        self.tp_group = get_tp_group().device_group
        self.dummy_vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=1,
            ),
            compilation_config=SimpleNamespace(
                static_forward_context=None,
            ),
            other_setting="value"
        )
        self.config = config

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    # should split ops in Decoder Layer
    def _forward_ms_op_input_layernorm(
            self,
            hidden_states: torch.Tensor,
            residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        return hidden_states, residual

    def _forward_ms_op_attn(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            residual: torch.Tensor,
            kv_cache: Optional[torch.Tensor] = None,
            attn_metadata: Optional[AttentionMetadata] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.dummy_vllm_config.compilation_config.static_forward_context = get_forward_context().no_compile_layers
        with set_forward_context(attn_metadata, self.dummy_vllm_config):
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
            )
        if hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # We scale both hidden_states and residual before
            # rmsnorm, and rmsnorm result would not affect by scale.
            hidden_states *= 1. / self.routed_scaling_factor
            if self.layer_idx == 0:
                # The residual is shared by all layers, we only scale it on
                # first layer.
                residual *= 1. / self.routed_scaling_factor
        return hidden_states, residual

    def _forward_ms_op_post_attn_layernorm(
            self,
            hidden_states: torch.Tensor,
            residual: Optional[torch.Tensor],
    ):
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        return hidden_states, residual

    def _forward_op_gating(
            self,
            hidden_states: torch.Tensor,
            attn_metadata: Optional[AttentionMetadata] = None
    ) -> torch.Tensor:
        if attn_metadata is None:
            attn_metadata = get_forward_context().attn_metadata
        # when profile runs, force experts to load balanced tokens
        # to avoid high memory consumption on a single rank.
        # TODO: need a better flag to indicate whether in profile run or not.
        if attn_metadata is None:
            # for profile run
            self.is_prefill = True
            self.enable_force_load_balance = True
        else:
            # is_prefill = attn_metadata.num_prefills > 0
            is_prefill = False
            self.enable_force_load_balance = False
            if hasattr(attn_metadata, 'with_prefill_across_dp'):
                self.is_prefill = is_prefill or attn_metadata.with_prefill_across_dp

        num_tokens, hidden_dim = hidden_states.shape

        if self.tp_size > 1:
            # pass
            num_tokens, hidden_size = hidden_states.shape
            if num_tokens < self.tp_size:
                hidden_states = nn.functional.pad(
                    hidden_states, (0, 0, 0, self.tp_size - num_tokens))
            chunk_hidden_states = torch.tensor_split(hidden_states,
                                                     self.tp_size,
                                                     dim=0)
            chunked_hidden_states_sizes = [x.shape[0] for x in chunk_hidden_states]
            local_hidden_states = chunk_hidden_states[self.tp_rank]
        else:
            local_hidden_states = hidden_states
            chunked_hidden_states_sizes = None

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.mlp.gate(local_hidden_states)

        # NOTE: now npu_moe_gating_top_k can only support `group_count=256` pattern
        mlp_config = self.config
        if mlp_config.num_experts == 256:
            topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
                router_logits,
                k=mlp_config.num_experts_per_tok,  # topk当前写8
                bias=self.mlp.gate.e_score_correction_bias,
                k_group=mlp_config.topk_group,  # fix: 4
                group_count=mlp_config.n_group,  # fix 8
                group_select_mode=1,  # 0: group中的最大; 1: topk2.sum(fix)
                renorm=0,  # 0: softmax->topk(fix); 1: topk->softmax
                norm_type=1,  # 0: softmax; 1: sigmoid(fix)
                # out_flag=False, # todo new api; 第三个输出是否输出
                # y2_flag=False, # old api; 第三个输出是否输出
                routed_scaling_factor=1,
                eps=float(1e-20))
        else:
            topk_weights, topk_ids = select_experts(
                hidden_states=local_hidden_states,
                router_logits=router_logits,
                top_k=mlp_config.num_experts_per_tok,
                use_grouped_topk=False,
                renormalize=mlp_config.norm_topk_prob,
                topk_group=getattr(mlp_config, "topk_group", None),
                num_expert_group=getattr(mlp_config, "n_group", None),
                custom_routing_function=None,
                scoring_func=getattr(mlp_config, "scoring_func", 'softmax'),
                e_score_correction_bias=getattr(self.mlp.gate, "e_score_correction_bias", None)
            )

        topk_weights = topk_weights.to(hidden_states.dtype)
        # this is a naive implementation for experts load balance so as
        # to avoid accumulating too much tokens on a single rank.
        # currently it is only activated when doing profile runs.
        if self.enable_force_load_balance:
            topk_ids = torch.randint_like(topk_ids, 0, self.config.num_experts)

        return topk_weights, topk_ids, local_hidden_states, chunked_hidden_states_sizes

    def _forward_op_grouped_mlp(
            self, dispatched_input, tokens_per_expert
    ):
        return apply_mlp(
            dispatched_input,
            self.mlp.experts.w13_weight,
            self.mlp.experts.w2_weight,
            tokens_per_expert
        )

    def _forward_combine_comm(
            self, hidden_states, microbatch_id, num_tokens, chunked_hidden_states_sizes
    ):
        token_dispatcher = self.mlp.experts.token_dispatchers[microbatch_id]
        final_hidden_states, _ = token_dispatcher.token_unpermutation(hidden_states)
        if hasattr(self.mlp, 'routed_scaling_factor'):
            final_hidden_states = final_hidden_states * self.mlp.routed_scaling_factor

        if self.tp_size > 1:
            final_hidden_states = gather_from_sequence_parallel_region(final_hidden_states, self.tp_group,
                                                                       chunked_hidden_states_sizes)
            if num_tokens < self.tp_size:
                final_hidden_states = final_hidden_states[:num_tokens]

        if hasattr(self.mlp, "shared_experts"):
            final_hidden_states = final_hidden_states + token_dispatcher.cached_shared_expert_output
            token_dispatcher.cached_shared_expert_output.untyped_storage().resize_(0)
            token_dispatcher.cached_shared_expert_output = None

        final_hidden_states = final_hidden_states.view(num_tokens, -1)

        return final_hidden_states

    def _forward_ms_layer_alltoallv_finegrained(
            self,
            positions: List[torch.Tensor],
            hidden_states: List[torch.Tensor],
            residual: List[torch.Tensor],
            attn_metadata: List[AttentionMetadata],
            kv_cache: Optional[torch.Tensor] = None,
    ):
        layer_index, ms_metadata, attn_metadata = get_multistream_layer_context(
        )
        assert layer_index >= 0 and ms_metadata is not None
        num_micro_batchs = ms_metadata.ms_config.num_micro_batches
        assert len(positions) == num_micro_batchs
        assert len(hidden_states) == num_micro_batchs
        assert residual is not None
        assert attn_metadata is not None
        num_tokens = [None] * num_micro_batchs
        hidden_dims = [None] * num_micro_batchs
        topk_weights, topk_ids = [None] * num_micro_batchs, [None] * num_micro_batchs
        tokens_per_expert = [None] * num_micro_batchs
        dispatched_input = [None] * num_micro_batchs
        shared_expert_output = [None] * num_micro_batchs
        router_expert_output = [None] * num_micro_batchs
        chunked_hidden_states_sizes = [None] * num_micro_batchs
        token_dispatchers = self.mlp.experts.token_dispatchers
        has_shared_expert = hasattr(self.mlp, 'shared_experts')

        def discard_tensor(tensor):
            if isinstance(tensor, torch.Tensor):
                tensor = [tensor]
            for t in tensor:
                t.untyped_storage().resize_(0)

        # block 1 : attention
        # block 2 : Router Gating
        # block 3 : Token DisPatch
        # the attn computation of microbatch 1 can be overlapped with the moe
        # communication in the previous layer, and the attn computation of microbatch 2
        # can be overlapped with the attn communication of microbatch 1
        for i in range(num_micro_batchs):
            forward_context = get_forward_context()
            layer_index, ms_metadata, attn_metadata = get_multistream_layer_context(
            )
            ms_metadata.try_wait_event(layer_index - 1, i, MSEventKey.FFN_AR_FINISH)
            forward_context.attn_metadata = attn_metadata[i]

            # input layernorm
            hidden_states[i], residual[
                i] = self._forward_ms_op_input_layernorm(
                hidden_states[i], residual[i])
            # attention and tp allreduce
            hidden_states[i], residual[i] = self._forward_ms_op_attn(
                positions[i], hidden_states[i], residual[i], kv_cache,
                attn_metadata[i])
            # post attention layer norm
            hidden_states[i], residual[i] = self._forward_ms_op_post_attn_layernorm(
                hidden_states[i], residual[i]
            )
            num_tokens[i], hidden_dims[i] = hidden_states[i].shape
            # If TP is enabled, hidden_states will be chunked.
            topk_weights[i], topk_ids[i], dispatched_input[i], chunked_hidden_states_sizes[i] = self._forward_op_gating(
                hidden_states[i], attn_metadata[i])
            token_dispatchers[i].preprocess_and_permtute1(
                dispatched_input[i], topk_weights[i], topk_ids[i],
                shared_experts=None, shared_experts_input=None
            )
            # Launch DisPatch Comm in a New Stream.
            dispatch_context = MultiStreamStepMetadata(
                comm_stream=ms_metadata.communicate_stream,
                before_comm_event=ms_metadata.ms_events[layer_index][i][
                    MSEventKey.MOE_BEFORE_COMM],
                after_comm_event=ms_metadata.ms_events[layer_index][i][
                    MSEventKey.MOE_AFTER_COMM],
            )
            dispatch_context.before_comm_event.record()
            # print_with_sync(f'begin token dispatch{i}...', torch.distributed.get_rank())
            with torch.npu.stream(dispatch_context.comm_stream):
                dispatch_context.comm_stream.wait_event(dispatch_context.before_comm_event)
                token_dispatchers[i].dispatch_alltoall()
                dispatched_input[i], tokens_per_expert[i] = token_dispatchers[i].permute2()
                dispatch_context.after_comm_event.record()

        # print_with_sync('begin experts...', torch.distributed.get_rank())
        # block 4 : Router Experts Computation
        # block 5 : Token Combine Communication
        for i in range(num_micro_batchs):
            ms_metadata.try_wait_event(layer_index, i, MSEventKey.MOE_AFTER_COMM)
            discard_tensor(hidden_states[i])
            router_expert_output[i] = self._forward_op_grouped_mlp(dispatched_input[i], tokens_per_expert[i])
            discard_tensor(dispatched_input[i])

            # Launch Combine Comm in a New Stream.
            combine_context = MultiStreamStepMetadata(
                comm_stream=ms_metadata.communicate_stream,
                before_comm_event=ms_metadata.ms_events[layer_index][i][
                    MSEventKey.FFN_COM_FINISH],
                after_comm_event=ms_metadata.ms_events[layer_index][i][
                    MSEventKey.FFN_AR_FINISH],
            )
            combine_context.before_comm_event.record()
            ms_metadata.try_wait_event(layer_index, i, MSEventKey.MOE_SE_COMM_FINISH)
            with torch.npu.stream(combine_context.comm_stream):
                combine_context.comm_stream.wait_event(combine_context.before_comm_event)
                hidden_states[i] = self._forward_combine_comm(
                    router_expert_output[i], i, num_tokens[i], chunked_hidden_states_sizes[i]
                )
                ms_metadata.ms_events[layer_index][i][MSEventKey.FFN_AR_FINISH] = combine_context.comm_stream.record_event()

        return hidden_states, residual


@support_torch_compile
class CustomQwen3DBOMoEModel(Qwen3MoeModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens")
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen3MoeDecoderLayerDBO(config=config,
                                                   cache_config=cache_config,
                                                   quant_config=quant_config,
                                                   prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

        # dbo related members
        if VLLM_ASCEND_ENABLE_DBO:
            self.use_mla = False
            self.multistream_config = MultiStreamConfig()
            multistream_metadata = make_multistream_metadata_ds(
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                causal_lm=getattr(config, "causal_lm", True),
                multistream_config=self.multistream_config,
            )
            self.ms_pre_layer = MultiStreamPreTransformerLayer(
                multistream_metadata)
            self.ms_post_layer = MultiStreamPostTransformerLayer(
                multistream_metadata)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        num_normal_layers = (0 if VLLM_ASCEND_ENABLE_DBO and self.can_run_ms()
                             else self.end_layer - self.start_layer)

        moe_start_layer = self.start_layer + num_normal_layers
        for i in range(self.start_layer, min(moe_start_layer, self.end_layer)):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, residual)

        if moe_start_layer < self.end_layer:
            # if we enable multistream/dbo, process sparse layers here
            hidden_states, residual = self._forward_ms_layers(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                moe_start_layer=moe_start_layer
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def can_run_ms(self):
        attn_metadata = get_forward_context().attn_metadata
        # enable prefill overlap
        with_prefill = get_forward_context().with_prefill
        if attn_metadata is None or not with_prefill or not attn_metadata.enable_dbo_across_dp:
            return False

        return True

    def _forward_ms_layers(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            residual: torch.Tensor,
            moe_start_layer: int,
            kv_caches: Optional[List[torch.Tensor]] = None,
    ):

        if moe_start_layer == self.end_layer:
            return hidden_states, residual

        attn_metadata, [positions, hidden_states,
                        residual] = self.ms_pre_layer(
            [positions, hidden_states, residual], )
        num_micro_batch = len(attn_metadata)
        # the rest layers
        for i in range(moe_start_layer, self.end_layer):
            layer = self.layers[i]
            ms_layer_forward_func = layer._forward_ms_layer_alltoallv_finegrained
            # print("get_called......")
            hidden_states, residual = ms_layer_forward_func(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                attn_metadata=attn_metadata,
            )
            advance_step_multistream_layer_context()

        layer_index, ms_metadata, attn_metadata = get_multistream_layer_context()
        for i in range(num_micro_batch):
            ms_metadata.try_wait_event(layer_index - 1, i, MSEventKey.FFN_AR_FINISH)


        [hidden_states,
         residual] = self.ms_post_layer([hidden_states, residual], )
        return hidden_states, residual


class CustomQwen3MoeForCausalLMDBO(Qwen3MoeForCausalLM):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
        "experts":
            ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = CustomQwen3DBOMoEModel(vllm_config=vllm_config,
                                            prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      quant_config=quant_config)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)
    
    def forward(self, *args, **kwargs):
        if "graph_enable" in kwargs:
            kwargs.pop('graph_enable')
        return super().forward(*args, **kwargs)



