from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, tensor_model_parallel_all_gather
from vllm.model_executor.layers.fused_moe.utils import is_model_fused_shared_expert_compatible
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2Model,
    DeepseekV2MoE,
    GlmMoeDsaForCausalLM,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    make_empty_intermediate_tensors_factory,
    make_layers,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.mla.index_group import (
    SparseMLAIndexGroupBuilder,
    get_sparse_mla_index_group_max_rows,
)

from vllm_ascend.utils import enable_dsa_cp, is_rot_weight_used
from vllm_ascend.worker.v2 import pp_transport


@support_torch_compile
class AscendDeepSeekMTP(DeepSeekMTP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.is_rot_weight_used = is_rot_weight_used(vllm_config)
        if self.is_rot_weight_used:
            self.rot = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if self.is_rot_weight_used:
            hidden_states = self.rot(hidden_states)
        return super().forward(input_ids, positions, hidden_states, intermediate_tensors, inputs_embeds, spec_step_idx)

    def _maybe_set_own_lm_head(self, loaded_weights: set[str]) -> None:
        """Expose a checkpoint-provided MTP head to the runner.

        DeepSeekMTP always constructs ``shared_head``, so module existence does
        not prove head ownership. Treat it as independent only when its weight
        was loaded from the checkpoint.
        """
        mtp_layer_idx = self.model.mtp_start_layer_idx
        own_head_weight = f"model.layers.{mtp_layer_idx}.shared_head.head.weight"
        if own_head_weight not in loaded_weights:
            # GLM-5.3 and friends ship no MTP head, leaving shared_head.head at
            # its allocation-time contents. Record that so the proposer shares
            # the target head instead of inspecting those values.
            self.has_own_lm_head = False
            return

        self.has_own_lm_head = True
        mtp_layer = self.model.layers[str(mtp_layer_idx)]
        self.lm_head = mtp_layer.shared_head.head

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if self.quant_config is not None and (cache_scale_mapper := self.quant_config.get_cache_scale_mapper()):
            weights = cache_scale_mapper.apply(weights)

        weights_mapper = WeightsMapper(
            orig_to_new_prefix={"rot.": f"model.layers.{self.config.num_hidden_layers}.rot."},
        )
        loaded_weights = super().load_weights(weights_mapper.apply(weights))
        self._maybe_set_own_lm_head(loaded_weights)
        return loaded_weights

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        if "rot" in name:
            name = name.replace(f"model.layers.{spec_layer}.rot.", "rot.")
            return name
        return super()._rewrite_spec_layer_name(spec_layer, name)


class AscendDSACPDecoderLayer(DeepseekV2DecoderLayer):
    input_is_sequence_parallel: bool = False
    gather_sequence_parallel_output: bool = False

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            not self.input_is_sequence_parallel
            or llama_4_scaling is not None
            or not self.self_attn.mla_attn.supports_sequence_parallel_input
        ):
            # Preserve the first replicated -> SP transition and configurations
            # whose attention/projection contract requires the upstream path.
            hidden_states, residual = super().forward(positions, hidden_states, residual, llama_4_scaling)
        else:
            # Attention and MoE share the same contiguous TP token shard. The
            # backend owns any head reduction; a second model-level RS is invalid.
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
            output_scale = 1.0 / self.routed_scaling_factor if hidden_states.dtype == torch.float16 else 1.0
            hidden_states = self.self_attn.mla_attn.forward_sequence_parallel(
                positions, hidden_states, output_scale=output_scale
            )
            # The backend scales partial projections before TP reduction to
            # preserve upstream's FP16 overflow protection.
            if hidden_states.dtype == torch.float16 and self.layer_idx == 0:
                residual *= 1.0 / self.routed_scaling_factor
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
            hidden_states = self.mlp(hidden_states, already_sequence_parallel=True)
        if self.gather_sequence_parallel_output:
            # A one-token shard can have the global row count. The producer's
            # layout, rather than tensor shape, determines boundary restoration.
            combined = tensor_model_parallel_all_gather(torch.cat([hidden_states, residual], dim=-1), 0)
            hidden_states, residual = combined[: positions.shape[0]].chunk(2, dim=-1)
            hidden_states, residual = hidden_states.contiguous(), residual.contiguous()
        return hidden_states, residual


@support_torch_compile
class AscendDSACPModel(DeepseekV2Model):
    """GLM V1 model retaining SP/DSA-CP shards across compatible layers."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # The upstream constructor hardcodes its decoder class. Construct the
        # Ascend layers here, retaining inherited forward and weight loading.
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = current_platform.device_type
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.is_v32 = hasattr(config, "index_topk")
        topk_indices_buffer = (
            torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk,
                dtype=torch.int32,
                device=self.device,
            )
            if self.is_v32
            else None
        )
        index_group_builder = (
            SparseMLAIndexGroupBuilder(topk_indices_buffer, get_sparse_mla_index_group_max_rows(vllm_config))
            if topk_indices_buffer is not None
            else None
        )
        self.embed_tokens = (
            VocabParallelEmbedding(
                self.vocab_size, self.hidden_size, quant_config=quant_config, prefix=f"{prefix}.embed_tokens"
            )
            if get_pp_group().is_first_rank
            else PPMissingLayer()
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: AscendDSACPDecoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
                index_group_builder=index_group_builder,
            ),
            prefix=f"{prefix}.layers",
        )
        # Only MoE layers with a supported projection can share SP token shards.
        local_layers = self.layers[self.start_layer : self.end_layer]
        for layer in local_layers:
            o_proj = layer.self_attn.o_proj
            o_proj.reduce_results = False
            layer.use_sequence_parallel_moe = (
                isinstance(layer.mlp, DeepseekV2MoE)
                and layer.mlp.is_sequence_parallel
                and getattr(config, "llama_4_scaling", None) is None
                and layer.self_attn.mla_attn.supports_sequence_parallel_input
            )
            o_proj.reduce_results = not layer.use_sequence_parallel_moe
        # Restore full states before a non-SP consumer or the local layer exit.
        for index, layer in enumerate(local_layers):
            layer.input_is_sequence_parallel = (
                layer.use_sequence_parallel_moe and index > 0 and local_layers[index - 1].use_sequence_parallel_moe
            )
            layer.gather_sequence_parallel_output = layer.use_sequence_parallel_moe and (
                index + 1 == len(local_layers) or not local_layers[index + 1].use_sequence_parallel_moe
            )
        self.is_fused_shared_expert_enabled = is_model_fused_shared_expert_compatible(self.layers, DeepseekV2MoE, "mlp")
        self.norm = (
            RMSNorm(self.hidden_size, eps=config.rms_norm_eps) if get_pp_group().is_last_rank else PPMissingLayer()
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], self.hidden_size
        )
        self.aux_hidden_state_layers = tuple[int, ...]()
        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 0)
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
        self.use_mha = config.model_type == "deepseek" or all(dim == 0 for dim in (qk_nope_head_dim, qk_rope_head_dim))
        self.num_redundant_experts = vllm_config.parallel_config.eplb_config.num_redundant_experts

        # Preserve Ascend's existing auxiliary-state and index-cache contract.
        self._use_upstream_aux_relay = vllm_config.use_v2_model_runner
        self.topk_indices_buffer = topk_indices_buffer
        transport_data_types = (
            pp_transport.PPTransportDataType.AUX_HIDDEN_STATES,
            pp_transport.PPTransportDataType.TOPK_INDICES,
        )
        pp_transport.configure_pp_topk_transport(self, transport_data_types)
        self.make_empty_intermediate_tensors = pp_transport.make_empty_intermediate_tensors(
            self, self.make_empty_intermediate_tensors, transport_data_types
        )


class AscendGlmMoeDsaForCausalLM(GlmMoeDsaForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        if (
            not vllm_config.use_v2_model_runner
            and vllm_config.parallel_config.use_sequence_parallel_moe
            and enable_dsa_cp()
        ):
            self.model_cls = AscendDSACPModel
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if vllm_config.use_v2_model_runner and vllm_config.parallel_config.pipeline_parallel_size > 1:
            # EPLB maps and expert weights must describe the same local layers.
            self.num_moe_layers = len(self.moe_layers)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = WeightsMapper(orig_to_new_prefix={"rot.": None})
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=mapper)
