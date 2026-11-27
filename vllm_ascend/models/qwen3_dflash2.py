from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import nn
from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.model_executor.models.qwen2 import Qwen2MLP
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
    _resolve_layer_attention,
)
from vllm.model_executor.models.qwen3_dflash2 import DFlashGroupedConv as UpstreamDFlashGroupedConv
from vllm.model_executor.models.utils import maybe_prefix
from vllm.v1.worker.gpu.spec_decode.dflash2.config import DFlash2SpeculatorConfig

from vllm_ascend.models.kimi_k3.kimi_k3 import AscendKimiMLAAttention, store_mla_draft_context_kv
from vllm_ascend.models.quarot_utils import align_draft_weights
from vllm_ascend.ops.kimi_mla import kimi_mla_draft_window
from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla
from vllm_ascend.ops.vocab_parallel_embedding import (
    VocabParallelMode,
    gather_lm_head_hidden_states,
    gather_lm_head_outputs,
)
from vllm_ascend.utils import get_rotation_path

_MAX_EXACT_FP32_TOKEN_ID = 1 << 24


def _grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    # Taps outside the current query block cannot contribute. Skipping them
    # also keeps single-request short drafts from padding beyond the input.
    for tap in range(1, min(taps, block_size)):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        # Multiplying a cross-request NaN or Inf by zero would retain a NaN.
        shifted = torch.where((position >= tap).view(-1, 1, 1), shifted, 0)
        output += coefficients[:, tap] * shifted
    return output.flatten(-2)


class DFlashGroupedConv(UpstreamDFlashGroupedConv):
    def _convolve(self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int) -> torch.Tensor:
        return _grouped_conv(
            hidden_states,
            delta,
            self.base_kernel[side],
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )


class DFlash2Qwen3DecoderLayer(DFlashQwen3DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config,
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        self._init_convs(vllm_config, config, quant_config, prefix)

    def _init_convs(
        self, vllm_config: VllmConfig, config, quant_config: QuantizationConfig | None, prefix: str
    ) -> None:
        draft_config = config.dflash_config
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        conv_args = dict(
            hidden_size=config.hidden_size,
            taps=int(draft_config["conv_kernel_size"]),
            group_size=int(draft_config["conv_group_size"]),
            block_size=1 + speculative_config.num_speculative_tokens,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=quant_config,
        )
        self.attention_conv = DFlashGroupedConv(**conv_args, prefix=maybe_prefix(prefix, "attention_conv"))
        self.mlp_conv = DFlashGroupedConv(**conv_args, prefix=maybe_prefix(prefix, "mlp_conv"))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = self.attention_conv.finish(hidden_states, coefficients)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
        return hidden_states, residual


def _score_edges(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    successors = successor_table[candidate_ids]
    predecessor_ids = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits[:, :, None] + torch.einsum("blpr,blcr->blpc", predecessors * hidden[:, :, None], successors)


# Context-free calls bypass any internal ACLGraphWrapper's capture/replay.
# Compiled device operations still participate in an active outer draft capture.
@support_torch_compile
class CandidateSelector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        top_k: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False)
        self.successor_codebook = nn.Parameter(torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False)
        self.hidden_projection = ReplicatedLinear(
            hidden_size,
            rank,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "hidden_projection"),
            return_bias=False,
        )

    def forward(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states is [num_reqs, num_steps, H]. NPU unquantized_gemm's Dynamo
        # fake is 2D-only, so flatten to [tokens, H], project, then restore (B, L).
        # torch._check ties independently marked dynamic batch dims so
        # predecessors [B, L, K, R] can broadcast with projected hidden.
        torch._check(hidden_states.shape[0] == candidate_ids.shape[0])
        torch._check(hidden_states.shape[1] == candidate_ids.shape[1])
        hidden = self.hidden_projection(hidden_states.flatten(0, 1))
        hidden = hidden.view(*hidden_states.shape[:-1], -1)
        return _score_edges(
            self.predecessor_codebook,
            self.successor_codebook,
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
            self.top_k,
        )


class DFlash2Qwen3Model(DFlashQwen3Model):
    # vLLM #52816 switched the parent constructor from its module global to
    # this class factory. Declare the main-lane factory so DFlash2 decoder
    # layers are built.
    decoder_layer_cls = DFlash2Qwen3DecoderLayer

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )

        draft_config = self.config.dflash_config
        self.input_embedding_scale = float(draft_config.get("input_embedding_scale", 1.0))
        # The selector has different inputs from the draft backbone, so it
        # must use a separate compilation cache namespace.
        with set_model_tag("dflash2_candidate_selector"):
            self.candidate_selector = CandidateSelector(
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
                rank=int(draft_config["selector_rank"]),
                top_k=int(draft_config["selector_top_k"]),
                params_dtype=vllm_config.model_config.dtype,
                prefix=maybe_prefix(prefix, "candidate_selector"),
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().embed_input_ids(input_ids) * self.input_embedding_scale


def _maybe_gather_lm_head_hidden_states(
    hidden_states: torch.Tensor,
    lm_head: nn.Module,
    capacity: int,
) -> torch.Tensor:
    """Gather request rows only for fine-grained LM-head TP."""
    if getattr(lm_head, "parallel_mode", None) is VocabParallelMode.FINE_GRAINED:
        return gather_lm_head_hidden_states(hidden_states, capacity)
    return hidden_states


def _gather_lm_head_candidates(
    values: torch.Tensor,
    ids: torch.Tensor,
    lm_head: nn.Module,
    num_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect vocabulary-shard candidates and return each owner's global TopK."""
    top_k = values.shape[-1]
    parallel_mode = getattr(lm_head, "parallel_mode", VocabParallelMode.STANDARD)
    if parallel_mode is VocabParallelMode.REPLICATED or (
        parallel_mode is VocabParallelMode.STANDARD and get_tensor_model_parallel_world_size() == 1
    ):
        return values, ids
    # Interleave scores and IDs for one exchange when FP32 preserves both.
    # Larger IDs and FP64 scores retain their own dtype in separate exchanges.
    packed = (
        values.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and lm_head.num_embeddings_padded <= _MAX_EXACT_FP32_TOKEN_ID
    )
    # Packed scores and IDs complete this TP column's candidates in one exchange.
    output_gather_group = getattr(lm_head, "output_gather_group", None)
    if packed:
        score_dtype = values.dtype
        pairs = torch.stack((values.float(), ids.float()), dim=-1)
        pairs = gather_lm_head_outputs(pairs, parallel_mode, output_gather_group)
        values = pairs[..., 0].to(score_dtype)
        ids = pairs[..., 1].to(torch.int64)
    else:
        values = gather_lm_head_outputs(values, parallel_mode, output_gather_group)
        ids = gather_lm_head_outputs(ids, parallel_mode, output_gather_group)
    if parallel_mode is VocabParallelMode.FINE_GRAINED:
        values, ids = values[:num_rows], ids[:num_rows]
    # Global TopK over the union of each resident vocabulary shard's TopK.
    if values.shape[-1] > top_k:
        values, selected = torch.topk(values, top_k, dim=-1)
        ids = ids.gather(-1, selected)
    return values, ids


class DFlash2Qwen3MLADecoderLayer(DFlash2Qwen3DecoderLayer):
    """DFlash2 decoder layer with a DeepSeek-style MLA draft attention.

    The MLA draft checkpoint stores the A-projections as separate
    ``q_a_proj`` / ``kv_a_proj_with_mqa`` tensors and the norms/projections
    under the DeepSeek names (``kv_a_layernorm``, ``kv_b_proj`` ...), which
    match AscendKimiMLAAttention's module tree. The stacked mapper on
    DFlash2Qwen3MLAModel fuses the two A-projections when loading.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        if getattr(config, "q_lora_rank", None) is None:
            raise ValueError("DFlash2 MLA draft requires q_lora_rank in the draft config.")
        modes = [_resolve_layer_attention(config, i) for i in range(config.num_hidden_layers)]
        sliding_window, causal = modes[layer_idx]
        # One FlashMLA schedule is shared by all draft layers.
        uniform_causal = sliding_window is not None and sliding_window > 0 and causal and len(set(modes)) == 1
        with kimi_mla_draft_window(sliding_window, uniform_causal=uniform_causal):
            self.self_attn = AscendKimiMLAAttention(
                config=config,
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                use_output_gate=False,
                use_rope=True,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                non_causal_multi_token_decode=any(not layer_causal for _, layer_causal in modes),
                disable_mlapo=True,
            )
        self.self_attn.causal = causal
        self.mlp = Qwen2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self._init_convs(vllm_config, config, quant_config, prefix)


class DFlash2Qwen3MLAModel(DFlash2Qwen3Model):
    """DFlash2 draft model whose decoder layers use MLA draft attention."""

    decoder_layer_cls = DFlash2Qwen3MLADecoderLayer

    hf_to_vllm_mapper = replace(
        DFlashQwen3Model.hf_to_vllm_mapper,
        orig_to_new_stacked={
            **DFlashQwen3Model.hf_to_vllm_mapper.orig_to_new_stacked,
            ".q_a_proj": (".fused_qkv_a_proj", 0),
            ".kv_a_proj_with_mqa": (".fused_qkv_a_proj", 1),
        },
    )

    def _build_fused_kv_buffers(self) -> None:
        """The MLA writer uses the loaded per-layer projections directly."""

    def context_kv_modules(self) -> list[nn.Module]:
        modules: list[nn.Module] = [self.hidden_norm]
        for layer in self.layers:
            attn = layer.self_attn
            modules.extend((attn._attention_layer, attn.fused_qkv_a_proj, attn.kv_a_layernorm))
            # A strided cache can select PROLOG_V3 after pruning. The fused op
            # consumes both Q and KV, even when only the cache writes are used.
            if attn.impl.supports_prolog_v3_writer():
                modules.extend((attn.q_a_layernorm, attn.q_b_proj))
        if self.use_aux_hidden_state:
            modules.append(self.fc)
        return modules

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        if context_states.numel() == 0:
            return
        cos, sin = get_cos_and_sin_mla(context_positions)
        store_mla_draft_context_kv(self.layers, self.hidden_norm(context_states), cos, sin, context_slot_mapping)


class DFlash2Qwen3ForCausalLM(DFlashQwen3ForCausalLM):
    # vLLM #52816 likewise routes the draft model through this class factory.
    model_cls = DFlash2Qwen3Model

    speculator_config_class = DFlash2SpeculatorConfig

    # Share the target LM head so compute_candidates can top-k the full vocab.
    has_own_lm_head = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        draft_hf_config = speculative_config.draft_model_config.hf_config
        dflash_config = getattr(draft_hf_config, "dflash_config", None) or {}
        if dflash_config.get("attention_mode") == "mla":
            self.model_cls = DFlash2Qwen3MLAModel
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        draft_config = self.config.dflash_config
        self.output_multiplier = float(draft_config.get("output_multiplier", 1.0))
        softcap = float(draft_config.get("final_logit_softcapping") or 0.0)
        self.final_logit_softcapping = softcap if softcap > 0 else None
        self.lmhead_tp_candidate_capacity = (
            vllm_config.scheduler_config.max_num_seqs * vllm_config.speculative_config.num_speculative_tokens
        )

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        names = []
        for layer in self.model.layers:
            attn = layer.self_attn
            inner = getattr(attn, "attn", None)
            names.append(inner.layer_name if inner is not None else attn.layer_name)
        return names

    def post_process(self, vllm_config: VllmConfig) -> None:
        if get_rotation_path(vllm_config) is None:
            return
        if not hasattr(self.model, "fc"):
            raise ValueError("DFlash2 with a QuaRot target requires auxiliary hidden states.")
        align_draft_weights(self, self.model.fc, vllm_config)

    def compute_candidates(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(self.lm_head.quant_method, UnquantizedEmbeddingMethod):
            raise ValueError("DFlash2 requires an unquantized target LM head for candidate TopK.")

        selector = self.model.candidate_selector
        num_rows = hidden_states.shape[0]
        hidden_states = _maybe_gather_lm_head_hidden_states(
            hidden_states, self.lm_head, self.lmhead_tp_candidate_capacity
        )
        logits = self.lm_head.quant_method.apply(self.lm_head, hidden_states, bias=None)
        num_pad = self.lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")
        values, ids = torch.topk(logits, selector.top_k, dim=-1)
        ids = ids.to(torch.int64) + self.lm_head.shard_indices.org_vocab_start_index

        values, ids = _gather_lm_head_candidates(values, ids, self.lm_head, num_rows)

        values = values.float() * self.output_multiplier
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            values = torch.tanh(values / cap) * cap
        return ids, values
