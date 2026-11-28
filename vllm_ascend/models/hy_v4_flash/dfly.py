# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from weakref import ref

import torch
import torch_npu
from torch import nn
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.models.hy_v4.nvidia.model import HYV4DecoderLayer
from vllm.models.hy_v4.nvidia.moe import HYV4FeedForward
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_ascend.device.device_config import is_950
from vllm_ascend.hisparse.c8 import C8_ROW_BYTES
from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.quantization.methods.base import PreparedLinearInput
from vllm_ascend.quantization.methods.w8a8.fp8_block import AscendFp8BlockLinearMethod

from .attention import AscendHYV4FlashAttention
from .attention_ops import _head_bmm
from .dfly_ops import compact_draft_c8_rows, dual_rms_concat, gated_residual, rounded_add_rms_norm

logger = init_logger_ascend(__name__)

CHECKPOINT_PREFIX = "model.dfly_layers."
MODEL_PREFIX = "dfly"
FP8 = torch.float8_e4m3fn
# Bound the temporary gathered KV and FP32 value tensors for wide draft graphs.
ATTENTION_BATCH_CHUNK = 8


def draft_swiglu_quant(gate_up, scale_alg):
    return torch_npu.npu_swiglu_mx_quant(
        gate_up,
        activate_left=True,
        swiglu_mode=0,
        dst_type=torch.float8_e4m3fn,
        scale_alg=scale_alg,
    )


def draft_c8_attention(query, rows, selected, sink, scale):
    """Fuse packed C8 dequantization and attention on gathered draft candidates."""
    from cann_ops_transformer import kv_quant_sparse_flash_attention

    rows, indices, lengths, active = compact_draft_c8_rows(rows, selected)
    batch, width = rows.shape[:2]
    packed = rows.view(batch, width, 1, C8_ROW_BYTES).view(torch.float8_e4m3fn)
    result, _, _ = kv_quant_sparse_flash_attention(
        query[:, None].contiguous(),
        packed,
        packed,
        indices,
        scale,
        2,
        2,
        actual_seq_lengths_kv=lengths,
        sparse_block_size=1,
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=0,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        rope_head_dim=64,
        sinks=sink.contiguous(),
    )
    return result[:, 0].masked_fill(~active[:, None, None], 0)


def draft_add_rms_norm(hidden, residual, norm):
    """Keep the residual out of place while fusing its addition and norm."""
    if (
        not is_950()
        or hidden.device.type != "npu"
        or hidden.dtype != torch.bfloat16
        or hidden.shape != residual.shape
        or hidden.dtype != residual.dtype
        or norm.weight.dtype != hidden.dtype
    ):
        updated_residual = hidden + residual
        return norm(updated_residual), updated_residual
    return rounded_add_rms_norm(hidden, residual, norm)


class DFlyAttention(AscendHYV4FlashAttention):
    def __init__(self, vllm_config, config, layer_idx, prefix):
        super().__init__(vllm_config, config, layer_idx, prefix, reuse_target=True)
        self.fuse_swa_q_norm = is_950() and hasattr(torch.ops.npu, "npu_rms_norm_dynamic_mx_quant")
        self.target_attention = None
        self.context = None
        self._decode_index_constants = {}
        self.post_proj_norm = (
            RMSNorm(config.hidden_size, config.rms_norm_eps) if config.dfly_attn_post_proj_norm is not False else None
        )
        logger.info("DFly layer %d uses fused Q normalization and C8 attention", layer_idx)

    def bind_target(self, source):
        # Weak ownership: registering the target as a submodule would make it
        # part of draft weight traversal and post-processing.
        self.target_attention = ref(source)

    def _forward_paged(self, hidden, q_c, q_nope, q_pe, kv_c, k_pe, positions, out):
        if self.context is None:
            return
        block_size, block_table, indices, query_positions = self.context
        source = self.target_attention()
        assert source is not None
        cache = source.kv_cache
        cache = cache.view(-1, source.cache_width)
        batch, width = query_positions.shape
        query, wv = self._absorb_query(q_nope, q_pe)
        query_kv = self._pack_kv(kv_c, k_pe).reshape(batch, width, -1)
        flat_query_kv = query_kv.flatten(0, 1)
        share_history = (
            source.use_pd_decode
            and is_950()
            and query.device.type == "npu"
            and self.use_fp8_cache
            and cache.dtype == torch.uint8
            and source.kv_storage_width == query_kv.shape[-1] == C8_ROW_BYTES
            and 1 < width < indices.shape[-1]
        )
        for first in range(0, batch, ATTENTION_BATCH_CHUNK):
            last = min(first + ATTENTION_BATCH_CHUNK, batch)
            selected = indices[first:last]
            tokens = slice(first * width, last * width)
            if share_history:
                # build_reuse_indices repeats history across the query axis;
                # only the causal tail changes. Keep that history shared in KV.
                history = selected[:, 0, :-width].clamp_min(0)
                history_rows = self._gather(cache, block_table[first:last], block_size, history, slice(C8_ROW_BYTES))
                rows = torch.cat((history_rows, query_kv[first:last]), dim=1)
                context = draft_c8_attention(query[tokens], rows, selected, self.learnable_sink_param, self.scale)
                out[tokens].copy_(_head_bmm(context, wv).reshape_as(out[tokens]))
                continue
            start = query_positions[first:last, :1, None]
            is_query = selected >= start
            # Substitute temporary draft rows for causal queries after reading
            # the rest of the candidates from the target's cache.
            histories = selected.masked_fill(is_query, 0).clamp_min(0)
            rows = self._gather(
                cache, block_table[first:last], block_size, histories.flatten(1), slice(source.kv_storage_width)
            )
            rows = rows.view(*selected.shape, source.kv_storage_width)
            offsets = (selected - start).clamp(0, width - 1)
            row_key = ("draft_rows", first, last, width, indices.device)
            row_offsets = self._decode_index_constants.get(row_key)
            if row_offsets is None:
                row_offsets = torch.arange(first * width, last * width, width, device=indices.device)[:, None, None]
                self._decode_index_constants[row_key] = row_offsets
            local = row_offsets + offsets
            draft_rows = flat_query_kv.index_select(0, local.flatten()).reshape_as(rows)
            rows = torch.where(is_query[..., None], draft_rows, rows)
            if not (
                is_950() and query.device.type == "npu" and rows.dtype == torch.uint8 and rows.shape[-1] == C8_ROW_BYTES
            ):
                self._attend(query[tokens], selected.flatten(0, 1), rows.flatten(0, 1), wv, out[tokens])
                continue
            context = draft_c8_attention(
                query[tokens],
                rows.flatten(0, 1),
                selected.flatten(0, 1),
                self.learnable_sink_param,
                self.scale,
            )
            out[tokens].copy_(_head_bmm(context, wv).reshape_as(out[tokens]))


class DFlyFeedForward(HYV4FeedForward):
    """Dense draft FFN with fused activation and MXFP8 quantization."""

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        scheme = getattr(self.down_proj.quant_method, "quant_method", None)
        if not (
            is_950()
            and gate_up.device.type == "npu"
            and gate_up.dtype == torch.bfloat16
            and gate_up.ndim == 2
            and gate_up.shape[-1] % 64 == 0
            and isinstance(scheme, AscendFp8BlockLinearMethod)
            and scheme.use_mxfp8
            and scheme.mxfp8_method.group_size == 32
            and not any(getattr(self.down_proj, "mxfp8_tp_padding", (0, 0)))
        ):
            return self.down_proj(self.act_fn(gate_up))[0]
        quantized, scale = draft_swiglu_quant(gate_up, scheme.mxfp8_method.dynamic_mx_quant_scale_alg)
        return self.down_proj(PreparedLinearInput(quantized, scale, gate_up.dtype))[0]


class DFlyLayer(HYV4DecoderLayer):
    def __init__(self, vllm_config, config, index):
        nn.Module.__init__(self)
        prefix = f"{MODEL_PREFIX}.layers.{index}"
        hidden = config.hidden_size
        eps = config.rms_norm_eps
        self.enable_ihc = False
        self.input_layernorm = RMSNorm(hidden, eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.self_attn = DFlyAttention(vllm_config, config, index, f"{prefix}.self_attn")
        self.mlp = None

    def _forward_normal(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = draft_add_rms_norm(hidden_states, residual, self.input_layernorm)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = draft_add_rms_norm(hidden_states, residual, self.post_attention_layernorm)
        return self.mlp(hidden_states), residual

    def materialize_mlp(self, config, quant_config, index, width):
        self.mlp = DFlyFeedForward(
            config.hidden_size, width, config.hidden_act, quant_config, prefix=f"{MODEL_PREFIX}.layers.{index}.mlp"
        )


class DFlyFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size
        self.enorm = RMSNorm(hidden, config.rms_norm_eps)
        self.hnorm = RMSNorm(hidden, config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(2 * hidden, hidden, bias=False, prefix=f"{MODEL_PREFIX}.mtp_fusion.eh_proj")
        self.gate = nn.Parameter(torch.empty(hidden), requires_grad=False)

    def forward(self, anchor, hidden):
        fused, _ = self.eh_proj(dual_rms_concat(anchor, hidden, self.enorm, self.hnorm))
        return gated_residual(anchor, fused, self.gate)


class DFlyHiddenCorrection(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size
        width = config.hidden_correction_intermediate_size
        prefix = f"{MODEL_PREFIX}.hidden_correction"
        self.hidden_norm = RMSNorm(hidden, config.rms_norm_eps)
        self.embed_norm = RMSNorm(hidden, config.rms_norm_eps)
        self.gate_up_proj = MergedColumnParallelLinear(
            2 * hidden, [width, width], bias=False, prefix=f"{prefix}.gate_up_proj"
        )
        self.down_proj = RowParallelLinear(width, hidden, bias=False, prefix=f"{prefix}.down_proj")

    def forward(self, hidden, previous_embed):
        combined = dual_rms_concat(hidden, previous_embed, self.hidden_norm, self.embed_norm)
        gate_up, _ = self.gate_up_proj(combined)
        correction, _ = self.down_proj(torch_npu.npu_swiglu(gate_up, dim=-1))
        return hidden + correction


class FlashDFlyDraft(nn.Module):
    fall_back_to_pt_during_load = False
    allow_patterns_overrides = ["*.safetensors"]
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    weight_mapper = WeightsMapper(
        orig_to_new_stacked={".gate_proj.": (".gate_up_proj.", 0), ".up_proj.": (".gate_up_proj.", 1)}
    )

    def __init__(self, vllm_config):
        super().__init__()
        self.config = config = vllm_config.model_config.hf_text_config
        self.quant_config = vllm_config.quant_config
        self.layers = nn.ModuleList(DFlyLayer(vllm_config, config, i) for i in range(config.num_hidden_layers))
        logger.info("HYV4 DFly always uses residual AddRMSNorm for %d layers and final norm", len(self.layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mtp_fusion = DFlyFusion(config)
        self.hidden_correction = DFlyHiddenCorrection(config)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.logits_processor.use_all_gather = True
        transfer = vllm_config.kv_transfer_config
        self.logits_processor.use_npu_fp32_head = (
            transfer is not None and transfer.kv_role == "kv_consumer" and is_950()
        )
        self.ffn_width = None

    def forward(self, embeddings, positions):
        hidden, residual = embeddings, None
        for layer in self.layers:
            hidden, residual = layer(positions, hidden, residual)
        return draft_add_rms_norm(hidden, residual, self.norm)[0]

    @staticmethod
    def is_unused_checkpoint_weight(name):
        return not name.startswith(CHECKPOINT_PREFIX) or name.startswith(f"{CHECKPOINT_PREFIX}confidence_head.")

    def _materialize_mlp(self, width):
        if width < 1 or width % get_tensor_model_parallel_world_size():
            raise ValueError("DFly FFN width must be positive and divisible by TP")
        if self.config.dfly_intermediate_size not in (None, width) or self.ffn_width not in (None, width):
            raise ValueError("Inconsistent DFly FFN width")
        if self.ffn_width is None:
            with torch.device(self.norm.weight.device), set_default_torch_dtype(self.norm.weight.dtype):
                for i, layer in enumerate(self.layers):
                    layer.materialize_mlp(self.config, self.quant_config, i, width)
            self.ffn_width = width

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        draft_weights = {}
        for name, tensor in weights:
            if self.is_unused_checkpoint_weight(name):
                continue
            name = name.removeprefix(CHECKPOINT_PREFIX)
            if name.split(".", 1)[0].isdigit():
                name = f"layers.{name}"
            elif name.startswith("final_norm."):
                name = f"norm.{name.removeprefix('final_norm.')}"
            if name in draft_weights:
                raise ValueError(f"Duplicate DFly tensor {name}")
            draft_weights[name] = tensor

        norm_names = {f"layers.{i}.self_attn.post_proj_norm.weight" for i in range(len(self.layers))}
        norms = norm_names & draft_weights.keys()
        if norms and norms != norm_names:
            raise ValueError("DFly requires consistent attention post-projection norms")
        if not norms and self.config.dfly_attn_post_proj_norm is None:
            for layer in self.layers:
                layer.self_attn.post_proj_norm = None

        for i in range(len(self.layers)):
            prefix = f"layers.{i}"
            for projection in ("gate_proj", "up_proj", "down_proj"):
                name = f"{prefix}.mlp.{projection}.weight"
                weight = draft_weights[name]
                shape = weight.shape if projection != "down_proj" else weight.T.shape
                if len(shape) != 2 or shape[1] != self.config.hidden_size:
                    raise ValueError(f"Invalid DFly FFN weight shape: {name} {tuple(weight.shape)}")
                self._materialize_mlp(shape[0])

        expected = {}
        for name, parameter in self.named_parameters():
            names = [name]
            if ".gate_up_proj." in name:
                names = [name.replace(".gate_up_proj.", f".{proj}.") for proj in ("gate_proj", "up_proj")]
            for source in names:
                expected[source] = parameter.dtype
        missing = expected.keys() - draft_weights.keys()
        extra = draft_weights.keys() - expected.keys()
        if missing or extra:
            raise ValueError(f"DFly checkpoint mismatch: missing={sorted(missing)}, unexpected={sorted(extra)}")
        for name, dtype in expected.items():
            tensor = draft_weights[name]
            if (dtype == FP8) != (tensor.dtype == FP8):
                raise ValueError(f"DFly FP8 storage mismatch for {name}: {tensor.dtype} vs {dtype}")
        return AutoWeightsLoader(self).load_weights(draft_weights.items(), mapper=self.weight_mapper)
