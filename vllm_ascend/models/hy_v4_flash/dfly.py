# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from weakref import ref

import torch
from vllm_ascend.device.device_config import is_950
from torch import nn
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, ReplicatedLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.models.hy_v4.nvidia.model import HYV4DecoderLayer
from vllm.models.hy_v4.nvidia.moe import HYV4FeedForward
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_ascend.hisparse.c8 import C8_ROW_BYTES, dequantize_c8_rows

from .attention import AscendHYV4FlashAttention

logger = init_logger(__name__)

CHECKPOINT_PREFIX = "model.dfly_layers."
MODEL_PREFIX = "dfly"
FP8 = torch.float8_e4m3fn
# Bound the temporary gathered KV and FP32 value tensors for wide draft graphs.
ATTENTION_BATCH_CHUNK = 8


class DFlyAttention(AscendHYV4FlashAttention):
    def __init__(self, vllm_config, config, layer_idx, prefix):
        super().__init__(vllm_config, config, layer_idx, prefix, reuse_target=True)
        self.target_attention = None
        self.context = None
        self.post_proj_norm = (
            RMSNorm(config.hidden_size, config.rms_norm_eps) if config.dfly_attn_post_proj_norm is not False else None
        )

    def bind_target(self, source):
        # Weak ownership: registering the target as a submodule would make it
        # part of draft weight traversal and post-processing.
        self.target_attention = ref(source)

    def _shared_draft_rows(self, cache, table, block_size, selected, positions, draft):
        width = draft.shape[1]
        # build_reuse_indices repeats history and the local window across all
        # queries; only the last width indices contain causal draft positions.
        history = selected[:, 0, :-width].clamp_min(0)
        # Masked draft positions read logical row zero in the original path.
        history_and_fallback = torch.cat((history, torch.zeros_like(history[:, :1])), dim=1)
        packed = self._gather(cache, table, block_size, history_and_fallback, slice(C8_ROW_BYTES))
        history_and_fallback, draft_decoded = dequantize_c8_rows(torch.cat((packed, draft), dim=1)).split(
            (packed.shape[1], width), dim=1
        )
        history_decoded = history_and_fallback[:, None, :-1].expand(-1, width, -1, -1)
        fallback = history_and_fallback[:, None, -1:]
        use_draft = selected[..., -width:] >= positions[:, :1, None]
        tail = torch.where(use_draft[..., None], draft_decoded[:, None], fallback)
        return torch.cat((history_decoded, tail), dim=2)

    def _forward_paged(self, hidden, q_c, q_nope, q_pe, kv_c, k_pe, positions, out, *, absorbed=False):
        if self.context is None:
            return
        block_size, block_table, indices, query_positions = self.context
        source = self.target_attention()
        assert source is not None
        cache = source.kv_cache
        cache = cache.view(-1, source.cache_width)
        batch, width = query_positions.shape
        query, wv = self._absorb_query(q_nope, q_pe, absorbed=absorbed)
        query_kv = self._pack_kv(kv_c, k_pe).reshape(batch, width, -1)
        share_history = (
            source.use_pd_decode
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
                logger.info_once("HYV4 DFly reuses each request's packed C8 history across draft positions.")
                decoded = self._shared_draft_rows(
                    cache, block_table[first:last], block_size, selected, query_positions[first:last], query_kv[first:last]
                )
                self._attend_decoded(query[tokens], selected.flatten(0, 1), decoded.flatten(0, 1), wv, out[tokens])
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
            local = torch.arange(first, last, device=indices.device)[:, None, None] * width + offsets
            draft_rows = query_kv.flatten(0, 1).index_select(0, local.flatten()).reshape_as(rows)
            rows = torch.where(is_query[..., None], draft_rows, rows)
            self._attend(query[tokens], selected.flatten(0, 1), rows.flatten(0, 1), wv, out[tokens])


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

    def materialize_mlp(self, config, quant_config, index, width):
        self.mlp = HYV4FeedForward(
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
        fused, _ = self.eh_proj(torch.cat((self.enorm(anchor), self.hnorm(hidden)), dim=-1))
        return anchor + self.gate * fused


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
        self.act = SiluAndMul()

    def forward(self, hidden, previous_embed):
        combined = torch.cat((self.hidden_norm(hidden), self.embed_norm(previous_embed)), dim=-1)
        gate_up, _ = self.gate_up_proj(combined)
        correction, _ = self.down_proj(self.act(gate_up))
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
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mtp_fusion = DFlyFusion(config)
        self.hidden_correction = DFlyHiddenCorrection(config)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.logits_processor.use_all_gather = True
        transfer = vllm_config.kv_transfer_config
        self.logits_processor.use_npu_fp32_head = (
            transfer is not None and transfer.kv_role == "kv_consumer"
            and is_950()
        )
        self.ffn_width = None

    def forward(self, embeddings, positions):
        hidden, residual = embeddings, None
        for layer in self.layers:
            hidden, residual = layer(positions, hidden, residual)
        return self.norm(hidden + residual)

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
