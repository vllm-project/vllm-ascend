# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HYV4 Flash Engram with request-local n-grams and TP-sharded row storage."""

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from vllm.distributed import (
    get_dp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.hy_v4_flash.reference import decode_e4m3fn, hash_ngrams


class FlashEngramLayout:
    def __init__(self, config):
        self.layer_ids = tuple(config.engram_layer_ids or ())
        self.max_ngram_size = config.engram_max_ngram_size
        self.n_heads = config.engram_n_head_per_ngram
        self.head_dim = config.engram_n_embed_per_ngram // self.n_heads
        self.pad_token_id = config.engram_pad_id
        self.vocab_size = config.vocab_size
        self.seed = config.engram_seed
        self.lookup_table = None
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("engram_pad_id is outside the vocabulary")
        groups = config.engram_layer_head_vocab_sizes
        self.primes = tuple(groups.get(str(layer), groups.get(layer)) for layer in self.layer_ids)
        flat = [[p for order in group for p in order] for group in self.primes]
        self.offsets = np.array([np.cumsum([0, *sizes[:-1]]) for sizes in flat], dtype=np.int64)
        if any(sum(sizes) > np.iinfo(np.int32).max for sizes in flat):
            raise ValueError("Engram flattened bucket offsets must fit int32")

    def load_lookup(self, value):
        if value.dtype not in (torch.int32, torch.int64) or value.ndim != 1 or not 0 < value.numel() <= self.vocab_size:
            raise ValueError("Invalid frozen Engram token map")
        lookup = value.detach().cpu().numpy()
        if np.any(lookup < 0) or np.any(lookup > np.iinfo(np.int32).max):
            raise ValueError("Invalid frozen Engram token map")
        if self.lookup_table is not None:
            if not np.array_equal(self.lookup_table, lookup):
                raise ValueError("Engram layers must share the same token map")
            return False
        self.compressed_vocab_size = int(lookup.max()) + 1
        self.lookup_table = lookup.copy()
        self.token_map = lookup.tolist()
        self.token_map.extend(range(len(self.token_map), self.vocab_size))
        bound = max(1, (np.iinfo(np.int64).max // self.compressed_vocab_size) // 2)
        self.multipliers = np.stack(
            [
                np.random.default_rng(self.seed + 10007 * layer).integers(
                    0, bound, size=self.max_ngram_size, dtype=np.int64
                )
                * 2
                + 1
                for layer in self.layer_ids
            ]
        )

        return True


def head_range(sizes, tp, rank):
    if not 1 <= tp <= len(sizes) or not 0 <= rank < tp:
        raise ValueError("Engram requires at least one complete hash head per TP rank")
    prefix = np.cumsum([0, *sizes]).tolist()
    boundaries = [0]
    for r in range(1, tp):
        target = prefix[-1] * r / tp
        candidates = range(boundaries[-1] + 1, len(sizes) - (tp - r) + 1)
        boundaries.append(min(candidates, key=lambda i: abs(prefix[i] - target)))
    boundaries.append(len(sizes))
    return boundaries[rank], boundaries[rank + 1]


class FlashNgramHashState(nn.Module):
    def __init__(self, config, *, dummy=False):
        super().__init__()
        layout = FlashEngramLayout(config)
        self.layout = layout
        self.lookback_depth = self.layout.max_ngram_size - 1
        self.pad_id = 0
        self.visual_token_ids: tuple[int, ...] = ()
        self.register_buffer(
            "token_map",
            torch.zeros(config.vocab_size, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "multipliers",
            torch.zeros(
                len(self.layout.layer_ids),
                self.layout.max_ngram_size,
                dtype=torch.int64,
            ),
            persistent=False,
        )
        for name, values in (
            ("primes", self.layout.primes),
            ("offsets", self.layout.offsets),
        ):
            self.register_buffer(name, torch.tensor(values, dtype=torch.int64), persistent=False)
        if dummy:
            self.load_lookup(torch.arange(config.vocab_size, device="cpu"))

    def load_lookup(self, value):
        if self.layout.load_lookup(value):
            self.token_map.copy_(
                torch.as_tensor(
                    self.layout.token_map,
                    dtype=torch.int32,
                    device=self.token_map.device,
                )
            )
            self.multipliers.copy_(torch.as_tensor(self.layout.multipliers, device=self.multipliers.device))
            self.pad_id = self.layout.token_map[self.layout.pad_token_id]

    def prepare_hashes_with_output(self, input_ids, lookback, layer_name, output):
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict) or layer_name not in metadata:
            output.fill_(-1)
            return
        self.prepare_batch_hashes(input_ids, lookback, metadata[layer_name].query_start_loc, output)

    def prepare_batch_hashes(self, input_ids, lookback, query_start_loc, output):
        output.fill_(-1)
        if self.layout.lookup_table is None:
            raise ValueError("Missing checkpoint Engram token map")
        if lookback is None or lookback.shape[1] != self.lookback_depth:
            raise ValueError("Flash Engram requires per-request lookback_token_ids")
        for request, (start, stop) in enumerate(zip(query_start_loc, query_start_loc[1:])):
            if start == stop:
                continue
            output[:, start:stop].copy_(
                hash_ngrams(
                    input_ids[start:stop],
                    lookback[request],
                    self.token_map,
                    self.multipliers,
                    self.primes,
                    self.offsets,
                    self.pad_id,
                    self.layout.pad_token_id,
                    self.visual_token_ids,
                )
            )


class FlashEmbedding(nn.Module):
    def __init__(self, sizes, dim, dtype, cpu_offload, dp_shared_memory=False):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.head_start, self.head_end = head_range(sizes, self.tp_size, get_tensor_model_parallel_rank())
        self.vocab_start = sum(sizes[: self.head_start])
        self.vocab_end = sum(sizes[: self.head_end])
        self.num_embeddings = sum(sizes)
        self.n_hash_cols = len(sizes)
        self.dim = dim
        self.cpu_offload = cpu_offload
        self._shared_group = None
        self._shared_buffers = {}
        if dp_shared_memory:
            if not cpu_offload:
                raise ValueError("Flash Engram dp_shared_memory requires CPU offload")
            group = get_dp_group()
            if group.world_size <= 1:
                raise ValueError("Flash Engram dp_shared_memory requires DP > 1")
            if not all(in_the_same_node_as(group.cpu_group)):
                raise ValueError("Flash Engram shared DP peers must be on the same node")
            self._shared_group = group
        rows = self.vocab_end - self.vocab_start
        self.embedding = nn.Module()
        self.embedding.weight = nn.Parameter(
            self._allocate_storage(
                (0,) if dtype is None else (rows, dim),
                torch.uint8 if dtype == torch.float8_e4m3fn else torch.bfloat16,
                "weight",
            ),
            requires_grad=False,
        )
        self.fp8 = dtype == torch.float8_e4m3fn
        self.embedding.register_parameter("weight_scale_inv", None)
        if self.fp8:
            self.embedding.weight_scale_inv = nn.Parameter(
                self._allocate_storage((rows,), torch.bfloat16, "weight_scale_inv"), requires_grad=False
            )
        for parameter in self.embedding.parameters():
            set_weight_attrs(parameter, {"weight_loader": self._load_shard})

    def _allocate_storage(self, shape, dtype, leaf):
        if self._shared_group is not None and shape[0]:
            from vllm_ascend.models.deepseek_v41.engram.npu import SharedUvaBuffer

            buffer = SharedUvaBuffer(shape, dtype, torch.device("npu", torch.npu.current_device()), self._shared_group)
            self._shared_buffers[leaf] = buffer
            logger.info(
                "HYV4 Flash Engram DP shared table: dp_rank=%d tp_rank=%d field=%s rows=%d bytes=%d backing=%s",
                self._shared_group.rank_in_group,
                get_tensor_model_parallel_rank(),
                leaf,
                shape[0],
                buffer.tensor.numel() * buffer.tensor.element_size(),
                buffer.shm.name,
            )
            return buffer.tensor
        parameter = getattr(self.embedding, leaf, None)
        storage = {"device": parameter.device} if parameter is not None else {}
        if self.cpu_offload:
            storage = {"device": "cpu", "pin_memory": True}
        return torch.empty(shape, dtype=dtype, **storage)

    def _copy_shared_weight(self, parameter, shard):
        group = self._shared_group
        if group is None:
            default_weight_loader(parameter, shard)
            return
        error = None
        if group.rank_in_group == 0:
            try:
                default_weight_loader(parameter, shard)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        errors = [None] * group.world_size
        # Publish the one writer's completed copy (or error) before either DP
        # replica can look up its own tokens from the shared CPU table.
        dist.all_gather_object(errors, error, group=group.cpu_group)
        if any(errors):
            raise RuntimeError(f"Flash Engram shared table loading failed: {errors}")

    def close_host_offload(self):
        for leaf, buffer in tuple(self._shared_buffers.items()):
            buffer.close()
            parameter = getattr(self.embedding, leaf)
            parameter.data = torch.empty(0, dtype=parameter.dtype, device="cpu")
            del self._shared_buffers[leaf]

    def load_weight(self, leaf, value):
        if leaf == "weight_scale_inv" and self.embedding.weight_scale_inv is None:
            self.embedding.weight_scale_inv = nn.Parameter(
                self.embedding.weight.new_empty(0, dtype=torch.bfloat16),
                requires_grad=False,
            )
            set_weight_attrs(self.embedding.weight_scale_inv, {"weight_loader": self._load_shard})
        self._load_shard(getattr(self.embedding, leaf), value)

    def _load_shard(self, parameter, value):
        is_weight = parameter is self.embedding.weight
        expected = (self.num_embeddings, self.dim) if is_weight else (self.num_embeddings,)
        dtypes = (torch.bfloat16, torch.float8_e4m3fn) if is_weight else (torch.bfloat16,)
        if tuple(value.shape) != expected or value.dtype not in dtypes:
            raise ValueError(f"Invalid Engram table: {tuple(value.shape)}/{value.dtype}")
        storage_dtype = torch.uint8 if value.dtype == torch.float8_e4m3fn else value.dtype
        if parameter.numel() and parameter.dtype != storage_dtype:
            raise ValueError("Engram table dtype changed during weight reload")
        if is_weight:
            if value.dtype != torch.float8_e4m3fn and self.embedding.weight_scale_inv is not None:
                raise ValueError("BF16 Engram table must not have FP8 row scales")
        elif self.embedding.weight.numel() and not self.fp8:
            raise ValueError("BF16 Engram table must not have FP8 row scales")
        if parameter.numel() == 0:
            parameter.data = self._allocate_storage(
                (self.vocab_end - self.vocab_start, *expected[1:]),
                storage_dtype,
                "weight" if is_weight else "weight_scale_inv",
            )
        shard = value[self.vocab_start : self.vocab_end]
        if storage_dtype == torch.uint8:
            shard = shard.view(torch.uint8)
        self._copy_shared_weight(parameter, shard)
        if is_weight:
            self.fp8 = value.dtype == torch.float8_e4m3fn
        if is_weight and self.fp8 and self.embedding.weight_scale_inv is None:
            self.embedding.weight_scale_inv = nn.Parameter(
                parameter.new_empty(0, dtype=torch.bfloat16), requires_grad=False
            )
            set_weight_attrs(self.embedding.weight_scale_inv, {"weight_loader": self._load_shard})

    def forward(self, hashes):
        indices = hashes[:, self.head_start : self.head_end].long() - self.vocab_start
        local = indices.clamp(0, self.vocab_end - self.vocab_start - 1)
        weight = self.embedding.weight
        flat_local = local.flatten().to(weight.device)
        rows = weight.index_select(0, flat_local).view(*indices.shape, self.dim).to(hashes.device)
        if self.fp8:
            scale = self.embedding.weight_scale_inv
            scales = scale.index_select(0, flat_local).view_as(indices).to(hashes.device)
            rows = decode_e4m3fn(rows) * scales.float().unsqueeze(-1)
        return rows.float()


class FlashEngram(nn.Module):
    def __init__(self, vllm_config, layout, layer_id, prefix):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        index = layout.layer_ids.index(layer_id)
        sizes = [p for order in layout.primes[index] for p in order]
        storage = vllm_config.engram_config
        if storage.disk_offload or storage.embedding_across_dp:
            raise ValueError("Flash Engram does not support DP head sharding or disk offload")
        dtype = None
        if vllm_config.load_config.load_format == "dummy":
            if storage.dp_shared_memory:
                raise ValueError("Flash Engram DP shared memory requires checkpoint loading")
            dtype = torch.float8_e4m3fn if config.engram_fp8_embedding else torch.bfloat16
        self.multi_head_embedding = FlashEmbedding(
            sizes, layout.head_dim, dtype, storage.cpu_offload, storage.dp_shared_memory
        )
        self.hidden_size, self.hc_mult = config.hidden_size, config.hc_mult
        local = self.multi_head_embedding.head_end - self.multi_head_embedding.head_start
        self.lookup_per_head_norms = nn.ModuleList([nn.RMSNorm(layout.head_dim) for _ in range(local)])
        self.value_proj = nn.Linear(local * layout.head_dim, self.hidden_size)
        self.key_projs = nn.ModuleList(
            [nn.Linear(local * layout.head_dim, self.hidden_size) for _ in range(self.hc_mult)]
        )
        self.norm_key = nn.ModuleList([nn.RMSNorm(self.hidden_size) for _ in range(self.hc_mult)])
        self.norm_query = nn.ModuleList([nn.RMSNorm(self.hidden_size) for _ in range(self.hc_mult)])
        self.unified_pre_gamma = nn.Parameter(torch.empty(self.hidden_size))
        self.unified_per_hc_scales = nn.ParameterList(
            [nn.Parameter(torch.empty(self.hidden_size)) for _ in range(self.hc_mult)]
        )
        self.total_width = len(sizes) * layout.head_dim
        self.start = self.multi_head_embedding.head_start * layout.head_dim
        self.stop = self.multi_head_embedding.head_end * layout.head_dim
        self.rank = get_tensor_model_parallel_rank()
        for projection in [self.value_proj, *self.key_projs]:
            set_weight_attrs(projection.weight, {"weight_loader": self._load_projection})
            set_weight_attrs(projection.bias, {"weight_loader": self._load_bias})

    def _load_projection(self, parameter, value):
        if value.shape != (self.hidden_size, self.total_width):
            raise ValueError("Flash Engram projection weight shape mismatch")
        default_weight_loader(parameter, value[:, self.start : self.stop])

    def _load_bias(self, parameter, value):
        default_weight_loader(parameter, value if self.rank == 0 else torch.zeros_like(value))

    def forward(self, hidden, hashes, embedding_rows=None):
        rows = self.multi_head_embedding(hashes) if embedding_rows is None else embedding_rows
        eps = torch.finfo(torch.float32).eps
        gamma = torch.stack([norm.weight for norm in self.lookup_per_head_norms]).float()
        local = self.multi_head_embedding
        local_hashes = hashes[:, local.head_start : local.head_end]
        x = rows.masked_fill(local_hashes.unsqueeze(-1) < 0, 0)
        x = (x * (x.square().mean(-1, keepdim=True) + eps).rsqrt() * gamma).to(hidden.dtype)
        x = x.flatten(1)
        gates = []
        for channel in range(self.hc_mult):
            key = self.key_projs[channel](x)
            if local.tp_size > 1:
                key = tensor_model_parallel_all_reduce(key)
            gate = (self.norm_key[channel](key) * self.norm_query[channel](hidden[:, channel])).sum(-1)
            gate = gate * self.hidden_size**-0.5
            gates.append((gate.abs().clamp_min(1e-6).sqrt() * gate.sign()).sigmoid())
        gate = torch.stack(gates, dim=1)
        value = self.value_proj(x)
        if local.tp_size > 1:
            value = tensor_model_parallel_all_reduce(value)
        value = value.float()
        signal = self.unified_pre_gamma.float() * value * (value.square().mean(-1, keepdim=True) + eps).rsqrt()
        scales = torch.stack(list(self.unified_per_hc_scales)).float()
        result = ((gate.float().unsqueeze(-1) + scales) * signal.unsqueeze(1)).to(hidden.dtype)
        return hidden + result.masked_fill((hashes < 0).all(-1)[:, None, None], 0)
