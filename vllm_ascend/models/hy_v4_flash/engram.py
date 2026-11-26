# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HYV4 Flash Engram with request-local n-grams and TP-sharded row storage."""

import numpy as np
import torch
import torch.distributed as dist
import torch_npu
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
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.device.device_config import is_950

from .pd_role import uses_prefill_ops

_CANN_ENGRAM_MAX_TOKENS = 512
_CANN_RMS_ALIGNMENT = 32


def _engram_inverse_rms(value, weight, epsilon, use_cann):
    if (
        use_cann
        and value.device.type == "npu"
        and value.dtype == weight.dtype == torch.float32
        and value.ndim in (2, 3)
        and 0 < value.shape[0] <= _CANN_ENGRAM_MAX_TOKENS
        and value.shape[-1] % _CANN_RMS_ALIGNMENT == 0
        and value.is_contiguous()
        and weight.shape == (value.shape[-1],)
        and weight.device == value.device
        and weight.is_contiguous()
        and not torch.is_autocast_enabled("npu")
    ):
        # gamma does not affect rstd. Keep the original multiply order below.
        return torch_npu.npu_rms_norm(value, weight, epsilon)[1]
    return (value.square().mean(-1, keepdim=True) + epsilon).rsqrt()


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
    use_fused_prefill = False

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
        batch = metadata[layer_name]
        starts = batch.query_start_loc
        starts_device = batch.query_start_loc_device
        if (
            self.use_fused_prefill
            and HAS_TRITON
            and input_ids.device.type == "npu"
            and starts_device is not None
            and starts_device.device == input_ids.device
            and self.layout.lookup_table is not None
            and lookback is not None
            and batch.prefill_reqs
            and all(batch.prefill_reqs)
            and 128 <= starts[-1] <= input_ids.numel()
            and lookback.shape == (len(starts) - 1, self.lookback_depth)
            and self.lookback_depth == 2
            and self.primes.shape[-1] == 8
            and all(t.is_contiguous() for t in (input_ids, lookback, output))
        ):
            from vllm_ascend.ops.triton.engram_hash import flash_prefill_hash

            flash_prefill_hash(
                input_ids,
                lookback,
                starts_device,
                starts[-1],
                self.token_map,
                self.multipliers,
                self.primes,
                self.offsets,
                output,
                self.pad_id,
                self.layout.pad_token_id,
                self.visual_token_ids,
                query_starts=starts,
            )
            logger.info_once("HYV4 Prefill uses the fused Engram hash")
            return
        self.prepare_batch_hashes(input_ids, lookback, starts, output)

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
    def __init__(self, sizes, dim, dtype, cpu_offload, dp_shared_memory=False, *, map_host_weight=False):
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
        self._host_buffers = {}
        self.map_host_weight = map_host_weight
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
            self._host_buffers[leaf] = buffer
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
        if self.map_host_weight and shape[0]:
            from vllm_ascend.models.deepseek_v41.engram.npu import HostUvaBuffer

            try:
                buffer = HostUvaBuffer(shape, dtype, torch.device("npu", torch.npu.current_device()))
            except RuntimeError as exc:
                # HostUvaBuffer cleans up before re-raising registration errors.
                # Never hide its cleanup-error subclass or unrelated failures.
                if type(exc) is not RuntimeError or not str(exc).startswith("aclrtHostRegisterV2 failed: rc=507899 "):
                    raise
                self.map_host_weight = False
                logger.warning("HYV4 Engram host registration unsupported; retaining CPU gather: %s", exc)
            else:
                self._host_buffers[leaf] = buffer
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
        for leaf, buffer in tuple(self._host_buffers.items()):
            buffer.close()
            parameter = getattr(self.embedding, leaf)
            parameter.data = torch.empty(0, dtype=parameter.dtype, device="cpu")
            del self._host_buffers[leaf]

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

    def prefetch_rows(self, hashes):
        from .engram_prefetch import gather_mapped_rows

        return gather_mapped_rows(
            hashes,
            self._host_buffers["weight"],
            self._host_buffers.get("weight_scale_inv") if self.fp8 else None,
            self.head_start,
            self.head_end,
            self.vocab_start,
            self.vocab_end,
        )

    def gather_rows(self, hashes):
        indices = hashes[:, self.head_start : self.head_end].long() - self.vocab_start
        local = indices.clamp(0, self.vocab_end - self.vocab_start - 1)
        weight = self.embedding.weight
        flat_local = local.flatten().to(weight.device)
        return weight.index_select(0, flat_local).view(*indices.shape, self.dim), flat_local

    def forward(self, hashes):
        rows, flat_local = self.gather_rows(hashes)
        rows = rows.to(hashes.device)
        if self.fp8:
            scale = self.embedding.weight_scale_inv
            scales = scale.index_select(0, flat_local).view(rows.shape[:-1]).to(hashes.device)
            rows = decode_e4m3fn(rows) * scales.float().unsqueeze(-1)
        return rows.float()


class FlashEngram(nn.Module):
    def __init__(self, vllm_config, layout, layer_id, prefix):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        use_cann = vllm_config.additional_config.get("hy4flash_cann_engram_rms", False)
        if not isinstance(use_cann, bool):
            raise ValueError("hy4flash_cann_engram_rms must be a boolean")
        transfer = vllm_config.kv_transfer_config
        self.use_cann_inverse_rms = (
            use_cann
            and is_950()
            and hasattr(torch_npu, "npu_rms_norm")
            and transfer is not None
            and transfer.kv_role == "kv_consumer"
        )
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
            sizes,
            layout.head_dim,
            dtype,
            storage.cpu_offload,
            storage.dp_shared_memory,
            map_host_weight=storage.cpu_offload and HAS_TRITON and is_950() and uses_prefill_ops(vllm_config),
        )
        self.hidden_size, self.hc_mult = config.hidden_size, config.hc_mult
        self._use_small_aiv = HAS_TRITON and is_950() and uses_prefill_ops(vllm_config)
        self.register_buffer("_lookup_gamma", None, persistent=False)
        self.register_buffer("_key_gamma", None, persistent=False)
        self.register_buffer("_query_gamma", None, persistent=False)
        self.register_buffer("_packed_kv_weight", None, persistent=False)
        self.register_buffer("_packed_kv_bias", None, persistent=False)
        self.register_buffer("_pre_gamma", None, persistent=False)
        self.register_buffer("_residual_scales", None, persistent=False)
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

    @torch.no_grad()
    def pack_prefill_weights(self):
        if self._use_small_aiv:
            self._lookup_gamma = torch.stack([norm.weight for norm in self.lookup_per_head_norms]).float()
            self._pre_gamma = self.unified_pre_gamma.float()
            self._residual_scales = torch.stack(list(self.unified_per_hc_scales)).float()

            if all(
                norm.eps is None and norm.weight.dtype == torch.bfloat16
                for norm in (*self.norm_key, *self.norm_query)
            ):
                self._key_gamma = torch.stack([norm.weight for norm in self.norm_key])
                self._query_gamma = torch.stack([norm.weight for norm in self.norm_query])

            if self.multi_head_embedding.tp_size == 4 and self.hc_mult == 4 and self.hidden_size == 3072:
                projections = (*self.key_projs, self.value_proj)
                if all(p.weight.shape[1] == 384 and p.weight.dtype == torch.bfloat16 for p in projections):
                    self._packed_kv_weight = torch.cat([p.weight for p in projections])
                    self._packed_kv_bias = torch.cat([p.bias for p in projections])

    def _project(self, projection, x):
        result = projection(x)
        if self.multi_head_embedding.tp_size > 1:
            result = tensor_model_parallel_all_reduce(result)
        return result

    def forward(self, hidden, hashes, embedding_rows=None):
        rows = self.multi_head_embedding(hashes) if embedding_rows is None else embedding_rows
        eps = torch.finfo(torch.float32).eps
        use_fused = (
            self._use_small_aiv
            and hidden.device.type == "npu"
            and hidden.dtype == torch.bfloat16
            and hidden.is_contiguous()
            and hidden.shape[0] > 0
            and self.hc_mult == 4
            and self.hidden_size == 3072
            and self._lookup_gamma is not None
            and self._lookup_gamma.device == hidden.device
        )
        gamma = (
            self._lookup_gamma
            if use_fused
            else torch.stack([norm.weight for norm in self.lookup_per_head_norms]).float()
        )
        local = self.multi_head_embedding
        local_hashes = hashes[:, local.head_start : local.head_end]
        # Mapped prefetch writes zeros for negative hashes while it gathers.
        x = rows if embedding_rows is not None else rows.masked_fill(local_hashes.unsqueeze(-1) < 0, 0)
        use_lookup_kernel = (
            use_fused
            and x.shape[0] >= 512
            and x.dtype == gamma.dtype == torch.float32
            and x.is_contiguous()
            and gamma.is_contiguous()
        )
        fuse_inverse = (
            use_lookup_kernel
            and x.shape[1:] == (4, 96)
            and gamma.shape == x.shape[1:]
            and not self.use_cann_inverse_rms
        )
        inverse = None if fuse_inverse else _engram_inverse_rms(x, gamma[0], eps, self.use_cann_inverse_rms)
        if use_lookup_kernel and (inverse is None or (inverse.dtype == torch.float32 and inverse.is_contiguous())):
            from .prefill_ops.engram_norm import apply_lookup_norm

            x = apply_lookup_norm(x, gamma, hidden.dtype, eps, inverse=inverse)
        else:
            x = (x * inverse * gamma).to(hidden.dtype)
        x = x.flatten(1)
        # Cap the batched workspace based on the current token count.
        use_batched_norm = (
            use_fused and 512 <= hidden.shape[0] <= 2048 and self._key_gamma is not None
            and x.dtype is torch.bfloat16
        )
        gate_scale = self.hidden_size**-0.5
        # This TP4 message range keeps the BF16 collective's tested reduction order.
        packed_projection = (
            use_fused and self._packed_kv_weight is not None
            and x.dtype is torch.bfloat16 and x.is_contiguous()
            and 64 <= x.shape[0] <= 1024
        )
        if packed_projection:
            projected = nn.functional.linear(x, self._packed_kv_weight, self._packed_kv_bias)
            projected = tensor_model_parallel_all_reduce(projected)
            packed = projected.unflatten(-1, (self.hc_mult + 1, self.hidden_size))
            *packed_keys, value = packed.transpose(0, 1).contiguous().unbind(0)

        # Match CANN reductions for the two full Prefill scheduling buckets.
        if use_fused and x.shape[0] in (2048, 4096) and self._key_gamma is not None:
            from .prefill_ops.engram_gate import gate_dot, gate_value_inverse
            from .prefill_ops.small_aiv import engram_gate_residual

            dots = tuple(
                gate_dot(
                    packed_keys[channel] if packed_projection else self._project(projection, x),
                    hidden, self._key_gamma[channel], self._query_gamma[channel], channel, eps,
                )
                for channel, projection in enumerate(self.key_projs)
            )
            if not packed_projection:
                value = self._project(self.value_proj, x)
            inverse = gate_value_inverse(value, eps)
            return engram_gate_residual(
                hidden, hashes, dots, self._residual_scales, value, self._pre_gamma, inverse, gate_scale,
            )

        if use_batched_norm:
            from .prefill_ops.engram_norm import engram_batched_dots

            keys = tuple(packed_keys) if packed_projection else tuple(
                self._project(projection, x) for projection in self.key_projs
            )
            dots = engram_batched_dots(keys, hidden, self._key_gamma, self._query_gamma)
        else:
            dots = []
            for channel, projection in enumerate(self.key_projs):
                key = packed_keys[channel] if packed_projection else self._project(projection, x)
                query = hidden[:, channel]
                key_norm = self.norm_key[channel]
                query_norm = self.norm_query[channel]
                if (
                    use_fused
                    and hidden.shape[0] >= 512
                    and key.dtype is torch.bfloat16
                    and key.is_contiguous()
                    and key_norm.eps is None
                    and query_norm.eps is None
                    and key_norm.weight.dtype == query_norm.weight.dtype == torch.bfloat16
                    and key_norm.weight.is_contiguous()
                    and query_norm.weight.is_contiguous()
                ):
                    from .prefill_ops.engram_norm import engram_normalized_dot

                    dot = engram_normalized_dot(key, query, key_norm.weight, query_norm.weight)
                else:
                    dot = (key_norm(key) * query_norm(query)).sum(-1)
                dots.append(dot)

        if not packed_projection:
            value = self._project(self.value_proj, x)
        simple_dots = use_fused and all(dot.dtype is torch.bfloat16 and dot.is_contiguous() for dot in dots)
        contiguous_value_norm = (
            use_fused
            and value.dtype == torch.bfloat16
            and value.is_contiguous()
            and 512 <= value.shape[0] <= 4096
        )
        if contiguous_value_norm and simple_dots:
            from .prefill_ops.engram_norm import engram_value_inverse
            from .prefill_ops.small_aiv import engram_gate_residual

            inverse = engram_value_inverse(value, eps)
            return engram_gate_residual(
                hidden, hashes, dots, self._residual_scales, value, self._pre_gamma, inverse, gate_scale,
            )

        if use_fused:
            from .prefill_ops.small_aiv import engram_gate_epilogue, engram_gates, engram_residual

            if simple_dots:
                gate = engram_gates(dots, gate_scale)
            else:
                gate = hidden.new_empty((hidden.shape[0], self.hc_mult))
                for channel, dot in enumerate(dots):
                    engram_gate_epilogue(dot, gate, channel, gate_scale)
        else:
            scaled = [dot * gate_scale for dot in dots]
            gate = torch.stack([(g.abs().clamp_min(1e-6).sqrt() * g.sign()).sigmoid() for g in scaled], dim=1)
        if contiguous_value_norm:
            from .prefill_ops.engram_norm import engram_value_inverse

            inverse = engram_value_inverse(value, eps)
            return engram_residual(
                hidden, hashes, gate, self._residual_scales, value,
                gamma=self._pre_gamma, inverse=inverse,
            )
        value = value.float()
        if use_fused:
            signal = self._pre_gamma * value * (value.square().mean(-1, keepdim=True) + eps).rsqrt()
            return engram_residual(hidden, hashes, gate, self._residual_scales, signal)
        gamma = self.unified_pre_gamma.float()
        inverse = _engram_inverse_rms(value, gamma, eps, self.use_cann_inverse_rms)
        signal = gamma * value * inverse
        scales = torch.stack(list(self.unified_per_hc_scales)).float()
        result = ((gate.float().unsqueeze(-1) + scales) * signal.unsqueeze(1)).to(hidden.dtype)
        return hidden + result.masked_fill((hashes < 0).all(-1)[:, None, None], 0)
