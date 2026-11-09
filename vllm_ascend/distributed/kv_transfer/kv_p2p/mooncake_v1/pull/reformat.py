# SPDX-License-Identifier: Apache-2.0
"""Pull-side KV cache reformat subsystem.

These helpers used to live inside the pull connector's KV cache receiving
thread. They are moved verbatim into a mixin so the receiving thread can inherit
them while the module that owns the thread stays focused on transfer execution.
The mixin only reads and writes the host thread state through self; it never
imports the entry module (that would create an import cycle).
"""

import sys
from collections.abc import Callable
from typing import Any

import torch
import torch_npu
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import enable_custom_op

# The pull entry module used to import ``is_conv_state_dim_first`` into its own
# namespace, and existing mamba-transfer tests patch that module attribute.
# Resolve the helper through the entry module at call time so those patches keep
# intercepting the call after the move; fall back to the real helper when this
# module is imported standalone.
_ENTRY_MODULE = "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector"


def is_conv_state_dim_first() -> bool:
    entry = sys.modules.get(_ENTRY_MODULE)
    fn: Callable[[], bool] | None = getattr(entry, "is_conv_state_dim_first", None) if entry is not None else None
    if fn is None:
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first as real_fn

        fn = real_fn
    return fn()


# Late-bound by the pull entry module once MooncakeConnectorWorker exists.
# The moved method bodies keep the original qualifier verbatim, and importing
# the entry module here would create a cycle.
MooncakeConnectorWorker: Any = None


class KvCacheReformatMixin:
    def _stash_pending_reformat(
        self,
        request_id: str,
        shard_idx: int,
        ready_attention_group_reformat_block_ids: list[tuple[int, list[list[int]], int, list[int]]],
    ) -> None:
        with self.pending_reformat_lock:
            self.pending_reformat[request_id][shard_idx] = ready_attention_group_reformat_block_ids

    def _reformat_pending_kv_caches(self, request_id: str) -> None:
        with self.pending_reformat_lock:
            shard_reformats = self.pending_reformat.pop(request_id, {})
        for shard_idx in sorted(shard_reformats):
            logger.debug(
                "Reformatting KV cache after all pulls completed. request_id=%s shard_idx=%s",
                request_id,
                shard_idx,
            )
            self._apply_kv_cache_reformat(shard_reformats[shard_idx])

    def _apply_kv_cache_reformat(
        self,
        ready_attention_group_reformat_block_ids: list[tuple[int, list[list[int]], int, list[int]]],
    ) -> None:
        if not ready_attention_group_reformat_block_ids:
            return

        gqa_reformat_groups = [
            (group_idx, grouped_local_block_ids, num_group_pulls, layer_indices)
            for (
                group_idx,
                grouped_local_block_ids,
                num_group_pulls,
                layer_indices,
            ) in ready_attention_group_reformat_block_ids
            if num_group_pulls > 1
            and not MooncakeConnectorWorker._group_skip_kv_reformat(self.kv_group2layeridx[group_idx][0])
        ]

        if self.is_hma_required:
            for group_idx, grouped_local_block_ids, num_group_pulls, layer_indices in gqa_reformat_groups:
                num_reformat_blocks = sum(len(block_ids) for block_ids in grouped_local_block_ids)
                logger.debug(
                    "Reformat hybrid linear KV cache for GQA attention group. "
                    "group_idx=%s, num_group_pulls=%s, num_block_groups=%s, num_reformat_blocks=%s, "
                    "layer_indices=%s",
                    group_idx,
                    num_group_pulls,
                    len(grouped_local_block_ids),
                    num_reformat_blocks,
                    layer_indices,
                )
                group_kv_caches = self._get_group_kv_caches(group_idx, layer_indices)
                if not group_kv_caches:
                    continue
                self.reformat_kv_cache_hybrid_linear_torch(grouped_local_block_ids, num_group_pulls, group_kv_caches)
            return

        uniform_num_pulls = {num_group_pulls for _, _, num_group_pulls, _ in ready_attention_group_reformat_block_ids}
        if len(uniform_num_pulls) != 1:
            raise RuntimeError(
                f"Non-hybrid Mooncake KV reformat expects uniform group pulls, but got {uniform_num_pulls}."
            )

        num_group_pulls = next(iter(uniform_num_pulls))
        need_cat_cache = num_group_pulls > 1
        need_nz_cache = get_ascend_config().enable_kv_nz
        if not (need_cat_cache or need_nz_cache):
            return

        use_fused_op = get_ascend_config().enable_transpose_kv_cache_by_block
        for group_idx, reformat_block_ids, _, layer_indices in ready_attention_group_reformat_block_ids:
            group_spec = self.kv_group2layeridx[group_idx][0]
            if MooncakeConnectorWorker._group_skip_kv_reformat(group_spec):
                continue
            group_kv_caches = self._get_group_kv_caches(group_idx, layer_indices)
            if not group_kv_caches:
                continue
            if use_fused_op and enable_custom_op():
                if need_cat_cache:
                    self.reformat_kv_cache_with_fused_op(reformat_block_ids, num_group_pulls, group_kv_caches)
                if need_nz_cache:
                    self.reformat_kv_cache(reformat_block_ids, num_group_pulls, False, need_nz_cache, group_kv_caches)
            else:
                self.reformat_kv_cache(
                    reformat_block_ids,
                    num_group_pulls,
                    need_cat_cache,
                    need_nz_cache,
                    group_kv_caches,
                )

    @torch.no_grad()
    def reformat_kv_cache_hybrid_linear_torch(
        self, block_ids: list[list[int]], tp_num_need_pulls: int, group_kv_caches
    ):
        flat_block_ids = [item for sublist in block_ids for item in sublist]
        if not flat_block_ids or tp_num_need_pulls == 1:
            return
        device = list(self.kv_caches.values())[0][0].device
        block_ids_tensor = torch.tensor(flat_block_ids, dtype=torch.long, device=device)
        num_blocks = block_ids_tensor.numel()

        def _transpose_cache_by_block(cache: torch.Tensor):
            # The transferred cache is laid out as
            # [block, split, token, head_per_split, dim]. Restore it to
            # [block, token, split, head_per_split, dim] in the selected blocks.
            selected = cache.index_select(0, block_ids_tensor)
            block_size = cache.shape[1]
            transposed = (
                selected.reshape(num_blocks, tp_num_need_pulls, block_size, -1)
                .transpose(1, 2)
                .contiguous()
                .reshape_as(selected)
            )
            cache.index_copy_(0, block_ids_tensor, transposed)

        for _, (k_cache_layer, v_cache_layer) in group_kv_caches.items():
            _transpose_cache_by_block(k_cache_layer)
            _transpose_cache_by_block(v_cache_layer)

    def _append_mamba_transfer_meta(
        self,
        src_list: list[int],
        dst_list: list[int],
        length_list: list[int],
        group_spec: dict[str, Any],
        src_layer_base_addr: list[int],
        dst_layer_base_addr: list[int],
        block_len: list[int],
        block_stride: list[int],
        remote_block_stride: list[int],
        remote_block_id: int,
        local_block_id: int,
        tp_num_need_pulls: int,
        remote_tp_offset: int,
    ) -> None:
        remote_tp_size = self.tp_size * tp_num_need_pulls
        assert remote_tp_size >= self.tp_size, "Mamba prefill TP size must be >= decode TP size."
        assert remote_tp_size % self.tp_size == 0, "Mamba prefill TP size must be divisible by decode TP size."

        remote_conv_addr, remote_ssm_addr = dst_layer_base_addr[:2]
        local_conv_addr, local_ssm_addr = src_layer_base_addr[:2]
        local_conv_len, local_ssm_len = block_len[:2]
        local_conv_stride, local_ssm_stride = block_stride[:2]
        remote_conv_stride, remote_ssm_stride = remote_block_stride[:2]

        tp_ratio = tp_num_need_pulls
        remote_conv_len = local_conv_len // tp_ratio
        remote_ssm_len = local_ssm_len // tp_ratio

        if tp_ratio == 1:
            src_list.extend(
                [
                    local_conv_addr + local_block_id * local_conv_stride,
                    local_ssm_addr + local_block_id * local_ssm_stride,
                ]
            )
            dst_list.extend(
                [
                    remote_conv_addr + remote_block_id * remote_conv_stride,
                    remote_ssm_addr + remote_block_id * remote_ssm_stride,
                ]
            )
            length_list.extend([remote_conv_len, remote_ssm_len])
            return

        conv_shape = group_spec["shapes"][0]
        conv_dtype_size = group_spec["dtype_sizes"][0]

        hf_text_config = self.vllm_config.model_config.hf_text_config
        linear_attn_config = getattr(hf_text_config, "linear_attn_config", None)
        if linear_attn_config is not None:
            projection_width = linear_attn_config["num_heads"] * linear_attn_config["head_dim"]
            remote_conv_sizes = [projection_width // remote_tp_size] * 3
        else:
            remote_num_key_heads = hf_text_config.linear_num_key_heads // remote_tp_size
            remote_num_value_heads = hf_text_config.linear_num_value_heads // remote_tp_size
            remote_conv_sizes = [
                remote_num_key_heads * hf_text_config.linear_key_head_dim,
                remote_num_key_heads * hf_text_config.linear_key_head_dim,
                remote_num_value_heads * hf_text_config.linear_value_head_dim,
            ]

        remote_conv_width = sum(remote_conv_sizes)
        remote_conv_offsets = [0, remote_conv_sizes[0], sum(remote_conv_sizes[:2])]
        if is_conv_state_dim_first():
            state_len = conv_shape[1]
            for remote_conv_offset, remote_conv_size in zip(remote_conv_offsets, remote_conv_sizes):
                local_addr_offset = (
                    (remote_conv_offset * tp_ratio + remote_tp_offset * remote_conv_size) * state_len * conv_dtype_size
                )
                remote_addr_offset = remote_conv_offset * state_len * conv_dtype_size
                src_list.append(local_conv_addr + local_block_id * local_conv_stride + local_addr_offset)
                dst_list.append(remote_conv_addr + remote_block_id * remote_conv_stride + remote_addr_offset)
                length_list.append(remote_conv_size * state_len * conv_dtype_size)
        else:
            state_len = conv_shape[0]
            for state_idx in range(state_len):
                for remote_conv_offset, remote_conv_size in zip(remote_conv_offsets, remote_conv_sizes):
                    remote_addr_offset = (state_idx * remote_conv_width + remote_conv_offset) * conv_dtype_size
                    local_addr_offset = (
                        (state_idx * remote_conv_width + remote_conv_offset) * tp_ratio
                        + remote_tp_offset * remote_conv_size
                    ) * conv_dtype_size
                    src_list.append(local_conv_addr + local_block_id * local_conv_stride + local_addr_offset)
                    dst_list.append(remote_conv_addr + remote_block_id * remote_conv_stride + remote_addr_offset)
                    length_list.append(remote_conv_size * conv_dtype_size)

        src_list.append(
            local_ssm_addr + local_block_id * local_ssm_stride + remote_tp_offset * local_ssm_len // tp_num_need_pulls
        )
        dst_list.append(remote_ssm_addr + remote_block_id * remote_ssm_stride)
        length_list.append(remote_ssm_len)

    def _get_group_kv_caches(self, group_idx: int, layer_indices: list[int] | None = None) -> dict[str, Any]:
        if layer_indices is None:
            _, layer_indices = self.kv_group2layeridx[group_idx]
        layer_index_set = set(layer_indices)
        model_type = self.vllm_config.model_config.hf_text_config.model_type
        num_attn_module = 2 if model_type in ("longcat_flash", "longcat_flash_ngram") else 1
        from vllm.v1.worker.utils import extract_layer_index

        def layer_in_group(layer_name: str) -> bool:
            if "mtp" in layer_name:
                return any(layer_idx >= self.num_layers for layer_idx in layer_index_set)
            return extract_layer_index(layer_name, num_attn_module) in layer_index_set

        return {
            layer_name: layer_cache for layer_name, layer_cache in self.kv_caches.items() if layer_in_group(layer_name)
        }

    @staticmethod
    def _get_kv_cache_dims_from_tensors(kv_caches: dict[str, Any]) -> tuple[int, int, int]:
        """Return (num_kv_heads, k_head_dim, v_head_dim) from registered KV cache tensors."""
        k_cache, v_cache = next(iter(kv_caches.values()))
        return int(k_cache.shape[-2]), int(k_cache.shape[-1]), int(v_cache.shape[-1])

    def reformat_kv_cache_with_fused_op(
        self,
        block_ids: list[list[int]],
        tp_num_need_pulls: int,
        kv_caches: dict[str, Any] | None = None,
    ):
        if kv_caches is None:
            kv_caches = self.kv_caches
        k_cache = list(kv_caches.values())[0][0]
        device = k_cache.device
        num_kv_head, head_dim, _ = self._get_kv_cache_dims_from_tensors(kv_caches)
        block_size = self.vllm_config.cache_config.block_size
        layers = len(kv_caches)
        flat_block_ids = [item for sublist in block_ids for item in sublist]
        block_ids_tensor = torch.tensor(flat_block_ids, dtype=torch.int64, device=device)

        k_caches = []
        v_caches = []
        for _, (k_cache_layer, v_cache_layer) in kv_caches.items():
            k_caches.append(k_cache_layer)
            v_caches.append(v_cache_layer)

        torch.ops._C_ascend.transpose_kv_cache_by_block(
            k_caches, v_caches, block_ids_tensor, block_size, num_kv_head, head_dim, tp_num_need_pulls, layers
        )

    def reformat_kv_cache(
        self,
        block_ids: list[list[int]],
        tp_num_need_pulls: int,
        need_cat_cache: bool = False,
        need_nz_cache: bool = False,
        kv_caches: dict[str, Any] | None = None,
    ):
        if kv_caches is None:
            kv_caches = self.kv_caches
        k_cache = list(kv_caches.values())[0][0]
        dtype = k_cache.dtype
        device = k_cache.device
        num_kv_heads, k_head_dim, v_head_dim = self._get_kv_cache_dims_from_tensors(kv_caches)

        flat_block_ids = [item for sublist in block_ids for item in sublist]
        block_ids_tensor = torch.tensor(flat_block_ids, dtype=torch.int32, device=device)
        num_blocks = len(flat_block_ids)
        num_tokens = num_blocks * self.block_size

        # Create device tensors for copy operations
        block_table = block_ids_tensor.view(1, -1)
        block_len_tensor = torch.tensor([num_tokens], dtype=torch.int32, device=device)
        seq_start_tensor = torch.tensor([0], dtype=torch.int32, device=device)

        k_buffer = torch.empty((num_tokens, num_kv_heads, k_head_dim), dtype=dtype, device=device)
        v_buffer = torch.empty((num_tokens, num_kv_heads, v_head_dim), dtype=dtype, device=device)

        # Create slot mapping for reshape operations
        block_offsets = torch.arange(0, self.block_size, dtype=torch.int32, device=device)
        slot_mapping = (
            block_offsets.reshape((1, self.block_size)) + block_ids_tensor.reshape((num_blocks, 1)) * self.block_size
        ).flatten()

        # FIXME: Right now, if we skip synchronization at this point, the system
        # will crash in GQA scenarios. However, we still haven't identified the
        # root cause.
        torch.npu.synchronize()

        # Process each layer in the KV cache
        for _, (k_cache_layer, v_cache_layer) in kv_caches.items():
            # Load cache data into buffers
            torch_npu.npu_gather_pa_kv_cache(
                k_cache_layer,
                v_cache_layer,
                block_table,
                block_len_tensor,
                seq_offset=seq_start_tensor,
                key=k_buffer,
                value=v_buffer,
            )
            if need_cat_cache:
                self._cat_kv_cache(
                    k_cache_layer,
                    v_cache_layer,
                    k_buffer,
                    v_buffer,
                    tp_num_need_pulls,
                    num_blocks,
                    num_tokens,
                    slot_mapping,
                    num_kv_heads,
                )
            if need_nz_cache:
                self._nz_kv_cache(
                    k_cache_layer,
                    v_cache_layer,
                    k_buffer,
                    v_buffer,
                    slot_mapping,
                    num_kv_heads,
                    k_head_dim,
                    v_head_dim,
                )
        # Clean up buffers
        del k_buffer, v_buffer

    def _cat_kv_cache(
        self,
        k_cache_layer,
        v_cache_layer,
        k_buffer,
        v_buffer,
        tp_num_need_pulls,
        num_blocks,
        num_tokens,
        slot_mapping,
        num_kv_heads: int,
    ):
        def _transpose_kv_cache_between_head(buffer: torch.Tensor) -> torch.Tensor:
            buffer = buffer.view(num_blocks, tp_num_need_pulls, self.block_size, -1)
            buffer.transpose_(1, 2)
            return buffer.contiguous().view(num_tokens, num_kv_heads, -1)

        # Transpose KV cache
        k_buffer = _transpose_kv_cache_between_head(k_buffer)
        v_buffer = _transpose_kv_cache_between_head(v_buffer)

        # Reshape and cache the processed buffers
        torch_npu.npu_scatter_pa_kv_cache(
            key=k_buffer,
            value=v_buffer,
            key_cache=k_cache_layer,
            value_cache=v_cache_layer,
            slot_mapping=slot_mapping,
            cache_mode="Norm",
        )

    def _nz_kv_cache(
        self,
        k_cache_layer,
        v_cache_layer,
        k_buffer,
        v_buffer,
        slot_mapping,
        num_kv_heads: int,
        k_head_dim: int,
        v_head_dim: int,
    ):
        nz_fmt_last_dim = 16
        k_cache_layer = k_cache_layer.view(
            -1, k_head_dim * num_kv_heads // nz_fmt_last_dim, self.block_size, nz_fmt_last_dim
        )
        v_cache_layer = v_cache_layer.view(
            -1, v_head_dim * num_kv_heads // nz_fmt_last_dim, self.block_size, nz_fmt_last_dim
        )
        torch_npu.npu_scatter_pa_kv_cache(k_buffer, v_buffer, k_cache_layer, v_cache_layer, slot_mapping)
