from typing import Any

import torch
import torch.distributed as dist
import torch_npu
from vllm.config import VllmConfig
from vllm.distributed import get_dcp_group
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.distributed.utils import get_decode_context_model_parallel_world_size


def get_dcp_local_seq_lens(
    seq_lens: torch.Tensor,
    dcp_size: int,
    interleave_size: int,
) -> torch.Tensor:
    """Return the interleave-aware KV length of every DCP rank."""
    tiled = seq_lens.unsqueeze(-1)
    rank_offsets = torch.arange(
        dcp_size,
        dtype=seq_lens.dtype,
        device=seq_lens.device,
    )
    base = tiled // interleave_size // dcp_size * interleave_size
    remainder = tiled - base * dcp_size
    return base + torch.clamp(
        remainder - rank_offsets * interleave_size,
        0,
        interleave_size,
    )


class ReplicatedKVMetadataMixin:
    """Build full-cache addresses using the consumer's own spec and buffers."""

    def _init_replicated_view(
        self,
        kv_cache_spec: AttentionSpec,
        kernel_block_size: int,
        dcp_size: int,
        max_num_reqs: int,
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        self.dcp_size = dcp_size
        self.device = device
        self.replicated_view_block_size = kernel_block_size
        if kv_cache_spec.block_size % self.replicated_view_block_size != 0:
            raise RuntimeError(
                "SFA replicated view requires the KV cache block size "
                f"({kv_cache_spec.block_size}) to be divisible by "
                f"{self.replicated_view_block_size}."
            )
        self.blocks_per_phys_block = kv_cache_spec.block_size // self.replicated_view_block_size
        max_num_input_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        max_model_len = vllm_config.model_config.max_model_len
        total_cp_size = self.dcp_size
        # The generic vLLM BlockTable may expose global-width storage, while
        # the DCP physical KV layout only populates rank-local block columns.
        self.max_local_block_table_cols = (
            cdiv(max_model_len, kv_cache_spec.block_size * total_cp_size) * self.blocks_per_phys_block
        )
        max_replicated_block_table_cols = self.max_local_block_table_cols * total_cp_size
        self.block_table_replicated_view_buf: torch.Tensor = torch.empty(
            (max_num_reqs, max_replicated_block_table_cols),
            dtype=torch.int32,
            device=device,
        )
        self.arange_buffer: torch.Tensor = torch.arange(
            max_replicated_block_table_cols,
            dtype=torch.int32,
            device=device,
        )
        self.slot_mapping_replicated_view_buf: torch.Tensor = torch.empty(
            (max_num_input_tokens,),
            dtype=torch.int32,
            device=device,
        )

    def _get_dcp_local_block_table(self, block_table: torch.Tensor, num_reqs: int) -> torch.Tensor:
        local_cols = min(block_table.shape[1], self.max_local_block_table_cols)
        return block_table[:num_reqs, :local_cols]

    def _ensure_replicated_view_buffers(
        self,
        num_reqs: int,
        num_input_tokens: int,
        local_block_table_cols: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_table_cols = local_block_table_cols * self.dcp_size
        if (
            self.block_table_replicated_view_buf.shape[0] < num_reqs
            or self.block_table_replicated_view_buf.shape[1] < block_table_cols
        ):
            raise RuntimeError(
                f"Replicated view buffer is too small: "
                f"block_table_replicated_view_buf.shape={self.block_table_replicated_view_buf.shape}, "
                f"num_reqs={num_reqs}, block_table_cols={block_table_cols}"
            )
        if self.slot_mapping_replicated_view_buf.shape[0] < num_input_tokens:
            raise RuntimeError(
                f"Replicated view buffer is too small: "
                f"slot_mapping_replicated_view_buf.shape={self.slot_mapping_replicated_view_buf.shape}, "
                f"num_input_tokens={num_input_tokens}"
            )
        return (
            self.block_table_replicated_view_buf[:num_reqs, :block_table_cols],
            self.arange_buffer[:block_table_cols],
            self.slot_mapping_replicated_view_buf[:num_input_tokens],
        )

    def _build_block_table_replicated_view(
        self,
        dcp_block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        num_reqs = dcp_block_table.shape[0]
        local_block_table_cols = dcp_block_table.shape[1]
        block_table_replicated_view, replicated_col_idx, _ = self._ensure_replicated_view_buffers(
            num_reqs,
            0,
            local_block_table_cols,
        )

        total_cp_size = self.dcp_size
        blocks_per_phys_block = self.blocks_per_phys_block
        local_col_idx = (
            replicated_col_idx // (total_cp_size * blocks_per_phys_block) * blocks_per_phys_block
            + replicated_col_idx % blocks_per_phys_block
        )
        rank_in_replicated_view = (replicated_col_idx // blocks_per_phys_block) % total_cp_size

        local_logical_blocks = torch.index_select(dcp_block_table, 1, local_col_idx)
        if blocks_per_phys_block == 1:
            replicated_blocks = local_logical_blocks * total_cp_size + rank_in_replicated_view
        else:
            local_sub_blocks = local_logical_blocks % blocks_per_phys_block
            local_phys_blocks = local_logical_blocks // blocks_per_phys_block
            replicated_blocks = (
                local_phys_blocks * total_cp_size + rank_in_replicated_view
            ) * blocks_per_phys_block + local_sub_blocks

        valid_req_mask = (seq_lens[:num_reqs].to(device=self.device) > 0).to(replicated_blocks.dtype).view(-1, 1)
        replicated_blocks = replicated_blocks * valid_req_mask
        block_table_replicated_view.copy_(replicated_blocks)
        return block_table_replicated_view

    def _build_slot_mapping_replicated_view(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        block_table_replicated_view: torch.Tensor,
    ) -> torch.Tensor:
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        num_actual_tokens = min(common_attn_metadata.num_actual_tokens, num_input_tokens)
        local_block_table_cols = block_table_replicated_view.shape[1] // self.dcp_size
        _, _, slot_mapping_replicated_view = self._ensure_replicated_view_buffers(
            num_reqs,
            num_input_tokens,
            local_block_table_cols,
        )
        slot_mapping_replicated_view.fill_(-1)
        if num_actual_tokens == 0:
            return slot_mapping_replicated_view

        query_lens = (
            common_attn_metadata.query_start_loc[1 : num_reqs + 1] - common_attn_metadata.query_start_loc[:num_reqs]
        )
        req_indices = torch.repeat_interleave(
            torch.arange(num_reqs, dtype=torch.int32, device=self.device),
            query_lens.to(device=self.device),
            output_size=num_input_tokens,
        )[:num_actual_tokens]
        if req_indices.numel() == 0:
            return slot_mapping_replicated_view

        num_actual_tokens = min(num_actual_tokens, req_indices.shape[0])
        req_indices = req_indices[:num_actual_tokens]
        positions = common_attn_metadata.positions[:num_actual_tokens].to(
            device=self.device,
            dtype=torch.int32,
        )
        logical_block_idx = positions // self.replicated_view_block_size
        block_offsets = positions % self.replicated_view_block_size
        block_table_indices = req_indices * block_table_replicated_view.shape[1] + logical_block_idx
        block_numbers = block_table_replicated_view.flatten()[block_table_indices]
        slot_mapping_replicated_view[:num_actual_tokens] = (
            block_numbers * self.replicated_view_block_size + block_offsets
        )
        return slot_mapping_replicated_view


class DCPMetadataBuilderMixin:
    """Shared DCP metadata access for backend-specific metadata builders."""

    dcp_size: int
    dcp_rank: int

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        dcp_group = get_dcp_group()
        self.dcp_size = dcp_group.world_size
        self.dcp_rank = dcp_group.rank_in_group

    @staticmethod
    def _require_dcp_metadata(
        common_attn_metadata: Any,
    ) -> Any:
        dcp_metadata = common_attn_metadata.context_parallel_metadata
        if dcp_metadata is None or dcp_metadata.num_computed_tokens_of_dcp is None:
            raise AssertionError("DCP metadata must be populated.")
        return dcp_metadata

    def _get_dcp_context_lens(
        self,
        common_attn_metadata: Any,
        *,
        start: int = 0,
        end: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        dcp_metadata = self._require_dcp_metadata(common_attn_metadata)
        context_lens = torch.as_tensor(
            dcp_metadata.num_computed_tokens_of_dcp[start:end],
            dtype=torch.int32,
            device=device,
        )
        return context_lens.reshape(-1, self.dcp_size)

    def _get_dcp_rank_context_lens(
        self,
        common_attn_metadata: Any,
        *,
        start: int = 0,
        end: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        return self._get_dcp_context_lens(
            common_attn_metadata,
            start=start,
            end=end,
            device=device,
        )[:, self.dcp_rank]


class DCPImplMixin:
    """Shared DCP group lifecycle and collectives for attention backends."""

    # vLLM #55780 defaults implementations to no DCP; these implement it.
    supports_dcp = True

    dcp_size: int
    dcp_rank: int

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dcp_group = get_dcp_group()
        self.dcp_size = self.dcp_group.world_size
        self.dcp_rank = self.dcp_group.rank_in_group
        self.dcp_device_group = self.dcp_group.device_group if self.dcp_size > 1 else None

    def _dcp_all_gather(
        self,
        tensor: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        if self.dcp_size == 1:
            return tensor
        return self.dcp_group.all_gather(tensor.contiguous(), dim)

    def _dcp_all_gather_fragments(
        self,
        *tensors: torch.Tensor,
        dim: int,
    ) -> tuple[torch.Tensor, ...]:
        if not tensors:
            return ()
        split_sizes = [tensor.shape[-1] for tensor in tensors]
        gathered = self._dcp_all_gather(
            torch.cat(tensors, dim=-1),
            dim,
        )
        return torch.split(gathered, split_sizes, dim=-1)

    def _merge_dcp_attention_output(
        self,
        attn_output: torch.Tensor,
        softmax_lse: torch.Tensor,
        head_size: int,
    ) -> torch.Tensor:
        return _npu_attention_update(
            head_size,
            _process_attn_out_lse(
                attn_output,
                softmax_lse,
                dcp_size=self.dcp_size,
                dcp_device_group=self.dcp_device_group,
            ),
            dcp_size=self.dcp_size,
        )


def _process_attn_out_lse(
    attn_output: torch.Tensor,
    softmax_lse: torch.Tensor,
    *,
    dcp_size: int | None = None,
    dcp_device_group=None,
) -> torch.Tensor:
    if dcp_size is None:
        dcp_size = get_decode_context_model_parallel_world_size()
    if dcp_size > 1 and dcp_device_group is None:
        dcp_device_group = get_dcp_group().device_group
    softmax_lse = softmax_lse.to(torch.float32)
    attn_output = attn_output.to(torch.float32)
    # Concat out&lse: [bs,num_heads,v_head_dim] + [bs,num_heads,1] -> [bs,num_heads,v_head_dim+1]
    attn_out_lse = torch.cat([attn_output, softmax_lse], dim=-1)
    if dcp_size > 1:
        # permute: [bs, num_heads, v_head_dim+1] -> [num_heads, v_head_dim+1, bs]
        attn_out_lse = attn_out_lse.permute([1, 2, 0]).contiguous()
        attn_out_lse_all2all = torch.empty_like(attn_out_lse)
        dist.all_to_all_single(
            attn_out_lse_all2all,
            attn_out_lse,
            group=dcp_device_group,
        )
        attn_out_lse = attn_out_lse_all2all.permute([2, 0, 1])

    return attn_out_lse


def _npu_attention_update(
    head_size,
    attn_out_lse: torch.Tensor,
    *,
    dcp_size: int | None = None,
) -> torch.Tensor:
    if dcp_size is None:
        dcp_size = get_decode_context_model_parallel_world_size()
    # [S, DCP * H, D+1]
    S, H_total, D_plus_1 = attn_out_lse.shape
    H = H_total // dcp_size
    D = head_size
    assert D_plus_1 == D + 1
    # [S, DCP, H, D+1] -> [DCP, S, H, D+1]
    x = attn_out_lse.view(S, dcp_size, H, D_plus_1)
    x = x.permute(1, 0, 2, 3).contiguous()
    # Split out lse
    out_flat, lse_flat = torch.split(x, [D, 1], dim=-1)  # [N, S, H, D], [N, S, H, 1]
    #    out: [N, S, H, D] -> [N, S*H, D]
    #    lse: [N, S, H, 1] -> [N, S*H]
    out_flat = out_flat.flatten(1, 2)  # [N, S*H, D]
    lse_flat = lse_flat.flatten(1, -1)  # [N, S*H]
    #  unbind to list
    out_list = out_flat.unbind(0)  # [S*H, D]
    lse_list = lse_flat.unbind(0)  # [S*H]
    attn_out, _ = torch_npu.npu_attention_update(lse_list, out_list, 0)
    attn_out = attn_out.view(-1, H, D)
    return attn_out


def _npu_attn_out_lse_update(attn_lse_mask, attn_lse_nomask, attn_out_mask, attn_out_nomask):
    T = attn_out_mask.shape[0]
    N = attn_out_mask.shape[1]
    D = attn_out_mask.shape[2]
    attn_out_mask, attn_lse_mask = _out_lse_reshape(attn_out_mask, attn_lse_mask)
    attn_out_nomask, attn_lse_nomask = _out_lse_reshape(attn_out_nomask, attn_lse_nomask)
    attn_out_mask = attn_out_mask.to(torch.float32)
    attn_out_nomask = attn_out_nomask.to(torch.float32)
    attn_lse_mask = attn_lse_mask.to(torch.float32)
    attn_lse_nomask = attn_lse_nomask.to(torch.float32)
    attn_output = [attn_out_nomask, attn_out_mask]
    attn_lse = [attn_lse_nomask, attn_lse_mask]
    update_type = 0
    output, _ = torch_npu.npu_attention_update(attn_lse, attn_output, update_type)
    output = output.view(T, N, D)
    return output


def _out_lse_reshape(attn_out: torch.Tensor, attn_lse: torch.Tensor) -> torch.Tensor:
    attn_out = attn_out.contiguous().view(attn_out.shape[0] * attn_out.shape[1], attn_out.shape[2])
    attn_lse = attn_lse.contiguous().view(attn_lse.shape[0] * attn_lse.shape[1] * attn_lse.shape[2])
    return attn_out, attn_lse


def _update_out_and_lse(out_list: torch.Tensor, lse_list: torch.Tensor) -> torch.Tensor:
    """LSE_final = log(sum(exp(LSE_i))), O_final = sum(exp(LSE_i - LSE_final) * O_i)
    Args:
        out_list: shape = [N, batch_size, num_heads, head_size]
        lse_list: shape = [N, batch_size, num_heads, 1]
    Returns:
        out_final: shape = [batch_size, num_heads, head_size]
        lse_final: shape = [batch_size, num_heads, 1]
    """
    lse_final = torch.logsumexp(lse_list, dim=0, keepdim=False)
    out_final = torch.sum(torch.exp(lse_list - lse_final) * out_list, dim=0)
    return out_final, lse_final
