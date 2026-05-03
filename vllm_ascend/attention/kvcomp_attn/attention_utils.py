import torch
from vllm.config import get_current_vllm_config_or_none

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.worker.kvcomp_utils import (
    KVCompMetaData,
    recover_request_lengths,
)


def build_kvcomp_metadata(
    kvcomp_meta: KVCompMetaData,
    common_meta: AscendCommonAttentionMetadata,
) -> None:
    num_reqs = common_meta.num_reqs
    kvcomp_meta.num_actual_tokens = common_meta.num_actual_tokens

    kvcomp_meta.slot_mapping = common_meta.slot_mapping
    kvcomp_meta.seq_lens_gpu = common_meta.seq_lens[:num_reqs]

    real_batch_size = kvcomp_meta.seq_lens_gpu.shape[0]
    assert num_reqs == real_batch_size, "the len of seq_lens_gpu is not equal with batch_size"

    kvcomp_meta.actual_query_lens = recover_request_lengths(common_meta.query_start_loc[: num_reqs + 1]).to(torch.int32)

    runtime_seq_lens_list = kvcomp_meta.seq_lens_gpu.tolist()

    if kvcomp_meta.num_actual_tokens < real_batch_size:
        runtime_seq_lens_list = runtime_seq_lens_list[: kvcomp_meta.num_actual_tokens] + [0] * (
            real_batch_size - kvcomp_meta.num_actual_tokens
        )
        kvcomp_meta.slot_mapping[kvcomp_meta.num_actual_tokens :] = -1

    runtime_max_len = max(runtime_seq_lens_list) if runtime_seq_lens_list else 0
    kvcomp_meta.max_seq_len_for_hamming = (
        common_meta.max_seq_len if common_meta.max_seq_len is not None else runtime_max_len
    )
    kvcomp_meta.block_tables_for_hamming = common_meta.block_table_tensor[:real_batch_size]

    top_k_cpu = kvcomp_meta.topk_for_hamming_full_cpu[:real_batch_size].clone()
    if kvcomp_meta.num_actual_tokens < real_batch_size:
        top_k_cpu[kvcomp_meta.num_actual_tokens :] = 0

    runtime_seq_lens_cpu = torch.tensor(runtime_seq_lens_list, dtype=torch.int32)
    chunk_size = kvcomp_meta.kvcomp_config.chunk_size
    remainder = runtime_seq_lens_cpu % chunk_size

    new_seq_lens = torch.where(
        remainder == 0,
        chunk_size * top_k_cpu,
        chunk_size * (top_k_cpu - 1) + remainder,
    )
    kvcomp_meta.seq_lens_from_hamming = new_seq_lens.tolist()

    q_start_loc_slice = common_meta.query_start_loc[:real_batch_size].to(torch.int64)

    torch.where(
        kvcomp_meta.slot_mapping[:real_batch_size] >= 0,
        torch.ones_like(q_start_loc_slice, dtype=torch.bool),
        torch.zeros_like(q_start_loc_slice, dtype=torch.bool),
        out=kvcomp_meta.valid_query_mask[:real_batch_size],
    )
    torch.where(
        kvcomp_meta.valid_query_mask[:real_batch_size],
        kvcomp_meta.actual_query_lens[:real_batch_size],
        torch.zeros_like(kvcomp_meta.actual_query_lens[:real_batch_size]),
        out=kvcomp_meta.seq_lens_for_reshape[:real_batch_size],
    )
    common_meta.kvcomp_metadata = kvcomp_meta


def reshape_and_cache_kvcomp(kvcomp_meta: KVCompMetaData | None, layer_index: int | None, key: torch.Tensor):
    assert kvcomp_meta is not None
    assert layer_index is not None

    if kvcomp_meta.hashk_caches[layer_index] is None:
        return None
    hash_encoder = kvcomp_meta.hash_encoder
    num_tokens = kvcomp_meta.num_actual_tokens

    hashk = hash_encoder.compute_hash(key[:num_tokens])
    hashk_op = hashk.transpose(0, 1).reshape(-1, hashk.shape[-1]).contiguous()

    hashk_cache_op = kvcomp_meta.hashk_caches[layer_index]
    real_batch_size = kvcomp_meta.seq_lens_gpu.shape[0]

    torch.ops._C_ascend.npu_reshape_and_cache_bnsd(
        hashk_op,
        hashk_cache_op,
        kvcomp_meta.slot_mapping[:num_tokens],
        kvcomp_meta.seq_lens_for_reshape[:real_batch_size],
        hashk_cache_op,
    )
    return hashk_cache_op


def get_kvcomp_decode_params(
    layer_index: int | None,
    kvcomp_meta: KVCompMetaData | None,
    query: torch.Tensor,
    key: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lengths_kv: list[int],
):
    assert kvcomp_meta is not None
    assert layer_index is not None

    if kvcomp_meta.hashk_caches[layer_index] is None:
        return block_table, actual_seq_lengths_kv

    kv_config = kvcomp_meta.kvcomp_config
    hash_encoder = kvcomp_meta.hash_encoder
    real_batch_size = kvcomp_meta.seq_lens_gpu.shape[0]

    if kv_config.vllm_hash_attention_skip_layers[layer_index]:
        return kvcomp_meta.hamming_output, kvcomp_meta.seq_lens_from_hamming

    hashk_cache_op = reshape_and_cache_kvcomp(kvcomp_meta, layer_index, key)

    hashq = hash_encoder.compute_hash(query[:real_batch_size])
    hashq_op = hashq.unsqueeze(2).contiguous()

    new_block_table = torch.ops._C_ascend.npu_hamming_dist_top_k(
        hashq_op,
        hashk_cache_op,
        None,
        kvcomp_meta.topk_for_hamming_full[:real_batch_size],
        kvcomp_meta.seq_lens_gpu[:real_batch_size],
        kvcomp_meta.chunk_sizes_for_hamming_full[:real_batch_size],
        kvcomp_meta.max_seq_len_for_hamming,
        kvcomp_meta.sink,
        kvcomp_meta.recent,
        None,
        kvcomp_meta.block_tables_for_hamming,
        kvcomp_meta.valid_query_mask[:real_batch_size],
        kvcomp_meta.hamming_output[:real_batch_size],
    )

    new_block_table = new_block_table.squeeze(1).contiguous()
    kvcomp_meta.hamming_output = new_block_table

    return new_block_table, kvcomp_meta.seq_lens_from_hamming


def is_enable_hamming_sparse():
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return False
    additional_config = vllm_config.additional_config if vllm_config.additional_config is not None else {}
    enable_hamming_sparse = additional_config.get("enable_hamming_sparse", False)
    enable_hamming_sparse = enable_hamming_sparse and not vllm_config.speculative_config
    return enable_hamming_sparse
