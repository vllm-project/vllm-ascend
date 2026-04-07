import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.triton.triton_utils import extract_slice, get_element, get_vectorcore_num, insert_slice


@triton.jit
def gated_split_qkv_rmsnorm_rope_kernel(
    input_gm_ptr,                # 输入张量 [batch, 2*q_hidden + 2*kv_hidden]
    q_gm_ptr,                    # Q 输出
    gate_gm_ptr,                 # gate 输出（新增）
    k_gm_ptr,                    # K 输出
    v_gm_ptr,                    # V 输出
    q_weight_ptr,                # Q RMSNorm 权重 [head_dim]
    q_bias_ptr,                  # Q bias（可选）
    k_weight_ptr,                # K RMSNorm 权重 [head_dim]
    k_bias_ptr,                  # K bias（可选）
    batch_size,
    q_hidden_size: tl.constexpr,       # q_head_num * head_dim
    kv_hidden_size: tl.constexpr,      # kv_head_num * head_dim
    total_hidden_size: tl.constexpr,   # 2*q_hidden + 2*kv_hidden
    eps: tl.constexpr,
    BIAS: tl.constexpr,
    HEAD_DIM: tl.constexpr,            # head_dim (256 for Qwen3-Next)
    ROPE_DIM: tl.constexpr,            # 64 (head_dim // 4)
    HALF_ROPE_DIM: tl.constexpr,       # 32
    IS_PARTIAL_ROPE: tl.constexpr,     # True for Qwen3-Next
    num_vectorcore: tl.constexpr,
    batch_size_per_iter_per_vec: tl.constexpr,
    q_head_nums_per_iter_per_vec: tl.constexpr,
    k_head_nums_per_iter_per_vec: tl.constexpr,
    q_head_num: tl.constexpr,
    kv_head_num: tl.constexpr,
    positions_gm_ptr,
    cos_sin_cache_gm_ptr,
):
    row_pid = tl.program_id(0)

    batch_size_per_vec = tl.cdiv(batch_size, num_vectorcore) #
    iter_num_per_vec = tl.cdiv(batch_size_per_vec, batch_size_per_iter_per_vec) #
    input_batch_offset = row_pid * batch_size_per_vec #
    input_batch_offset_end = min(input_batch_offset + batch_size_per_vec, batch_size) #

    q_weight_values = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM))  #
    k_weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM))  #
    if BIAS:
        q_bias_values = tl.load(q_bias_ptr + tl.arange(0, HEAD_DIM))  #
        k_bias_values = tl.load(k_bias_ptr + tl.arange(0, HEAD_DIM))  #

    qgate_cols = tl.arange(0, q_hidden_size)
    kv_cols = tl.arange(0, kv_hidden_size)

    mblk_idx = tl.arange(0, batch_size_per_iter_per_vec) + input_batch_offset # 

    for iter in range(iter_num_per_vec):
        cur_mblk_idx = mblk_idx + iter * batch_size_per_iter_per_vec
        mmask = cur_mblk_idx < input_batch_offset_end

        pos = tl.load(positions_gm_ptr + cur_mblk_idx, mask=mmask)
        cos_sin_row = tl.zeros((batch_size_per_iter_per_vec, ROPE_DIM), dtype=tl.bfloat16)
        for i in range(batch_size_per_iter_per_vec):
            pos_val = get_element(pos, (i,))
            row_data = tl.load(cos_sin_cache_gm_ptr + pos_val * ROPE_DIM + tl.arange(0, ROPE_DIM))
            cos_sin_row = insert_slice(
                cos_sin_row,
                row_data[None, :],
                offsets=(i, 0),
                sizes=(1, ROPE_DIM),
                strides=(1, 1),
            )
            
        cos_sin_row = cos_sin_row.reshape(batch_size_per_iter_per_vec, 1, ROPE_DIM)
        cos = extract_slice(cos_sin_row, offsets=(0, 0, 0), sizes=(batch_size_per_iter_per_vec, 1, HALF_ROPE_DIM), strides=(1, 1, 1))
        sin = extract_slice(cos_sin_row, offsets=(0, 0, HALF_ROPE_DIM), sizes=(batch_size_per_iter_per_vec, 1, HALF_ROPE_DIM), strides=(1, 1, 1))

        qgatekv_offset = cur_mblk_idx[:, None] * total_hidden_size + tl.arange(0, total_hidden_size)[None, :]
        qgatekv_data = tl.load(input_gm_ptr + qgatekv_offset, mask=mmask[:, None])
        # q_data, gate_data, k_data, v_data = tl.split(qgatekv_data, [q_hidden_size, q_hidden_size, kv_hidden_size, kv_hidden_size], dim=-1)
        q_data = extract_slice(qgatekv_data, offsets=(0, 0), sizes=(batch_size_per_iter_per_vec, q_hidden_size), strides=(1, 1))
        gate_data = extract_slice(qgatekv_data, offsets=(0, q_hidden_size), sizes=(batch_size_per_iter_per_vec, q_hidden_size), strides=(1, 1))
        k_data = extract_slice(qgatekv_data, offsets=(0, 2 * q_hidden_size), sizes=(batch_size_per_iter_per_vec, kv_hidden_size), strides=(1, 1))
        v_data = extract_slice(qgatekv_data, offsets=(0, 2 * q_hidden_size + kv_hidden_size), sizes=(batch_size_per_iter_per_vec, kv_hidden_size), strides=(1, 1))
        # qk_data = tl.zeros((batch_size_per_iter_per_vec, q_hidden_size + kv_hidden_size), dtype=tl.bfloat16)
        # qk_data[:, :q_hidden_size] = q_data
        # qk_data[:, q_hidden_size:] = k_data

        v_output_idx = cur_mblk_idx[:, None] * kv_hidden_size + kv_cols[None, :]
        gate_output_idx = cur_mblk_idx[:, None] * q_hidden_size + qgate_cols[None, :]
        tl.store(v_gm_ptr + v_output_idx, v_data, mask=mmask[:, None])
        tl.store(gate_gm_ptr + gate_output_idx, gate_data, mask=mmask[:, None])

        q_data = q_data.reshape(q_head_nums_per_iter_per_vec, HEAD_DIM).to(tl.float32)
        normalized_values_q = q_data
        normalized_values_q = normalized_values_q * normalized_values_q
        normalized_values_q = tl.sum(normalized_values_q, axis=1) / HEAD_DIM
        normalized_values_q = 1 / tl.sqrt(normalized_values_q + eps)
        normalized_values_q = q_data * normalized_values_q[:, None]

        k_data = k_data.reshape(k_head_nums_per_iter_per_vec, HEAD_DIM).to(tl.float32)
        normalized_values_k = k_data
        normalized_values_k = normalized_values_k * normalized_values_k
        normalized_values_k = tl.sum(normalized_values_k, axis=1) / HEAD_DIM
        normalized_values_k = 1 / tl.sqrt(normalized_values_k + eps)
        normalized_values_k = k_data * normalized_values_k[:, None]


        normalized_values_tmp = normalized_values_q.reshape(batch_size_per_iter_per_vec, q_head_num, HEAD_DIM)

        if BIAS:
            normalized_values_tmp = (normalized_values_tmp * q_weight_values + q_bias_values).to(tl.bfloat16)
        else:
            normalized_values_tmp = (normalized_values_tmp * q_weight_values).to(tl.bfloat16)

        # q rope
        values_tmp = tl.zeros((batch_size_per_iter_per_vec, q_head_num, ROPE_DIM), dtype=tl.bfloat16)
        x1 = extract_slice(
            normalized_values_tmp,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        x2 = extract_slice(
            normalized_values_tmp,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp = insert_slice(
            values_tmp,
            x1 * cos - x2 * sin,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp = insert_slice(
            values_tmp,
            x2 * cos + x1 * sin,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        q_output_idx = cur_mblk_idx[:, None] * q_hidden_size + qgate_cols[None, :]
        if IS_PARTIAL_ROPE:
            normalized_values_tmp = insert_slice(
                normalized_values_tmp,
                values_tmp,
                offsets=(0, 0, 0),
                sizes=(batch_size_per_iter_per_vec, q_head_num, ROPE_DIM),
                strides=(1, 1, 1),
            )
            tl.store(
                q_gm_ptr + q_output_idx,
                normalized_values_tmp.reshape(batch_size_per_iter_per_vec, q_hidden_size),
                mask=mmask[:, None],
            )
        else:
            tl.store(
                q_gm_ptr + q_output_idx,
                values_tmp.reshape(batch_size_per_iter_per_vec, q_hidden_size),
                mask=mmask[:, None],
            )

        # k rope
        normalized_values_tmp1 =  normalized_values_k.reshape(batch_size_per_iter_per_vec, kv_head_num, HEAD_DIM),

        if BIAS:
            normalized_values_tmp1 = (normalized_values_tmp1 * k_weight_values + k_bias_values).to(tl.bfloat16)
        else:
            normalized_values_tmp1 = (normalized_values_tmp1 * k_weight_values).to(tl.bfloat16)

        values_tmp2 = tl.zeros((batch_size_per_iter_per_vec, kv_head_num, ROPE_DIM), dtype=tl.bfloat16)

        x1 = extract_slice(
            normalized_values_tmp1,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        x2 = extract_slice(
            normalized_values_tmp1,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp2 = insert_slice(
            values_tmp2,
            x1 * cos - x2 * sin,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp2 = insert_slice(
            values_tmp2,
            x2 * cos + x1 * sin,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )

        k_output_idx = cur_mblk_idx[:, None] * kv_hidden_size + kv_cols[None, :]
        if IS_PARTIAL_ROPE:
            normalized_values_tmp1 = insert_slice(
                normalized_values_tmp1,
                values_tmp2,
                offsets=(0, 0, 0),
                sizes=(batch_size_per_iter_per_vec, kv_head_num, ROPE_DIM),
                strides=(1, 1, 1),
            )
            tl.store(
                k_gm_ptr + k_output_idx,
                normalized_values_tmp1.reshape(batch_size_per_iter_per_vec, kv_hidden_size),
                mask=mmask[:, None]
            )
        else:
            tl.store(
                k_gm_ptr + k_output_idx,
                values_tmp2.reshape(batch_size_per_iter_per_vec, kv_hidden_size),
                mask=mmask[:, None]
            )


def gated_split_qkv_rmsnorm_rope_impl(
    input: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    融合 Q+gate 拆分、RMSNorm 和 RoPE 的 Triton kernel 包装器。
    输入布局：[Q, gate, K, V] 连续排列。
    输出：(q_output, gate_output, k_output, v_output)
    """
    # print(f"input: {input}, cos_sin_cache: {cos_sin_cache}; postions: {postions}; q_weight: {q_weight.shape}; k_weight: {k_weight.shape}")
    num_vectorcore = get_vectorcore_num()
    rope_dim = cos_sin_cache.shape[-1]          # Qwen3-Next 中 rope_dim = head_dim // 4
    batch_size = input.shape[0]
    BIAS = q_bias is not None
    IS_PARTIAL_ROPE = rope_dim != head_dim      # 对于 Qwen3-Next 为 True

    total_hidden_size = q_hidden_size * 2 + kv_hidden_size * 2

    q_output = torch.empty(batch_size, q_hidden_size, device=input.device, dtype=input.dtype)
    gate_output = torch.empty(batch_size, q_hidden_size, device=input.device, dtype=input.dtype)
    k_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
    v_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)

    q_head_num = q_hidden_size // head_dim
    kv_head_num = kv_hidden_size // head_dim

    UB_SIZE = 87040
    element_size = input.element_size()

    # MAX_POSITION_EMBEDDINGS = [262144]
    # NUM_TOKENS = [1, 4, 8, 16, 1024, 2048, 4096, 10240, 20480]
    # NUM_QKV_HEADS = [(16, 2)]
    # HEAD_SIZES = [256]
    # ROPE_DIMS = [64]
    # 20480 + 1536 + 256 + 1024 + 8192 + 1024 = 32512
    if IS_PARTIAL_ROPE:
        factor = (5 * q_hidden_size + 3 * kv_hidden_size + rope_dim * 4 + q_head_num * rope_dim) + (2 * q_hidden_size + 2 * kv_hidden_size)
    else:
        factor = (5 * q_hidden_size + 3 * kv_hidden_size + rope_dim * 2 + q_head_num * rope_dim // 2) + (2 * q_hidden_size + 2 * kv_hidden_size)

    batch_size_per_iter_per_vec = max(1, int(UB_SIZE / element_size) // factor)

    # qgate_head_num_sum = int(q_head_num + q_head_num) # 
    q_head_nums_per_iter_per_vec = batch_size_per_iter_per_vec * q_head_num # 

    # kv_head_num_sum = int(q_head_num + q_head_num) # 
    k_head_nums_per_iter_per_vec = batch_size_per_iter_per_vec * kv_head_num # 

    # v tiling
    v_batch_size_per_iter_per_vec = UB_SIZE / torch.bfloat16.itemsize // (kv_hidden_size + 1) # 

    grid = (num_vectorcore, 1, 1)

    q_weight = 1.0 + q_weight
    k_weight = 1.0 + k_weight

    gated_split_qkv_rmsnorm_rope_kernel[grid](
        input,
        q_output,
        gate_output,
        k_output,
        v_output,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        batch_size,
        q_hidden_size,
        kv_hidden_size,
        total_hidden_size,
        eps,
        BIAS,
        head_dim,
        rope_dim,
        rope_dim // 2,
        IS_PARTIAL_ROPE,
        num_vectorcore,
        int(batch_size_per_iter_per_vec),
        q_head_nums_per_iter_per_vec,
        k_head_nums_per_iter_per_vec,
        q_head_num,
        kv_head_num,
        positions,
        cos_sin_cache,
    )
    return q_output, gate_output, k_output, v_output


def gated_split_qkv_rmsnorm_rope_impl_fake(
    input: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = input.shape[0]
    q_output = torch.empty(batch_size, q_hidden_size, device=input.device, dtype=input.dtype)
    gate_output = torch.empty(batch_size, q_hidden_size, device=input.device, dtype=input.dtype)
    k_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
    v_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
    return q_output, gate_output, k_output, v_output


direct_register_custom_op(
    op_name="gated_qkv_rmsnorm_rope",
    op_func=gated_split_qkv_rmsnorm_rope_impl, 
    fake_impl=gated_split_qkv_rmsnorm_rope_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)