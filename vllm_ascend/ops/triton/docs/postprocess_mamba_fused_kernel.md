# PostprocessMambaFusedKernel

## 产品支持情况

| 产品 | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

> 说明：`postprocess_mamba_fused_kernel` 是 vLLM Mamba 后处理路径使用的 Triton fused kernel。310P 场景不调用该 kernel，运行时通过 `vllm_ascend.patch.worker.patch_mamba_utils` 中的 CPU/Torch fallback 路径保持语义一致。

## 功能说明

- 算子功能：`postprocess_mamba_fused_kernel` 融合执行 Mamba decode/spec-decode 后处理阶段的决策计算和状态拷贝，避免原 `postprocess_mamba` 流程中因 CPU-GPU 同步带来的开销。
- 该 kernel 针对每个活跃请求和每个 Mamba state 执行一次程序实例，运行网格为：

    $$
    Grid=(num\_reqs,\ num\_layers \times num\_state\_types)
    $$

    其中 `program_id(0)` 表示 batch/request 维度，`program_id(1)` 表示扁平化后的 `state_idx = layer_idx * num_state_types + state_type_idx`。

- 后处理决策公式：

    当 `PRECOMPUTED_NEW_COMPUTED=False` 时：

    $$
    num\_tokens\_running\_state = num\_computed\_tokens + num\_scheduled\_tokens - num\_draft\_tokens
    $$

    $$
    new\_num\_computed\_tokens = num\_tokens\_running\_state + num\_accepted\_tokens - 1
    $$

    当 `PRECOMPUTED_NEW_COMPUTED=True` 时，`num_computed_tokens_ptr` 已经保存后处理后的 `new_num_computed_tokens`：

    $$
    num\_tokens\_running\_state = new\_num\_computed\_tokens - num\_accepted\_tokens + 1
    $$

    之后统一计算：

    $$
    aligned\_new\_computed = \left\lfloor \frac{new\_num\_computed\_tokens}{block\_size} \right\rfloor \times block\_size
    $$

    $$
    needs\_copy = aligned\_new\_computed \ge num\_tokens\_running\_state
    $$

    若 `needs_copy=True`，则：

    $$
    accept\_token\_bias = aligned\_new\_computed - num\_tokens\_running\_state
    $$

    $$
    dest\_block\_idx = \frac{aligned\_new\_computed}{block\_size} - 1
    $$

- Conv state 拷贝规则：当 `state_conv_widths_ptr[state_idx] > 0` 时，当前 state 被视为 conv state。kernel 将源 block 中从 `accept_token_bias` 开始的滑窗状态拷贝到目标 block 的起始位置：

    $$
    state[block\_table[row,\ src\_block\_idx],\ accept\_token\_bias:] \rightarrow
    state[block\_table[row,\ dest\_block\_idx],\ :conv\_width - accept\_token\_bias]
    $$

    当 `CONV_STATE_DIM_FIRST=True` 时，conv state 使用 DS 布局 `state[block, dim, state_len]`，kernel 按 dim 行逐行拷贝；否则按单段连续内存拷贝。

- Temporal state 拷贝规则：当 `state_conv_widths_ptr[state_idx] == 0` 时，当前 state 被视为 temporal state。kernel 使用 `accept_token_bias` 修正源 block 索引：

    $$
    state[block\_table[row,\ src\_block\_idx + accept\_token\_bias]] \rightarrow
    state[block\_table[row,\ dest\_block\_idx]]
    $$

- 当 `src_block_idx == dest_block_idx` 且 `state_idx == 0` 时，kernel 会把对应请求的 `num_accepted_tokens` 更新为 1，用于和 Python 参考实现中 `num_accepted_tokens_cpu[i] = 1` 的语义保持一致。
- 当 `src_block_idx == dest_block_idx` 且 `accept_token_bias == 0` 时，源和目标范围完全重合，状态拷贝是 no-op，kernel 仅保留必要的 `num_accepted_tokens` 更新。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|----------------------------|-----------|----------------------------------------------------------------------|----------------|------------|
| `num_accepted_tokens_ptr` | 输入/可选输出 | 每个请求本轮接受的 token 数。`num_accepted_tokens_out_ptr` 为空且 `src_block_idx == dest_block_idx` 时，kernel 会原地将对应请求写为 1。 | INT32 | ND |
| `mamba_state_idx_ptr` | 输入 | 每个请求当前保存 running state 的源逻辑 block 索引，即公式中的 `src_block_idx`。 | INT32 | ND |
| `num_scheduled_tokens_ptr` | 输入 | 每个请求本轮调度的 token 数。仅在 `PRECOMPUTED_NEW_COMPUTED=False` 时读取。 | INT32 | ND |
| `num_computed_tokens_ptr` | 输入 | `PRECOMPUTED_NEW_COMPUTED=False` 时表示进入后处理前的已计算 token 数；`PRECOMPUTED_NEW_COMPUTED=True` 时表示已经预先计算好的 `new_num_computed_tokens`。 | INT32 | ND |
| `num_draft_tokens_ptr` | 输入 | 每个请求本轮 draft token 数。仅在 `PRECOMPUTED_NEW_COMPUTED=False` 时读取。 | INT32 | ND |
| `block_table_ptrs_ptr` | 输入 | 每个 Mamba group 的 block table 基地址数组。每个元素是对应 group 的持久化 `int32[max_reqs, max_blocks]` block table 的 `data_ptr`。 | INT64 | ND |
| `block_table_stride_req` | 输入 | block table 中相邻请求行之间的 stride，单位为 `int32` 元素个数。 | INT64 | - |
| `state_base_addrs_ptr` | 输入/输出 | 每个 state tensor 的基地址数组。kernel 通过该地址读取源 state 并写入目标 state。数组索引为 `state_idx`。 | INT64 | ND |
| `state_block_strides_ptr` | 输入 | 每个 state tensor 中相邻物理 block 的字节跨度。 | INT64 | ND |
| `state_elem_sizes_ptr` | 输入 | 每个 state tensor 的单元素字节数。 | INT64 | ND |
| `state_inner_sizes_ptr` | 输入 | 每个 state tensor 除 block/滑窗轴之外的内部元素数量。conv state 用于计算单次滑窗拷贝元素数，temporal state 用于计算自然 block 数据大小。 | INT64 | ND |
| `state_conv_widths_ptr` | 输入 | 每个 state 的 conv width。值大于 0 表示 conv state；值等于 0 表示 temporal state。 | INT32 | ND |
| `state_group_indices_ptr` | 输入 | `state_idx` 到 Mamba group 索引的映射，用于选择对应 group 的 block table。 | INT32 | ND |
| `state_dim_row_count_ptr` | 输入 | DS conv 布局下每个 block 的 dim 行数。仅在 `CONV_STATE_DIM_FIRST=True` 且当前 state 为 conv state 时读取。 | INT32 | ND |
| `state_dim_row_stride_ptr` | 输入 | DS conv 布局下相邻 dim 行之间的字节跨度。仅在 `CONV_STATE_DIM_FIRST=True` 且当前 state 为 conv state 时读取。 | INT64 | ND |
| `num_accepted_tokens_out_ptr` | 可选输出 | `src_block_idx == dest_block_idx` 时的 `num_accepted_tokens` 更新输出缓冲区。传入空指针时改为原地更新 `num_accepted_tokens_ptr`。 | INT32 | ND |
| `idx_mapping_ptr` | 可选输入 | V2 model runner / PP 场景下的 `batch_idx -> req_idx` 映射。`HAS_IDX_MAPPING=True` 时读取；元素为 -1 表示跳过该 batch row。 | INT32 | ND |
| `num_reqs` | 输入 | 当前 batch 中活跃请求行数。该参数是运行时标量，不作为 constexpr，以避免不同 batch size 触发重复编译。 | INT32 | - |
| `block_size` | 属性 | Mamba cache 的 block size，由模型配置决定，同一次模型初始化后固定。 | INT32 | - |
| `COPY_BLOCK_SIZE` | 属性 | 字节拷贝循环的固定分块大小，用于控制每次 `tl.load` / `tl.store` 的 byte 数。 | INT32 | - |
| `CONV_STATE_DIM_FIRST` | 属性 | 表示 conv state 是否采用 DS 布局 `state[block, dim, state_len]`。True 时按 dim 行逐行拷贝；False 时按单段连续内存拷贝。 | BOOL | - |
| `HAS_IDX_MAPPING` | 属性 | True 表示 `program_id(0)` 是 batch row，需要通过 `idx_mapping_ptr` 转为 request-state slot；False 表示 `program_id(0)` 直接为请求索引。默认值为 False。 | BOOL | - |
| `PRECOMPUTED_NEW_COMPUTED` | 属性 | True 表示 `num_computed_tokens_ptr` 已经保存后处理后的 `new_num_computed_tokens`，此时不读取 `num_scheduled_tokens_ptr` 和 `num_draft_tokens_ptr`。默认值为 False。 | BOOL | - |
| Mamba state tensors | 输出 | 通过 `state_base_addrs_ptr` 间接写入的 conv/temporal state 目标 block。kernel 不返回独立 tensor，而是原地更新持久化 Mamba cache。 | FLOAT16、BFLOAT16、FLOAT32 等实际 state dtype | ND |

## 约束说明

- 该 kernel 是 vLLM Mamba 内部后处理 fused kernel，不是独立的 `torch.ops.custom`/aclnn 对外接口。正常通过 `MambaSpecDecodeGPUContext.run_fused_postprocess` 或 vLLM 上层 Mamba 后处理路径间接调用。
- 该 kernel 依赖 Triton-Ascend 运行时；310P 场景不调用该 kernel，改走 `patch_mamba_utils.py` 中的 CPU/Torch fallback。
- grid 必须为 `(num_reqs, num_layers * num_state_types)`，其中第二维长度必须和所有 state metadata 数组的扁平化长度一致。
- `num_reqs` 表示活跃 batch row 数。`HAS_IDX_MAPPING=True` 时，`req_idx` 来自 `idx_mapping_ptr[batch_idx]`，可能是稀疏 request-state slot；只对 `batch_idx` 做 `num_reqs` 边界检查，不对 `req_idx` 使用 `num_reqs` 约束。
- `idx_mapping_ptr` 中的 -1 是跳过哨兵值，仅在 `HAS_IDX_MAPPING=True` 时有效。
- `PRECOMPUTED_NEW_COMPUTED=True` 时，`num_scheduled_tokens_ptr` 和 `num_draft_tokens_ptr` 可以为空，但不能在 kernel 内被读取；`PRECOMPUTED_NEW_COMPUTED=False` 时二者必须是有效设备指针。
- `num_accepted_tokens_out_ptr` 可以为空。为空时，`src_block_idx == dest_block_idx` 的更新写回 `num_accepted_tokens_ptr`；非空时写入输出缓冲区。
- block table 的元素类型为 `INT32`，但 kernel 会在参与地址计算前将 `src_block_id` 和 `dest_block_id` 扩展为 `INT64`，避免大 Mamba cache 下 `block_id * state_block_stride` 发生 32 位溢出。
- 每个 Mamba group 拥有独立分配的物理 block table，`state_group_indices_ptr[state_idx]` 必须能映射到 `block_table_ptrs_ptr` 中的合法 group。
- `state_block_strides_ptr` 是页面跨度，可能大于实际 state 数据大小。temporal state 的拷贝大小使用 `state_inner_sizes_ptr * state_elem_sizes_ptr`，而不是直接使用 `state_block_stride`。
- `state_conv_widths_ptr[state_idx] > 0` 表示 conv state，`state_conv_widths_ptr[state_idx] == 0` 表示 temporal state。
- `CONV_STATE_DIM_FIRST=True` 且当前 state 为 conv state 时，`state_dim_row_count_ptr` 和 `state_dim_row_stride_ptr` 必须有效；否则 kernel 使用单段连续区域复制路径。
- 状态拷贝按 `uint8` 字节粒度执行。为规避 triton-ascend `PtrOffsetInfo::AxisInfo` 在循环内 pointer cast 上的分析问题，指针类型转换需要保留在循环外或行级入口处。
- 当 `src_block_idx == dest_block_idx` 且 `accept_token_bias == 0` 时，状态拷贝被视为 no-op，kernel 只保留 `num_accepted_tokens` 的语义更新。
- 输入指针、输出指针、block table 和 metadata tensor 需要位于设备侧并满足上层 `MambaSpecDecodeGPUContext` 初始化约束；不支持用该 kernel 直接处理主机侧 numpy/CPU tensor。

## 调用示例

| 调用方式 | 样例代码 | 说明 |
|:----------------------------|:-----------|:-----------|
| vLLM Mamba patch 路径 | [`patch_mamba_utils.py`](../../../patch/worker/patch_mamba_utils.py) | 非 310P 场景下将 `vllm_ascend.ops.triton.mamba.postprocess.postprocess_mamba_fused_kernel` 安装为 `vllm.v1.worker.mamba_utils.postprocess_mamba_fused_kernel`。 |
| 单卡 e2e 测试 | [`test_postprocess_mamba.py`](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_postprocess_mamba.py) | 构造 `MambaSpecDecodeGPUContext`、block table、conv/temporal state，并通过 `run_fused_postprocess` 验证 kernel 结果与 Python `postprocess_mamba` 参考实现一致。 |
| 310P fallback 语义守护 | [`test_mamba_align_fallback_310p_source.py`](../../../../tests/ut/_310p/test_mamba_align_fallback_310p_source.py) | 310P 不调用 Triton fused kernel，测试确保 CPU/Torch fallback 保留同样的决策和 state copy 语义。 |

典型调用流程如下：

```python
import torch
from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext

import vllm_ascend.patch.worker.patch_mamba_utils  # noqa: F401

# 1. 上层根据 KVCacheConfig、forward_context 和 state copy funcs 初始化 metadata。
gpu_ctx = MambaSpecDecodeGPUContext.create(
    max_num_reqs=max_num_reqs,
    kv_cache_config=kv_cache_config,
    num_state_types=2,
    device=torch.device("npu:0"),
    make_buffer=make_buffer,
)
gpu_ctx.initialize_from_forward_context(
    kv_cache_config,
    forward_context,
    mamba_state_copy_funcs,
    block_tables,
)

# 2. decode/spec-decode 后处理阶段由 context 触发 fused kernel。
gpu_ctx.run_fused_postprocess(
    num_reqs=num_reqs,
    num_accepted_tokens_gpu=num_accepted_tokens_gpu,
    mamba_state_idx_gpu=mamba_state_idx_gpu,
    num_scheduled_tokens_gpu=num_scheduled_tokens_gpu,
    num_computed_tokens_gpu=num_computed_tokens_gpu,
    num_draft_tokens_gpu=num_draft_tokens_gpu,
)
```

该示例展示的是上层推荐调用方式。直接调用 `postprocess_mamba_fused_kernel[grid](...)` 时，需要自行准备参数说明表中的所有指针、stride、state metadata、constexpr 属性和 grid 维度。
