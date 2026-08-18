# `compute_slot_mapping_fused_kernel` 算子文档

## 1、算子功能介绍

`compute_slot_mapping_fused_kernel` 是 vllm-ascend 在 Triton 上实现的 **多组（multi-group）融合 KV Cache 槽位映射算子**，文件位置：`vllm_ascend/ops/triton/slot_mapping.py`。

当模型存在多个 KV Cache 组（例如 MLA 场景下主 KV Cache 与每个 expert 的 KV Cache）时，vLLM 需要为每组分别调用单组 `_compute_slot_mapping_kernel` 完成「逻辑位置 → 物理 slot id」的映射。在 Ascend NPU 上，多次 kernel 启动会带来可观的启动开销。本算子将 **N 组单组 kernel 启动融合为一次 2D-grid 启动**：

- `grid = (num_reqs + 1, num_groups)`，其中 `program_id(0)` 对应请求、`program_id(1)` 对应 KV Cache 组；
- 各组通过预缓存的参数指针数组（block_table 指针、stride、block_size、slot_mapping 指针等）直接定位各自的输入/输出；
- 每个组内的单组计算逻辑与 `_compute_slot_mapping_kernel`（`vllm_ascend/ops/triton/compute_slot_mapping.py`，PR #13048）完全对齐：
    - `pos` 强转为 int32 以降低标量运算开销；
    - `TOTAL_CP_WORLD_SIZE == 1` 时走专门的非 CP 快速路径（无交错计算）；
    - 使用窗口化 block_table 加载 + `tl.gather` 修复非连续访问。

生产路径中，`MultiGroupBlockTable`（`vllm_ascend/worker/block_table.py`）会在 `__init__` 阶段预缓存各组参数张量，并在解码热路径中直接调用本 kernel，避免每次 `torch.tensor(…)` 构造开销。测试用途的 `launch_slot_mapping_fused` 帮助函数位于 `tests/ut/ops/a2/test_slot_mapping.py`。

### 多组融合的计算说明

融合 kernel 按组（`program_id(1)`）并行执行与单组 kernel 相同的计算：

1. **请求区间**：`req_idx = program_id(0)`，通过 `query_start_loc_ptr` 得到该请求的 token 区间 `[start_idx, end_idx)`；
2. **组内映射**：`group_idx = program_id(1)`，从各参数指针数组取第 `group_idx` 个元素得到本组的 block_table / block_size / slot_mapping；
3. **CP 处理**：当 `TOTAL_CP_WORLD_SIZE > 1` 时，按与单组 kernel 相同的 interleave 规则判断 token 归属，非本 rank 的 token 写入 `PAD_ID`；
4. **padding**：最后一个 `program_id(0) == num_reqs` 的 program 负责把各组 `slot_mapping` 中 `[num_tokens, max_num_tokens)` 区间填充为 `PAD_ID`。

---

## 2、参数含义介绍

### 2.1 运行时参数（设备侧张量/标量）

| 参数名 | 形状 / 类型 | 含义 |
| --- | --- | --- |
| `num_tokens` | int | 当前 batch 中实际的 token 数量。 |
| `max_num_tokens` | int | 预分配的最大 token 数（各组 `slot_mapping` 的容量）。 |
| `query_start_loc_ptr` | `[num_reqs + 1]`, int32 | 每条请求在 batch 中的累加起始位置，`req_idx` 的 token 区间为 `[query_start_loc[req_idx], query_start_loc[req_idx+1])`。 |
| `positions_ptr` | `[num_tokens]`, int64 | 每个 token 在其所属请求中的逻辑序列位置。 |
| `group_block_table_ptrs` | `[num_groups]`, int64 | 各组 block_table 的 GPU 数据指针。 |
| `group_block_table_strides` | `[num_groups]`, int32 | 各组 block_table 一行的元素个数（`max_num_blocks_per_req`）。 |
| `group_block_sizes` | `[num_groups]`, int32 | 各组 Attention 内核使用的逻辑块大小。 |
| `group_slot_mapping_ptrs` | `[num_groups]`, int64 | 各组输出 slot_mapping 的 GPU 数据指针。 |
| `group_kv_cache_block_sizes` | `[num_groups]`, int32 | 各组 KV Cache 物理分配块大小。 |
| `group_blocks_per_kv` | `[num_groups]`, int32 | 各组一个物理 KV 块包含的逻辑块数量。 |

### 2.2 编译期常量参数（`tl.constexpr`）

| 参数名 | 含义 |
| --- | --- |
| `TOTAL_CP_WORLD_SIZE` | Context Parallel 通信域的总 rank 数；为 `1` 时走非 CP 路径。 |
| `TOTAL_CP_RANK` | 当前设备在 CP 通信域中的 rank。 |
| `CP_KV_CACHE_INTERLEAVE_SIZE` | CP 模式下 KV Cache 的交错大小。 |
| `PAD_ID` | 无效槽位的填充值（通常为 `PAD_SLOT_ID = -1`），用于 padding 区域与非本 rank 槽位。 |
| `TILE_BLOCK_SIZE` | Triton 循环中的 tile 大小，控制单次迭代处理的 token 数。 |
| `BLOCK_TABLE_WINDOW_SIZE` | 一次性加载到寄存器的 block_table 窗口大小（需 ≥ `TILE_BLOCK_SIZE / min_block_size + 1`，并取 2 的幂以便 Triton 向量化）。 |

### 2.3 启动网格

```python
grid = (num_reqs + 1, num_groups)
```

前 `num_reqs` 行分别处理一条请求，最后一行负责 padding；`num_groups` 列对应各组 KV Cache。

---

## 3、算子使用示例

生产环境中，`MultiGroupBlockTable.compute_slot_mapping_fused`（`vllm_ascend/worker/block_table.py`）负责预缓存参数并启动 kernel：

```python
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.ops.triton.slot_mapping import compute_slot_mapping_fused_kernel

device = "npu"

# ---- 1. 预缓存参数（生产路径由 MultiGroupBlockTable.__init__ 完成）----
num_groups = 2
group_block_table_ptrs = torch.tensor([...], dtype=torch.int64, device=device)
group_block_table_strides = torch.tensor([...], dtype=torch.int32, device=device)
group_block_sizes = torch.tensor([...], dtype=torch.int32, device=device)
group_slot_mapping_ptrs = torch.tensor([...], dtype=torch.int64, device=device)
group_kv_cache_block_sizes = torch.tensor([...], dtype=torch.int32, device=device)
group_blocks_per_kv = torch.tensor([...], dtype=torch.int32, device=device)

# ---- 2. 输入张量 ----
num_reqs = 2
num_tokens = 8
max_num_batched_tokens = 512
query_start_loc = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)
positions = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int64, device=device)

# ---- 3. 编译期常量 ----
TILE_BLOCK_SIZE = 1024
BLOCK_TABLE_WINDOW_SIZE = 32  # _next_power_of_2(TILE_BLOCK_SIZE / min_block_size + 1)

# ---- 4. 启动融合 kernel（一次启动处理所有组）----
grid = (num_reqs + 1, num_groups)
compute_slot_mapping_fused_kernel[grid](
    num_tokens,
    max_num_batched_tokens,
    query_start_loc,
    positions,
    group_block_table_ptrs,
    group_block_table_strides,
    group_block_sizes,
    group_slot_mapping_ptrs,
    group_kv_cache_block_sizes,
    group_blocks_per_kv,
    TOTAL_CP_WORLD_SIZE=1,
    TOTAL_CP_RANK=0,
    CP_KV_CACHE_INTERLEAVE_SIZE=1,
    PAD_ID=PAD_SLOT_ID,                # 通常为 -1
    TILE_BLOCK_SIZE=TILE_BLOCK_SIZE,
    BLOCK_TABLE_WINDOW_SIZE=BLOCK_TABLE_WINDOW_SIZE,
)
```

### 单元测试参考

独立的 `launch_slot_mapping_fused` 帮助函数（按需构造参数数组的便捷封装）以及融合结果与逐组计算的等价性验证，见 `tests/ut/ops/a2/test_slot_mapping.py`。
