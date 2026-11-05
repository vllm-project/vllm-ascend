# `_compute_slot_mapping_kernel` 算子文档

## 1、算子功能介绍

`_compute_slot_mapping_kernel` 是 vllm-ascend 在 Triton 上实现的 KV Cache 槽位映射算子，文件位置：`vllm_ascend/ops/triton/compute_slot_mapping.py`。

在大模型推理过程中，KV Cache 通常采用 **block-based（分块）** 的物理存储方式：逻辑上属于同一序列的 token 会被分散到若干物理块中，块号由 `block_table` 维护。Attention 算子需要拿到每个 token 在物理 KV Cache 中的 **slot id（槽位号）**，才能正确地读写对应的 K/V 向量。本算子即用于完成「逻辑位置 → 物理 slot id」的映射计算。

具体行为：

- 每个 `program_id(0) == req_idx` 处理一条请求（一个 sequence）。
- 通过 `query_start_loc_ptr` 得到该请求在 batch 中的 token 起止区间 `[start_idx, end_idx)`。
- 对区间内每个 token：
    - 读取其逻辑序列位置 `pos`；
    - 根据 `block_size` 将 `pos` 拆分为 `block_indices`（块内逻辑块号）与 `slot_offsets`（块内偏移）；
    - 通过 `block_table` 把逻辑块号转换为物理块号 `block_numbers`；
    - 计算 `slot_id = block_numbers * block_size + slot_offsets`，写回 `slot_mapping_ptr`。
- 当 `TOTAL_CP_WORLD_SIZE > 1`（启用 Context Parallel，CP）时，会按 `CP_KV_CACHE_INTERLEAVE_SIZE` 交错布局把虚拟块切分到各 rank：仅属于本 rank 的 token 写入真实 slot id，其余写入 `PAD_ID`（详见下文「多 CP 域下的计算说明」）。
- **最后一个 program**（`req_idx == num_programs(0) - 1`）专门用于把 `slot_mapping_ptr` 中 `[num_tokens, max_num_tokens)` 这段填充为 `PAD_ID`

### 多 CP 域下的计算说明

当启用 Context Parallel（`TOTAL_CP_WORLD_SIZE > 1`）时，KV Cache 会以 **interleave（交错）** 方式分散存储到各 rank 上：序列中第 `i` 个 token 的 K/V 始终存放在 `dcp_rank == (i // CP_KV_CACHE_INTERLEAVE_SIZE) % TOTAL_CP_WORLD_SIZE` 的设备上。为保证各 rank 仅访问本地显存，本算子在 CP 模式下引入「虚拟块」概念并按以下步骤完成映射：

1. **虚拟块划分**：将 `TOTAL_CP_WORLD_SIZE` 个 rank 的物理块在逻辑上合并为一个虚拟块
   - `virtual_block_size = KV_CACHE_BLOCK_SIZE * TOTAL_CP_WORLD_SIZE`
   - `virtual_block_indices = pos // virtual_block_size`（虚拟块号）
   - `virtual_block_offsets = pos % virtual_block_size`（虚拟块内偏移）

2. **归属判定**：通过 interleave 粒度判断当前 token 是否属于本 rank
   - `is_local = (virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK`
   - `is_local == False` 的 token 最终写入 `PAD_ID`，不占用本 rank 的 `block_table` 访问。

3. **本地偏移换算**：将虚拟偏移中属于本 rank 的部分抽取为本地物理偏移
   - `local_block_offsets = (virtual_block_offsets // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)) * CP_KV_CACHE_INTERLEAVE_SIZE + (virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE)`
   - 该公式把每个虚拟块内本 rank 拥有的若干 interleave 段拼接成连续的本地偏移。

4. **逻辑块号与块内偏移**：基于本地偏移计算 `block_table` 索引
   - `block_indices = virtual_block_indices * BLOCKS_PER_KV_BLOCK + local_block_offsets // block_size`
   - `slot_offsets = local_block_offsets % block_size`

5. **物理块号查询与 slot id 组装**：通过 `block_table` 把逻辑块号转换为物理块号
   - `block_numbers = gather(block_table, block_indices)`
   - `slot_ids = block_numbers * block_size + slot_offsets`
   - 最终写入：`slot_ids = where(is_local, slot_ids, PAD_ID)`

---

## 2、参数含义介绍

### 2.1 运行时参数（设备侧张量/标量）

| 参数名 | 形状 / 类型 | 含义 |
| --- | --- | --- |
| `num_tokens` | int | 当前 batch 中实际的 token 数量。 |
| `max_num_tokens` | int | 预分配的最大 token 数。 |
| `query_start_loc_ptr` | `[num_reqs + 1]`, int32 | 每条请求在 batch 中的累加起始位置，类似 CSR 的 `indptr`，`req_idx` 的 token 区间为 `[query_start_loc[req_idx], query_start_loc[req_idx+1])`。 |
| `positions_ptr` | `[num_tokens]`, int64 | 每个 token 在其所属请求中的逻辑序列位置。 |
| `block_table_ptr` | `[max_num_reqs, max_num_blocks_per_req]`, int32（展平） | 块表，将「逻辑块号」映射为「物理块号」；按行展开存储。 |
| `block_table_stride` | int | `block_table` 一行的元素个数，即 `max_num_blocks_per_req`。 |
| `block_size` | int | Attention 内核使用的逻辑块大小（一个逻辑块包含多少 token）。 |
| `slot_mapping_ptr` | `[max_num_tokens]`, int32 | 输出张量，写入每个 token 对应的物理 KV Cache slot id。 |

### 2.2 编译期常量参数（`tl.constexpr`）

| 参数名 | 含义 |
| --- | --- |
| `KV_CACHE_BLOCK_SIZE` | KV Cache 物理分配块大小（物理 block 的大小）。 |
| `BLOCKS_PER_KV_BLOCK` | 一个物理 KV 块包含多少个逻辑 block，满足 `KV_CACHE_BLOCK_SIZE = BLOCKS_PER_KV_BLOCK * block_size`。 |
| `TOTAL_CP_WORLD_SIZE` | Context Parallel 通信域的总 rank 数；为 `1` 时走非 CP 路径。 |
| `TOTAL_CP_RANK` | 当前设备在 CP 通信域中的 rank。 |
| `CP_KV_CACHE_INTERLEAVE_SIZE` | CP 模式下 KV Cache 的交错大小，用于将虚拟块按 interleave 粒度切分到各 rank。 |
| `PAD_ID` | 无效槽位的填充值（通常为 `PAD_SLOT_ID = -1`），用于 padding 区域与非本 rank 槽位。 |
| `TILE_BLOCK_SIZE` | Triton 循环中的 tile 大小，控制单次迭代处理的 token 数。 |
| `BLOCK_TABLE_WINDOW_SIZE` | 一次性加载到寄存器的 block_table 窗口大小（需 ≥ `TILE_BLOCK_SIZE / block_size + 1`，并取 2 的幂以便 Triton 向量化）。 |

### 2.3 启动网格

```python
grid = (num_reqs + 1,)
```

前 `num_reqs` 个 program 分别处理一条请求，最后一个 program 负责 padding。

---

## 3、算子使用示例

假设：

- `block_size = 128`，`KV_CACHE_BLOCK_SIZE = 128`，未启用 CP（`TOTAL_CP_WORLD_SIZE = 1`）；
- 当前 batch 有 2 条请求，token 区间分别为 `[0, 5)` 与 `[5, 10)`；
- `max_num_batched_tokens = 8192`，需要把多余位置填充为 `PAD_ID`。

```python
import torch
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.ops.triton.compute_slot_mapping import (
    _compute_slot_mapping_kernel,
    _next_power_of_2,
)

device = "npu"


# ---- 1. 输入张量 ----
num_reqs = 2
num_tokens = 10
max_num_batched_tokens = 8192
block_size = 128

# 每条请求的 token 起止位置（CSR 风格 indptr）
query_start_loc = torch.tensor([0, 5, 10], dtype=torch.int32, device=device)

# 每个 token 在其请求中的逻辑位置
positions = torch.tensor(
    [0, 1, 2, 3, 4,   # 请求 0
     0, 1, 2, 3, 4],  # 请求 1
    dtype=torch.int64, device=device,
)

# 块表：逻辑块号 -> 物理块号，形状 [max_num_reqs, max_num_blocks_per_req]
max_num_reqs = 64
max_num_blocks_per_req = 320
block_table = torch.randint(
    0, 320, (max_num_reqs, max_num_blocks_per_req), dtype=torch.int32, device=device
)

# ---- 2. 输出张量 ----
slot_mapping = torch.zeros(max_num_batched_tokens, dtype=torch.int32, device=device)

# ---- 3. 编译期常量 ----
KV_CACHE_BLOCK_SIZE = 128
BLOCKS_PER_KV_BLOCK = KV_CACHE_BLOCK_SIZE // block_size   # = 1
TILE_BLOCK_SIZE = 1024
BLOCK_TABLE_WINDOW_SIZE = _next_power_of_2(
    cdiv(TILE_BLOCK_SIZE, block_size) + 1
)

# ---- 4. 启动 kernel ----
grid = (num_reqs + 1,)
_compute_slot_mapping_kernel[grid](
    num_tokens,
    max_num_batched_tokens,
    query_start_loc,
    positions,
    block_table,
    block_table.stride(0),
    max_num_blocks_per_req
    block_size,
    slot_mapping,
    KV_CACHE_BLOCK_SIZE=KV_CACHE_BLOCK_SIZE,
    BLOCKS_PER_KV_BLOCK=BLOCKS_PER_KV_BLOCK,
    TOTAL_CP_WORLD_SIZE=1,
    TOTAL_CP_RANK=0,
    CP_KV_CACHE_INTERLEAVE_SIZE=1,
    PAD_ID=PAD_SLOT_ID,                # 通常为 -1
    TILE_BLOCK_SIZE=TILE_BLOCK_SIZE,
    BLOCK_TABLE_WINDOW_SIZE=BLOCK_TABLE_WINDOW_SIZE,
)

# slot_mapping[:num_tokens] 为各 token 的物理 slot id
# slot_mapping[num_tokens:max_num_batched_tokens] 被填充为 PAD_ID
print(slot_mapping[:num_tokens])
```

### 启用 Context Parallel 的注意事项

当 `TOTAL_CP_WORLD_SIZE > 1` 时：

- 物理块大小被虚拟化为 `virtual_block_size = KV_CACHE_BLOCK_SIZE * TOTAL_CP_WORLD_SIZE`；
- 仅 `is_local == True`（属于本 `TOTAL_CP_RANK`）的 token 才会写入真实 slot id，其余写 `PAD_ID`；
- `block_indices` 与 `slot_offsets` 通过 `local_block_offsets` 重新换算，需要保证 `BLOCKS_PER_KV_BLOCK` 与 `CP_KV_CACHE_INTERLEAVE_SIZE` 配置正确。
