# DeepSeek V4 层间混合组件复用适配方案

## 1. 背景

层间 KV Cache 复用允许执行时序不重叠的层共用物理内存。当前
vLLM-Ascend main 的层复用 planner 会构造共享成员列表，并将其传给
`KVCacheTensor.layers`：

```python
shared_by = [layer_name_a, layer_name_b]
KVCacheTensor(layers=shared_by, ...)
```

在本文讨论的 Ascend 层复用路径中，`layers` 表示这些逻辑组件应使用同一块
KV Cache 原始内存。它与旧版本 `KVCacheTensor.shared_by` 的共享语义一致，
只是字段名称和构造接口发生了变化。

本文只适用于这种 shared-by 风格的分配路径：一个输出 descriptor 对应一次
物理分配，`layers` 中的成员绑定到该分配。把 `layers` 解释为公共 backing 中
多个独立 packed region 的路径不在本次适配范围内。

DeepSeek V4 的不同物理层可能使用不同压缩比、dtype 或 scale 布局。只要这些层
位于同一个时序复用 slot，并且对应组件能够从一段连续 raw tensor 构造自己的
view，它们就可以复用同一块原始内存。共享内存按 lane 中最大的组件分配，每个
组件只使用自身所需的前缀。

## 2. 当前 DeepSeek V4 的真实表示

### 2.1 Indexer cache spec

当前 DeepSeek V4 的 `AscendDeepseekV4IndexerCache.get_kv_cache_spec()` 返回
`AscendMLAAttentionSpec`，不是 `AscendSFAIndexerCacheSpec`。

它通过以下字段描述 BF16/C8 等差异：

```text
dtype
scale_dim
scale_dtype
tokens_per_state
page_size_bytes
```

model runner 首先给该组件一段连续的 `torch.int8` raw tensor，随后现有
`_adjust_kv_layout()` 根据该组件自己的 spec 生成 K、scale 以及需要的完整
cache view。因此，本方案不会在 `kv_cache_raw_tensors` 中提前构造
`(K, scale)` tuple。

### 2.2 每个物理层可能包含多个组件

一个 DeepSeek V4 DSA 物理层可能包含：

```text
.attn
.indexer.k_cache
.indexer.compressor.state_cache
.compressor.state_cache
.swa_cache
```

并不是每个物理层都一定包含全部五个组件。planner 必须处理实际出现的所有
named spec，不能只处理 `main` 和 `indexer`，也不能丢弃当前保存在
`extra_main_specs` 中的组件。

## 3. 目标

本次适配实现以下能力：

1. planner 接受一个输入 `KVCacheTensor` 描述多个 layer/component；
2. planner 枚举每个物理层实际存在的全部 cache 组件；
3. 执行时序不重叠且 raw 表示兼容的不同 spec 组件可以进入同一个 lane；
4. 输出 descriptor 使用 `KVCacheTensor.layers` 表达共享成员；
5. model runner 为一个共享 descriptor 只申请一块 raw tensor；
6. descriptor 中的每个组件只绑定自身所需长度的 raw tensor view；
7. 现有 `_adjust_kv_layout()` 继续根据各组件 spec 生成最终 view；
8. 暂不支持跨类型共享的组件仍然完整保留；
9. 相同 spec 的现有路径和非层复用场景保持不变。

## 4. 非目标

本次适配不处理：

- 通过 `offset`、`layer_stride` 或 `block_stride` 表达的 packed layout；
- 任意 `AttentionSpec`、`MambaSpec` 和 hidden-state cache 之间的跨类型共享；
- 同一时刻可能被多个物理层访问的内存共享；
- 改变 KV Cache block 数量、scheduler block 管理或传输协议；
- 改变现有 KV Cache alignment；
- 为未来 cache 类型设计通用内存布局框架。

## 5. 术语

- **物理层**：模型执行顺序中的一层，包括普通 transformer 层和参与执行的
  MTP 层。
- **组件**：一个具有独立 layer name 和 cache spec 的逻辑 cache，例如
  `.attn` 或 `.indexer.k_cache`。
- **slot**：一组执行时序不重叠、允许复用内存的物理层。
- **role**：组件名去掉物理层编号后得到的语义名称，例如
  `.indexer.k_cache`。
- **lane**：同一个 slot 中，role 相同且 raw 表示兼容的一组组件。
- **descriptor**：一个 `KVCacheTensor`。本文范围内，其 `layers` 列出
  共用同一块 raw tensor 的组件名称。
- **raw tensor**：model runner 实际申请的 `torch.int8` 字节内存。

## 6. 设计边界

### 6.1 planner 决定共享关系

planner 负责：

- 根据物理层执行时序建立 slot；
- 枚举 slot 中每个物理层的全部组件；
- 根据 role 和支持的 raw 表示建立 lane；
- 计算每个 lane 所需的最大字节数；
- 输出 `KVCacheTensor(layers=[...], size=...)`。

planner 不申请 device tensor，也不生成 attention backend 最终使用的 view。

### 6.2 model runner 执行共享计划

model runner 负责：

- 为一个输出 descriptor 申请一次 raw tensor；
- 将该 raw tensor 的正确长度前缀绑定给 descriptor 中的每个组件；
- 将绑定结果保存到 `kv_cache_raw_tensors[layer_name]`；
- 继续使用现有 reshape 逻辑生成最终 KV Cache view。

model runner 不重新推导层复用关系，也不改写 descriptor 的 `layers`。

## 7. 组件兼容规则

第一版只增加 DeepSeek V4 连续 raw tensor 的跨-spec共享，不推广到任意 cache
类型。

### 7.1 DeepSeek V4 连续 raw 组件

同时满足以下条件的组件可以进入同一个 lane：

- 属于同一个时序复用 slot；
- role 相同；
- spec 是 DeepSeek V4 使用连续 raw tensor 的
  `AscendMLAAttentionSpec`；
- 现有 reshape 路径能够仅根据该组件自己的 spec 解释一段连续 raw tensor。

它们可以具有不同的：

```text
dtype
scale_dim
scale_dtype
tokens_per_state
block_size
head_size
page_size_bytes
```

这些字段决定每个组件自身的 raw 长度和最终 view，不要求共享成员具有相同形状。
例如 C4/C128 main cache，或者具有不同 dtype/scale 布局的
`.indexer.k_cache`，都可以在 role 相同且生命周期不重叠时使用同一个 lane。

对应的 key 保持简单：

```python
reuse_key = (role, "dsv4_contiguous_raw")
```

这里不把 block size、head size 或 dtype 放入 key，因为每个成员会绑定自己的
精确长度，并由自己的 spec 完成 reshape。

### 7.2 其他组件

其他组件第一版采用保守规则：

```python
reuse_key = (role, type(spec), spec)
```

也就是只有 role、spec 类型和 spec 值全部相同时才进入同一个 lane。不同的
state cache 或 SWA spec 不做跨类型共享，但必须保留为其他 lane 或 singleton
lane，不能从输出中丢失。

如果某种 spec 当前分配路径不支持共享，即使 spec 相同，也将其拆成 singleton
lane。DeepSeek V4 的 state/SWA cache 属于 attention spec，规格完全相同时可走
现有 compressed-cache 分配路径共享；规格不同时仍分开。第一版不为其他 cache
类型增加新的通用 binder。

## 8. planner 适配

### 8.1 展开输入 descriptor

输入 descriptor 可能已经包含多个 layer/component。planner 应遍历完整的
`layers` 列表：

```python
tensors_by_name = {}
for tensor in old_tensors:
    layers = get_kv_cache_tensor_layers(tensor)
    if not layers:
        raise ValueError("KV cache tensor descriptor has no layers")
    for layer_name in layers:
        if layer_name in tensors_by_name:
            raise ValueError(
                f"{layer_name} belongs to multiple KV cache tensors"
            )
        tensors_by_name[layer_name] = tensor
```

输入 descriptor 原有的共享分组只用于找到每个组件的来源。最终共享分组由新的
slot/lane 计划替换，不能将旧分组和新分组求并集，否则可能形成错误的传递共享。

### 8.2 枚举全部组件

planner 不再只读取每个物理层的 `main` 和 `indexer`。它需要遍历该物理层
实际存在的全部 `NamedKVCacheSpec`：

```python
for slot_id, physical_layers in enumerate(buffer_slots):
    for physical_layer in physical_layers:
        for named_spec in layer_cache_specs[physical_layer]:
            ...
```

现有 `main/indexer/extra_main_specs` 可以在过渡期继续作为解析结果，但生成 lane
时必须把三部分全部展开。更直接的实现是让
`layer_cache_specs[physical_layer]` 保存一个 named spec tuple。本次不要求为
此增加额外辅助类型。

### 8.3 按组件建立 lane

支持共享的组件按以下 key 分组：

```python
(slot_id, reuse_key)
```

不支持共享的组件为每个组件生成独立 key，从而保留为 singleton lane。

### 8.4 验证组件集合

生成 lane 后必须满足：

```python
planned_names == set(tensors_by_name)
```

该检查保证：

- 输入 descriptor 中的每个组件都进入了输出计划；
- `extra_main_specs` 不会被遗漏；
- planner 不会生成输入中不存在的组件；
- 同一个组件不会出现在多个输出 lane 中。

### 8.5 生成输出 descriptor

每个 lane 生成一个输出 descriptor：

```python
KVCacheTensor(
    layers=[component.layer_name for component in components],
    size=max(
        component.allocation_size_bytes
        for component in components
    ),
)
```

对于 DeepSeek V4 连续 raw tensor：

```python
component.allocation_size_bytes = (
    spec.page_size_bytes * kv_cache_config.num_blocks
)
```

这是该组件传给现有 reshape 路径的完整 raw tensor 长度。scheduler 和
`KVCacheManager` 仍只管理 `kv_cache_config.num_blocks` 个 block，不使用
输入 descriptor 中可能存在的额外容量扩大 block 数量。

如果后续支持包含多个内部 raw view 且需要额外 padding 的组件，
`allocation_size_bytes` 必须取所有 view 的最大
`offset_bytes + size_bytes`，不能只取各 view 大小之和。本次 DSV4 连续 raw
tensor 不需要这类内部 padding 计算。

## 9. model runner 适配

### 9.1 触发条件

DeepSeek V4 层复用分配路径只处理满足以下条件的 descriptor：

- planner 已经实际改写了本次 KV Cache 配置；
- descriptor 中所有成员都是 DeepSeek V4 连续 raw 组件；
- descriptor size 能容纳所有成员的 `allocation_size_bytes`。

planner 是否实际改写通过函数返回值在初始化调用链中局部传递，不保存为 model
runner 的长期状态，也不通过 `offset`、`layer_stride` 等几何字段猜测。
其他类型的 descriptor 继续走各自现有分配路径。

### 9.2 申请并绑定公共 raw tensor

一个混合 lane 只申请一次内存：

```python
allocation_size = max(
    component.allocation_size_bytes
    for component in components
)
raw_tensor = self._allocate_int8_cache_tensor(
    allocation_size,
    KV_CACHE_TENSOR_ALIGNMENT,
)
```

然后给每个成员绑定其自身所需的前缀：

```python
for component in components:
    component_raw = raw_tensor[
        : component.allocation_size_bytes
    ]
    kv_cache_raw_tensors[component.layer_name] = component_raw
```

这些 component raw tensor 的长度可以不同，但底层 storage 相同。因为 slot 中的
物理层不会同时使用该内存，不同成员可以在各自执行期间覆盖同一段字节。

不能把完整的最大 raw tensor 直接交给较小组件。否则现有代码通过
`raw_tensor.numel() / spec.page_size_bytes` 推导 block 数量时，较小组件可能
得到大于 `kv_cache_config.num_blocks` 的错误形状。

### 9.3 保留现有 reshape

绑定后，每个 DSV4 组件在 `kv_cache_raw_tensors` 中仍然对应一个
`torch.Tensor`：

```text
BF16 component -> exact-length raw tensor view
C8 component   -> exact-length raw tensor view
```

后续 `_reshape_kv_cache_tensors()` 根据当前 layer 的
`AscendMLAAttentionSpec` 调用现有 `_adjust_kv_layout()`，生成该组件需要的
K、scale 和其他 view。本次不重新实现这段 reshape 逻辑。

## 10. 示例

假设物理层复用关系为：

```text
Slot A: Layer1, Layer3
Slot B: Layer2, Layer4
```

每个层包含 main 组件，部分层还包含 indexer 和 state 组件：

```text
Layer1.main:    C4，  100 MiB
Layer1.indexer: BF16，80 MiB
Layer1.state:   StateSpec-A，16 MiB

Layer2.main:    C128，60 MiB
Layer2.state:   StateSpec-B，8 MiB

Layer3.main:    C128，60 MiB
Layer3.indexer: C8，  48 MiB
Layer3.state:   StateSpec-D，16 MiB

Layer4.main:    C4，  100 MiB
Layer4.state:   StateSpec-C，12 MiB
```

planner 输出：

```text
Lane (A, main):
  layers = [Layer1.main, Layer3.main]
  size   = max(100 MiB, 60 MiB) = 100 MiB

Lane (A, indexer):
  layers = [Layer1.indexer, Layer3.indexer]
  size   = max(80 MiB, 48 MiB) = 80 MiB

Lane (A, Layer1.state):
  layers = [Layer1.state]
  size   = 16 MiB

Lane (A, Layer3.state):
  layers = [Layer3.state]
  size   = 16 MiB

Lane (B, main):
  layers = [Layer2.main, Layer4.main]
  size   = max(60 MiB, 100 MiB) = 100 MiB

Lane (B, StateSpec-B):
  layers = [Layer2.state]
  size   = 8 MiB

Lane (B, StateSpec-C):
  layers = [Layer4.state]
  size   = 12 MiB
```

对于 Lane A 的 indexer，model runner 只申请一块 80 MiB raw tensor：

```text
Layer1.indexer -> raw_tensor[:80 MiB]
Layer3.indexer -> raw_tensor[:48 MiB]
```

两者共享底层 storage，之后分别由自己的 spec 和现有
`_adjust_kv_layout()` 生成最终 view。这个示例假定这些 state cache 规格不同，
因此它们保守地保留为 singleton。如果 Layer1.state 和 Layer3.state 是完全
相同且现有分配路径支持共享的 attention spec，两者也可以形成一个 exact-spec
lane。所有 state 组件都会完整保留。

## 11. 错误处理

以下情况应直接报错：

- descriptor 的 `layers` 为空；
- 一个组件名出现在多个输入 descriptor 中；
- planner 组件集合和输入 descriptor 组件集合不一致；
- layerwise DSV4 descriptor 含有不支持的 spec；
- 组件绑定长度超过公共 raw tensor。

以下情况保持不共享即可：

- role 不同；
- 非 DSV4 连续 raw 组件且 spec 不同；
- 当前 allocator 不支持共享的 cache 类型；
- slot 中某个组件没有兼容成员；
- 配置没有形成真实的跨层 lane。

## 12. 测试方案

### 12.1 planner 单元测试

1. 使用真实的 DSV4 `AscendMLAAttentionSpec` 构造测试；
2. C4/C128 main spec 按 slot 进入同一 main lane；
3. 不同 dtype/scale 布局的 indexer spec 进入同一 indexer lane；
4. lane descriptor size 取成员 `allocation_size_bytes` 的最大值；
5. 一个输入 descriptor 包含多个 `layers` 时，全部组件都被展开；
6. 一个 DSV4 物理层的五类组件均被保留；
7. 不兼容 state/SWA 组件保留为不同 lane 或 singleton；
8. 输入组件重复时报错；
9. planner 输出组件集合与输入集合完全一致；
10. MTP layer 与普通层一样参与 slot/lane 规划。

### 12.2 model runner 单元测试

1. 混合 DSV4 lane 只调用一次 raw tensor 分配；
2. 不同长度的 component raw view 具有相同底层 storage；
3. 每个 component raw view 的长度等于其
   `allocation_size_bytes`；
4. 较小组件保持 `kv_cache_config.num_blocks`，不会得到额外 block；
5. 现有 `_adjust_kv_layout()` 能从各自 raw view 生成正确结果；
6. 相同 spec descriptor 仍走原有路径；
7. singleton state/SWA descriptor 仍走原有路径；
8. 不启用层复用时分配结果不变。

检查共享关系时应比较 raw view 的底层 storage，例如
`untyped_storage().data_ptr()`；不能要求所有子 view 的
`Tensor.data_ptr()` 都相同。

### 12.3 集成验证

至少验证一个同时包含 C4、C128、BF16/C8 indexer、state cache 和 MTP 的
DeepSeek V4 配置：

- 初始化 KV Cache 成功；
- 实际 allocation 数量与输出 descriptor 计划一致；
- 共享成员使用相同底层 storage；
- singleton 或不同 lane 使用不同 storage；
- 所有输入组件都能在最终 `kv_caches` 中找到；
- prefill、decode 和 MTP 结果正确；
- layerwise KV transfer 循环运行时没有地址越界、数据污染或 3102 非法访问。

## 13. 修改范围

修改限制在以下位置：

1. `layerwise_cache_layout.py`
   - 展开输入 descriptor 的全部 `layers`；
   - 枚举每个物理层的全部 named spec；
   - 为 DSV4 连续 raw 组件生成兼容 key；
   - 其他组件使用保守 key 或 singleton；
   - 按 `(slot_id, reuse_key)` 生成 lane；
   - 验证输入和输出组件集合一致；
   - 输出最大尺寸的共享 descriptor。
2. `model_runner_v1.py`
   - 通过 planner 的局部返回值区分标准 DSV4 layout 和层复用计划；
   - 每个 DSV4 组件 lane descriptor 申请一次 raw tensor；
   - 为每个成员绑定自身所需的前缀；
   - 继续使用现有 `_adjust_kv_layout()`。
3. 对应单元测试
   - 覆盖全部组件保留、planner 分组、raw storage 共享和 reshape 接入。

不修改 scheduler、block manager、connector 协议和现有 alignment 配置。

## 14. 实施顺序

1. 先让 planner 展开多-owner descriptor，并保证全部组件原样保留；
2. 将 lane 规划从 `main/indexer` 两类扩展为遍历全部 named spec；
3. 增加 DSV4 连续 raw 组件兼容 key 和 planner 测试；
4. 增加 model runner 的单 raw tensor、不同长度前缀绑定；
5. 增加 storage、组件完整性和 reshape 测试；
6. 最后进行 DeepSeek V4 NPU 集成验证。

## 15. 结论

本方案不引入新的通用 layout API，也不处理 packed descriptor。
`KVCacheTensor.layers` 继续表达共享成员，planner 负责生成完整且安全的 lane，
model runner 负责申请一次 raw tensor，并按每个组件的真实长度绑定 view。

针对当前 DeepSeek V4，适配重点是：

1. 使用真实的 `AscendMLAAttentionSpec` 和连续 raw tensor 路径；
2. 枚举并保留物理层的全部 cache 组件；
3. C4/C128 或不同 dtype/scale 布局通过各自 spec 解释同一底层 storage；
4. 暂不支持跨类型共享的 state/SWA 组件保持保守 lane 或 singleton。

这样既能完成 DeepSeek V4 组件级层复用，也把修改限制在 planner、现有 DSV4
分配路径和对应测试中。
