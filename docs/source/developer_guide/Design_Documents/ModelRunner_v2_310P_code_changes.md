# 310P Model Runner V2 代码改动说明

本文说明 310P Model Runner V2 第一版重构及后续联调修复涉及的代码文件、改动内容和改动原因。

## 问题与修复总览

| 阶段 | 报错或问题 | 根因 | 修复位置 |
| --- | --- | --- | --- |
| 配置创建 | `Model Runner V2 requires Triton` | 上游在 Worker 创建前执行全局 `HAS_TRITON` 门禁 | `patch_use_v2_model_runner.py` |
| KV Cache reshape | `shape ... is invalid for input of size ...` | 公共 reshape 逻辑把 NZ 尾部物理分形维度 `16` 误替换为逻辑 `head_size_v` | `worker/v2/attn_utils.py` |
| KV Cache 算子 | `ReshapeAndCacheOperation CheckIniMatch Failed`，cache 实际为 ND | 对 raw tensor 执行 `view()` 只能改变逻辑 shape，不能生成 FRACTAL_NZ 物理格式 | `_310p/worker/v2/model_runner.py` |
| ACLGraph 初始化 | `NPUModelRunner310V2` 缺少 `uniform_decode_query_len` | 当前配套 vLLM V2 接口版本只保证存在 `decode_query_len` | `_310p/worker/v2/model_runner.py` |
| ACLGraph capture | `The pageable memory copy task does not support graph capture` | Decode metadata 使用 pageable CPU `seq_lens`，attention 在图内执行 H2D | `_310p/attention/metadata_builder.py` |
| Eager prefill | `tensor.hostData is null` | 前一版 ACLGraph 修复无条件把 PrefillNoCache 的 host `seq_lens` 替换成了 NPU tensor | `_310p/attention/metadata_builder.py` |

## 1. 310P Worker 入口

### `vllm_ascend/_310p/worker_310p.py`

改动内容：

- 新增 `_create_model_runner()`。
- `use_v2_model_runner=True` 时延迟导入并创建 `NPUModelRunner310V2`。
- `use_v2_model_runner=False` 时继续创建原有的 `NPUModelRunner310`。
- 分别记录实际使用的 V1 或 V2 runner。

改动原因：

- 310P Worker 原来固定使用 Model Runner V1，需要通过已有 V2 开关接入新的 V2 runner。
- V2 runner 使用延迟导入，避免 V1 启动时加载 V2 kernel registry 和 V2 实现。
- V1 的设备初始化、workspace 初始化和 runner 构造参数保持不变。

## 2. 310P V2 包结构

### `vllm_ascend/_310p/worker/__init__.py`

改动内容：

- 新增 310P Worker 子包声明。

改动原因：

- 为 310P 专用 V2 Worker 实现提供独立目录，不在该文件中引入注册或导入副作用。

### `vllm_ascend/_310p/worker/v2/__init__.py`

改动内容：

- 新增 310P Model Runner V2 子包声明。
- 不主动导入 V2 runner 或 kernel registry。

改动原因：

- Python 导入任意 V2 子模块时都会先执行该文件，因此必须保持无副作用。
- 避免 Model Runner V1 进程因为加载某个类型而提前注册 V2 kernel 或导入默认 Triton
  kernel。

## 3. 310P V2 Runner

### `vllm_ascend/_310p/worker/v2/model_runner.py`

改动内容：

- 新增 `NPUModelRunner310V2`，继承公共 `NPUModelRunner`。
- 导入该 runner 时调用 `register_310p_kernels()`；没有上游 dispatcher 时为 no-op。
- 通过 `request_state_cls` 使用 `Ascend310PRequestState`。
- 通过 `aclgraph_manager_cls` 使用 `ModelAclGraphManager310`。
- 校验第一版配置范围，只允许 TP，暂不进入 prefix cache、MTP 和其他未适配路径。
- 使用 CPU request-state mirror 准备 prefill input IDs、positions、sequence lengths 和
  logits indices。
- 使用 310P BlockTables 准备输入 block table 和 slot mapping。
- 使用 greedy sampler 完成首版 token 选择。
- 使用原生 PyTorch/NPU indexing 完成 sampled token、total length 和
  `num_computed_tokens` 回写。
- 使用 `AscendKVBlockZeroer310V2` 适配 V2 KV block-size 参数布局。

改动原因：

- 上游输入准备、状态更新等路径包含 Triton kernel，310P 只能通过子类和类属性扩展点
  替换。
- 310P 已经能从 scheduler 获取 CPU metadata，应直接使用这些 mirror，避免从 NPU
  Tensor 反向 D2H。
- 类级差异、RequestState 和 ACLGraph manager 不属于函数级 kernel dispatcher 的适用
  范围，需要通过继承和类属性扩展。

## 4. Triton Kernel Dispatcher 解耦

### `vllm_ascend/worker/v2/block_table.py`

改动内容：

- 不改动该文件，保持与主线一致：默认 `_compute_slot_mappings_kernel` 仍是多 cache
  group raw-pointer ABI，也不添加 `@pluggable_kernel`。

改动原因：

- [vLLM PR #43048](https://github.com/vllm-project/vllm/pull/43048) 尚未合入主线，导入
  `vllm.model_executor.triton_dispatcher` 会让插件在主线 vLLM 上 `ImportError`。
- 310P 通过 `patch_v2/patch_block_table.py` 整类替换 BlockTables，不需要函数级分发，
  公共 kernel 因此没有改 ABI 的理由。
- 保持 ABI 不变也让 910B/910C 的单次 kernel launch 行为和
  `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_slot_mapping.py`
  继续有效。

### `vllm_ascend/_310p/worker/v2/kernel_registry.py`

改动内容：

- 以 `try/except ImportError` 探测 `vllm.model_executor.triton_dispatcher`，导出
  `HAS_TRITON_DISPATCHER`。
- 提供 `KERNEL_IMPLS`（kernel 全限定名 → 310P 实现）和 `register_310p_kernels()`。
- 首版 `KERNEL_IMPLS` 为空，`register_310p_kernels()` 为 no-op 并返回空元组。

改动原因：

- 310P 首版可达的 Triton 路径在上游都没有 `@pluggable_kernel`，只能通过类/模块级替换
  覆盖，dispatcher 无法减少这些改动。
- 保留唯一接入点：PR #43048 合入后只需登记 kernel 名与实现，调用侧保持
  `kernel[grid](...)`，并可删除对应子类覆写；未合入时不产生任何导入依赖。

## 5. 310P BlockTables

### `vllm_ascend/_310p/worker/v2/block_table.py`

改动内容：

- 新增 `Ascend310PBlockTables`。
- 使用 CPU Tensor 保存 block table 和 block 数量，作为 block-table metadata 的所有者。
- `append_block_ids()` 直接更新 CPU block table。
- `gather_block_tables()` 根据 request index 在 CPU 聚合输入 block table，再复制到固定的
  NPU buffer。
- `compute_slot_mappings()` 调用同文件内的 `_compute_group_slot_mappings()`，按 cache
  group 用 NumPy 生成 slot ID，再一次性 H2D。
- CPU owner tensor 同时保存 NumPy 视图，避免每步重新包装。
- 支持多个 KV cache group。
- dummy block table 和 slot mapping 复用持久化 NPU Tensor，保持 ACLGraph 所需的固定
  地址。
- 不导入 `vllm_ascend/worker/v2/block_table.py`，UT 对该约束做静态校验。

改动原因：

- 上游 BlockTables 的 staged write、gather 和 slot mapping 都是 Triton 实现，310P 需要整
  类替换，而不是替换单个 kernel。
- CPU owner 可以直接消费 scheduler metadata，避免热路径 D2H；NumPy 实现与 310P Model
  Runner V1 的 slot mapping 算法一致。
- 不导入公共 V2 block table，确保 310P 进程不会执行该模块的 `@triton.jit` 定义。

### `vllm_ascend/patch/worker/patch_v2/patch_block_table.py`

改动内容：

- 310P 设备将上游 V2 `model_runner.BlockTables` 替换为 `Ascend310PBlockTables`。
- 非 310P 设备继续使用公共 `AscendBlockTables`。

改动原因：

- 310P 必须整类替换上游 BlockTables 的 staged-write、gather 和 slot-mapping Triton
  路径。
- patch 目标属于 Model Runner V2，不改变 310P Model Runner V1 使用的 BlockTables。

## 6. RequestState 和 Staged Write

### `vllm_ascend/_310p/worker/v2/states.py`

改动内容：

- 新增 `Ascend310PStagedWriteTensor`。
- 使用 CPU owner 保存待写数据，并记录 dirty row。
- 普通状态通过 `index_copy_` 更新 NPU Tensor。
- 大型 token buffer 继续使用仓库已有的 310P UVA wrapper。
- 新增 `Ascend310PRequestState`，替换上游 RequestState 中的 Triton-backed
  `StagedWriteTensor`。
- 保存 `num_computed_tokens` 的 NumPy、CPU Tensor 和 NPU Tensor 镜像。

改动原因：

- 上游 `StagedWriteTensor.apply_write()` 使用 Triton kernel，且没有函数级替换入口。
- RequestState 是 prefill、decode、chunked prefill 和 ACLGraph 的公共基础，310P 必须提供
  无 Triton 的状态更新路径。

## 7. 多模态 RoPE 和 ModelState

### `vllm_ascend/_310p/worker/v2/rope.py`

改动内容：

- 新增 `Ascend310PRopeState`。
- 支持 MRoPE 和 XDRoPE position state。
- 多模态 prefill positions 保存在 CPU-owned staged-write buffer。
- decode positions 根据 `num_computed_tokens` 和 RoPE delta 在 CPU 计算。
- 将 positions 写入固定的 NPU buffer。
- 新增 `get_310p_rope_state()`，根据模型配置创建对应 RoPE state。

改动原因：

- 上游多维 RoPE position 准备使用 Triton kernel，只能通过 RoPE state 子类替换。
- Qwen3-VL 首版多模态适配需要无 Triton MRoPE 路径。

### `vllm_ascend/_310p/worker/v2/model_state.py`

改动内容：

- 新增 `Ascend310PModelState`。
- 初始化多模态 encoder cache 和 encoder runner。
- 使用 `Ascend310PRopeState`，不创建上游 Triton-backed 多维 RoPE state。
- `prepare_inputs()` 使用 CPU request metadata 准备多维 positions。
- 为第一版提供 310P greedy sampler。

改动原因：

- 将 310P 多模态和 RoPE 差异隔离在 `_310p` 目录。
- 避免在公共 `AscendModelState` 中散落 `is_310p()` 分支。

### `vllm_ascend/worker/v2/model_states/__init__.py`

改动内容：

- hybrid 模型选择 `AscendMambaHybridModelState`。
- 非 hybrid 的 310P 模型选择 `Ascend310PModelState`。
- 其他设备和普通模型继续使用原有 `AscendModelState`。

改动原因：

- Qwen3.5-4B 需要 hybrid model state。
- Qwen3-VL 等非 hybrid 310P 模型需要 310P 多模态 RoPE state。
- 将设备和模型类型选择集中在 model-state factory。

## 8. Qwen3.5 Hybrid ModelState

### `vllm_ascend/worker/v2/model_states/mamba_hybrid.py`

改动内容：

- 新增 `AscendMambaHybridModelState`，继承上游 `MambaHybridModelState`。
- 保留上游 hybrid state 初始化和 cache 生命周期。
- 使用 Ascend `build_attn_metadata()` 构造 Attention、GDN/Mamba metadata。
- 准备 prefill/decode 标志、accepted-token 数量和 draft-token metadata。
- 使用原生 Tensor indexing 替换上游 Triton accepted-token scatter kernel。

改动原因：

- Qwen3.5-4B 同时包含 Full Attention 和 GDN/Mamba layer，需要 hybrid cache metadata。
- 上游 metadata builder 不包含 Ascend Attention backend 需要的扩展字段。
- 上游 accepted-token scatter 是 Triton kernel，需要 310P 侧提供等价实现。

## 9. Greedy Sampling

### `vllm_ascend/_310p/worker/v2/sampler.py`

改动内容：

- 新增 `Ascend310PGreedySampler`。
- 使用 `torch.argmax()` 选择 token。
- 检查并拒绝 temperature、top-k、top-p、min-p、penalties、logprob 等第一版未适配
  参数。
- 不进入 vLLM Ascend sampling 目录中的 Triton kernel。

改动原因：

- 第一版真实模型推理需要基本 greedy token 生成能力。
- 完整后处理计划在后续版本适配，本次不为未支持能力提前添加空注册实现。
- 明确拒绝未支持参数可以避免静默改变用户请求语义。

## 10. ACLGraph

### `vllm_ascend/_310p/worker/v2/aclgraph.py`

改动内容：

- 新增 `ModelAclGraphManager310`。
- 复用公共 graph manager 接口和上游 graph capture/replay 基础逻辑。
- 不执行主线 Ascend Attention graph-task handle 更新。
- 保存 capture sizes，并增加 310P ACLGraph replay 日志。

改动原因：

- 310P 使用直接 NPU 算子 capture，不使用主线 Ascend Attention 的 graph handle。
- 复用公共 handle 更新逻辑会访问 310P 不存在的 graph 参数。

## 11. KV Block Zeroer

### `vllm_ascend/_310p/worker/v2/kv_block_zeroer.py`

改动内容：

- 新增 `AscendKVBlockZeroer310V2`，继承现有 V1 `AscendKVBlockZeroer310`。
- 将 V2 的 `list[int]` block-size 参数转换为 V1 zeroer 需要的
  `list[list[int]]`。

改动原因：

- V1 和 V2 的 `kernel_block_sizes` 参数结构不同。
- 兼容逻辑放在 V2 专用适配器中，避免修改 V1 共用 zeroer 的代码和行为。

## 12. 公共 Model Runner V2 扩展点

### `vllm_ascend/patch/platform/patch_use_v2_model_runner.py`

改动内容：

- 保留通过 `VLLM_USE_V2_MODEL_RUNNER` 显式选择 V2 runner 的属性补丁。
- 新增 `_validate_v2_model_runner()` 平台补丁。
- 310P 跳过上游全局 `HAS_TRITON` 硬门禁。
- 310P 继续执行上游 `_get_v2_model_runner_unsupported_features()` 校验。
- 非 310P 完整调用上游原始 `_validate_v2_model_runner()`。

改动原因：

- 上游 `VllmConfig` 在 Worker 创建前要求 `HAS_TRITON=True`，导致 310P 即使已经提供
  非 Triton 实现也无法完成配置创建。
- 该校验独立于任何 kernel 替换机制，必须由 Ascend 平台只针对 310P 接管。
- 只跳过 Triton 条件而不跳过其他 feature gate，避免放开上游尚未支持的 V2 配置。

### `vllm_ascend/worker/v2/model_runner.py`

改动内容：

- 增加 `request_state_cls` 类级扩展点。
- 将输入准备封装为可覆盖方法：
  - `_prepare_prefill_inputs()`；
  - `_prepare_pos_seq_lens()`；
  - `_combine_sampled_and_draft_tokens()`。
- 将已经存在的 CPU/NumPy metadata 传给这些扩展方法。
- 增加 `aclgraph_manager_cls` 类级扩展点。

改动原因：

- 这些调用最终进入 vLLM 上游的 Triton kernel，只能在 runner 层提供可覆盖方法。
- 310P 需要使用 scheduler CPU mirror 准备输入，不能从 device Tensor 反向 D2H。
- 非 310P 默认实现仍调用原来的上游函数，保持原有调用语义。

## 13. 非 MLA KV Cache 的 NZ shape 修复

### `vllm_ascend/worker/v2/attn_utils.py`

报错现象：

```text
RuntimeError: shape '[572, 64, 128, 128]' is invalid for input of size 74973184
```

错误发生在 `_reshape_kv_cache_v2()` 创建 `v_cache` view 时。

改动内容：

- 新增 `_get_non_mla_kv_cache_shapes()`，统一计算非 MLA K/V Cache shape。
- 当 `head_size_v == head_size` 时，K/V 完整复用 backend 返回的物理 shape。
- 只有 V 的逻辑 head size 确实与 K 不同时，才调整 V Cache 的最后一维。
- `_reshape_kv_cache()` 和 `_reshape_kv_cache_v2()` 共同使用该 helper。

改动原因：

- 310P backend 返回的 KV Cache shape 为：

  ```text
  [2, num_blocks, num_kv_heads * head_size / 16, block_size, 16]
  ```

- 最后一维 `16` 是 FRACTAL_NZ 的物理分形维度，不是逻辑 `head_size`。
- 原逻辑只要检测到 `head_size_v` 属性，就将最后一维 `16` 替换为
  `head_size_v`。对于 K/V head size 相同的 Qwen 模型，这会无故扩大 V Cache
  shape，导致目标 shape 的元素数与 raw storage 不一致。
- 该修复保留所有 backend-specific 物理维度，同时继续兼容真正的非对称 K/V
  head-size 模型。

### `tests/ut/worker/test_attn_utils_v2.py`

改动内容：

- 增加 310P 风格 NZ shape 的回归测试。
- 验证 K/V head size 相同时，尾部物理维度 `16` 不会被覆盖。
- 验证 K/V head size 不同时仍生成独立的 V Cache shape。

改动原因：

- 防止公共 KV Cache reshape 后续再次将 backend 物理布局误判为逻辑布局。

## 14. 310P V2 完整 KV Cache 初始化和 FRACTAL_NZ 分配

### `vllm_ascend/_310p/worker/v2/model_runner.py`

报错现象：

```text
ReshapeAndCacheOperation CheckIniMatch Failed
Actual Inputs: key_cache(float16, nd), value_cache(float16, nd)
Supported Combs: key_cache(float16, fractal_nz),
                 value_cache(float16, fractal_nz)
```

随后出现的 ATB setup failure 和 Segmentation Fault 是算子参数校验失败后的次生错误，
首要问题是传入的 KV Cache 格式为 ND，而不是 FRACTAL_NZ。

改动内容：

- 在 `NPUModelRunner310V2` 中完整覆盖 `initialize_kv_cache()`。
- 深拷贝并保存 `KVCacheConfig`，避免修改 Engine 侧传入的配置对象。
- 根据每个 KV Cache group 计算 block size 和最大 block 数。
- 初始化 310P attention backend、kernel block size、310P BlockTables 和
  ACLGraph manager。
- 新增 `_allocate_kv_cache_tensors_310p()`：
  - Attention KV Cache 使用 `torch_npu.empty_with_format()` 分别分配 K/V；
  - ACL format 显式指定为 `ACL_FORMAT_FRACTAL_NZ`；
  - shape 使用 `AscendAttentionBackend310.get_kv_cache_shape()` 的物理布局；
  - Mamba/SSM state cache 保持 ND，并按 page storage 建立 `as_strided` view；
  - 处理 shared KV Cache layer，确保共享层引用同一份 cache；
  - 校验所有预期 layer 均已完成初始化。
- 分配完成后继续执行 `bind_kv_cache()`、KV connector 初始化和 PCP manager
  初始化，保持 V2 KV Cache 的完整生命周期。
- 第一版明确拒绝 asymmetric K/V head size，避免在没有验证 310P NZ V Cache
  布局前静默生成错误格式。

改动原因：

- `tensor.view()` 和 `reshape()` 只能改变逻辑 shape，不能将底层 ND storage
  转换成 FRACTAL_NZ。
- 310P `ReshapeAndCacheOperation` 的输入 key/value 是 ND，但目标 cache 和输出
  cache 必须是 FRACTAL_NZ。
- MRV1 在 KV Cache 初始化阶段直接按 NZ 格式分配；MRV2 若继续使用公共 raw
  tensor reshape 路径，最终得到的仍然是 ND cache。
- 因此不能只修正 shape，必须由 310P V2 runner 接管完整 KV Cache 初始化和
  物理格式分配。

### `tests/ut/_310p/test_model_runner_v2_310p.py`

改动内容：

- 增加 310P attention KV Cache 使用 FRACTAL_NZ 分配的测试。
- 验证 K/V 分别通过 `torch_npu.empty_with_format()` 分配。
- 验证传入的 shape 保留 NZ 尾部物理分形维度 `16`，并显式使用 ACL format 29。

改动原因：

- 防止后续重构重新退回 raw ND tensor 加 `view()` 的分配方式。
- Mamba state、共享层、完整初始化生命周期仍需要在后续补充独立 UT，并在真实
  310P 环境通过启动和请求验证。

## 15. `uniform_decode_query_len` 上游版本兼容

### `vllm_ascend/_310p/worker/v2/model_runner.py`

报错现象：

```text
AttributeError: 'NPUModelRunner310V2' object has no attribute
'uniform_decode_query_len'
```

错误发生在 `initialize_kv_cache()` 调用
`resolve_cudagraph_mode_and_sizes()` 时。

改动内容：

- 新增 `_get_uniform_decode_query_len()`：

  ```python
  return getattr(self, "uniform_decode_query_len", self.decode_query_len)
  ```

- ACLGraph mode/size 解析统一调用该兼容方法。

改动原因：

- 不同 vLLM V2 开发版本对 uniform decode query length 的属性暴露存在差异。
- 310P 代码最初参考了包含 `uniform_decode_query_len` 的版本，但实际配套版本只保证
  `decode_query_len` 存在。
- 对普通首版 decode，两者表达的都是图捕获使用的固定 decode query length，因此可在
  属性不存在时安全回退。
- 使用局部兼容 helper，不修改公共 MRV2，也不向 MRV1 注入新属性。

### `tests/ut/_310p/test_model_runner_v2_310p.py`

改动内容：

- 验证存在 `uniform_decode_query_len` 时优先使用该值。
- 验证属性不存在时回退到 `decode_query_len`。

改动原因：

- 同时覆盖新旧上游接口，防止后续升级 vLLM 时再次出现启动期属性错误。

## 16. ACLGraph 内 pageable `seq_lens` H2D 修复

### `vllm_ascend/_310p/attention/metadata_builder.py`

报错现象：

```text
aclrtAllocatorGetByStream failed. Parameter stream is invalid.
Asynchronous copy task failed.
The pageable memory copy task does not support graph capture.
```

Python 栈指向：

```python
attn_metadata.seq_lens = attn_metadata.seq_lens.to(
    device=query.device,
    non_blocking=True,
)
```

改动内容：

- 在调用公共 builder 后，仅对 DecodeOnly、ChunkedPrefill、PrefillCacheHit 和
  SpecDecoding 将以下字段绑定到 `AscendCommonAttentionMetadata` 的设备侧常驻
  view：

  ```python
  attn_metadata.seq_lens = common_attn_metadata.seq_lens[:num_reqs]
  attn_metadata.query_start_loc = (
      common_attn_metadata.query_start_loc[: num_reqs + 1]
  )
  ```

- DecodeOnly、ChunkedPrefill、PrefillCacheHit 和 SpecDecoding 使用相同的设备侧
  输入 buffer。
- PrefillNoCache 保留公共 builder 生成的 host `seq_lens`。
- ChunkedPrefill 需要的 pinned host `query_lens_cpu` 逻辑保持不变。

改动原因：

- MRV2 已经在 `_prepare_pos_seq_lens()` 中把 CPU 长度写入预分配的
  `input_buffers.seq_lens` NPU tensor。
- 公共 Ascend metadata builder 为了 CPU metadata 计算，优先选择
  `seq_lens_cpu`，而 MRV2 的该 tensor 来自普通 CPU/NumPy 共享内存，属于
  pageable memory。
- 修复前 310P builder 只在 ChunkedPrefill 和 SpecDecoding 中重新绑定设备
  `seq_lens`；DecodeOnly 提前返回，导致 paged attention 在图内才执行
  pageable CPU 到 NPU 的 `.to()`。
- ACL Graph 不允许捕获 pageable H2D copy。应复用 graph capture 前已经更新的
  NPU 常驻 buffer，而不是在 attention forward 内临时转换。
- 不能对所有 attention state 无条件绑定设备 tensor。310P PrefillNoCache 进入
  ATB SelfAttention encoder 路径，`seq_lens` 用于构造 host 参数；传入 NPU tensor
  会使 ATB 报 `tensor.hostData is null`，并导致 eager prefill 失败。
- 因此 metadata 必须保持分阶段契约：PrefillNoCache 使用 host `seq_lens`，paged
  和 splitfuse 路径使用常驻 NPU `seq_lens`。
- MRV1 的长度准备使用 pinned CPU mirror 和常驻 NPU `self.seq_lens`；本次修改让
  MRV2 明确采用相同的 graph-safe 数据流。

影响范围：

- 代码位于 310P metadata builder，不改变其他 Ascend 机型的 builder。
- 该 builder 也可能被 310P MRV1 使用；绑定到已存在的设备侧
  `common_attn_metadata.seq_lens` 与原有算子输入语义一致，并减少一次潜在 H2D，
  不改变 MRV1 的长度值和 KV Cache 行为。

### `tests/ut/_310p/attention/test_attention_v1_310.py`

改动内容：

- 增加 DecodeOnly metadata 回归测试。
- 验证 `seq_lens` 和 `query_start_loc` 与 common metadata 的设备侧 view
  共享 storage，而不是使用父类构造的 CPU tensor。
- 增加 PrefillNoCache 回归测试，验证其继续使用父类生成的 host `seq_lens`。

改动原因：

- 防止未来调整 splitfuse 分支时再次让 DecodeOnly 绕过设备 buffer 绑定，或把
  PrefillNoCache 错误切换成设备 tensor。

## 17. 联调后的数据流约束

后续修改必须保持以下约束：

1. 310P Attention KV Cache 必须在创建时就是 FRACTAL_NZ；不能依赖
   `view()`、`reshape()` 或算子 forward 中的临时格式转换。
2. K/V Cache shape 必须保留 backend 返回的物理分形维度；逻辑 head size
   不能直接覆盖物理维度。
3. ACLGraph capture/replay 使用的 `seq_lens`、`query_start_loc`、block table、
   slot mapping 必须来自预分配的常驻 NPU buffer，并在进入图前原地更新。
4. 图内不得出现 pageable CPU 到 NPU 的拷贝、临时设备 tensor 分配或依赖动态
   Python 值的数据准备。
5. 310P V2 的上游接口兼容应集中在 `_310p` 专用类中，避免改变其他设备和
   310P MRV1 的执行路径。
6. 每个运行时问题都必须补充对应 UT；NPU 环境还需分别验证 eager 和 ACLGraph
   的真实请求，不能只以服务启动成功作为通过标准。
7. 本任务后续每次代码修改都必须在同一轮同步更新本文档，记录改动文件、问题现象、
   根因、实现方式、影响范围、测试结果和仍未验证的内容。

## 18. FULL_DECODE_ONLY attention 长度 buffer 刷新

问题现象：

- eager 精度正常。
- `FULL_DECODE_ONLY` 首个生成 token 正常，后续稳定进入乱码或重复符号。
- 对比定位显示 prefill、首个 decode 的 token、position 和运行时长度元数据一致，但图模式
  首个 decode 的 attention hidden states 已经与 eager 不同。

根因：

- 310P paged attention 采用 direct-op ACLGraph capture，不注册公共 attention graph task
  参数更新机制。
- 图捕获 metadata 绑定 capture-time `seq_lens` tensor 的固定地址；运行时 metadata 可能使用
  另一块 tensor。只更新 runtime tensor 无法改变图实际读取的 capture tensor，部分 capture
  bucket 因而继续读取 dummy context length。
- stream 同步只能保证已有写操作完成，不能解决 capture tensor 与 runtime tensor 地址不同。

### `vllm_ascend/_310p/worker/v2/model_state.py`

改动内容：

- 在 `Ascend310PModelState` 实例中记录 FULL graph 捕获使用的所有 `seq_lens` tensor。
- 按 `data_ptr()` 管理不同物理地址；同一地址出现不同 bucket view 时保留 `numel()` 最大的
  view，避免依赖上游 capture bucket 的排序。
- capture 阶段只登记 buffer；仅在非 capture 的 `CUDAGraphMode.FULL` 准备阶段，将当前
  runtime padded `seq_lens` 原位复制到全部 capture buffer，并把剩余尾部清零。
- eager 和 piecewise 不执行刷新。

改动原因：

- ACLGraph replay 只能读取捕获时绑定的固定 tensor 地址，必须在 replay 前刷新这些地址的内容。
- 状态归属 310P direct-op graph contract，因此放在 `Ascend310PModelState`，不向公共 model
  state 热路径增加设备判断。

公共路径说明：

- `vllm_ascend/worker/v2/model_states/default.py` 最终保持不变；310P capture buffer
  状态和刷新逻辑全部收敛在专用 model state 中。

影响范围：

- 仅影响 310P MRV2 默认模型状态的 FULL graph replay。
- 不修改 310P MRV1，不影响其他 Ascend 机型，也不改变公开 API 和配置。

### `tests/ut/_310p/test_model_runner_v2_310p.py`

改动内容：

- 验证不同地址的 capture buffer 均会刷新，短 runtime batch 后的尾部会清零。
- 验证同一地址的不同长度 view 始终保留最大 view。
- 验证 eager 不刷新 capture buffer，FULL runtime 才刷新。

验证状态：

- 服务器原始图请求修复后输出正确；连续 20 次确定性请求结果一致且有效。
- 服务器目标回归测试结果为 `1 passed, 19 deselected`，并通过 `py_compile` 和
  `git diff --check`。本次重构后的新增 UT 仍需重新在服务器环境执行。
- 并发请求在 310P V2 `postprocess_sampled()` 中可能存在 `idx_mapping` 与
  `query_start_loc` 长度不一致问题；该问题与本次 attention 首次偏差独立，暂未修改。

## 19. FULL graph 多并发采样后处理 padding 对齐

问题现象：

- Qwen3-8B eager 多并发正常。
- `FULL_DECODE_ONLY` 单请求能够运行；多并发进入 `num_tokens=16` 的图后，
  `postprocess_sampled()` 报 `tensor a (15) must match tensor b (16)`。

根因：

- FULL graph 按固定 capture bucket 重放。15 个真实 decode 请求进入 16-request bucket 时，
  `idx_mapping` 被补齐到 16，尾部使用 `-1` sentinel。
- `query_start_loc` 仍只保存真实请求边界，长度为 16，因此差分后的 `query_lens` 长度为 15。
- 原实现直接使用长度为 16 的 `idx_mapping >= 0` mask 筛选长度为 15 的 `query_lens`，
  导致维度不匹配。eager 不做 bucket padding，所以不会进入该故障路径。

### `vllm_ascend/_310p/worker/v2/model_runner.py`

改动内容：

- 新增 `_get_valid_query_lens()`，在 `idx_mapping` 与 `query_start_loc` 共同描述的请求区间内
  计算 query length。
- 只使用共同区间内的 `idx_mapping` mask；FULL graph 尾部 `-1` padding 不参与
  `num_computed_tokens` 更新。
- 正好命中 capture bucket 时保持原有行为不变。

改动原因：

- `query_start_loc` 描述真实调度请求，`idx_mapping` 在图模式还承担固定 bucket padding，
  两者不能无条件假设长度相同。
- 修复放在 310P MRV2 专用后处理内，不修改上游 `post_update()` 和公共 MRV2 路径，
  因而不影响 310P MRV1及其他机型。

### `tests/ut/_310p/test_model_runner_v2_310p.py`

改动内容：

- 增加 15 个真实请求进入 16-request graph bucket 的回归测试，验证尾部 `-1` 被忽略。
- 增加请求数正好命中 capture bucket 的测试，验证所有 query length 均被保留。

验证状态：

- `py_compile`、`ruff check` 和 `git diff --check` 已通过。
- 当前 Windows Python 环境未安装 `pytest`，目标 UT 尚未在本机执行，需在服务器开发环境补跑。
- 真实 310P TP2 eager、单请求 graph 和 15→16 bucket 多并发仍需在服务器环境验证。

## 20. 异步调度校验放开

问题现象：

- vLLM 上游 MRV2 和当前 Ascend MRV2 公共路径已经支持 async scheduling。
- 310P MRV2 在 runner 初始化阶段仍将 `scheduler_config.async_scheduling=True` 判定为
  首版范围外特性，服务尚未进入执行路径就抛出 `NotImplementedError`。

根因：

- 310P 首版开发时使用了保守的特性白名单，异步调度尚未完成验证，因此增加了设备专用拦截。
- 后续公共 MRV2 已提供异步调度需要的两批 in-flight buffer、异步输出 copy stream/Event、
  `AsyncModelRunnerOutput` 返回路径和 executor 能力声明，但 310P 的旧校验没有同步移除。

### `vllm_ascend/_310p/worker/v2/model_runner.py`

改动内容：

- 删除 `_validate_first_release_config()` 对 `async_scheduling` 的设备专用拒绝。
- 继续复用上游 MRV2 的异步输出和调度机制，不在 310P 中复制另一套实现。

安全性说明：

- 310P sampler 产生的 device tensor 在上游 `AsyncOutput` 中由独立 copy stream 异步复制，
  `torch.cuda.Stream/Event` 已由 Ascend MRV2 兼容层映射到 NPU 实现。
- 310P 的 sampled-token、`total_len` 和 `num_computed_tokens` 状态仍在主 stream 上于下一步前完成
  回写；长度 CPU mirror 使用独立 stream/Event，并在下一批 `_update_seq_lens_cpu()` 读取前同步。
- 当前首版仍不支持 MTP、PP、DP、prefix cache 等既有拦截特性；本次只放开普通生成场景的
  async scheduling，不扩大其他功能范围。
- 修改只位于 310P MRV2 专用配置校验，不影响 310P MRV1 和其他机型。

### `tests/ut/_310p/test_model_runner_v2_310p.py`

改动内容：

- 增加 `async_scheduling=True` 能通过 310P 首版配置校验的回归测试。
- 既有非 TP 并行、MTP、prefix cache 和 EP 拒绝测试保持不变。

验证状态：

- `py_compile`、`ruff check` 和 `git diff --check` 已通过。
- 当前 Windows Python 环境未安装 `pytest`，目标 UT 尚未在本机执行，需在服务器开发环境补跑。
- 真实 310P 环境仍需分别验证 TP1/TP2、eager/ACLGraph、单请求/多并发，以及流式输出。

## 21. Qwen3.5 hybrid model state 的无 Triton RoPE 路由

问题现象：

- Qwen3.5-4B 在 `profile_run()` 阶段启动失败，尚未完成 KV Cache 初始化。
- Python 堆栈进入上游 `vllm/v1/worker/gpu/mm/rope.py`，执行
  `_prepare_rope_positions_kernel[(num_reqs,)](...)` 时抛出
  `TypeError: 'function' object is not subscriptable`。

根因：

- Qwen3.5-4B 是 hybrid attention/GDN 模型，同时使用多维 RoPE。
- model state 工厂优先匹配 `model_config.is_hybrid`，返回公共
  `AscendMambaHybridModelState`，因此没有进入已有的 310P
  `Ascend310PModelState/Ascend310PRopeState` 路径。
- 公共 hybrid state 继承上游 `DefaultModelState`，其 `RopeState.prepare_positions()`
  调用上游 Triton kernel，310P 无法执行。
- 不能简单让所有 310P 模型都返回默认 `Ascend310PModelState`，否则会丢失 GDN/Mamba
  metadata、`num_accepted_tokens_gpu` 和 hybrid 后处理语义。

### `vllm_ascend/_310p/worker/v2/model_state.py`

改动内容：

- 抽取 `_Ascend310PModelStateMixin`，集中提供：
  - 310P model state 公共初始化；
  - `Ascend310PRopeState` 创建及 CPU position 准备；
  - FULL graph capture `seq_lens` buffer 登记与 replay 前刷新；
  - 310P 首版 greedy sampler。
- `Ascend310PModelState` 改为组合该 mixin 与公共 `AscendModelState`，标准 attention
  模型行为保持不变。
- 新增 `Ascend310PMambaHybridModelState`，组合该 mixin 与
  `AscendMambaHybridModelState`：
  - RoPE position 准备走 310P 无 Triton 路径；
  - attention/GDN metadata 和 `postprocess_state()` 继续继承公共 hybrid 实现；
  - 初始化 hybrid 所需的 `num_accepted_tokens_gpu`。

改动原因：

- 只替换 310P 不可执行的 RoPE position kernel，保留已经适配的 Ascend hybrid/GDN
  数据流，避免复制整套 hybrid attention 实现。
- mixin 让默认模型和 hybrid 模型共享同一份 310P RoPE、ACLGraph buffer 与 sampler
  契约，防止两条路径后续修复不一致。

### `vllm_ascend/worker/v2/model_states/__init__.py`

改动内容：

- hybrid 分支内增加 310P 子分支，返回 `Ascend310PMambaHybridModelState`。
- 非 310P hybrid 仍返回原有 `AscendMambaHybridModelState`；非 hybrid 路由不变。

影响范围：

- 只改变 310P MRV2 hybrid 模型的 model state 类型。
- 不修改 vLLM 上游源码，不影响 310P MRV1，也不改变其他 Ascend 机型的 hybrid 路径。
- 本问题发生在 profile dummy run，与 async scheduling 开关无关。

### `tests/ut/_310p/test_model_runner_v2_310p.py`

改动内容：

- 验证 310P hybrid 配置由工厂路由到专用 hybrid model state。
- 验证专用 state 仍继承 `AscendMambaHybridModelState`，保留公共 hybrid 行为。

验证状态：

- `py_compile`、`ruff check` 和 `git diff --check` 已通过。
- 当前 Windows 本地 Python 环境未安装 `pytest`，目标 UT 未在本地执行，需在服务器环境补充执行。
- 真实 310P Qwen3.5-4B 仍需依次验证 profile、eager 请求、ACLGraph 捕获/重放、
  TP1/TP2、chunked prefill、多并发和 async scheduling。

## 22. Qwen3.5 MRoPE cos/sin slice 初始化

问题现象：

- 第 21 节修复后，Qwen3.5-4B 已越过上游 Triton positions kernel，但在首次
  profile forward 的 `AscendMRotaryEmbedding310.forward_oot()` 中报错：
  `MRoPE cos/sin slices are not initialized`。

根因：

- 310P 的 MRoPE 算子不直接用 positions 索引 cache，而是读取预先构造且地址稳定的
  `_mrope_cos_slice` 和 `_mrope_sin_slice`。
- MRV1 在 `NPUModelRunner310._model_forward()` 调用
  `prepare_mrope_cos_sin_slices_from_runner()`，保证每次模型 forward 前刷新 slice。
- MRV2 310P 新增的 CPU RoPE 路径只生成了 positions，没有同步迁移 MRV1 的 slice
  准备步骤，因此算子在 profile 阶段检测到 slice 为空。

修改内容：

- `vllm_ascend/_310p/worker/v2/model_state.py`
  - 在 310P `prepare_inputs()` 得到最终 padded positions 后，仅当
    `model_config.uses_mrope` 时调用 `prepare_mrope_cos_sin_slices_from_runner()`。
  - slice 在模型 forward 前完成刷新，并继续使用既有的固定容量 storage，满足后续
    ACLGraph 捕获和重放对地址稳定性的要求。
  - 修改局限于 310P MRV2 ModelState，不修改公共 MRV2 execute 路径，不影响 MRV1
    或其他设备。
- `tests/ut/_310p/test_model_runner_v2_310p.py`
  - 增加回归测试，验证 MRoPE positions 准备后、模型 forward 前必定构造 cos/sin
    slice，并将同一 positions tensor 传入模型。

验证状态：

- `py_compile`、`ruff check` 和 `git diff --check` 已通过。
- 当前 Windows 本地 Python 环境未安装 `pytest`，新增目标 UT 需在服务器执行。
- 真实 310P 需重新验证 Qwen3.5-4B profile、eager 和 ACLGraph。

## 23. Qwen3.5 Hybrid 上游状态初始化契约

问题现象：

- Qwen3.5-4B 已完成启动和 profile，但真实请求加入时在上游
  `MambaHybridModelState.add_request()` 报错：
  `Ascend310PMambaHybridModelState has no attribute _align_mode`。

根因：

- 第 21 节的首版实现为避开上游 Triton RoPE，手工复制了 Default/Hybrid 构造逻辑。
- 服务器所使用的上游 vLLM 已在 Hybrid ModelState 中增加 `_align_mode` 等初始化契约，
  而本地配套源码版本尚无该字段。手工复制构造逻辑无法自动跟随上游演进。
- 该问题与 RoPE kernel 执行无关：构造上游 RoPE state 本身不会启动 Triton，真正需要
  避免的是请求阶段调用上游 `prepare_positions()`。

修改内容：

- `vllm_ascend/_310p/worker/v2/model_state.py`
  - 310P Hybrid state 改为先调用完整的 `AscendMambaHybridModelState.__init__()`，继承
    当前上游全部 Hybrid 字段和未来新增契约。
  - 父类初始化完成后、首个请求进入前，将上游 RoPE state 替换为
    `Ascend310PRopeState`，并依据新 RoPE state 重建 MM pruner。
  - 标准 attention 的310P state仍沿用轻量初始化；MRV1和其他设备不进入该类。
- `tests/ut/_310p/test_model_runner_v2_310p.py`
  - 增加回归测试，验证专用 Hybrid state 必须执行完整父类构造，然后替换310P RoPE
    state并初始化图模式 seq-lens buffer。

验证状态：

- `py_compile`、`ruff check` 和 `git diff --check` 已通过。
- 当前 Windows 本地 Python 环境未安装 `pytest`，新增目标 UT 需在服务器执行。
- 真实 310P 需重新验证 Qwen3.5-4B 请求加入、eager、ACLGraph 和多并发。

## 24. Qwen3.5 GDN metadata 的包装模型前缀对齐

问题现象：

- Qwen3.5-4B 已进入真实请求 prefill，但310P GDN `_forward_core()` 使用
  `attn_metadata[self.prefix]` 时出现：
  `KeyError: language_model.model.layers.0.linear_attn`。

根因：

- Qwen3.5 外层模型通过 `language_model` 包装文本模型，GDN模块保存的完整 prefix 为
  `language_model.model.layers.*.linear_attn`。
- Hybrid KV Cache和 attention group 使用文本模型内部注册名生成 metadata；当前服务器
  对应键为去除包装层后的 `model.layers.*.linear_attn`。
- 两个名字表示同一 GDN层，但310P GDN沿用按模块完整 prefix直接索引的实现，导致
  metadata已经生成却无法命中。

修改内容：

- `vllm_ascend/_310p/worker/v2/model_state.py`
  - 在310P ModelState完成 attention metadata构建后，扫描模型中的
    `*.linear_attn` prefix。
  - 当模块 prefix与现有 metadata键存在唯一的点分隔后缀匹配时，为完整 prefix增加
    指向同一 metadata对象的别名；不复制 tensor或 metadata内容。
  - 已存在的精确键不覆盖；存在多个候选时不猜测，避免错误地把不同层绑定在一起。
  - 修复限定在310P MRV2 ModelState，不改变公共 attention builder、GDN算子、MRV1
    或其他设备的键语义。
- `tests/ut/_310p/test_model_runner_v2_310p.py`
  - 增加包装 prefix唯一匹配回归测试。
  - 增加多候选时拒绝建立别名的边界测试。

验证状态：

- `py_compile`、`ruff check`和`git diff --check`已通过。
- 当前 Windows本地 Python环境未安装`pytest`，新增目标 UT需在服务器执行。
- 真实310P需重新验证 Qwen3.5-4B eager prefill/decode，再验证 ACLGraph。

临时调试补充：

- 上述后缀别名未命中时，在310P GDN执行原始字典索引前，仅当 `self.prefix`缺失才通过
  `print(..., flush=True)`输出当前 prefix和 `attn_metadata`全部 keys，随后保留原始
  `KeyError`行为。该打印用于确认真实命名差异，定位完成后应删除。

## 25. GDN metadata缺失的现场证据与别名方案回退

现场打印结果：

- GDN `self.prefix`为 `language_model.model.layers.0.linear_attn`。
- `attn_metadata`包含的8个 key全部为
  `language_model.model.layers.{3,7,11,15,19,23,27,31}.self_attn.attn`。
- metadata key保留了完整 `language_model`包装前缀，因此第24节关于“包装前缀不一致”
  的推断被现场证据否定。
- 真正现象是 GDN/linear-attention metadata整个 group没有被构建，而不是已经构建后
  无法通过别名命中。

本轮调整：

- 回退第24节实现的 GDN metadata后缀别名逻辑及其两项测试。
- 保留310P GDN报错点的条件打印。
- 当最终 metadata中没有任何 `linear_attn` key时，在310P ModelState中额外打印：
  - `kv_cache_config.kv_cache_groups`的 layer names；
  - `init_attn_backend()`生成的 `attn_groups` layer names；
  - 最终 metadata keys。
- 该诊断用于判断 GDN group是在 KV Cache配置阶段缺失、attention backend初始化阶段
  缺失，还是 metadata builder阶段遗漏。定位完成后应删除临时打印。

影响范围：

- 仅310P MRV2临时诊断路径；不修改MRV1和其他设备功能。

## 26. 补回 MRV2遗漏的 GDN KV Cache spec

最终现场证据：

- `kv_cache_groups`从引擎下发时就只包含8个 Full Attention层。
- `attn_groups`和最终 metadata与该输入完全一致，因此 backend初始化及 metadata builder
  没有丢层。
- Qwen3.5的24个 `*.linear_attn`层在 `get_kv_cache_spec()`收集阶段已被遗漏，导致既
  没有 GDN state cache，也没有 GDN metadata。

修改内容：

- `vllm_ascend/_310p/worker/v2/model_runner.py`
  - 310P MRV2覆盖 `get_kv_cache_spec()`，首先保留上游收集结果。
  - 遍历同一 `static_forward_context`，仅对上游结果中缺失的 `*.linear_attn`层调用其
    `get_kv_cache_spec(vllm_config)`并补回有效 spec。
  - 不覆盖上游已有 spec，不影响 Full Attention层，也不修改公共MRV2和MRV1。
- 删除 `model_state.py`和 `gdn_310.py`中的两处临时 debug print。
- `tests/ut/_310p/test_model_runner_v2_310p.py`
  - 增加回归测试，验证遗漏的 linear-attention spec会被补回、已有 Full Attention
    spec保持不变且非 linear-attention模块不会被额外调用。

验证状态：

- `py_compile`、`ruff check`和`git diff --check`已通过。
- 当前 Windows本地 Python环境未安装`pytest`，新增目标 UT需在服务器执行。
- 真实310P需确认启动日志中的 KV Cache groups同时包含 Full Attention和GDN层，并
  重新验证 Qwen3.5 eager prefill/decode。

## 27. 非投机 Hybrid模型的 KV block zeroer初始化

问题现象：

- 补回 Qwen3.5 GDN `MambaSpec`后，真实请求在上游 MRV2
  `update_requests()`中触发 `assert self.kv_block_zeroer is not None`。

根因：

- GDN state cache使 `kv_cache_config.needs_kv_cache_zeroing=True`，调度器会下发
  `new_block_ids_to_zero`，要求新分配或复用的 Mamba/GDN block在使用前清零。
- 当前 vllm-ascend公共 Worker仅在 Eagle3且投机 token数大于1时调用
  `_init_kv_zero_meta()`；Qwen3.5首版配置 `speculative_config=None`，因此310P MRV2
  zeroer实现存在但没有初始化。

修改内容：

- `vllm_ascend/_310p/worker/v2/model_runner.py`
  - 在310P KV Cache完成分配和 `bind_kv_cache()`后调用
    `_init_kv_zero_meta_if_needed()`。
  - 仅当 `kv_cache_config.needs_kv_cache_zeroing`为真时初始化
    `AscendKVBlockZeroer310V2`元数据。
  - 初始化放在 cache绑定之后，保证 zeroer读取到各 GDN层已绑定的 state cache。
  - 不修改公共 Worker的 Eagle3逻辑，不影响MRV1和其他设备。
- `tests/ut/_310p/test_model_runner_v2_310p.py`
  - 参数化验证需要清零时恰好初始化一次，不需要时不初始化。

验证状态：

- `py_compile`、`ruff check`和`git diff --check`已通过。
- 当前 Windows本地 Python环境未安装`pytest`，新增目标 UT需在服务器执行。
- 真实310P需重新验证 Qwen3.5 eager首个请求、请求结束后的 block复用和连续请求。

## 28. MRV2 zeroer的 pin-memory能力检测

问题现象：

- 第27节开始初始化 zeroer后，启动阶段报错：
  `NPUModelRunner310V2 has no attribute pin_memory`。

根因：

- 310P MRV1 runner保存 `self.pin_memory`实例字段，原 zeroer适配沿用了该字段。
- 当前上游 MRV2 runner不再提供此属性，而是在初始化 zeroer时动态调用
  `is_pin_memory_available()`；310P V2适配需要遵守新的 runner契约。

修改内容：

- `vllm_ascend/_310p/worker/v2/model_runner.py`
  - `_init_kv_zero_meta()`改为使用上游 MRV2相同的
    `is_pin_memory_available()`返回值构造 `AscendKVBlockZeroer310V2`。
  - 不向 MRV2 runner人为添加 `pin_memory`字段，也不改变310P MRV1实现。
- `tests/ut/_310p/test_model_runner_v2_310p.py`
  - 验证 V2 zeroer使用动态能力检测结果，并完成 metadata初始化调用。

验证状态：

- `py_compile`、`ruff check`和`git diff --check`已通过。
- 当前 Windows本地 Python环境未安装`pytest`，新增目标 UT需在服务器执行。
- 真实310P需重新验证 Qwen3.5 KV Cache初始化及首个 eager请求。

## 29. Hybrid shared cache 槽按实际 Spec 拆分分配

问题现象：

- Qwen3.5-4B 初始化完成后，GDN 的 `causal_conv1d` 报错：
  `convStates` 的实际 format 为 `FRACTAL_NZ`，而 310P 算子只支持 ND。

根因：

- 上游 Hybrid KV Cache 对齐逻辑允许一个 `KVCacheTensor.shared_by` 同时包含
  Full Attention 层和 GDN/Mamba 层；这是逻辑槽共享，不代表两类 cache 可以复用同一个
  tensor 对象或物理格式。
- 310P MRV2 首版 allocator 只读取 `shared_by` 的第一个 layer spec。若第一个 layer 是
  Full Attention，就分配 NZ 格式 K/V cache，再把同一个 tuple 绑定给该槽内全部 layer。
- GDN 因而把 NZ attention cache 当作卷积 state 使用，最终在
  `aclnnCausalConv1dV310` 的 format 校验阶段失败。
- MRV1 会遍历共享槽内的全部 layer：attention 独立分配 NZ K/V，linear attention
  独立分配 ND state，因此没有该问题。

修改内容：

- `vllm_ascend/_310p/worker/v2/model_runner.py`
  - 对每个 `shared_by` 槽中的 layer 按实际 cache spec 分组，而不是只采用首个 spec。
  - Attention 的分组键同时包含 backend 和 kernel block size，避免规格相同但实际布局要求
    不同的 attention layer 被错误合并。
  - 每个 Attention 分组独立分配 `FRACTAL_NZ` K/V cache；每个 Mamba/GDN 分组独立分配
    ND state cache，只向同组 layer 绑定对应对象。
  - 保留相同 spec/layout layer 的共享语义，也保留 `shared_layers` 的最终别名绑定。
  - 修改仅位于 310P MRV2 allocator，不改变 MRV1、公共 MRV2 或其他机型的缓存分配。
- `tests/ut/_310p/test_model_runner_v2_310p.py`
  - 增加 mixed `shared_by` 回归测试：同一逻辑槽同时包含 Full Attention 和 Mamba/GDN。
  - 验证 attention 得到 NZ K/V tuple、GDN 得到独立 ND state list，且两者不是同一对象。

验证状态：

- `py_compile` 和 `ruff check` 已通过。
- 当前 Windows 本地 Python 环境未安装 `pytest`，目标 UT 需在服务器环境补充执行。
- 真实 310P 需重新验证 Qwen3.5-4B eager，并继续验证 ACLGraph、TP 和并发场景。

## 30. Paged Attention kernel block size 的310P约束

问题现象：

- Qwen3.5-4B 的 Attention KV Cache 格式修正后，Paged Attention setup 报错：
  `head_size of keyCache should be no greater than 256 and block_size * head_size`
  `should be no greater than 128 * 128`。

根因：

- 310P Paged Attention 要求 `head_size <= 256`，并要求
  `kernel_block_size * head_size <= 128 * 128`。
- 公共 MRV2 `init_attn_backend()` 返回的默认 kernel block size 没有应用该310P专属限制。
- 对 head size 为256的 Qwen3.5，kernel block size 128会得到 `128 * 256`，超过算子上限；
  必须使用后端支持的64 block，使乘积降为 `64 * 256 = 128 * 128`。
- MRV1 的 `may_reinitialize_input_batch()` 已经按该公式过滤后端支持的 block sizes，
  310P MRV2此前遗漏了相同的设备约束。

修改内容：

- `vllm_ascend/_310p/worker/v2/model_runner.py`
  - 增加 `_ATTENTION_BLOCK_SIZE_LIMIT` 常量和
    `_adjust_kernel_block_sizes_310p()`。
  - 在公共 backend 初始化完成后、BlockTable和KV Cache创建前，对每个Attention组重新选择
    满足310P限制的最大受支持 kernel block size。
  - head size为256时从 `[128, 64]` 中选择64；head size超过256或没有合法block时提前抛出
    明确的 `NotImplementedError`，避免进入ATB后发生不透明的setup失败。
  - 调整后的同一 `kernel_block_sizes` 同时传递给BlockTable、NZ Cache分配和zeroer，保证三者布局一致。
- `tests/ut/_310p/test_model_runner_v2_310p.py`
  - 增加 head size为256时必须选择64 kernel block的回归测试。

验证状态：

- `py_compile`、`ruff check`和`git diff --check`已通过。
- 当前 Windows本地 Python环境未安装`pytest`，目标 UT需在服务器执行。
- 真实310P需重新验证 Qwen3.5-4B eager和ACLGraph，并确认日志中KV Cache采用64 kernel block。

## 31. GDN/Mamba state cache 的连续 stride 修复

问题现象：

- Qwen3.5-4B 在 eager 模式下已不再出现 cache format 或 Paged Attention setup 报错，
  但 `temperature=0` 的确定性请求仍从首轮生成开始输出乱码。
- eager 模式同样异常，因此该问题与 ACLGraph 捕获或重放无关。

根因：

- 310P MRV2 为每个 Mamba/GDN state 构造 `torch.as_strided()` 视图时，曾把
  `kv_cache_spec.page_size_bytes / dtype_size` 作为第一维 block stride。
- `page_size_bytes` 是整个 Hybrid cache page 的分配单位，可能同时覆盖 convolution state、
  recurrent state 和对齐填充；它不是单个 state tensor 相邻 block 之间的元素跨度。
- 与此同时，`storage_offset_bytes` 又按照每个 state 完整连续 tensor 的大小向后移动，导致
  state 起始偏移采用“按 state 连续分区”布局，而第一维 stride 采用“按 page 交错”布局，
  两种地址语义互相矛盾。
- 结果是 state 的 shape、dtype 和 ND format 都合法，算子不会在参数检查阶段报错，
  但 GDN 会读取错误的状态地址并造成模型精度异常。

MRV1 对照：

- MRV1 使用
  `raw_tensor[start_idx:target_idx].view(dtype).view(target_shape)`，为每个 state 从 raw buffer
  中划分完整的连续区域。
- 对形状 `(num_blocks, *shape)` 的 state，正确第一维 stride 是 `prod(shape)`，即
  `torch.empty(target_shape).stride()[0]`，而不是整个 Hybrid page 的元素数。

修改内容：

- `vllm_ascend/_310p/worker/v2/model_runner.py`
  - 删除 `elements_per_page = kv_cache_spec.page_size_bytes // dtype_size`。
  - 将 Mamba/GDN state 视图第一维改为 tensor 自身的连续 stride：
    `stride=(stride[0], *stride[1:])`。
  - 保留基于 `storage_offset_bytes` 的 state 分区，使 raw cache 布局与 MRV1 一致：
    `[state0 的全部 blocks][state1 的全部 blocks][padding]`。
  - 修改仅影响310P MRV2的 Mamba/GDN state视图，不改变Attention NZ cache、MRV1或其他机型。

验证状态：

- 需要在真实310P上重新验证 Qwen3.5-4B eager首token、连续decode和多请求场景。
- eager精度恢复后，再继续验证 ACLGraph、TP和并发场景。

## 32. 第一版合入准备：测试矩阵与文档入口

改动内容：

- `tests/e2e/pull_request/one_card/_310p/test_model_runner_v2_310p.py`
  - dense/hybrid 参数化增加 `Qwen3.5-2B`。
  - VL 增加 `Qwen3-VL-2B-Instruct`，并补 `FULL_DECODE_ONLY` 图模式（encoder eager、decode 捕获）。
- `tests/e2e/pull_request/four_card/_310p/test_model_runner_v2_310p.py`
  - 同步 TP2 dense/hybrid/VL；新增 TP2 W8A8 图模式、Qwen3.5-27B TP4 图模式和
    Qwen3-Embedding-8B TP2 pooling 图模式。
- `tests/e2e/pull_request/four_card/_310p/test_model_runner_v2_moe_310p.py`
  - 新增 Qwen3-30B-A3B TP2 eager/图、Qwen3.5-35B-A3B TP4 eager/图。
- 上述文本、VL、量化和 MoE 用例在同一 runner 内连续发送第二个请求，覆盖首请求完成后的
  request-state 清理、block-table condense 和 ACL Graph replay。
- `tests/ut/_310p/quantization/test_w8a8sc_310.py`
  - 覆盖 row-parallel `tp_rank != 0` 时 quant_bias 置零。
- `_310p/quantization/methods/w8a8_dynamic.py`
  - MoE 权重在 NZ 转换前统一为 `[E,K,N]`。
  - dynamic linear 在 row-parallel 非 0 rank 不重复应用 bias。
  - 310P dynamic **linear** 保持 ND `[N,K]` 并反量化到 fp16：GE 图模式会把
    NZ `[K,N]` 重铺成 `QuantBatchMatmulV3_NZ_NZ` kernel 21（Qwen3.5-2B TP2）。
- `_310p/quantization/modelslim_config.py`
  - 静态 W8A8/W8A8SC MoE 启动失败时给出 W8A8_DYNAMIC 指引。
  - 将 `tid2eid` 继续传给 MoE quant method，保留专家映射语义。
- `docs/source/developer_guide/Design_Documents/ModelRunner_v2_310P_pr_notes.md`
  - 新增社区合入注释：改动结构、支持矩阵、测试清单、评审清单。

改动原因：

- 第一版验收面从 Qwen3-Dense 扩展到 VL/MoE/Qwen3.5，需要对应 E2E 而不是只依赖手工 serve。
- 合入文档需要英文入口，方便社区评审对照文件与门禁，而不是只保留联调过程记录。

未纳入本轮：

- 临时精度 A/B 开关已移除；W8A8-Dynamic **MoE** 仍使用 `[E,K,N]` grouped-matmul
  NZ。Dense linear 在 310P 上改为 ND fp16 dequant，避免 GE 编译
  `QuantBatchMatmulV3_NZ_NZ`。真实检查点精度仍需在 310P 服务器上按验收矩阵确认。
- Prefix cache / MTP 的阶段开关环境变量（第一版保持启动拒绝）。

## 33. 去除对未合入 Triton dispatcher 的依赖

问题现象：

- 分支代码在导入期依赖 `vllm.model_executor.triton_dispatcher`，而
  [vLLM PR #43048](https://github.com/vllm-project/vllm/pull/43048) 至今未合入主线，
  在主线 vLLM 上 `import vllm_ascend.worker.v2.block_table` 和
  `tests/ut/_310p/test_model_runner_v2_310p.py` 都会 `ImportError`。
- 为配合 dispatcher ABI，公共 `_compute_slot_mappings_kernel` 曾从多 cache group
  raw-pointer 改为单 cache group Tensor，导致主线已有的
  `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_slot_mapping.py`
  参数不再匹配，也让 910B/910C 由单次 launch 变成按 cache group 逐次 launch。

修改内容：

- `vllm_ascend/worker/v2/block_table.py`
  - 回退到主线实现：无 `@pluggable_kernel`，保留原 raw-pointer ABI 与单次 launch。
- `vllm_ascend/_310p/worker/v2/block_table.py`
  - 新增模块级 `_compute_group_slot_mappings()`，用 NumPy 生成 slot ID，算法与 310P
    Model Runner V1 的 `_compute_slot_mapping_numpy()` 一致。
  - `compute_slot_mappings()` 直接调用该函数，不再延迟导入公共 Triton kernel。
  - CPU owner tensor 增加 NumPy 视图（`block_tables_np` 等），gather 与 slot mapping
    共用，避免每步重复 `tensor.numpy()`。
- `vllm_ascend/_310p/worker/v2/kernel_registry.py`
  - 改为可选接入点：`try/except ImportError` 探测 dispatcher，导出
    `HAS_TRITON_DISPATCHER`、`KERNEL_IMPLS` 和 `register_310p_kernels()`。
  - 首版 `KERNEL_IMPLS` 为空，注册函数是 no-op。
- `vllm_ascend/_310p/worker/v2/model_runner.py`
  - 由“导入 registry 触发副作用”改为显式调用 `register_310p_kernels()`。
- `tests/ut/_310p/test_model_runner_v2_310p.py`
  - 删除 `_get_kernel_impl` 断言，改为校验无 dispatcher 时注册为 no-op。
  - 新增静态校验：310P V2 BlockTables 不导入任何 triton 模块，也不导入公共 V2
    block table。
  - 新增多 cache group slot mapping 与空 batch padding 用例。

改动原因：

- 插件不能依赖未合入的上游 API；310P 已经通过 `patch_v2/patch_block_table.py` 整类
  替换 BlockTables，函数级 dispatcher 对第一版没有增量价值。
- 公共 kernel 保持主线形态，910B/910C 行为和既有 nightly 用例都不受 310P 适配影响。
- dispatcher 合入后仍可在 `kernel_registry.py` 一处接入，无需回改调用侧。

## 34. Qwen3.5-2B-W8A8 图模式 QuantBatchMatmulV3 故障

问题现象：

- Qwen3.5-2B-W8A8 TP2 + `FULL_DECODE_ONLY` 在 `profile_run` 触发
  `QuantBatchMatmulV3_NZ_NZ_int8_int8_fp16_high_performance_21`
  （hash `5247287448945562503`）。Eager 同检查点可跑通。
- Qwen3.5-4B-W8A8 / 9B-W8A8 TP2 图模式原本即可跑通。

根因：

- 310P dynamic linear 把权重存成 format-29 的 `[K,N]`（NZ(`[N,K]`) 再 transpose）。
- Eager `npu_quant_matmul` 接受该布局；GE/`torch.compile` 会按 `[K,N]` 重新 NZ
  铺砖，打到 kernel 21。2B fused `qkv_proj` 的 KV shard 为 N=256，4B/9B 为 N=512。

修改：

- `_310p/quantization/methods/w8a8_dynamic.py` 的 **linear** 方案改为保持 ND
  `[N,K]`，int8×scale 反量化后走 `F.linear`。MoE grouped-matmul NZ 不变。
- `quantization/modelslim_config.py` 拒绝 MLX/JANG 的 `bits`+`group_size`
  配置，避免误入 Ascend ModelSlim。

验证：

- Qwen3.5-2B-W8A8 TP2 图模式两次 curl 200，decode 走 ACL Graph `num_tokens=1`。
- 改动后复测 Qwen3.5-9B-W8A8 TP2 图模式，两次 curl 200。

## 35. Qwen3 W8A8SC MRv2 图模式验收

问题现象：

- 本地 W8A8SC 检查点是 ModelSlim `save_sharded_state_310.py` 预切分压缩权重，
  必须 `--load-format sharded_state`，且 TP 必须与 shard 目录一致。

修改：

- 运行时仍走既有 310P 方案 `AscendW8A8SCLinearMethod310`
  (`npu_matmul_compress_dequant`)。MRv2 无需新 kernel。
- 将 W8A8SC e2e 从 dense W8A8/W8A8-Dynamic 参数化中拆出，显式传
  `load_format="sharded_state"`。
- 新增 Qwen3-8B TP2、Qwen3-32B TP4、Qwen3-VL-4B TP1、Qwen3-VL-8B TP2
  ACL Graph e2e。
- `tests/ut/_310p/quantization/test_w8a8sc_310.py` 取消 apply skip
  （PTA 26 已有该算子；用例仍 mock）。

验证（`FULL_DECODE_ONLY` `[1,16]`，docker `v0230_mrv2_dy`）：

- Qwen3-8B-W8A8SC TP2（NPU 6,7）两次 curl 200。
- Qwen3-VL-8B-W8A8SC TP2 文本 + 图像 curl 200。
- Qwen3-VL-4B-W8A8SC TP1（仅有 TP1 shard）文本 + 图像 curl 200。
- Qwen3-32B-W8A8SC TP4（仅有 TP4 shard；NPU 2,3,6,7）两次 curl 200。

