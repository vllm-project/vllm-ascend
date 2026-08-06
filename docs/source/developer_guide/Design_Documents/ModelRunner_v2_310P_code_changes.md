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
- 导入该 runner 时加载 `kernel_registry.py`，注册首版需要的 310P kernel 实现。
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

- 上游输入准备、状态更新等路径仍包含未添加 `@pluggable_kernel` 的 Triton kernel，无法
  通过 dispatcher 注册替换。
- 310P 已经能从 scheduler 获取 CPU metadata，应直接使用这些 mirror，避免从 NPU
  Tensor 反向 D2H。
- 类级差异、RequestState 和 ACLGraph manager 不属于函数级 kernel dispatcher 的适用
  范围，需要通过继承和类属性扩展。

## 4. 可插拔 Slot Mapping Kernel

### `vllm_ascend/worker/v2/block_table.py`

改动内容：

- 在 vLLM Ascend 自有 `_compute_slot_mappings_kernel` 前增加：

  ```python
  @pluggable_kernel
  @triton.jit
  ```

- 将 kernel ABI 从多 cache group raw-pointer 数组调整为单 cache group、显式
  block-table Tensor。
- `AscendBlockTables.compute_slot_mappings()` 按 cache group 调用该 kernel。
- 没有平台注册实现时仍执行默认 Triton kernel。

改动原因：

- PR #43048 的 dispatcher 只能替换显式添加 `@pluggable_kernel` 的对象。
- 原 raw-pointer ABI 无法让 310P Python/PyTorch 注册函数访问实际 block-table Tensor。
- 使用显式 Tensor 参数后，默认 Triton 实现和 310P 原生实现可以共享
  `kernel[grid](...)` 调用形式。

### `vllm_ascend/_310p/worker/v2/kernel_registry.py`

改动内容：

- 使用 `register_kernel()` 注册以下全限定名：

  ```text
  vllm_ascend.worker.v2.block_table._compute_slot_mappings_kernel
  ```

- 提供 CPU/PyTorch slot-mapping 实现。
- 注册函数保持与默认 Triton kernel 相同的参数语义，并接收 dispatcher 传入的 `grid`。
- 根据 token position、block size 和物理 block number 生成 slot ID。

改动原因：

- 310P 不执行 Triton，需要在不修改 vLLM 上游的前提下替换 vLLM Ascend 自有 kernel。
- registry 只在真正导入 310P V2 runner 时加载，避免影响 V1。

## 5. 310P BlockTables

### `vllm_ascend/_310p/worker/v2/block_table.py`

改动内容：

- 新增 `Ascend310PBlockTables`。
- 使用 CPU Tensor 保存 block table 和 block 数量，作为 block-table metadata 的所有者。
- `append_block_ids()` 直接更新 CPU block table。
- `gather_block_tables()` 根据 request index 在 CPU 聚合输入 block table，再复制到固定的
  NPU buffer。
- `compute_slot_mappings()` 调用可插拔 slot-mapping kernel；310P 实际进入 registry 中的
  CPU/PyTorch 实现。
- 支持多个 KV cache group。
- dummy block table 和 slot mapping 复用持久化 NPU Tensor，保持 ACLGraph 所需的固定
  地址。
- 默认 Triton kernel 在 `compute_slot_mappings()` 内延迟导入。

改动原因：

- 上游 BlockTables 的 staged write 和 gather kernel 尚未提供 dispatcher，不能只替换
  slot-mapping kernel。
- CPU owner 可以直接消费 scheduler metadata，避免热路径 D2H。
- 延迟导入确保仅加载该类不会给 V1 引入 dispatcher 或 Triton kernel 定义。

### `vllm_ascend/patch/worker/patch_v2/patch_block_table.py`

改动内容：

- 310P 设备将上游 V2 `model_runner.BlockTables` 替换为 `Ascend310PBlockTables`。
- 非 310P 设备继续使用公共 `AscendBlockTables`。

改动原因：

- 310P 必须替换上游 BlockTables 中尚未开放 dispatcher 的 staged-write 和 gather
  Triton 路径。
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

- 上游 `StagedWriteTensor.apply_write()` 使用未开放 dispatcher 的 Triton kernel。
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

- 上游多维 RoPE position 准备使用未开放 dispatcher 的 Triton kernel。
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
- 上游 accepted-token scatter kernel 尚未开放 dispatcher。

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

- 上游 `VllmConfig` 在 Worker 创建前要求 `HAS_TRITON=True`，导致 310P 即使已经注册
  非 Triton kernel 实现也无法完成配置创建。
- dispatcher 不会自动移除该配置校验，因此必须由 Ascend 平台只针对 310P 接管。
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

- 这些调用最终进入 vLLM 上游未添加 `@pluggable_kernel` 的 Triton kernel，当前无法使用
  dispatcher 替换。
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
