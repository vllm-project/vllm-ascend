# 310P Model Runner V2 代码改动说明

本文仅说明 310P Model Runner V2 第一版重构涉及的代码文件、改动内容和改动原因。

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
