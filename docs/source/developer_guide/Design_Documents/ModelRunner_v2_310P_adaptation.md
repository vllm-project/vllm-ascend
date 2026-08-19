# 310P Model Runner V2 分阶段适配指导

## 1. 背景与目标

仓库中的通用 Model Runner V2 已完成 Ascend 适配，但 310P 仍固定使用
`NPUWorker310 + NPUModelRunner310`（Model Runner V1）。310P 不支持 Triton，现有 V1
还包含独立的输入准备、Attention、ACL Graph、多模态、采样、KV Cache、量化和权重
NZ 逻辑，不能仅通过替换 runner 基类完成迁移。

本适配最初计划依赖上游 [vLLM PR #43048](https://github.com/vllm-project/vllm/pull/43048)
引入的 Triton kernel dispatcher：调用方继续使用 `kernel[grid](...)`，平台插件通过
`register_kernel()` 注册 PyTorch、NumPy、`torch_npu` 或 ACLNN 实现。该 PR 至今未合入
vLLM 主线，因此**第一版不依赖该机制**，改为使用仓库既有的类级别与模块级别替换边界
（详见第 4 章）。dispatcher 合入后的接入点保留在
`_310p/worker/v2/kernel_registry.py`。

310P Model Runner V2 的最终目标是与 310P Model Runner V1 **能力完全对齐，不新增、
不减少**。为了控制首版风险，采用分阶段交付：

- **第一版**：TP、NZ、多模态、chunked prefill、ACL Graph、W8A8/W8A8SC；
- **第二版**：补齐 prefix cache、完整后处理和 MTP，达到最终 V1 能力对齐；
- V1 当前不支持的功能不进入 V2 适配范围。

核心文本模型以 Qwen3-8B 和 Qwen3.5-4B 为重点。由于这两个 checkpoint 本身不是视觉
语言模型，第一版使用仓库现有的 Qwen3-VL-8B-Instruct 作为多模态验收模型。

本文是实施和验收指导，不表示相关能力已经在 310P Model Runner V2 上验证通过。

## 2. 310P V1 能力边界

### 2.1 最终必须与 V1 对齐的能力

| 能力分类 | 310P V1 当前范围 | Model Runner V2 最终要求 |
| --- | --- | --- |
| 并行策略 | 仅 TP | 仅 TP，不增加其他并行策略 |
| 权重布局 | Weight NZ | 保持相同转换策略与例外 |
| 多模态 | 支持 | 保持现有模型和输入类型范围 |
| ACL Graph | 支持 | 保持现有 capture/replay 语义 |
| 推测解码 | 仅 MTP | 仅 MTP，不增加其他方法 |
| Prefix cache | 支持 | 保持现有 block 复用语义 |
| Chunked prefill | 支持 | 保持现有调度与 Attention 语义 |
| 后处理 | 支持 | 参数和输出语义与 V1 一致 |
| 量化 | 支持 | 首版重点 W8A8、W8A8SC |

### 2.2 不在当前适配范围的能力

以下能力既不作为首版目标，也不作为后续 V1 对齐目标：

- DP、EP、PP、CP、DCP、PCP；
- FlashComm1、MC2、EPLB；
- Eagle、ngram、DFlash、DSpark 等非 MTP 推测解码；
- MLA、SFA、DSA 等 310P V1 当前未支持的 Attention backend；
- LoRA、sleep mode、UVA offload 等未纳入 310P V1 当前范围的能力；
- 为 Model Runner V2 单独增加、但 V1 不具备的新用户功能。

若通用 V2 默认开启这些能力，310P 必须在配置校验阶段明确拒绝，不能静默进入未验证
路径，也不能为了“复用 V2”扩大 310P 的产品范围。

## 3. 分版本范围

### 3.1 第一版范围

第一版必须支持：

- TP：TP1 冒烟、TP2 硬门禁；
- Weight NZ；
- 多模态文本+图片请求；
- chunked prefill；
- ACL Graph，至少 `FULL_DECODE_ONLY`；
- W8A8、W8A8SC；
- 完成真实请求所必需的 greedy 基础采样。

第一版重点模型及职责如下：

| 模型 | 第一版主要职责 |
| --- | --- |
| `Qwen/Qwen3-8B` | 标准 dense Attention、TP、FP16、chunked prefill、ACL Graph |
| `Qwen/Qwen3.5-4B` | GDN/Mamba + Full Attention hybrid、TP、chunked prefill、ACL Graph |
| `Qwen/Qwen3-VL-8B-Instruct` | 多模态文本+图片、ViT/encoder、TP |
| `vllm-ascend/Qwen3-8B-W8A8` | W8A8、Weight NZ、ACL Graph |
| `vllm-ascend/Qwen3-8B-w8a8sc-310-vllm-tp1` | W8A8SC、Weight NZ |

模型路径必须在实施前确认可访问并固定 revision。若实际 checkpoint 名称变化，更新测试
配置即可，不改变功能验收责任。

### 3.2 第一版暂缓、第二版补齐

- prefix cache；
- 完整后处理；
- MTP。

第一版允许使用 greedy 生成验证模型输出，但不因此声明“后处理已适配”。第二版完成后
才可声明 Model Runner V2 与 310P V1 功能完全对齐。

### 3.3 第二版范围

第二版只补齐 V1 已有能力：

- prefix cache；
- temperature/random、top-k、top-p、min-p；
- repetition、frequency、presence penalty；
- bad words、logit bias；
- sampled token logprobs、prompt logprobs；
- structured output 中 V1 已支持的 grammar bitmask 范围；
- MTP eager 与 ACL Graph；
- MTP rejection sampling、KV block 清零和 hybrid cache state 更新。

第二版不增加 Eagle、ngram、DFlash 或其他 speculative decoding 方法。

## 4. Triton 替换策略

### 4.1 为什么第一版不使用 dispatcher

PR #43048 尚未合入 vLLM 主线，一旦在 `import` 期依赖
`vllm.model_executor.triton_dispatcher`，插件在主线 vLLM 上会直接 `ImportError`。同时
dispatcher 只能替换显式添加 `@pluggable_kernel` 的对象，而 310P 首版可达的 Triton 路径
（staged write、gather、多维 RoPE、采样）在上游都没有该装饰器，仅靠 dispatcher 也无法
覆盖。因此第一版全部使用仓库既有的替换边界：

| 替换边界 | 使用位置 | 说明 |
| --- | --- | --- |
| 类替换（patch 模块属性） | `patch_v2/patch_block_table.py` | 310P 使用 `Ascend310PBlockTables` |
| 类属性扩展点 | `request_state_cls`、`aclgraph_manager_cls` | runner 内注入 310P 实现 |
| 子类覆写 | `Ascend310PModelState`、`Ascend310PRopeState` | 多模态 position/state |
| 模块属性替换 | `patch_v2/patch_triton.py` | 采样类 kernel 全平台替换 |

这些边界与仓库其他 NPU 适配一致，不额外引入新的分发机制。

### 4.2 目录边界与 dispatcher 接入点

第一版采用以下目录边界：

```text
vllm_ascend/
├── worker/v2/block_table.py                # 公共默认 Triton kernel，保持与主线一致
└── _310p/worker/v2/
    ├── kernel_registry.py                  # dispatcher 合入后的唯一接入点
    ├── model_runner.py                     # 仅保留 runner 契约差异
    ├── block_table.py                      # CPU-owned V2 BlockTables + NumPy slot mapping
    ├── states.py                           # 无 Triton staged-write state
    ├── model_state.py                      # 310P 多模态 RoPE/model state
    └── aclgraph.py                         # 310P V2 graph 策略
```

`kernel_registry.register_310p_kernels()` 只在导入 310P V2 runner 时调用一次；V1 路径不
导入该模块。当前 `KERNEL_IMPLS` 为空，函数为 no-op；PR #43048 合入后，只需登记
kernel 全限定名与 310P 实现，调用侧保持 `kernel[grid](...)`，并可同步删除对应的子类
覆写。允许设备判断存在于该入口，但不允许在 Model Runner V2 主流程和每个 kernel 调用
点散落 `is_310p()`。

### 4.3 替换机制的边界

无论使用类替换还是未来的 dispatcher，都不自动解决：

- 默认 Triton 模块能否在无 Triton 环境完成 import；
- 需要 CPU request-state mirror 的高层算法；
- graph capture 中 tensor 地址、shape 和 stream 顺序；
- 同一进程中按不同设备动态切换实现。

第一版必须建立两道门禁：

1. **Import gate**：310P 进程不导入定义默认 Triton kernel 的模块；确实需要导入时，镜像
   必须能完成 `@triton.jit` 的模块级定义。允许安装 Triton Python 包，但不允许在 310P 上
   编译或执行 Triton kernel；
2. **Invocation gate**：真实请求执行期间没有 Triton kernel 编译或调用。

310P V2 BlockTables 不导入 `vllm_ascend/worker/v2/block_table.py`，因此不受该模块的
`@triton.jit` 定义影响；单元测试对该约束做静态校验。若镜像完全不包含 Triton Python
包，则需要进一步提供 lazy/default stub 机制，不能通过伪造 `HAS_TRITON=True` 解决。

### 4.4 分版本 Kernel Inventory

| 路径 | 第一版 | 第二版 | 使用的替换边界 |
| --- | --- | --- | --- |
| Block table slot mapping | 必须 | 回归 | 310P BlockTables 内 NumPy 实现 |
| Block table staged write/gather | 必须 | 回归 | 310P BlockTables/RequestState |
| 多维 RoPE position | 多模态必须 | 回归 | `Ascend310PRopeState` |
| KV block zeroer | 仅非 MTP 可达路径 | MTP 清零 | zeroer 类/function |
| greedy sampling | 必须 | 回归 | sampler function |
| random/gumbel | 暂缓 | 必须 | sampler function/kernel |
| top-k/top-p/min-p | 暂缓 | 必须 | sampler function/kernel |
| penalties/bincount | 暂缓 | 必须 | function + kernel |
| bad words | 暂缓 | 必须 | function + kernel |
| logprobs | 暂缓 | 必须 | function + kernel |
| grammar bitmask | 暂缓 | 必须 | 模块属性替换 |
| rejection sampling | 禁用 | MTP 必须 | function + kernels |
| DFlash kernels | 禁用 | 禁用 | 不替换 |

310P slot mapping 直接消费 scheduler 和 CPU state 的 request index、`query_start_loc` 与
position，避免从 device tensor 反向 D2H；公共 `AscendBlockTables` 与其 raw-pointer ABI
保持主线形态不变。

## 5. 当前 V1 到 V2 的差异映射

| 能力 | 310P V1 实现 | V2 迁移要求 |
| --- | --- | --- |
| Runner 选择 | 310P worker 固定 V1 runner | worker 内按显式开关选择 V1/V2 |
| 输入准备 | CPU 计算 position、slot mapping、metadata | 接入 V2 RequestState/InputBatch，不做 D2H |
| Block table | NumPy 计算，支持多 cache group | 提供无 Triton V2 BlockTables |
| Attention | 310P paged/splitfuse NPU 算子 | 适配 V2 metadata/context |
| Qwen3.5 GDN | 310P GDN/causal-conv 算子 | eager、chunked prefill、graph 覆盖 |
| KV Cache | Attention K/V 分离，格式 29 | Attention NZ；Mamba 按 cache spec |
| 多模态 | 310P MM encoder/Attention 路径 | 接入 V2 MM input/encoder state |
| ACL Graph | 捕获完整 forward | 适配 V2 graph manager/descriptor |
| Weight NZ | post-load `maybe_trans_nz()` | 保证 V2 loader hook 一致 |
| 量化 | 310P ModelSlim methods | W8A8/W8A8SC 加载、post-load、forward |
| 后处理 | 310P 非 Triton采样实现 | 第二版按 V2 函数接口注册 |
| MTP | 310P proposer/rejection/zeroer | 第二版适配 V2 speculator |

## 6. 第一版实施流程

### 阶段 0：冻结 V1 基线

在 310P V1 上记录：

- Qwen3-8B FP16：TP1/TP2 eager、chunked prefill、ACL Graph；
- Qwen3.5-4B FP16：TP1/TP2 eager、chunked prefill、ACL Graph；
- Qwen3-VL-8B-Instruct FP16：TP1/TP2 文本+图片；
- Qwen3-8B W8A8：eager、ACL Graph；
- Qwen3-8B W8A8SC：真实权重推理；
- 参数 ACL format、KV cache format、显存峰值、graph capture size；
- 相同 prompt 下的 token 输出和错误边界。

若 V1 当前缺少某个组合，记录为“新增 V2 验收项”，不能假设已有可信基线。所有比较
固定 checkpoint revision、prompt、sampling params 和 seed。

### 阶段 1：打通 Worker 到 V2 Runner

在 `NPUWorker310.init_device()` 中按 `self.use_v2_model_runner` 选择 runner：

```python
if self.use_v2_model_runner:
    self.model_runner = NPUModelRunnerV2For310P(self.vllm_config, self.device)
else:
    self.model_runner = NPUModelRunner310(self.vllm_config, self.device)
```

要求：

- 平台仍选择 310P worker；
- 保留 310P device 初始化、关闭 JIT compile、workspace 和内存 profiling；
- 日志打印实际 worker/runner 类；
- 关闭 V2 开关时 V1 行为不变；
- 第一版对 prefix cache、完整后处理和 speculative config 给出范围内的明确提示；
- 对 V1 从未支持的并行/Attention/spec decode 方法直接拒绝。

### 阶段 2：替换第一版可达的 Triton 路径

第一版不修改 vLLM 上游，也不修改公共 `AscendBlockTables`，只在 `_310p` 目录内提供
无 Triton 实现，并通过既有 patch 与类扩展点接入。第一版的具体替换如下：

1. 310P `BlockTables` 使用 CPU request metadata 生成 block table 和 slot mapping；
2. `RequestState` 和多模态 RoPE 使用 CPU owner、脏行 H2D 和原生 NPU tensor 操作；
3. input IDs、position、sequence length 和 chunked-prefill 状态由 CPU mirror 构造；
4. greedy sampler 和 post-update 使用原生 PyTorch/NPU 算子；
5. KV block zeroer 使用直接 tensor 清零；
6. 仅在导入 310P V2 runner 时调用 `kernel_registry.register_310p_kernels()`；V1 runner 不
   导入该模块，同时禁止 310P 静默回退默认 Triton 实现。

随机采样、logprob、grammar、rejection sampling 和 DFlash 等首版不可达能力不提供 310P
实现，也不为了形式完整提前添加空实现。第二版适配后处理和 MTP 时再补齐对应替换。

### 阶段 3：适配 RequestState、InputBatch 和 Slot Mapping

通用 V2 slot mapping 消费 NPU tensor 并调用 Triton。310P 应：

1. 根据排序后的 `req_ids` 和 `scheduler_output.num_scheduled_tokens` 构造
   `num_scheduled_tokens_np`、`query_start_loc_np`；
2. 从 `req_states.num_computed_tokens_np` 加请求内 offset 生成 `positions_np`；
3. 生成与 token 顺序一致的 `req_indices_np`；
4. NumPy 计算 logical block 到 physical block 映射；
5. 写入预分配 int32 CPU slot-mapping buffer，一次性 H2D；
6. position、query start location、sequence length 写入已有 V2 buffer。

禁止对 NPU tensor 调用 `.cpu()`、`.numpy()` 或 `.item()` 生成 slot mapping。图 replay
使用的 buffer 地址必须固定。

UT 至少覆盖单 Attention cache group、Attention + Mamba 混合 group、padding、request
condense、chunked prefill 跨 step 以及 graph dummy request。

### 阶段 4：适配 Qwen3-8B

- `AscendAttentionBackend310` 消费 V2 `InputBatch`；
- prefill、chunked prefill、decode metadata 正确；
- K/V cache 分开并以 `ACL_FORMAT_FRACTAL_NZ` 分配；
- TP2 的 QKV shard、row/column parallel linear、LM head gather 正确；
- TP collective 在 eager 和 graph 中顺序一致；
- chunk 边界不重复或遗漏 token；
- 最终输出与 V1 基线一致。

### 阶段 5：适配 Qwen3.5-4B Hybrid

Qwen3.5-4B 必须覆盖：

- Full Attention 与 GDN/Mamba layer 的混合 cache spec；
- Mamba state/conv cache shape、dtype 和生命周期；
- 310P causal-conv、GDN gating、chunk/recurrent 路径；
- chunked prefill 跨 chunk 的 Mamba state 连续性；
- TP2 下 GDN/Mamba 参数 shard 与 collective；
- profile/dummy run 不生成越界 block table；
- graph replay 间 Mamba state、seq lens、Attention block table 同步更新；
- 非 MTP 场景不创建 speculative block。

### 阶段 6：适配 Weight NZ

Weight NZ 继续由 post-load hook 负责：

```text
model loader
  -> process_weights_after_loading()
  -> quant_method.process_weights_after_loading()
  -> maybe_trans_nz(weight)
  -> torch_npu.npu_format_cast(weight, 29)
```

要求：

1. 在 V1/V2 共用 NPU 初始化位置设置
   `torch.npu.config.allow_internal_format = True`；
2. V2 loader 调用前完成 post-load patch；
3. runner 不再次遍历全部 parameter 转 NZ；
4. 保留 FP32、meta、conv1d 和矩阵维度为 1 的 ND 例外；
5. TP2 每个 shard 都检查实际格式；
6. Weight NZ 与 Attention KV Cache NZ 分别验证。

当前 310P `_should_trans_nz()` 绕过 `weight_nz_mode`，对支持的非 FP32、非 meta 权重
始终尝试转 NZ。适配保持 V1 语义，不顺便修改配置行为。

### 阶段 7：适配 W8A8 和 W8A8SC

两种量化不能只以同一个 W8A8 测试代替：

- 分别验证 quant config 注册和方法选择；
- 分别验证 weight、scale、bias 加载；
- 分别验证 post-load pack/NZ 转换顺序；
- 禁止 runner 二次转换 packed weight；
- TP shard 后的 scale/bias 与 weight 对应；
- eager 与 graph 输出分别验证；
- 使用真实 checkpoint 做精度/输出门禁，dummy 只用于结构冒烟。

Qwen3-8B 是 W8A8/W8A8SC 的第一版硬门禁。若有稳定的 Qwen3.5-4B 对应量化
checkpoint，可增加回归，但不能用不存在或随机量化的权重阻塞首版。

### 阶段 8：适配多模态

第一版使用 `Qwen/Qwen3-VL-8B-Instruct`，要求：

- TP1/TP2 文本+图片请求；
- processor 输出、placeholder、prompt embeds 与 V2 request state 对齐；
- MM encoder 的 profile、实际执行和缓存生命周期正确；
- 310P MM encoder Attention/算子路径无 Triton；
- 图模式开始前先保证 eager 文本+图片真实请求通过；
- 同一服务先文本请求再图片请求、再文本请求，排除 stale MM state；
- 不通过临时减少视觉层数作为最终验收。

Qwen3-8B、Qwen3.5-4B 继续作为文本主模型；不能用它们的 text-only 请求声明多模态
支持。

### 阶段 9：适配 Chunked Prefill

两个文本主模型都必须覆盖：

- prompt 长度大于 `max_num_batched_tokens`，确保发生真实分块；
- 第一块、中间块、最后一块的 query start location 和 slot mapping；
- Qwen3-8B 跨 chunk Attention KV 写入；
- Qwen3.5 跨 chunk Mamba/GDN state 与 Attention KV 一致；
- TP1/TP2 与非 chunked 基线输出一致；
- eager 与 ACL Graph decode 组合；
- 多请求 mixed prefill/decode 不越界。

第一版不要求 prefix cache；不得把 prefix cache 命中当作 chunked prefill 验收。

### 阶段 10：适配 ACL Graph

310P V1 捕获完整 forward，并直接调用 310P NPU 算子。V2 应提供 310P graph manager/
策略，或在通用 manager 中建立明确硬件接口。

第一版要求：

- Qwen3-8B、Qwen3.5-4B TP1/TP2 `FULL_DECODE_ONLY`；
- Qwen3-8B W8A8、W8A8SC 的 graph smoke/正确性；
- capture/replay 的 input、block table、slot mapping、KV cache 地址固定；
- eager 与 graph greedy token 一致；
- chunked prefill 后 decode graph 正确；
- request finish/condense 后第一次 replay 不读旧 block table；
- Qwen3.5 Mamba state 在 replay 间正确更新；
- 日志存在实际 replay 证据；
- 默认 full+piecewise 至少 smoke，若 V1 对该组合不支持则不扩大范围。

多模态 ACL Graph 是否作为首版强门禁，严格跟随 310P V1 当前支持矩阵：V1 已支持的
组合必须对齐；V1 未支持的组合不得作为 V2 新增功能。

## 7. 第一版验收矩阵

| 模型 | 场景 | TP1 | TP2 | ACL Graph | 真实权重 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-8B FP16 | dense + chunked prefill | 必须 | 必须 | 必须 | 必须 |
| Qwen3.5-4B FP16 | hybrid + chunked prefill | 必须 | 必须 | 必须 | 必须 |
| Qwen3-VL-8B-Instruct FP16 | text+image | 必须 | 必须 | 按 V1 范围 | 必须 |
| Qwen3-8B W8A8 | quant + NZ | 必须 | 按 checkpoint/V1 范围 | 必须 | 必须 |
| Qwen3-8B W8A8SC | quant + NZ | 必须 | 按 checkpoint/V1 范围 | 按 V1 范围 | 必须 |

每个服务测试至少包含：

1. `/v1/models` 返回 200；
2. 首个和连续第二个真实请求返回 200、输出非空；
3. 与 V1 或可信基线比较 greedy token；
4. TP1/TP2 在相同条件下输出一致；
5. chunked prefill 确实发生，而不是 prompt 一次性调度；
6. graph 日志存在真实 replay；
7. 日志无 Triton 编译、调用或静默 fallback；
8. weight、Attention KV cache ACL format 正确；
9. 多模态测试包含真实图片；
10. prefix cache、完整后处理、MTP 未被错误声明为首版支持。

dummy load 可以先验证结构、算子和 API 路径，但最终结论必须由真实权重请求给出。

## 8. 第二版：补齐 V1 能力

### 8.1 Prefix Cache

- Qwen3-8B 重复前缀命中与无 prefix cache 输出一致；
- Qwen3.5 Attention block 与 Mamba cache 的复用边界正确；
- prefix cache 与 chunked prefill 组合；
- TP1/TP2；
- eager 与 V1 已支持的 ACL Graph 组合；
- block 释放、重用后无 stale state。

### 8.2 完整后处理

按 V1 已支持参数逐项迁移，不增加新 sampler 功能：

- temperature/random，固定 seed；
- top-k、top-p、min-p；
- repetition/frequency/presence penalty；
- bad words、logit bias；
- sampled token logprobs、prompt logprobs；
- V1 已支持范围内的 grammar bitmask/structured output。

通过第 4 章的替换边界提供 310P 非 Triton 实现；参数不能被静默忽略。TP1/TP2、
eager/graph 分别执行与 V1 的行为对比。

### 8.3 MTP

仅适配 MTP：

1. 初始化 V2 MTP speculator；
2. 适配 `1 + K` query length、draft token、expanded index mapping；
3. 适配 accepted/rejected token 的 state 回写；
4. 接入 310P PyTorch rejection/recovered-token 路径；
5. 接入非 Triton KV block zeroer；
6. 分别清理 Attention speculative block 和 Mamba speculative state；
7. 非 uniform batch 按 V1 语义回退 eager；
8. 建立 MTP graph capture shape；
9. 验证 TP1/TP2 输出、接受率和连续请求稳定性。

MTP checkpoint 缺失时标记“checkpoint missing”，不能用 dummy 声明支持。Eagle、ngram、
DFlash、DSpark 继续明确拒绝。

## 9. 回归要求

- 新增 310P V2 UT；
- 第一版模型/功能矩阵 E2E；
- 第二版 prefix cache、后处理、MTP E2E；
- 现有 310P V1 回归，确保行为未减少；
- 非 310P 通用 V2 回归，确保 310P 替换未污染其他平台；
- 配置拒绝测试，确保未支持能力不会误入；
- 无 Triton import/invocation 门禁；
- dummy 与真实权重证据分开记录；
- 理论 max model len 与当前硬件实测可运行长度分开记录。

## 10. 重点风险

- **上游 API 风险**：PR #43048 未合入且接口可能变化，禁止在导入期依赖该模块。
- **能力漂移风险**：不得因 V2 已有功能而扩大 310P 范围，也不得漏掉 V1 已有能力。
- **替换覆盖不足**：替换单个 kernel 不能让整个 V2 脱离 Triton，必须逐路径核对
  inventory。
- **Import 风险**：不调用 Triton 不代表无 Triton 环境可 import 默认模块。
- **全局替换风险**：模块属性替换和未来的 kernel 注册都是进程级生效，测试进程要隔离
  这些状态。
- **D2H 风险**：从 device position 反算 CPU slot mapping 会每步同步。
- **图地址风险**：replay 前重建 tensor 会破坏固定地址。
- **TP 风险**：单卡通过不能证明 shard、collective、LM head gather 正确。
- **Hybrid cache 风险**：Qwen3.5 同时包含 Attention 与 Mamba/GDN 状态。
- **多模态误判风险**：text-only 请求不能证明 MM encoder/processor 可用。
- **Chunked prefill 误判风险**：必须证明真实分块发生。
- **NZ 混淆风险**：Weight NZ 通过不代表 KV Cache NZ 正确。
- **量化二次转换风险**：runner 统一转 NZ 可能破坏 packed weight 和 scale 映射。
- **阶段范围风险**：第一版不应被 prefix cache、完整后处理或 MTP 阻塞；第二版必须
  补齐这些 V1 能力后才能声明最终对齐。

## 11. 完成标准

### 11.1 第一版完成

- 不依赖未合入的上游 PR，在主线 vLLM 上可直接导入运行；
- 显式开关选择 310P V2 runner，关闭时 V1 不变；
- 310P 真实请求中无 Triton 编译和执行；
- Qwen3-8B、Qwen3.5-4B TP1/TP2 通过；
- Qwen3.5 hybrid cache 和 chunked prefill 无 state 污染；
- Qwen3-VL-8B-Instruct TP1/TP2 文本+图片通过；
- Qwen3-8B W8A8、W8A8SC 真实权重通过；
- Weight NZ、Attention KV Cache NZ 分别有格式和正确性证据；
- `FULL_DECODE_ONLY` 按第一版矩阵完成真实 replay；
- slot mapping 不引入 D2H 热路径同步；
- 第一版暂缓能力和 V1 未支持能力均不会误入。

### 11.2 最终 V1 对齐完成

- 第一版全部能力持续通过；
- prefix cache、完整后处理、MTP 按 V1 范围通过；
- 非 MTP speculative decoding 仍被拒绝；
- 未增加 V1 不支持的并行、Attention 或模型执行能力；
- 现有 310P V1 和非 310P V2 回归不受影响。
