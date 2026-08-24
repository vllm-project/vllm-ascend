# vLLM Ascend 启用 LoRA 的启动资源开销分析

## 1. 分析范围

本文回答以下问题：启动 vLLM Ascend 时加入 `--enable-lora`，以及继续通过
`--lora-modules` 加载真实 adapter，是否会增加 CPU 内存、NPU HBM（下文也称
“显存”）、CPU 开销和启动时间；如果会，内存分配和耗时具体发生在哪里。

分析基于：

- vLLM Ascend 分支 `ds_lora_moe_v25`，提交 `2208e231c`；
- 仓库声明的配套 vLLM release 为 `v0.25.1`，提交 `752a3a504`；
- 默认 LoRA 参数：`max_loras=1`、`max_lora_rank=16`、
  `max_cpu_loras=max_loras`、`lora_dtype=base model dtype`。

本文是代码静态分析，不包含特定模型和 NPU 机器上的实测数值。实际数值还取决于
模型层数与维度、LoRA target modules、TP/EP、adapter rank、dtype、ACLGraph
capture sizes、CANN/PyTorch NPU allocator，以及每个节点上的 worker 数。

## 2. 结论

**会增加，而且仅设置 `--enable-lora`、尚未加载真实 adapter 时，就已经会增加
NPU HBM 和启动时间。** 加载真实 adapter 后，还会增加每个 worker 进程的 CPU
常驻内存和 adapter 加载时间。

| 资源 | 仅 `--enable-lora` | 再加载 `--lora-modules` | 主要原因 |
| --- | --- | --- | --- |
| NPU HBM | 明显增加 | 通常只增加少量瞬时峰值 | 启动时已经按 `max_loras × max_lora_rank` 为所有被包装层预分配 A/B 权重槽；真实权重写入已有槽位 |
| CPU 常驻内存 | 管理对象较小；warmup dummy adapter 通常是临时的 | 明显增加 | 每个 worker 在 CPU LRU cache 中保留真实 adapter，最多 `max_cpu_loras` 个 |
| CPU 启动开销 | 增加 | 继续增加 | 遍历/替换模块、创建零张量、dummy LoRA、读取配置和权重、dtype 转换、pack/reshape/pin memory |
| 启动时间 | 增加 | 继续增加 | NPU 权重槽清零、profile/warmup、额外 ACLGraph，以及静态 adapter 串行加载 |
| 运行期临时 HBM | 增加 | 相同 | LoRA shrink buffer、MoE routing tensors、算子 workspace 和图池 |
| KV cache 容量 | 可能下降 | 一般不再明显下降 | LoRA HBM 在 KV cache 定容前已经分配，会从同一 HBM 预算中扣除 |

需要区分两种内存：

1. **NPU 权重槽按配置上限分配**，主要由 `max_loras`、`max_lora_rank` 和被包装
   的模型层决定，与当前实际加载的 adapter 数和实际 rank 不完全一致。
2. **CPU adapter cache 按真实权重分配**，主要由真实 rank、target modules、dtype、
   `max_cpu_loras` 和 worker 数决定。

## 3. 启动调用链

```text
EngineArgs.create_engine_config
  -> --enable-lora 生成 LoRAConfig
  -> NPUModelRunner.load_model
     -> 加载 base model
     -> LoRAModelRunnerMixin.load_lora_model
        -> LRUCacheWorkerLoRAManager
        -> LoRAModelManager.__init__
           -> 创建 PunicaWrapperNPU 元数据张量
           -> 遍历 target modules
           -> 将 base layer 替换为 WithLoRA wrapper
           -> 为每层创建 max_loras 个、max_lora_rank 宽的 NPU A/B 槽
  -> determine_available_memory/profile_run
     -> 创建 dummy LoRA、执行 LoRA forward、统计 activation peak
  -> compile_or_warm_up_model/capture_model
     -> warmup 并捕获无 LoRA/有 LoRA ACLGraph
  -> OpenAI init_static_loras
     -> 每个 --lora-modules 依次 add_lora
        -> CPU 读取/转换/pack adapter
        -> 放入 CPU LRU cache
        -> copy_ 到预分配的 NPU slot
  -> API server ready
```

关键入口如下：

- `vllm_ascend/worker/model_runner_v1.py:3438-3516`：加载 base model 后，在
  `self.lora_config` 存在时调用 `load_lora_model`；因此 LoRA 权重槽计入日志中的
  `Loading model weights took ... GB`。
- vLLM `vllm/v1/worker/lora_model_runner_mixin.py:31-46`：创建
  `LRUCacheWorkerLoRAManager`。
- vLLM `vllm/lora/model_manager.py:141-142`：初始化 Punica wrapper 并替换模型层。
- vLLM `vllm/entrypoints/openai/api_server.py:342-355`：API 初始化期间等待
  `init_static_loras()`，所以 `--lora-modules` 的加载属于 server-ready 延迟。

## 4. NPU HBM 增加在哪里

### 4.1 最大项：每个被包装层的 LoRA A/B 权重槽

Ascend 在 `vllm_ascend/platform.py:865-867` 选择
`vllm_ascend.lora.punica_npu.PunicaWrapperNPU`，但权重槽创建主要复用 vLLM 的
LoRA layer。`LoRAModelManager._create_lora_modules()` 遍历模型，并调用
`from_layer(...)` 替换所有匹配 target 的层：

```python
# vLLM v0.25.1: vllm/lora/model_manager.py:395-461
for module_name, module in self.model.named_modules(remove_duplicate=False):
    if not self._match_target_modules(module_name):
        continue
    new_module = replace_submodule(
        self.model,
        module_name,
        from_layer(
            module,
            self.lora_slots,
            self.lora_config,
            packed_moduled_lst,
            self.model.config,
        ),
    )
```

`from_layer()` 在 wrapper 创建时立刻执行 `create_lora_weights()`：

```python
# vLLM v0.25.1: vllm/lora/utils.py:113-123
instance_layer = lora_cls(layer)
instance_layer.create_lora_weights(max_loras, lora_config, model_config)
return instance_layer
```

普通 linear 的实际分配如下：

```python
# vLLM v0.25.1: vllm/lora/layers/base_linear.py:129-150
self.lora_a_stacked = tuple(
    torch.zeros(
        max_loras, 1, lora_a_out_size, self.input_size,
        dtype=lora_config.lora_dtype, device=self.device,
    )
    for _ in range(self.n_slices)
)
self.lora_b_stacked = tuple(
    torch.zeros(
        max_loras, 1, lora_b_out_size, lora_config.max_lora_rank,
        dtype=lora_config.lora_dtype, device=self.device,
    )
    for _ in range(self.n_slices)
)
```

这说明：

- 分配发生在真实 adapter 加载之前；
- 使用的是 `max_lora_rank`，不是 adapter 的实际 rank；
- 使用的是 `max_loras` 个 slot，即使当前没有请求使用 LoRA；
- 默认 `target_modules=None` 时，会包装模型声明的所有受支持 LoRA modules，而
  不是只根据稍后加载的 adapter 文件决定；
- packed QKV、gate/up 等层有多个 slice，A/B 权重也按 slice 分配；
- base weight 由 wrapper 引用，不会因为包装而再复制一份 base model 权重。

对一个单 slice dense linear，单 device 的近似槽位大小为：

```text
bytes = max_loras × dtype_bytes
        × (lora_A_rank_local × input_size
           + output_size_local × max_lora_rank)
```

对所有被包装层求和即可。未启用 fully-sharded LoRA 时，
`lora_A_rank_local` 通常等于 `max_lora_rank`；启用后，部分层会按 TP 切分 rank 或
输出维度。最可靠的计算方式始终是直接按代码中每个 `lora_*_stacked.shape`
的 `numel × element_size` 求和。

#### Dense 示例

假设一个仅用于估算的 32 层 MHA 模型：`hidden_size=4096`、
`intermediate_size=11008`，每层对 q/k/v/o/gate/up/down 共 7 个矩阵启用 LoRA，
单卡、BF16、`max_loras=1`、`max_lora_rank=16`，则槽位约为：

```text
每层 = 16 × 2 bytes
       × [4 × (4096 + 4096) + 3 × (4096 + 11008)]
     = 2.3828125 MiB

32 层 = 76.25 MiB
```

这只是 projection 层示例，不包含 embedding、lm_head、额外 tower/connector、
allocator rounding 和 ACLGraph。若改为 `max_loras=4`、`max_lora_rank=64`，同一
模型的该项线性放大 16 倍，约为 `1.19 GiB`。

### 4.2 MoE 权重槽会再乘 local expert 数

MoE wrapper 的张量形状包含 `local_num_experts`。以 gated 2D MoE、未 fully
shard 为例，vLLM 在 `vllm/lora/layers/fused_moe.py:157-217` 创建：

```text
w13 A: 2 × [M, E_local, R, H]
w13 B: 2 × [M, E_local, I_local, R]
w2  A: 1 × [M, E_local, R, I_local]
w2  B: 1 × [M, E_local, H, R]
```

其中 `M=max_loras`、`R=max_lora_rank`、`H=hidden_size`、
`I_local=intermediate_size_per_partition`。近似为：

```text
MoE bytes per layer
  ≈ dtype_bytes × M × E_local × R × 3 × (H + I_local)
```

3D fused 格式的 w13 布局不同，应按
`vllm/lora/layers/fused_moe.py:461-493` 的真实 shape 计算。Ascend wrapper 位于
`vllm_ascend/lora/fused_moe.py:313-387`，它复用上述权重分配，只替换 forward
注入和 routing 实现。因此对于 MoE，`E_local`、EP 切分方式和每层 expert 数是
判断 LoRA HBM 的关键；其开销可能远大于 dense LoRA。

### 4.3 Punica 固定元数据张量

`PunicaWrapperNPU.__init__()` 调用上游 `PunicaWrapperBase.__init__()`。后者在
NPU 上预分配：

```python
# vLLM v0.25.1: vllm/lora/punica_wrapper/punica_base.py:138-160
_token_lora_indices      # [max_num_batched_tokens], int64
_sampler_indices         # [max_num_batched_tokens], int64
_sampler_indices_padded  # [max_num_batched_tokens], int64
_embeddings_indices      # [2, max_num_batched_tokens], int64
_seq_start_locs          # [max_num_seqs], int64
_seq_lengths             # [max_num_seqs], int64
_lora_indices_per_batch  # [max_num_seqs], int64
```

单个 language Punica wrapper 的固定大小近似为：

```text
40 × max_num_batched_tokens + 24 × max_num_seqs bytes
```

例如 `8192 tokens + 256 seqs` 约 `0.318 MiB`，通常远小于权重槽。多模态
tower/connector 可能创建额外 wrapper，见 vLLM
`vllm/lora/model_manager.py:173-271`。

每个 MoE wrapper 还会创建很小的 `adapter_enabled[max_loras + 1]` int32 张量，
见 vLLM `vllm/lora/layers/fused_moe.py:239-256`。

### 4.4 forward/profile/graph capture 的临时张量

这些不是 adapter 常驻权重，但会增加 profile peak、运行期 peak 或 ACLGraph
graph pool：

- Dense：`vllm_ascend/lora/punica_npu.py:345-353` 在没有外部 buffer 时，为每个
  slice 创建 `[num_tokens, max_lora_rank]` 的 FP32 shrink buffer。
- Logits：`vllm_ascend/lora/punica_npu.py:490-498` 创建同类 FP32 buffer。
- MoE：`vllm_ascend/lora/punica_npu.py:405-458` 创建/派生 `expert_idx`、
  `lora_idx_safe`、`enabled`、`combined_idx`，并为每个 slice 创建
  `[num_routed_rows, local_rank]` FP32 `shrink_out`。
- AllGather/AlltoAll MoE routing：`vllm_ascend/lora/fused_moe.py:62-154` 和
  `190-250` 会创建 pad、repeat、argsort、permutation 和 exchanged index 张量。
- 量化 MoE：`vllm_ascend/lora/quant_moe.py:250-300` 还会创建 dispatched mask、
  group ids 和 composite group list。

allocator 可复用很多逐层临时 buffer，因此不能把所有层的临时 tensor 简单相加；
应以 `torch.npu.max_memory_allocated()` 或启动 memory profiling 的 peak 为准。

### 4.5 LoRA 会挤占 KV cache 预算

LoRA wrapper 和权重槽在 `NPUModelRunner.load_model()` 内创建，而 worker 随后才在
`vllm_ascend/worker/worker.py:518-569` 执行 `profile_run()` 并计算可用于 KV cache
的空间。因此在相同 `gpu_memory_utilization` 下，常见结果不是进程最终 HBM 无限
增加，而是：

```text
LoRA 权重/metadata/activation/graph 增加
  -> non-KV memory 增加
  -> 可分配 KV cache bytes/blocks 减少
  -> 最大并发或可缓存 token 数下降
```

如果显式设置了固定 `--kv-cache-memory`，代码仍会 profile，但不会自动为 LoRA
缩减该值，见 `vllm_ascend/worker/worker.py:528-544`；原来贴近上限的配置可能因此
OOM。

## 5. CPU 内存增加在哪里

### 5.1 真实 adapter 常驻 CPU LRU cache

worker 在加载 adapter 时明确指定 `device="cpu"`：

```python
# vLLM v0.25.1: vllm/lora/worker_manager.py:142-154
lora = self._lora_model_cls.from_local_checkpoint(
    lora_path,
    expected_lora_modules,
    ...,
    device="cpu",
    dtype=self.lora_config.lora_dtype,
    ...,
)
```

加载后的 `LoRAModel` 被放入 `_registered_adapters` LRU cache。容量是
`max_cpu_loras`，默认等于 `max_loras`：

- vLLM `vllm/lora/model_manager.py:105-113`：创建 CPU registered cache 和 active
  NPU cache；
- vLLM `vllm/lora/model_manager.py:282-289`：capacity 对应
  `max_cpu_loras`，NPU slot 数对应 `max_loras`；
- vLLM `vllm/config/lora.py:108-116`：未设置时令
  `max_cpu_loras=max_loras`；
- vLLM `vllm/lora/worker_manager.py:291-318`：超出 CPU cache capacity 时淘汰
  oldest adapter，再激活到 NPU slot。

对 dense adapter，单个 CPU cache entry 的权重近似为：

```text
adapter bytes ≈ dtype_bytes × actual_rank
                × Σ真实 target matrices(input_dim + output_dim)
```

注意 CPU cache 使用 **实际 rank**，而 NPU 预分配使用 **max rank**。节点总 CPU
内存还要乘 worker 进程数。模型执行器会在各 worker 上执行 `add_lora`，所以 TP
多卡通常不是共享一份 CPU adapter：dense 权重可能在每个 TP worker 中各保留
一份。safetensors + EP MoE 路径会尽量只读取本 rank 的 local experts；`.bin/.pt`
则先整体加载后再裁剪。

### 5.2 加载峰值高于最终 cache 大小

`LoRAModel.from_local_checkpoint()` 的主要阶段位于 vLLM
`vllm/lora/lora_model.py:205-307`：

1. 读取 `adapter_config.json`；
2. 读取 safetensors 或 `torch.load(..., map_location="cpu")`；
3. 将每个 tensor 转为 LoRA dtype；
4. packed QKV/MoE 权重进行 pack、reshape、permute、contiguous 和 EP slice；
5. 在支持时将最终 CPU tensor 变为 pinned memory；
6. 删除不再需要的 unpacked/remote expert tensor。

`vllm/lora/model_manager.py:734-825` 显示 pack 和 pin memory 都可能产生新 storage。
因此 adapter 加载期间可能同时存在 raw tensor、dtype-converted tensor、packed/
contiguous tensor 和 pinned copy，CPU RSS 峰值会高于最终 LRU entry。`.bin/.pt`
无法在读取时跳过 remote experts，峰值通常比 safetensors 更不利；代码在
`vllm/lora/lora_model.py:273-293` 明确区分了这两条路径。

### 5.3 dummy LoRA CPU 内存通常是临时的

profile/warmup/graph capture 会创建 dummy adapter。vLLM
`vllm/v1/worker/lora_model_runner_mixin.py:93-130` 使用：

```python
num_loras = lora_config.max_loras
lora_warmup_rank = min(lora_config.max_lora_rank, 8)
for lr in lora_requests:
    self.lora_manager.add_dummy_lora(lr, rank=lora_warmup_rank)
...
self.lora_manager.remove_all_adapters()
```

dummy tensor 在 CPU 创建，见 vLLM `vllm/lora/model_manager.py:536-648`。同一次
context 内的多个 dummy ID 会通过 `clone()` 共享底层 CPU tensor，因此不是简单
的 `max_loras` 倍；但每个 NPU slot 仍会被 reset/copy。capture 完成后也会调用
`maybe_remove_all_loras()`，所以只启用 LoRA、未加载真实 adapter 时，最终不应有
真实 adapter 大小的 CPU 常驻 cache，但启动峰值和 CPU 工作量仍然存在。

## 6. CPU 使用率和同步开销

启用 LoRA 不会单独保留一个 CPU 核或额外创建一个固定后台计算线程，但会增加
以下 CPU 工作：

- 启动：模型 `named_modules()` 遍历、target 匹配、wrapper 替换和 Python 对象创建；
- adapter：JSON/safetensors I/O、key 校验、dtype 转换、pack/reshape/contiguous、
  LRU 管理；
- 每批请求：生成 prompt/token LoRA mapping、Python tuple/list/set/dict，更新 active
  adapter 和 Punica metadata；
- CPU 到 NPU：将 mapping 转为 tensor，并 copy 到预分配 metadata buffer；
- adapter 切换：按层 `reset_lora()` 并把 CPU 权重 copy 到 NPU slot。

每批 mapping 的代码证据：

- vLLM `vllm/v1/worker/gpu_input_batch.py:977-1000`：对 request LoRA id 执行
  NumPy `repeat`，再转成 tuple/set；
- vLLM `vllm/lora/punica_wrapper/utils.py:91-145`：创建 Python list/dict，逐 token
  映射 slot，随后 H2D；
- `vllm_ascend/lora/punica_npu.py:56-81`：额外用 Python set 计算 active MoE
  adapters 和 `no_lora`；
- scheduler 在 `vllm_ascend/core/recompute_scheduler.py:427-477` 等位置维护
  `scheduled_loras` 并检查 `max_loras`。

还需注意上游 `compute_meta()` 在 prefill metadata 路径中存在 NPU tensor
`.item()`：

```python
# vLLM v0.25.1: vllm/lora/punica_wrapper/utils.py:27-35
lora_indices_tensor, seq_length_tensor = torch.unique_consecutive(...)
max_length = seq_length_tensor.max().item()
token_nums = seq_length_tensor.sum().item()
```

Ascend `PunicaWrapperNPU.update_metadata()` 调用了上游 `super().update_metadata()`，
因此 prefill 会经过该逻辑。这里的两次 `.item()` 会让 CPU 等待 NPU 结果，是同步
延迟，不一定表现为 CPU 利用率很高，但会增加 host 侧阻塞和端到端 step 时间。

## 7. 启动时间增加在哪里

### 7.1 模型加载阶段

`vllm_ascend/worker/model_runner_v1.py:3450-3516` 的
`DeviceMemoryProfiler` 同时包住 base model 加载和 `load_lora_model()`。LoRA 增加：

- manager/Punica 初始化；
- 扫描并包装所有 target modules；
- 大量 `torch.zeros(..., device=npu)` 分配和清零；
- Ascend LoRA class 注册、MoE context 构建；
- 后续真实 adapter 激活时逐层 CPU->NPU copy。

因此可直接比较启用前后的日志：

```text
Loading model weights took X.XXXX GB
```

这个日志包含预分配 LoRA slot，但不包含稍后 OpenAI 层加载静态 adapter 的全部
耗时。

### 7.2 memory profile 和 warmup

`vllm_ascend/worker/worker.py:546-569` 执行 `profile_run()`。当前 Ascend
`_dummy_run()` 在 `vllm_ascend/worker/model_runner_v1.py:3281-3289` 中，只要存在
`lora_config`，就暂时强制 dummy run 使用 `max_loras`：

```python
with self.maybe_dummy_run_with_lora(
    self.lora_config,
    ...,
    num_active_loras=(
        self.lora_config.max_loras
        if self.lora_config is not None
        else num_active_loras
    ),
):
```

这会在启动 profile/warmup 中真正执行 LoRA 路径，并创建 dummy adapter、metadata
和 shrink/routing buffer，因而同时增加启动时间和 measured activation peak。

### 7.3 ACLGraph 数量和 graph pool

vLLM v0.25.1 的 `CompilationConfig.cudagraph_specialize_lora` 默认是 `True`。在
`vllm/v1/cudagraph_dispatcher.py:111-130`：

- 未启用 LoRA：capture case 为 `[0]`；
- 启用 LoRA、默认 `specialize_active_lora=False`：case 为
  `[0, max_loras + 1]`；
- 再设置 `specialize_active_lora=True`：会增加 `1, 2, 4, ...`（不超过
  `max_loras` 的 2 的幂）以及 `max_loras + 1` 等 active-count cases。

`initialize_cudagraph_keys()` 在
`vllm/v1/cudagraph_dispatcher.py:166-231` 对 capture sizes 和 LoRA cases 做笛卡尔积。
所以默认仅启用 LoRA 就会把 decoder/mixed 的 graph descriptor 数量大致从一组
变成两组；启用 active-count specialization 后还会更多。

这不表示 graph pool HBM 必然严格翻倍，因为不同图共享 pool，且大 shape 可复用
部分内存；但 graph capture 次数、启动时间和 per-graph metadata 会增加。
`vllm_ascend/worker/model_runner_v1.py:4686-4696` 复用上游 capture 流程，worker 在
`vllm_ascend/worker/worker.py:684-710` 调用它。启动日志可直接观察：

```text
Graph capturing finished in ... secs, took ... GiB
```

设置 `--enforce-eager` 可跳过 ACLGraph capture 的这部分开销，但不会取消 LoRA
权重槽、CPU cache、metadata 和 eager forward 临时 buffer。

若把 `cudagraph_specialize_lora` 设为 `False`，只捕获通用 LoRA-enabled case，
可减少启动时间和图数量；代价是 base-only batch 也使用 LoRA-enabled graph，运行
时可能多执行 no-op LoRA 路径。`specialize_active_lora=True` 则相反：用更多图换取
不同 active adapter 数下的运行性能。

### 7.4 静态 adapter 在 API ready 前串行加载

vLLM `OpenAIServingModels.init_static_loras()` 位于
`vllm/entrypoints/openai/models/serving.py:124-139`，它逐个 `await
load_lora_adapter()`；后者在 `167-198` 调用 engine `add_lora()`。API server 在
`vllm/entrypoints/openai/api_server.py:350-355` 等待全部静态 LoRA 完成后才继续初始化。

因此 `--lora-modules` 增加的磁盘/网络解析、每个 worker 的 CPU load/pack/pin、
CPU->NPU slot copy 都计入 server-ready 时间，而且多个静态 adapter 是依次提交的。

## 8. 各参数如何影响资源

| 参数 | NPU HBM | CPU RAM | 启动时间 | 备注 |
| --- | --- | --- | --- | --- |
| `max_loras` | 权重槽近似线性增加 | 默认会同时推高 `max_cpu_loras` | 清零/copy 增加；可能增加图 case | 表示单 batch 可同时 active 的 adapter 数 |
| `max_lora_rank` | 权重槽近似线性增加 | 真实 cache 取决于实际 rank | 分配/清零及 LoRA kernel 增加 | dummy warmup rank capped at 8，但 NPU slot 不 cap |
| `max_cpu_loras` | 不直接增加预分配 HBM | cache 容量线性增加 | 首次加载更多 adapter 时增加 | 必须不小于 `max_loras` |
| `lora_dtype` | 与 dtype bytes 成正比 | 与 dtype bytes 成正比 | dtype cast 成本不同 | `auto` 默认跟 base dtype |
| `lora_target_modules` | 可显著减少 | 真实 adapter target 同步减少 | 包装、分配、图中 LoRA op 减少 | 应确保覆盖 adapter 实际 target |
| `fully_sharded_loras` | 部分 A rank/B output 可按 TP 减小 | CPU cache 通常仍保留完整权重 | 可能降低分配但增加通信/初始化复杂度 | 运行期会引入 all-reduce/all-gather |
| `cudagraph_specialize_lora=False` | 减少 graph pool/metadata | 基本不变 | 减少 capture | base-only 请求可能走通用 LoRA graph |
| `specialize_active_lora=True` | 增加更多 graph | 基本不变 | 明显增加 capture | 用启动资源换运行性能 |
| `--enforce-eager` | 去掉 graph pool，但保留 LoRA slots | 基本不变 | 跳过 capture | 运行性能可能下降 |

另外，`vllm_ascend/lora/punica_npu.py:29-48` 在 310P 或
`max_lora_rank >= 128` 时切换到上游 torch LoRA ops，否则使用 Ascend 自定义 ops。
这会改变算子 warmup、workspace 和性能，做 rank 对比时不能只看 rank 的线性内存
公式。

## 9. 建议的实测拆分方法

要把开销分清，不要只比较“base”和“LoRA + adapter”两次。应保持模型、TP/EP、
batch/token 上限、dtype、quantization、KV cache 配置完全一致，分四组：

1. **A：base + eager**：不启用 LoRA，设置 `--enforce-eager`；
2. **B：LoRA-enabled + eager**：仅加 `--enable-lora`，不加载真实 adapter；
3. **C：LoRA-loaded + eager**：在 B 上增加 `--lora-modules`；
4. **D：base/LoRA graph 对照**：去掉 `--enforce-eager`，分别跑 A 和 C。

分别记录：

- server 进程启动到 health check ready 的 wall time；
- 所有 worker RSS 之和以及启动期间 peak RSS；
- `Loading model weights took ... GB`；
- memory profile 的 non-KV/activation peak；
- `Graph capturing finished ... took ... GiB`；
- 最终 KV cache bytes、block 数或可缓存 token 数；
- `npu-smi` 中每个 worker/NPU 的 HBM；
- 首次 adapter load 与 cache hit 后再次使用的耗时。

差值含义：

```text
B - A = wrapper + NPU slot + Punica metadata + dummy/profile 的纯 enable 开销
C - B = 真实 adapter CPU cache + load/pack/pin + CPU->NPU 激活开销
D(LoRA) - D(base) = LoRA ACLGraph/compile/warmup 增量
```

如果能在 worker 内取得 model，可用以下思路核对 LoRA storage。必须按 storage
去重，因为 MoE 的兼容字段可能只是底层 `w13_lora_*`/`w2_lora_*` 的 view：

```python
import torch


def iter_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_tensors(item)


storages = {}
for module_name, module in model.named_modules():
    for attr_name, value in vars(module).items():
        if "lora" not in attr_name and attr_name != "adapter_enabled":
            continue
        for tensor in iter_tensors(value):
            storage = tensor.untyped_storage()
            key = (tensor.device.type, tensor.device.index, storage.data_ptr())
            storages[key] = max(storages.get(key, 0), storage.nbytes())

print(f"unique LoRA storage: {sum(storages.values()) / 2**20:.2f} MiB")
```

该值用于解释 tensor storage，不等同于 allocator reserved memory；实际 HBM 还要看
`torch.npu.memory_allocated()`、`torch.npu.memory_reserved()` 和 `npu-smi`。

## 10. 降低开销的优先级

1. 把 `max_loras` 和 `max_lora_rank` 设为业务真正需要的最小值。这两个值直接决定
   NPU 权重槽上限，不要因为“可能以后会用”而设置得过大。
2. 用 `--lora-target-modules` 限制实际需要的模块，尤其是大模型和 MoE；先确认
   adapter targets 与部署配置一致。
3. 把 `max_cpu_loras` 控制在需要热缓存的 adapter 数；它不减少 NPU slot，但能
   降低 worker RSS。
4. 优先使用 safetensors，尤其是 EP MoE，降低全量反序列化和 remote expert 的 CPU
   峰值。
5. 启动时间或图池紧张时，比较
   `cudagraph_specialize_lora=False`；仅调试/低吞吐场景可用 `--enforce-eager`
   验证 graph 增量。
6. TP 场景评估 `--fully-sharded-loras`，但必须同时测通信开销和当前模型/Ascend
   路径兼容性，不能只按 HBM 结论启用。
7. 固定 `--kv-cache-memory` 时为 LoRA slot、profile peak 和 graph pool 重新留余量，
   不要沿用 base-only 的极限值。

## 11. 最终判断

- **NPU HBM：一定增加。** 只要 `lora_config` 存在，就会创建最大 rank/最大 slot
  的权重 bank；这是主要常驻开销。
- **CPU RAM：加载真实 adapter 后一定增加。** 每个 worker 有独立 CPU LRU cache；
  只 enable、不加载真实 adapter 时，最终常驻增量主要是管理对象，dummy 权重通常
  在 warmup/capture 后移除。
- **CPU：会增加工作和同步等待。** 没有独占 CPU 核，但启动解析/pack、每批 mapping
  和 prefill `.item()` 同步都存在。
- **启动时间：一定增加。** 至少包含 layer wrapping、NPU slot 清零、dummy profile/
  warmup；默认 ACLGraph specialization 还增加图数量；静态 adapter 会在 API ready
  前继续串行加载。
- **真实 adapter 通常不会再增加一份同等大小的常驻 NPU 权重。** 它被 copy 到已经
  预留的 slot；但加载瞬时峰值、graph/workspace 和 allocator reserved memory 仍可能
  变化。
