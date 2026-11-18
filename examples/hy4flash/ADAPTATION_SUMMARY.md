# Hy4 Flash 昇腾适配总结

更新于 2026-11-18。当前实机基线为 8 张 Ascend 950DT，每卡 96 GiB HBM。运行环境为 CANN 9.2.1、`torch_npu 2.10.0.post7`、Python 3.12.16，使用 vLLM MRV2、TP8 + EP 和 Hy4 Flash 全量权重。启动示例使用 `/home/hy4nxt` 作为权重路径。

## 精度差异有多大

**已测的 4K 和 16K 单路样本逐值对齐。短输入的差异已经能改变贪心输出。** 两边均使用 C8 和 DFly 5，控制输入和采样设置，以 eager 为对照。logprob 差按相同生成前缀统计。

| 输入场景 | 输出 token 差异 | 所选 token 的最大绝对 logprob 差 | 平均接受长度 eager → FULL |
| --- | --- | ---: | ---: |
| 5 token 单路 | 0 / 18 | 0.627 | 1.385 → 1.500 |
| 128 token 单路 | 18 / 24，第 4 个 token 首次分叉 | 0.205 | 3.571 → 3.286 |
| 128 token × 8 路 | 16 / 192，两路在第 14 个 token 首次分叉 | 0.297 | 4.038 → 3.852 |
| 509、2,047、2,051、4,093 token 混合长度 | 0 / 96 | 0.497 | 2.561 → 2.561 |
| 4,096 token 单路 | 0 / 200，logprob 和 top-5 也一致 | 0 | 2.971 → 2.971 |
| 16,384 token 单路 | 0 / 24，logprob 和 top-5 也一致 | 0 | 3.375 → 3.375 |

服务器以 128 并发、128K 上下文启动时，也复现了 4K 和 16K 的逐值对齐。大配置另测 128–130 token 的短输入，每路生成 24 个 token。

| 并发 | 实际回放图 | 有输出分叉的请求 | 所选 token 最大绝对 logprob 差 | 平均接受长度 eager → FULL | TPOT eager → FULL |
| --- | --- | --- | ---: | ---: | ---: |
| BS9 | BS16 | 3 / 9 | 0.746 | 4.291 → 4.423 | 163.9 → 58.2 ms |
| BS17 | BS32 | 7 / 17 | 0.518 | 3.620 → 3.860 | 292.0 → 108.6 ms |

BS9 有一路在首 token 已分叉，两轮 prefill 调度也有差异。表中对照是服务端到端的输出差异。公共 top-5 候选的最大绝对 logprob 差在小配置对照中为 3.039，在大配置并发对照中为 5.145。GPQA 和 MMMU-Pro 对最新 FULL 与 eager 的受控任务分数仍待测。

此前固定形状图的逐层定位中，6 个有效候选、C8 字节、反量化 KV、query、有效 QK logits 和 learned sink 一致。Eager 在 6 个有效位置和 1 个 sink 上做 softmax，固定形状路径把候选填充到 2,048 个位置，再加 1 个 sink。归约宽度由 7 变成 2,049，有效概率最大差约 `1.19e-7`。一处 FP32 context 从 `-0.068115241826` 变为 `-0.068115226924`，转 BF16 分别为 `-0.068359375` 和 `-0.06787109375`。这说明算子层的细微差异怎样传递到整网输出。

## 当前适配

| 模块 | 已接入的实现与实测范围 |
| --- | --- |
| 模型与权重 | 文本和视觉入口注册在 `vllm_ascend/models/hy_v4_flash`。当前 checkpoint 有 42 层语言主干，其中 12 层 IndexPool、30 层 SWA。加载原生 128×128 block FP8 权重，普通线性层和专家按 950DT 的 MXFP8 推理路径处理，MLA 的 `kv_b_proj` 保持 BF16。此前完成图片请求验证，最新 FULL 精度回归使用文本输入。 |
| SWA、IndexPool、分页 MLA | 基于 vLLM 页表写读 KV。IndexPool 增量缓存已完成的组，按本轮有效长度和因果范围召回。当前 attention 使用分页读取、QK、带 learned sink 的 softmax 和 PV。设备侧长度与页表随每步请求更新。 |
| C8 与小算子优化 | `fp8` KV 使用 C8，512 维 latent 采用每 128 维一个 FP32 scale，64 维 RoPE 保留 BF16。950DT 下自动选用符合条件的 C8 写入、分页读取与反量化、同宽验证批量计算、吸收投影和 iHC 融合。MoE 的 clamped SwiGLU 根据量化后端能力选择融合或逐步计算。 |
| FULL decode ACLGraph | 目标模型的 KV 写入、IndexPool、分页 MLA、iHC 和 MoE 随整图回放。默认精确捕获 1–8 路，更大的同宽批次复用下一档完整图。Prefill、混合 query 宽度及未命中的尾步使用 eager。 |
| DFly 5 | 此前讨论的 DFlash 5。Checkpoint 自带 5 层 draft，vLLM 方法名为 `dfly`。每轮提出最多 5 个草稿 token，目标模型最多每路联合验证 6 个 query。Draft 使用目标层的候选快照和分页 KV，draft attention 和后续计算均已入 FULL 图。 |
| Engram | 4 张 embedding 表按 TP 分片，默认用 pinned CPU 内存保存。图回放前按本轮 token 算 hash 并拷入查到的行。当前 checkpoint 的表为 BF16，约 126.7 GB 合计。 |
| 共享专家双流 | 共享专家和路由专家可用不同 NPU stream。此前同步调度的开关对照覆盖 10 个请求、120 个生成 token，token、logprob 与 top-5 均一致。最新 FULL 整网回归启用了双流。 |
| Prefix cache | 已在 eager 和此前的分段图中验证纯文本、SWA/IndexPool 长度边界和图片前缀命中。冷热请求采用不同 prefill 分块时概率可变，控制成相同分块后逐值一致。当前启动默认关闭，启用时须关闭 DFly。最新 FULL + prefix 的组合回归待补。 |

### 目标模型和 draft 的数据流

1. 每路 decode 的逻辑位置从设备上的本轮序列长度推导。若序列长度为 `end`，每路有 `width` 个 query，则第 `i` 个 query 的逻辑位置为 `end - width + i`。SWA 按窗口截取，IndexPool 在已完成的历史组中召回，并拼入当前因果可见的组尾。页表和 slot mapping 决定实际 KV 地址，填充行使用 mask。
2. C8 的 MLA 基础行由 512 字节 E4M3 latent、128 字节 BF16 RoPE 和 16 字节 FP32 scale 组成。IndexPool 在 sidecar 中保留 BF16 原始键、3 个 `int64` 位置及增量池化键。Attention 读取选中行并反量化，带 learned sink 的 softmax 将 sink 计入分母。
3. DFly 用目标验证中最后有效行作为 anchor，从目标第 25、29、33、37、41 层复用 Top-K 快照和 KV，层号从 0 开始。Draft 过滤 anchor 后的旧候选，每个 query 只访问可见的草稿块。草稿 KV 在 draft 生命周期内使用，目标验证更新目标缓存。候选构造曾与 GPU 参考逐索引比较 81 组批大小、query 宽度、Top-K 宽度和窗口组合，结果一致。
4. Engram 图回放前根据当前 token 和 lookback 计算 hash，将查到的行写入预分配缓冲。目标和 draft 分别提交一次完整图。1–8 路精确捕获，更大的同宽 decode 使用 vLLM batch padding，防止填充请求读写真实 KV。

相关实现位于 `vllm_ascend/models/hy_v4_flash/`。其中 `attention.py` 和 `index_pool_cache.py` 实现候选与 KV，`aclgraph.py` 派发目标图，`dfly.py` 和 `dfly_speculator.py` 实现 draft，`engram.py` 和 `model_state.py` 管理 Engram 存储及图输入。

## 启动与开关

在匹配的 Ascend 容器内执行启动脚本。下面以 `/opt/hy4flash` 为部署目录，容器名为 `hy4flash`。

```bash
docker exec -it hy4flash bash
```

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /opt/hy4flash
./start_hy4flash.sh
```

下面的 `export` 在**容器内同一个 shell** 执行，再运行脚本。从宿主机直接调用 `docker exec` 时，可用 `-e NAME=value` 传递这些变量。

| 特性 | 环境变量 | 默认 |
| --- | --- | --- |
| 图模式 | `export HY4FLASH_GRAPH_MODE=decode` 启用 FULL decode。用 `export HY4FLASH_GRAPH_MODE=eager` 让目标和 draft 都走 eager。 | `decode` |
| DFly 5 | `export HY4FLASH_ENABLE_DFLY=1` 开启，`export HY4FLASH_ENABLE_DFLY=0` 关闭。 | `1` |
| 共享专家双流 | `export HY4FLASH_SHARED_EXPERT_MULTISTREAM=1` 开启，`export HY4FLASH_SHARED_EXPERT_MULTISTREAM=0` 关闭。 | 实机服务器 `1`，分支所附脚本 `0` |
| KV 缓存类型 | `export HY4FLASH_KV_CACHE_DTYPE=fp8` 用 C8，`export HY4FLASH_KV_CACHE_DTYPE=bfloat16` 用 BF16 KV。 | `fp8` |

使用 `export 变量名=值` 可覆盖脚本的资源设置。

| 环境变量 | 实机服务器脚本默认 | 分支所附脚本默认 | 用途 |
| --- | --- | --- | --- |
| `HY4FLASH_MAX_NUM_SEQS` | `128` | `8` | 调度并发上限 |
| `HY4FLASH_MAX_MODEL_LEN` | `131072` | `4096` | 单请求上下文上限，包含输入与输出 |
| `HY4FLASH_MAX_NUM_BATCHED_TOKENS` | `4096` | `4096` | 每步调度 token 预算 |
| `HY4FLASH_GPU_MEMORY_UTILIZATION` | `0.85` | `0.85` | vLLM 显存预算 |
| `HY4FLASH_TP` | `8` | `8` | 张量并行度，脚本启用 EP |
| `HY4FLASH_CHECKPOINT` | `/home/hy4nxt` | 同左 | 权重路径 |
| `HY4FLASH_HOST` / `HY4FLASH_PORT` | `127.0.0.1` / `8000` | 同左 | 服务地址和端口 |
| `HCCL_BUFFSIZE` | `1024` | `1024` | 通信缓冲，单位 MiB |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | 同左 | NPU allocator 配置 |
| `VLLM_LOGGING_LEVEL` | `INFO` | 同左 | 日志级别 |

脚本会设置 `VLLM_USE_V2_MODEL_RUNNER=1`。服务名为 `hy4-flash`。图模式的 decode 捕获预算为 512 query token，DFly 5 下最多覆盖每路 6 query 的 85 路同宽批次。服务器的 128 并发是调度上限，128K 是配置上限，最新逐值对照覆盖到 16K 实际输入。

用分支所附脚本复现实机服务器设置。

```bash
export HY4FLASH_GRAPH_MODE=decode
export HY4FLASH_ENABLE_DFLY=1
export HY4FLASH_SHARED_EXPERT_MULTISTREAM=1
export HY4FLASH_KV_CACHE_DTYPE=fp8
export HY4FLASH_MAX_NUM_SEQS=128
export HY4FLASH_MAX_MODEL_LEN=131072
./start_hy4flash.sh
```

### 用 CLI 切换的特性

- **Prefix cache**。开启前设置 `export HY4FLASH_ENABLE_DFLY=0`，启动命令追加 `--enable-prefix-caching`。模型在启动时拒绝同时开启这两项。下面使用已验证的 eager 配置。默认启动传入 `--no-enable-prefix-caching`。

  ```bash
  export HY4FLASH_GRAPH_MODE=eager
  export HY4FLASH_ENABLE_DFLY=0
  ./start_hy4flash.sh --enable-prefix-caching
  ```

- **Engram CPU offload**。默认 `cpu_offload=true`，TP 本地表使用 pinned CPU 内存。传入 `--engram-config '{"cpu_offload":false}'` 将表放在 NPU，TP8 下仅表权重平均每卡约 15.8 GB。传入 `--engram-config '{"cpu_offload":true}'` 可显式开启。
- **文本或视觉**。默认加载视觉塔。启动追加 `--language-model-only` 可用于纯文本评测。
- **调度对照**。启动追加 `--no-async-scheduling` 可固定到同步调度。比较精度时还要核对实际 prefill 分块和 batch。

### 算子调试与实验开关

以下模型键通过 `--additional-config` JSON 传入。启动脚本已经构造该 JSON，手工传入整段配置时应带上原启动项，包括 `enable_mlapo: false`、`enable_fused_mc2: 0`、共享专家双流设置和 `ascend_compilation_config.enable_npugraph_ex: false`。

| 键或变量 | 当前默认 | 作用和实测状态 |
| --- | --- | --- |
| `hy4flash_precise_decode` | 开 DFly 时 `true` | 控制 decode 精度策略，包括 iHC 的逐行均方归约。 |
| `hy4flash_fuse_c8_write` | 具备 Triton 的 950DT + C8 路径为 `true` | 使用 Triton 合并 C8 数据和元数据写入，`false` 可回归原写法。 |
| `hy4flash_batch_verify_attention` | 满足 950DT + C8 等条件时为 `true` | 选择经验证的同宽 attention 批量路径和对应的 PV 归约策略，`false` 用于对照。 |
| `hy4flash_triton_mla_logits` | `false` | Triton logits 实验，单算子输出及回放已验证，实测性能收益不足。 |
| `hy4flash_cannbot_mla_prolog` | `false` | CANNBot BF16 prolog 实验。需要匹配容器 Python 和 CPU 架构的 CANNBot 运行时，并从 `PYTHONPATH` 导入 `mla_prolog_v4`。整网对照 336 个输出 token 中有 126 个位置不同。 |
| `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1` 或 JSON `enable_flashcomm1: true` | `0` / `false` | FlashComm1 / sequence parallel 实验已启动验证，观测到首 token 和后续输出差异。关闭时设变量为 `0` 并保持 JSON 为 `false` 或省略该键。 |

例如在服务器保持双流开启，临时对照 C8 写缓存路径。

```bash
./start_hy4flash.sh --additional-config '{
  "enable_mlapo": false,
  "enable_fused_mc2": 0,
  "multistream_overlap_shared_expert": true,
  "ascend_compilation_config": {"enable_npugraph_ex": false},
  "hy4flash_fuse_c8_write": false
}'
```

| 实验 | 结果 |
| --- | --- |
| C8 QSFA | 4 head、Top-K 2,048 的 strided PA 探针通过。相同页布局的连续与零拷贝读取结果逐值一致，和现行 C8 attention 对照时仍有反量化舍入与带 sink 的差异。 |
| SMLA | 容器测试的接口要求 head dim 512，Hy4 吸收后需要 576。MixedQuant 路径的测试布局要求 KV 行宽 608，Hy4 的 C8 基础行宽为 656。 |
| CANNBot BF16 prolog | 四组整网对照中，出现差异的位置分别为 5 / 24、44 / 192、62 / 96 和 15 / 24。 |
| FlashComm1 | 固定环境对照中，短请求有 12 / 24 个输出 token 位置不同，混合长度有 67 / 96 个位置不同。短请求相同前缀的所选 logprob 最大绝对差为 0.906。 |

C8 QSFA 和 SMLA 当前仅有算子探针。模型的 attention 使用分页读取和分步 MLA。

## 性能证据和当前边界

- **FULL decode**。同配置的 TP8 + EP、C8、DFly 5 单路对照中，4K 输入的 TPOT 从 144.5 降至 12.7 ms，16K 从 144.4 降至 12.7 ms。服务器大配置原有 49 个目标图档位，在第 16 张图遇到 CANN Runtime 207001 内存分配失败。改为 12 张目标图后，全图捕获结束，KV cache 容量为 1,363,904 token。BS9 和 BS17 分别复用 BS16 和 BS32 的 FULL 图。
- **C8 算子优化**。接入 FULL 前的对照中，融合写入与同宽 attention 优化覆盖短单路、短 BS8、混合长度和 16K 单路。336 个输出 token 的 token ID、logprob 和 top-5 逐值一致，16K 单路 TPOT 从 105.7 降至 71.0 ms。
- **历史 GSM8K**。100 条、5-shot、最多 8 路并发，使用当时的分段图和 DFly 5。9,310 轮验证共提出 46,550 个草稿 token，接受 21,795 个。平均接受长度 `1 + 21795 / 9310 = 3.341` token/轮，草稿接受率 46.82%。
- **Prefix cache**。先前的分段图和 eager 分别完成 10 个场景。65-token 热请求命中 64 token，2,048、2,049、2,052-token 请求分别命中 1,984、2,048、2,048 token，同图同文本命中 128 token，换图命中 0。冷热分块不同的 2,052-token 请求所选 logprob 最大差为 0.170681，冷请求强制按 `[2048, 4]` 切分后逐值一致。
- **并行限制**。当前模型校验要求 DP、PP、PCP、DCP 均为 1，EPLB、DBO、LoRA 和 KV transfer 暂未接入。TP2 + DP4 的试验中，大张量 HCCL AllGather 报 EI0006，`HCCL_BUFFSIZE=1024` 仍复现。独立 8-rank 程序中，BF16 `[32, 1536]` 通过，`[2048, 1536]` 在 AICPU 多通道建链阶段超时。DP 试验改动已回退。TP8 + EP 是当前整网验收配置。
- **模型回归**。最终 FULL 版本的模型参考测试在 950DT 容器内为 `14 passed`，覆盖设备侧长度、同宽 decode padding、分页读取、C8、候选与因果边界。

### Rank 0 profiling 摘要

4K 输入、200 输出在 prefill 后采集 6 步 decode，由 CANN msprof 导出。目标和 draft 每步各提交一次整图。

| msprof 统计项 | 次数 |
| --- | ---: |
| `aclmdlRIExecuteAsync` | 12 |
| `_write_c8_rows` | 252 |
| `_read_c8_rows` | 252 |
| `_copy_pool_keys` | 144 |

## 源码配套

分支以 `vllm-ascend` 的 `915d58873c0854c0550b56f66a126ca2cb114e71` 为基线，包含 Ascend 模型实现、运行时接入和测试。引擎侧采用 vLLM `ba15685ab7f4f5fbed8630bbb7441f8837e76fbe`，完整改动随分支放在 `patches/hy4flash-vllm-ba15685.patch`，覆盖模型配置、文本和视觉共用实现、Engram、处理器及参考计算。启动脚本位于 `examples/hy4flash/start_hy4flash.sh`。两份源码采用同级目录布局。

在具有配套 vLLM Ascend 编译依赖的环境中，准备有该分支的 `vllm-ascend` 源码和上述基线的 `vllm` 源码。对干净的 vLLM 基线应用一次补丁，再复制启动脚本到两份源码的上一级。

```bash
git -C vllm apply --check ../vllm-ascend/patches/hy4flash-vllm-ba15685.patch
git -C vllm apply ../vllm-ascend/patches/hy4flash-vllm-ba15685.patch
install -m 0755 vllm-ascend/examples/hy4flash/start_hy4flash.sh ./start_hy4flash.sh
```
