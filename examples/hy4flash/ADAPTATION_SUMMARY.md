# Hy4 Flash 昇腾适配总结

更新于 2026-11-18。当前实机基线为 8 张 Ascend 950DT，每卡 96 GiB HBM。运行环境为 CANN 9.2.1、`torch_npu 2.10.0.post7`、Python 3.12.16，使用 vLLM MRV2、TP8 + EP 和 Hy4 Flash 全量权重。启动示例使用 `/home/hy4nxt` 作为权重路径。

下文的 TP8 精度和性能数据对应上述基线。另在 CANN 9.3.0、Python 3.12.13、相同 Torch-NPU 版本的 8 张 950DT 上，完成了 TP4 × DP2 的启动和请求验证，包括 Engram DP 共享 CPU 表；该验证不替代下文的精度评测。

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

启动脚本使用 checkpoint 配置选择 dtype。当前 `/home/hy4nxt` 自动解析为 BF16。

下面的 `export` 在**容器内同一个 shell** 执行，再运行脚本。从宿主机直接调用 `docker exec` 时，可用 `-e NAME=value` 传递这些变量。

| 特性 | 环境变量 | 默认 |
| --- | --- | --- |
| 图模式 | `export HY4FLASH_GRAPH_MODE=decode` 启用 FULL decode。用 `export HY4FLASH_GRAPH_MODE=eager` 让目标和 draft 都走 eager。 | `decode` |
| DFly 5 | `export HY4FLASH_ENABLE_DFLY=1` 开启，`export HY4FLASH_ENABLE_DFLY=0` 关闭。 | `1` |
| 共享专家双流 | `export HY4FLASH_SHARED_EXPERT_MULTISTREAM=1` 开启，`export HY4FLASH_SHARED_EXPERT_MULTISTREAM=0` 关闭。 | `1` |
| KV 缓存类型 | `export HY4FLASH_KV_CACHE_DTYPE=fp8` 用 C8，`export HY4FLASH_KV_CACHE_DTYPE=bfloat16` 用 BF16 KV。 | `fp8` |

使用 `export 变量名=值` 可覆盖脚本的资源设置。

| 环境变量 | 实机服务器脚本默认 | 分支所附脚本默认 | 用途 |
| --- | --- | --- | --- |
| `HY4FLASH_MAX_NUM_SEQS` | `128` | `128` | 调度并发上限 |
| `HY4FLASH_MAX_MODEL_LEN` | `131072` | `131072` | 单请求上下文上限，包含输入与输出 |
| `HY4FLASH_MAX_NUM_BATCHED_TOKENS` | `4096` | `4096` | 每步调度 token 预算 |
| `HY4FLASH_GPU_MEMORY_UTILIZATION` | `0.85` | `0.85` | vLLM 显存预算 |
| `HY4FLASH_TP` | `8` | `8` | 张量并行度，脚本启用 EP |
| `HY4FLASH_CHECKPOINT` | `/home/hy4nxt` | 同左 | 权重路径 |
| `HY4FLASH_HOST` / `HY4FLASH_PORT` | `127.0.0.1` / `8000` | 同左 | 服务地址和端口 |
| `HCCL_BUFFSIZE` | `1024` | `1024` | 通信缓冲，单位 MiB |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | 同左 | NPU allocator 配置 |
| `VLLM_LOGGING_LEVEL` | `INFO` | 同左 | 日志级别 |

脚本会设置 `VLLM_USE_V2_MODEL_RUNNER=1`。服务名为 `hy4-flash`。图模式的 decode 捕获预算为 512 query token，DFly 5 下最多覆盖每路 6 query 的 85 路同宽批次。配对图回放已覆盖实际 130,944 token 输入。11 月 19 日评测服务使用 131,072 总上下文和 32 路调度上限，评测客户端并发为 16。

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
- **Engram DP 共享 CPU 表**。同机 TP4 × DP2 可传入 `--engram-config '{"cpu_offload":true,"dp_shared_memory":true}'`，使两个 DP 副本中相同 TP rank 的 worker 共用一份表分片。要求 DP peer 位于同一节点和共享内存命名空间，`/dev/shm` 能容纳表权重；当前 BF16 checkpoint 的 16 份共享分片合计约 118 GiB。显式传入 `dp_shared_memory:false` 时仍按 DP 副本各自存表，不再被强制改为 true。共享模式不支持 dummy checkpoint；`embedding_across_dp` 和磁盘 offload 仍未接入。
- **文本或视觉**。默认加载视觉塔。启动追加 `--language-model-only` 可用于纯文本评测。
- **调度对照**。启动追加 `--no-async-scheduling` 可固定到同步调度。比较精度时还要核对实际 prefill 分块和 batch。

### 算子调试与实验开关

以下模型键通过 `--additional-config` JSON 传入。启动脚本已经构造该 JSON，手工传入整段配置时应带上原启动项，包括 `enable_mlapo: false`、`enable_fused_mc2: 0`、共享专家双流设置和 `ascend_compilation_config.enable_npugraph_ex: false`。

| 键或变量 | 当前默认 | 作用和实测状态 |
| --- | --- | --- |
| `hy4flash_precise_decode` | 开 DFly 时 `true` | 控制 decode 精度策略，包括 iHC 的逐行均方归约。 |
| `hy4flash_fuse_c8_write` | 具备 Triton 的 950DT + C8 路径为 `true` | 使用 Triton 合并 C8 数据和元数据写入，`false` 可回归原写法。 |
| `hy4flash_gpu_c8_scale` | `false` | 使用 `fp8` KV 时可设为 `true`，按 GPU 的 2 次幂规则计算 C8 scale。覆盖目标模型和 DFly，沿用现有缓存布局。 |
| `hy4flash_batch_verify_attention` | 满足 950DT + C8 等条件时为 `true` | 选择经验证的同宽 attention 批量路径和对应的 PV 归约策略，`false` 用于对照。 |
| `hy4flash_share_swa_c8_reads` | 满足 950DT + C8 等条件时为 `true` | FULL 图中每路 6 query、8 个本地头、512 窗口的 SWA 共读 KV，`false` 可回到逐 query 读取。 |
| `hy4flash_fuse_indexpool_fp8` | 满足 950DT + C8 + FULL decode 等条件时为 `true` | 融合 IndexPool query 与 pooled key 的 128 维 Hadamard 和 E4M3 舍入，沿用 CANN 行 scale 与除法。支持连续 BF16 输入、最多 8192 行，`false` 可回到参考路径。 |
| `hy4flash_fuse_indexpool_select` | 满足 950DT + C8 + FULL decode 等条件时为 `true` | 压缩比 4、16 个 index heads、每路 6 个验证 query 共读池页并打分，保留现有 TopK、排序和候选拼接。 |
| `hy4flash_fuse_swa_q_norm` | `false` | 实验开关。将 SWA 的 query RMSNorm 和已有 MXFP8 投影的动态激活量化合并。BF16 投影和 IndexPool 层沿用原路径。 |
| `hy4flash_fuse_attn_output_gate` | `false` | 950DT FULL decode 的目标模型实验。融合 attention 输出门的 sigmoid、乘法和 `o_proj` 的 OCP MXFP8 激活量化。适用于已测的 1,024 和 2,048 本地宽度。 |
| `hy4flash_overlap_attn_output_gate` | 950DT FULL decode 为 `true` | 目标模型的输出 gate 投影与分页 attention 并行提交，`o_proj` 前等待 gate。 |
| `hy4flash_fuse_c8_qk` | `false` | 950DT C8 FULL decode 实验。融合 sparse MLA 页读取、反量化和 IEEE FP32 QK。每路 6 query、4 个本地头、2,048 候选，给 PV 保留 512 维 FP32 latent。 |
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

目标模型第 1 层（从 0 编号）的 `q_b_proj` 默认将 checkpoint 的 block FP8 权重解量为 BF16，推理使用 BF16 GEMM。可单独打开 GPU C8 scale 做 KV 精度对照。

```bash
./start_hy4flash.sh --language-model-only --additional-config '{
  "enable_mlapo": false,
  "enable_fused_mc2": 0,
  "multistream_overlap_shared_expert": true,
  "ascend_compilation_config": {"enable_npugraph_ex": false},
  "hy4flash_gpu_c8_scale": true
}'
```

此前 950DT 验证中，448 行 C8 输入在 eager 和图回放中得到相同的 scale 和 FP8 字节，逐项符合 GPU 写缓存公式的 CPU 参考。GPU C8 scale 和第 1 层 BF16 投影同时启用时，TP8 + EP、DFly 5 的全量语言模型通过 9 和 2,052 token 输入的生成测试，目标和 draft 都回放了 FULL 图。

| 实验 | 结果 |
| --- | --- |
| C8 QSFA | 4 head、Top-K 2,048 的 strided PA 探针通过。相同页布局的连续与零拷贝读取结果逐值一致，和现行 C8 attention 对照时仍有反量化舍入与带 sink 的差异。 |
| SMLA | 容器测试的接口要求 head dim 512，Hy4 吸收后需要 576。MixedQuant 路径的测试布局要求 KV 行宽 608，Hy4 的 C8 基础行宽为 656。 |
| CANNBot BF16 prolog | 四组整网对照中，出现差异的位置分别为 5 / 24、44 / 192、62 / 96 和 15 / 24。 |
| FlashComm1 | 固定环境对照中，短请求有 12 / 24 个输出 token 位置不同，混合长度有 67 / 96 个位置不同。短请求相同前缀的所选 logprob 最大绝对差为 0.906。 |

C8 QSFA 和 SMLA 当前仅有算子探针。模型的 attention 使用分页读取和分步 MLA。

## 性能证据和当前边界

- **IndexPool Hadamard 与 E4M3 舍入**。950DT TP8 + EP、C8、DFly 5 在相同预热的 FULL decode A/B 中，414 个固定续写位置、2,240 个采样原始 logits 及 DFly 计数逐值一致。四档主图 p50 时间下降 6.4%–7.4%。4K 单路、512 token BS8、513 token BS9 和 4K BS6 的 p50 TPOT 分别下降 5.79%、6.52%、6.19% 和 0.88%。单算子额外覆盖 E4M3 中点及相邻 FP32 值，144,384 个值逐位一致。
- **SWA 共读 KV**。950DT TP8 + EP、C8、DFly 5、同步调度的 FULL 图 A/B 中，23 个请求、414 个固定续写位置的 token ID、2,242 个采样的原始 logits（top-5 和固定续写 token）及 DFly 计数逐值一致。自由生成时，512 token × BS8 的 p50 TPOT 从 22.76 降至 22.20 ms，513 token × BS9 补齐到 BS16 的 p50 从 57.98 降至 56.56 ms。64 和 4K 单路基本持平。
- **SWA query RMSNorm 与动态 MXFP8 量化融合**。在共读版本上打开融合，固定续写的 414 个位置仍逐项一致。四组自由生成的 p50 TPOT 相对共读版变化介于 −0.23% 和 +0.15%，默认关闭。
- **Attention 输出 gate**。950DT TP8 + EP、C8、DFly 5、8K 上下文配置下，Triton 融合和旁流各自与基线的 360 个固定续写位置、1,949 个采样原始 logits、DFly 计数及自由生成一致。4K 单路 p50 TPOT 为基线 12.1042 ms、Triton 12.0461 ms、旁流 11.9375 ms。512 token BS8 和 513 token BS9 的旁流 p50 TPOT 分别下降 0.46% 和 0.39%。
- **Attention 输出 gate 的 128K 对照**。对 130,944 token 输入完成实际 prefill，生成 128 token。在同一请求的 24 个 decode 步上，用相同 KV 和输入配对回放三张目标图。8 个 rank 的目标 hidden states 逐值一致。Rank 0 主图 p50 为基线 32.3193 ms、Triton 32.2176 ms（−0.31%）、旁流 32.0118 ms（−0.95%）。本次测量范围为目标图回放。
- **C8 页读取与 QK**。对 6 个 query、4 个本地头和 2,048 候选，Triton 融合页读取、反量化和 IEEE FP32 QK。QK 核只将 512 维 FP32 latent 写回给 PV。单卡页读到 PV 从 288.04 降到 215.81 µs，QK 和 PV 逐值一致。PV 直接读 C8 页的尝试耗时 382.70 µs，出现 4 个 BF16 值差异。整网支持通过 `hy4flash_fuse_c8_qk` 单独开启。
- **IndexPool 分页打分和共读池页**。复用同一份池页为 6 个 query 打分，将 Cube 运算拆成 4+2 个 query，在 128 组 tile 内完成。原 TopK、排序和候选拼接在全有效、尾行 padding 和分数并列场景逐项一致。128K 配置下，实际 4K、16K 和 128K 时，页读、打分至 TopK 的回放耗时从 234.84、239.67 和 281.77 µs 降到 96.65、98.69 和 214.85 µs。满足形状条件的 FULL decode 路径默认开启。
- **Gate 旁流和 IndexPool 分页打分的默认路径**。950DT TP8 + EP、DFly 5 和 C8 下，对照旧路径、新默认和新默认加 C8 QK。396 个固定续写位置及 2,237 个采样 raw logits 逐值一致，短测自由生成和 DFly 计数一致。配对回放中，BS1、BS8、BS9 和 128K 的有效 hidden 与 5 层 DFly 候选逐值一致。实际 130,944 token 输入的主图 p50 为 32.44 → 31.47 → 30.21 ms。C8 QK 相对新默认的主图 p50 降幅为 3.97%–11.82%。
- **iHC inverse RMS**。48 行单卡图回放中 CANN 与 Triton 探针从现行逐行归约的 103 µs 降至约 14 µs。用 4 层 checkpoint 的 attention/MLP 门控权重及合成 BF16 输入继续验证，两种探针均改变门控和 BF16 残差，模型沿用现行逐行归约。
- **FULL decode**。同配置的 TP8 + EP、C8、DFly 5 单路对照中，4K 输入的 TPOT 从 144.5 降至 12.7 ms，16K 从 144.4 降至 12.7 ms。服务器大配置原有 49 个目标图档位，在第 16 张图遇到 CANN Runtime 207001 内存分配失败。改为 12 张目标图后，全图捕获结束，KV cache 容量为 1,363,904 token。BS9 和 BS17 分别复用 BS16 和 BS32 的 FULL 图。
- **C8 算子优化**。接入 FULL 前的对照中，融合写入与同宽 attention 优化覆盖短单路、短 BS8、混合长度和 16K 单路。336 个输出 token 的 token ID、logprob 和 top-5 逐值一致，16K 单路 TPOT 从 105.7 降至 71.0 ms。
- **历史 GSM8K**。100 条、5-shot、最多 8 路并发，使用当时的分段图和 DFly 5。9,310 轮验证共提出 46,550 个草稿 token，接受 21,795 个。平均接受长度 `1 + 21795 / 9310 = 3.341` token/轮，草稿接受率 46.82%。
- **Prefix cache**。先前的分段图和 eager 分别完成 10 个场景。65-token 热请求命中 64 token，2,048、2,049、2,052-token 请求分别命中 1,984、2,048、2,048 token，同图同文本命中 128 token，换图命中 0。冷热分块不同的 2,052-token 请求所选 logprob 最大差为 0.170681，冷请求强制按 `[2048, 4]` 切分后逐值一致。
- **并行限制与 DP 验证**。模型已放开 DP 校验，PP、PCP、DCP 仍要求为 1，EPLB、DBO、LoRA 和 KV transfer 暂未接入。TP4 × DP2 已在上述 CANN 9.3.0 环境完成启动、curl 和两路并发 4209-token 长输入验证；Engram 共享开启前后，短补全 32 token、长补全 16 token 的 token 和所选 logprob 逐值一致，两个 DP 都处理了请求。26 项相关单测通过。128 并发及 128K 上下文为启动配置上限，本次未做该规模压力验收。
- **历史 HCCL 问题**。此前 CANN 9.2.1 的 TP2 + DP4 试验中，大张量 HCCL AllGather 报 EI0006，`HCCL_BUFFSIZE=1024` 仍复现；独立 8-rank 程序中，BF16 `[32, 1536]` 通过，`[2048, 1536]` 建链超时。CANN 9.3.0 下对 `[32, 1536]`、`[2048, 1536]`、`[4096, 1536]` 各重复 3 次的 8-rank 原生 AllGather 均通过，本次未修改 HCCL 库，也未将其记为底层通信缺陷修复。
- **模型回归**。本轮模型参考测试在 950DT 容器内为 `16 passed`，覆盖设备侧长度、同宽 decode padding、分页读取、C8、候选与因果边界，以及 SWA 共读的跨页和 padding 场景。

### Rank 0 profiling 摘要

4K 输入、200 输出在 prefill 后采集 6 步 decode，由 CANN msprof 导出。目标和 draft 每步各提交一次整图。以下为开启 `all-optimizations.json` 的最新 Rank 0 记录。

| msprof 统计项 | 次数 |
| --- | ---: |
| `aclmdlRIExecuteAsync` | 12 |
| `_write_c8_rows` | 252 |
| `_read_c8_rows` | 180 |
| `_read_c8_qk` | 72 |
| `_score_paged_pool` | 72 |
| `_indexpool_hadamard` | 144 |
| `_indexpool_round_e4m3` | 144 |
| `_gate_dynamic_mx_quant` | 246 |
| `RmsNormDynamicMxQuant` | 180 |
| `SwigluMxQuant` | 246 |

### 全套优化的评测与启动

`all-optimizations.json` 保存 950DT、TP8 + EP、DFly 5 和 C8 的已验证配置，包含 Gate 旁流及量化融合、SWA 共读及 query RMSNorm 量化融合、IndexPool Hadamard 和分页打分、分页 C8 读取与 FP32 QK。第 1 层（从 0 编号）的 `q_b_proj` 固定使用 checkpoint 解量后的 BF16 权重。启动脚本根据 checkpoint 选择计算 dtype。

从两份源码的上一级目录，可按本轮评测的服务设置启动。

```bash
export HY4FLASH_MAX_NUM_SEQS=32
export HY4FLASH_MAX_MODEL_LEN=131072
export HY4FLASH_GPU_MEMORY_UTILIZATION=0.70
./start_hy4flash.sh --language-model-only --no-async-scheduling \
  --additional-config "$(cat vllm-ascend/examples/hy4flash/all-optimizations.json)"
```

EvalScope 1.15.0 使用并发 16、`temperature=0.9`、`top_p=1` 和 `top_k=-1`，完整评测 AIME2026 与 GPQA-Diamond 各三轮。每轮都重新生成输出。

| 数据集 | 第一轮 | 第二轮 | 第三轮 | 平均正确率 |
| --- | ---: | ---: | ---: | ---: |
| AIME2026 | 30 / 30 | 29 / 30 | 30 / 30 | 98.89% |
| GPQA-Diamond | 184 / 198 | 181 / 198 | 181 / 198 | 91.92% |

684 次作答均自然结束。AIME 第一轮的生成上限为 32,768，最长输出 30,452。其后五轮请求使用 131,072 的总上下文预算。AIME 第二轮最长输出 97,818 token。文本检查中，29 条 AIME 思考段出现了局部公式控制字符，最终答案段未检出这些字符。

16,384 输入、1,024 输出、8 路并发的单次无 profiler 测量，逐请求 TPOT 均值为 123.40 ms/token。8 路全部进入 decode 的 20.0 秒区间中，TPOT 均值为 27.39 ms/token，P50 为 24.62 ms/token。较早开始生成的请求会在其余请求分块 prefill 时累计等待，逐请求 TPOT 包含这段等待。

## 源码配套

分支以 `vllm-ascend` 的 `915d58873c0854c0550b56f66a126ca2cb114e71` 为基线，包含 Ascend 模型实现、运行时接入和测试。引擎侧采用 vLLM `ba15685ab7f4f5fbed8630bbb7441f8837e76fbe`，完整改动随分支放在 `patches/hy4flash-vllm-ba15685.patch`，覆盖模型配置、文本和视觉共用实现、Engram、处理器及参考计算。启动脚本位于 `examples/hy4flash/start_hy4flash.sh`。两份源码采用同级目录布局。

在具有配套 vLLM Ascend 编译依赖的环境中，准备有该分支的 `vllm-ascend` 源码和上述基线的 `vllm` 源码。对干净的 vLLM 基线应用一次补丁，再复制启动脚本到两份源码的上一级。

```bash
git -C vllm apply --check ../vllm-ascend/patches/hy4flash-vllm-ba15685.patch
git -C vllm apply ../vllm-ascend/patches/hy4flash-vllm-ba15685.patch
install -m 0755 vllm-ascend/examples/hy4flash/start_hy4flash.sh ./start_hy4flash.sh
```
