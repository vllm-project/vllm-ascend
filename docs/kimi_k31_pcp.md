# Kimi K3.1 KDA PCP 与 PD

本适配基于 `hust17yixuan/vllm-ascend-kimi` 的 `releases/v0.30.0`
（`635d196858ddeba42941bf9221105c25ede3bc27`），配对 vLLM v0.30.0
（`ced6857afa0ea7b2e3f0846a62e1394e90f15607`），使用 Model Runner V2。
实现参考 [PCP PR #18511](https://github.com/vllm-project/vllm-ascend/pull/18511)
和 [PD PR #19518](https://github.com/vllm-project/vllm-ascend/pull/19518)。

## Head 分片与 token 顺序

KDA 使用 `TP × PCP` 个 head 分片，分片编号为 `tp_rank * pcp_size + pcp_rank`。
Q/K/V、gate、卷积、SSM、输出投影均使用同一个分片编号；`f_a` 保持复制。
KDA 的 head 数必须能被 `TP × PCP` 整除。输出先在 PCP 组求和，再在 TP 组求和。

每个 KDA 层先 gather PCP 的局部 hidden states，恢复每个请求的因果 token 顺序，
计算所有请求对应的本地 head，再取回本 rank 的 token。KDA metadata 使用全局请求、
block tables 和 slot mappings；SFA 继续使用原有局部 query 与 KV gather 方案。
因此不改变 SFA 的 NoPE、RoPE、indexer 或 PCP cache 写入流程。

沿用本分支的 `pcp_decode_sharding` 判定：`PCP > 1 && DCP == 1` 时按请求切分
decode；DCP 大于 1 时使用原有复制 decode。KDA 在两种情况下都维护每个请求的本地
head 状态，包括本 rank 没有局部请求的情况，不需要关闭 decode sharding。

## 图执行

Gather/scatter 注册为具有符号 shape 的 custom op。Permutation indices 和 padding
mask 由 metadata builder 写入固定地址的 buffer，供图重放使用。KDA 的 decode
capture 容量按全局请求数计算，不修改 SFA 或 runner 的局部 capture 配置。

本分支 Kimi 模型默认可能将 `FULL_AND_PIECEWISE` 降为 `FULL_DECODE_ONLY`。
验证实际 PIECEWISE 捕获时设置已有开关 `VLLM_USE_BREAKABLE_CUDAGRAPH=1`，并检查日志
确实出现 PIECEWISE 和 FULL 两类捕获；仅传入配置不能作为 PIECEWISE 通过的证据。

## Mooncake V2 PD

Mamba 状态传输使用与 KDA 相同的 TP-major、PCP-minor 分片编号。
P/D 的 head 分片数分别为各自的 `TP × PCP`；两者需要具有整数比例。
相等时直接对应，大小不等时按 head 区间拆分或拼接。卷积分别处理 Q/K/V 区间，
SSM 按 head 维度处理，不将 PCP rank 当成相同状态的副本。

非池化 PD 采用先 P 后 D 的协议，完整传递 P 返回的 `kv_transfer_params`。
P 必须保留状态块直到 D 拉取完成。验证批量请求时，应给 P 足够的 KV 容量，或逐请求
转发到 D；等待整个 P 批次完成再转发时，容量不足会阻塞后续请求。

## Dummy 模型与验证边界

Dummy config 保留原模型全部 93 层、69 个 KDA 层、24 个 SFA 层及其顺序，
保留 hidden size 7168、KDA 96 heads × 128、conv size 4 和全部 SFA/indexer 维度。
仅减少专家数并移除权重量化描述，以 BF16 `load_format="dummy"` 加载。
常规验证使用 16 experts、top-k 16；四卡 PD 验证使用 4 experts、top-k 4。
不裁层，避免改变 hybrid cache 的分组及页布局。

2026-10-26 在受管控的四张空闲 A5 上验证：

| 场景 | 结果 |
| --- | --- |
| TP2 × PCP2、DCP1、eager | 10 个请求通过，含 300-token chunked prefill、并发和 decode |
| TP2 × PCP2、DCP1、FULL | 与 eager 的输出 token、cache groups、shape、stride 一致 |
| TP2 × PCP2、DCP1、PIECEWISE + FULL | 实际捕获两类图，与 eager 输出及缓存布局一致 |
| TP2 × PCP2、DCP2、FULL | 复制 decode 通过，输出与 DCP1 一致；使用 FP8 indexer cache |
| Mooncake V2：P(TP2) → D(PCP2) | 4 个请求完成；D 端分别命中 128、256、64、16 个远端缓存 token |
| 投影加载、token 重排、图 metadata、PD 地址规划 | 267 项相关单元测试通过 |

TP2 × PCP2 每 rank 的 KDA conv spec 为 `[3, 9216]`（BF16），
SSM spec 为 `[24, 128, 128]`（FP32）。实际缓存包含 117 个 layer entries
（69 KDA、24 SFA、24 indexer），以及一个 attention/indexer group 和三个各含
23 层的 Mamba groups。已核对解析后的原始与 dummy config 除权重相关字段外一致。

DCP2 使用 FP8 indexer cache 是本分支 A5 SFA 自定义算子的现有要求。
验证环境还需要匹配的 `local_kv_ops`、`_cann_ops_custom` 和 `vllm_ascend_C`，
通过 `ASCEND_CUSTOM_OPP_PATH`、`LD_LIBRARY_PATH` 加载。

上述结果是 dummy 权重的功能验证。量化真实权重精度、长上下文、prefix cache、
推测解码、多模态输入与吞吐性能不在本次实测范围内。
PD 实测覆盖真实 connector 的离线 LLM 接口与跨 TP/PCP 拓扑搬运；HTTP proxy 和
P/D 两侧同时 TP2 × PCP2 的八卡组合未实测。不同拓扑的 dummy 随机权重不保证一致，
PD 验收采用实际 transfer、远端 cache hit 和完整 decode，不声称其输出精度等价。

当前测试运行环境在 PD 请求完成后的进程清理阶段出现
`malloc(): unsorted double linked list corrupted` 和 worker 退出超时。
使用原始基线代码、P/D 均为 TP2/PCP1 重放同一批请求，也复现相同报错。
因此 PD 请求功能通过，正常关闭仍有已有问题，不能视为本次已修复。
