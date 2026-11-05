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
