# HiSparse 工作源码审阅快照

这里保留经过范围筛选的 HiSparse 集成差异源码、测试、native 构建支持及待验证 overlay。
2026-11-02 移除了正式 backend 未使用的自定义 BF16 Attention 核心、专用绑定和测试，
剩余 **150 个 payload 文件**。每个文件的 SHA256、来源及必要性分类见 `manifest.json`。

原完整工作版本仍可从提交
[`caed56ef4`](https://github.com/zhenwenqi2024/vllm-ascend/tree/caed56ef4fb417491ef7b8667a045dc7aa4c8241/benchmarks/hisparse/experimental/20261031)
查阅，不再将未使用的 Attention 实现保留在本功能 PR 的 diff 中。

## 目录与状态

| 目录 | 内容 | 验证边界 |
| --- | --- | --- |
| `source/vllm_ascend/` | 当前工作版本的 HiSparse、C8、Host DMA、P/D、lazy MTP、Resident 更新、视图复用及兼容依赖 | 包含未完成 NPU 验证的代码，不能将整组标为通过 |
| `source/csrc/` | Host 注册、native DMA 和 resolver 支持源码 | 不含编译产物或自定义 Attention 核心；各分支历史验证范围不同 |
| `source/tests/` | 集成工作分支对应测试 | 不是当前 rebased main 的自动测试；配套依赖和硬件要求仍适用 |
| `source/benchmarks/` | 历史 primitive qualification 工具 | 不等于模型精度或性能验收 |
| `candidates/follower-query-prepack-v1/` | 提前准备 follower 当前 Q，保留 KV-ready 等待的候选 | 48 项 CPU proxy 合同通过；未做 NPU graph/输出和模型性能验证 |
| `candidates/mtp-group-prefetch-a6-v1/` | 将你的 verification 整组预取移到当前冻结 Ascend adapter，仅放宽 gather 门禁 | 54 项 CPU 调用顺序/策略检查通过；当前组合未做 NPU/模型测试 |

`source` 保存的是实现文件，不是补丁文件：它含该版本的完整文件内容，方便逐行查看。
来源包括历史冻结运行树以及对应集成分支的测试和 native 支持。
清理中编辑过的文件另记原始路径、原 SHA256 和修改原因，不能再按原运行树整组认定为逐字相同。
没有复制机器配置、凭据、权重、二进制、原始 profiling 或私人日志。

自定义 Attention 的构建入口和注册已删除。两个归档 SFA 版本的 BF16 分支复用各自已有的
`_C_ascend.npu_sparse_flash_attention` bridge；C8 dispatch 和 bridge 定义保持不变。
仍被缓存寻址和索引压缩调用的两个辅助函数迁到 `ops/hisparse_layout.py`，函数定义保持不变。
以上均为非部署快照的范围清理，不改变正式 `vllm_ascend/` 路径，也不继承新的模型验证结果。

## 与正式路径的关系

当前正式 `vllm_ascend/` 路径仍保留已审查的基础接入和 CPU 页面描述符优化。
本目录不由 setup/plugin 导入，不改变服务默认行为，也不会自动应用覆盖。
完整实验版本依赖旧的冻结 Core 接口；直接覆盖 rebased main 会倒退新的 API 和修复。
因此先保留完整代码供审阅，确认正确性和独立收益后，将相应改动收敛到正式路径。

本目录的 `conftest.py` 防止 pytest 将冻结测试当成当前 checkout 的测试；
`ruff.toml` 和 pre-commit 排除仅保护快照字节，不代表这些文件通过了当前 lint/CI。
全仓库安全扫描并未因此禁用。SHA 校验与 Python 语法检查单独执行并记录。

## 配套版本

- 公共 vLLM merge：`17b5c9a97198275745961638a35e78aa1ad907ff`，仅带 PR62592。
- 最后完成模型实测的 tree：`b7bf04864e45cbc7255ba5b307455a91944e5a430759058f3f0e70beb6339063`。
- 当前待验证运行 tree：`a6da14f89646863a22b384128119b07470a12444f94b1ca7a6b426c552754399`。

最后完成的单 D C8/128 输入/8192 输出 N2，ON/OFF 平均 TPOT 劣化为 3.8293%；
两轮分别 1.9493% 和 5.7438%，输出与 MTP 采信率有差异，不能宣称稳定达到 3% 目标。
视图复用与 Q-prepack 不能继承上一版本的模型验证结论。

整组预取已有独立旧版本的两轮模型记录。重新核对其 8-rank、40 个中间 Target：
57 个 follower gather 全部提前到各组 leader attention 结束前，映射/打包前等待
从 0.968620 ms 降到 0.706404 ms。但首个 AllReduce 等待从 0.667246 ms
增加到 1.159374 ms，整轮没有证明净收益；不能将这项增加因果归到预取本身。
本目录保留这个有效的局部优化方向，当前 a6 组合仍需独立验证。

## 后续更新规则

保留失败及回归证据。每项新结果标明实际 tree、shape、拓扑、输出/MTP 可比性和验证范围。
验证有独立收益后更新 PR 结论并迁入正式路径；没有收益或尚未验证的代码继续保持实验状态。
本快照不构成新旧实验混合后的生产可用性承诺。
