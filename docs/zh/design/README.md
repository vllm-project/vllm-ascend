# vLLM-Ascend 设计文档（中文）

本目录存放尚未并入 Sphinx 英文站的中文设计说明。用户配置仍以
`docs/source/user_guide/configuration/additional_config.md` 为准。

## 活文档（DFX）

| 文档 | 内容 |
|------|------|
| [dfx_design.md](./dfx_design.md) | DFX 总览：Config / InputFilter / Detector / Dump / Report / Processor / I/O Snapshot |
| [dfx_ops.md](./dfx_ops.md) | 运维操作与排障速查 |
| [dumper_design.md](./dumper_design.md) | msprobe dump 生命周期与 PP/TP 齐步 |
| [anomaly_detection_design.md](./anomaly_detection_design.md) | 异常检测：SpecAcceptance / TokenLogprob(ILLDetector) / OutputSubstring / TokenRepeat |
| [async_scheduling_design.md](./async_scheduling_design.md) | 异步调度下的占位符、dump 时序与跨 rank OR（pending-OR 真相见 dumper） |
