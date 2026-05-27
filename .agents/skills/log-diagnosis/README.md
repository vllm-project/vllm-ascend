# 日志诊断链路

基于日志进行问题定位的技能组，按仓库分调度器，每个具体问题有独立诊断 skill。

## 包含技能

| 技能 | 说明 | 触发词 |
|------|------|--------|
| log-diagnosis-framework | 框架说明，定义组织方式和通用结构 | 日志定位、日志诊断、故障定位 |
| log-diagnosis-vllm | vLLM 日志诊断调度器，分发给子 skill | vllm日志、vllm诊断、vllm问题定位 |
| log-diagnosis-mindie | MindIE-PyMotor 日志诊断调度器 | MindIE日志、PyMotor日志 |
| log-diagnosis-skill-builder | 给定故障场景，自动生成诊断 skill 骨架 | 编写诊断skill、帮我写skill |
| log-diagnosis-vllm-inference-timeout | vLLM 推理超时诊断（子 skill） | vllm推理超时 |
| log-diagnosis-pd-link-establishment | PD 建链失败诊断（子 skill） | PD建链、建链失败 |
| log-diagnosis-shrink-p-reserve-d | 缩 P 保 D 异常诊断（子 skill） | 缩P保D |

## 架构图

```
log-diagnosis-framework（框架规范）
  │
  ├── log-diagnosis-vllm（调度器）
  │       └── log-diagnosis-vllm-inference-timeout
  │
  ├── log-diagnosis-mindie（调度器）
  │       ├── log-diagnosis-pd-link-establishment
  │       └── log-diagnosis-shrink-p-reserve-d
  │
  └── log-diagnosis-skill-builder（快速新建诊断方向）
```

## 使用方式

- 说 `vllm日志` / `MindIE日志` → 自动路由到对应调度器
- 说具体问题（如 `vllm推理超时`）→ 直接触发对应子 skill
- 说 `帮我写一个 xxx 的诊断 skill` → 调用 skill-builder 生成骨架
