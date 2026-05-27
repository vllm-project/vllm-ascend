# 日志分析链路

从代码仓库中提取日志，到生成诊断文档和故障模式库的全流程。

## 包含技能

| 技能 | 说明 | 触发词 |
|------|------|--------|
| log-analysis-search | 从代码仓库中搜索日志语句，提取原始日志及上下文 | 搜索日志、提取日志、整理日志语句 |
| log-analysis-structurize | 将原始日志整理为 9 列标准化表格 | 结构化日志、整理日志表格 |
| log-analysis-correct | 用实际环境日志校正代码提取的表格 | 校正日志、用实际日志校正 |
| log-analysis-document | 生成带 Mermaid 流程图的 Markdown 文档 | 生成Markdown、生成文档 |
| log-analysis-fault-mode | 从日志表格提炼故障模式，生成故障模式库 CSV | 提炼故障模式、故障模式库 |
| log-analysis-dispatcher | 串联以上 6 个原子 skill，完成全流程 | 整理日志全流程、日志全链路分析 |

## 流程图

```
代码仓库
  └─ search        提取原始日志
       └─ structurize  整理为 9 列表格
            └─ correct    用实际日志校正
                 └─ document   生成 Markdown 文档
                      └─ fault-mode  提炼故障模式库
```

## 使用方式

- 单独使用某个原子技能：直接说触发词
- 走完整链路：说 `整理日志全流程` 或 `日志全链路分析`，由 dispatcher 调度
