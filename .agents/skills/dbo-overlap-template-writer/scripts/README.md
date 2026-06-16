# DBO Overlap Template Writer - Scripts

This directory contains analysis scripts for vllm-ascend DBO overlap template development.

## Scripts Overview

### `analyze_ascend_profiling.py`

Main analysis script with 5 subcommands:

```bash
# Triage report (recommended default)
python3 analyze_ascend_profiling.py triage \
  --input /path/to/profiling_dir \
  --num-layers 10

# Kernel breakdown
python3 analyze_ascend_profiling.py breakdown \
  --input /path/to/profiling_dir \
  --top-k 50

# Communication analysis
python3 analyze_ascend_profiling.py comm \
  --input /path/to/profiling_dir \
  --num-layers 10

# Communication call stack extraction (OPTIMIZED)
python3 analyze_ascend_profiling.py stack \
  --input /path/to/profiling_dir \
  --top-comm-types 3   # Only extract stacks for top-3 frequent comm types (default)
  --max-stacks 50      # Maximum unique stacks to collect (default: 50)

# Profiling comparison
python3 analyze_ascend_profiling.py compare \
  --input-a /path/to/dir_a \
  --input-b /path/to/dir_b
```

## `stack` 子命令参数详解

`stack` 子命令采用两阶段优化策略，显著减少大文件处理时间和内存占用：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--top-comm-types` | 3 | 只提取频率最高的 K 种通信类型的 stack，0 表示提取所有 |
| `--max-stacks` | 50 | 最多收集多少个唯一 stack，0 表示不限制 |
| `--max-depth` | 20 | 最大调用栈深度 |
| `--full-stack` | False | 显示完整调用栈（含框架内部帧） |
| `--no-dedup` | False | 不去重相同调用栈的通信事件 |

### 工作原理

**阶段一：确定目标**（从 communication.json）
- 加载 communication.json，统计各通信类型频率
- 选出 top-K 高频通信类型作为目标
- 只将这些类型传入阶段二

**阶段二：精准提取**（从 trace_view.json）
- 单遍扫描 events 列表，同时构建 python_function 索引和收集目标 comm 事件
- 处理完后立即释放 events 列表引用（`del events`）
- 找到足够数量的唯一 stack 后提前退出（early termination）

## 性能对比

对于典型的 1GB `trace_view.json` 文件：

| 配置 | 时间 | 内存峰值 |
|------|------|----------|
| `--top-comm-types 0 --max-stacks 0`（提取所有） | ~60s | ~2GB |
| `--top-comm-types 3 --max-stacks 50`（默认） | ~15s | ~500MB |

## 使用场景

1. **General profiling analysis**: Use `analyze_ascend_profiling.py triage`
2. **Detecting missing DBO hooks**: Use `analyze_ascend_profiling.py stack`（默认参数即可）
3. **Large profiling files (>5GB)**: 减小 `--max-stacks` 或增大 `--top-comm-types` 以加速
4. **Extract all stacks**: Use `--top-comm-types 0 --max-stacks 0`

## Installation

```bash
# Required
pip install pandas

# Optional (for advanced JSON processing)
pip install ijson
```

## Related Documentation

- `references/profiling-guide.md`: Detailed profiling analysis guide
- `SKILL.md`: Complete skill workflow documentation