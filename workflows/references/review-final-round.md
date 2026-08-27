# 最终轮审查附加检查

> 本文件由 Reviewer 在最终轮审查（总分 >= 70 且无必须修复项）时读取执行。

---

## 交付件检查清单

**适用时机**：当审查预计通过（PASS / PASS WITH NOTES）时，在最终轮审查中执行。所有必选项必须满足才能判定 PASS。

| # | 交付件 | 路径 | 检查标准 |
|---|--------|------|----------|
| D1 | 算子源码 | `operators/{operator_name}/{operator_name}.asc` | 独立编译通过，无警告 |
| D2 | 构建文件 | `operators/{operator_name}/CMakeLists.txt` | 依赖项完整 |
| D3 | Golden 数据生成 | `operators/{operator_name}/gen_data.py` | 支持所有要求的 dtype |
| D4 | 运行脚本 | `operators/{operator_name}/run.sh` | 可正常执行 |
| D5 | 算子文档 | `operators/{operator_name}/README.md` | 包含：算子概述、数学公式、API 映射、编译运行指南、测试结果、已知限制 |
| D6 | 设计文档 | `operators/{operator_name}/docs/DESIGN.md` | 包含：需求分析、3D 抽象、API 映射、UB 规划、精度策略 |
| D7 | 开发计划 | `operators/{operator_name}/docs/PLAN.md` | 6 阶段全部标记完成，测试结果已记录 |
| D8 | 审查报告 | `operators/{operator_name}/docs/REVIEW.md` | 当前轮次审查报告已写入 |

## 代码清洁检查（最终轮专用）

| # | 检查项 | Grep 命令 | 要求 |
|---|--------|----------|------|
| C1 | printf/cout 残留 | `grep -n "printf\|cout" operators/{operator_name}/*.asc` | 无残留（或仅保留必要的错误提示） |
| C2 | TODO/FIXME 残留 | `grep -n "TODO\|FIXME\|HACK\|XXX" operators/{operator_name}/*.asc` | 无残留 |
| C3 | 注释掉的代码块 | 目视检查 | 无大段注释代码（允许少量说明性注释） |
| C4 | 调试用硬编码 | `grep -n "= 1;\|= 2;\|= 8;" operators/{operator_name}/*.asc` | 无调试用写死值 |

## 精度全覆盖验证（最终轮专用）

**独立运行所有 dtype x 所有 test case 组合**，记录完整结果：

```
对每个 dtype in {float32, float16, bfloat16}:
  对每个 test case in 要求的用例列表:
    1. python3 gen_data.py[shape] [dim] [dtype]
    2. ./build/{operator_name}_custom [shape] [dim] [dtype]
    3. python3 gen_data.py --verify [shape] [dim] [dtype]
    -> 记录: max_abs_err, max_rel_err, mismatch_count, PASS/FAIL
```

**结果汇总表**（必须在 REVIEW.md 中呈现）：

| dtype | Case | Shape | dim | Max Abs Err | Max Rel Err | Mismatch | 状态 |
|-------|------|-------|-----|-------------|-------------|----------|------|
| fp32  | 1    | ...   | ... | ...         | ...         | ...      | PASS/FAIL |

## 最终轮审查流程

```
1. 执行评分检查表（维度 1-7），计算总分
   │
2. 判断是否进入最终轮附加检查：
   ├── 总分 < 70 或存在必须修复项 → 跳到步骤 6（直接 FAIL，跳过附加检查）
   └── 总分 >= 70 且无必须修复项 → 继续步骤 3（执行最终轮附加检查）
   │
3. 执行交付件检查清单（D1-D8），确认全部存在且合格
4. 执行代码清洁检查（C1-C4），确认无调试残留
5. 执行精度全覆盖验证，记录完整结果表
   │
6. 汇总判定：
   ├── 总分 >= 80 + 无必须修复 + 交付件齐全 + 代码清洁 → PASS
   ├── 总分 70-79 + 无必须修复 + 交付件齐全 → PASS WITH NOTES（附建议）
   └── 其他 → FAIL（列出未满足项）
   │
7. 写入 REVIEW.md（首轮创建；复审轮次在文件末尾追加，禁止覆盖历史报告）
```
