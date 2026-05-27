---
name: log-analysis-fault-mode
description: 故障模式提炼器的适配器层。从日志表格中提炼故障模式，生成标准化故障模式库 CSV。触发词：提炼故障模式、生成故障模式库、故障模式库。
version: adaptor
---

# 故障模式提炼器（适配器）

本 skill 是**适配器层**，不存储故障模式库 schema 本身，而是定位外部 schema 文件并支持 custom schema 注入。

---

## 触发词

提炼故障模式、生成故障模式库、故障模式库

---

## 核心设计

```
外部 Schema 文件（Source of Truth）
    ↓
本适配器（加载 schema + 支持 custom override）
    ↓
故障模式生成工具（按 schema 输出 CSV）
```

- **外部版**：默认 schema 在 `external-schema.csv`（30 列），支持用户注入 custom schema
- **内部版**（后续）：公司正式故障模式库文档作为 Source of Truth，按公司字段口径生成 internal profile

---

## Schema 文件路径

| 版本 | 文档位置 |
|------|---------|
| 外部默认 Schema | `~/.hermes/skills/local/log-analysis/log-analysis-fault-mode/external-schema.csv` |

---

## Schema Profile 机制

### 默认行为

直接加载 `external-schema.csv` 作为 30 列故障模式库 schema。

### Custom Schema 注入

用户可提供自定义 schema CSV（格式：`列号,列名,说明,取值/示例`），适配器会：
1. 以 `external-schema.csv` 为基础
2. 合并用户提供的 custom schema
3. 冲突时以 custom schema 为准

调用时指定 `custom_schema_path` 即可启用自定义 schema。

### 内部版（后续）

公司正式故障模式库文档 → internal profile → 按公司字段口径生成

---

## 调用方式

### 查看默认 Schema

读取 `external-schema.csv` 获取默认 30 列定义，包括：
- 列1-8：基本信息（编号、错误码、名称、一~五级分类）
- 列9-17：故障描述（原因、影响、爆炸半径、等级、分类、场景）
- 列18-27：检测与处理（检测机制、处理逻辑、恢复、构造、定位）
- 列28-30：辅助信息（使能条件、可定位性、自动化程度）

### 生成故障模式库

调用本 skill 时，传入：
- 校正后的日志表格（`log-analysis-correct` 输出）
- 输出目录
- 功能名称
- 可选：`custom_schema_path` 指定自定义 schema

### 使用场景

| 场景 | 调用方式 |
|------|---------|
| 生成标准故障模式库 | 加载默认 external-schema.csv |
| 使用自定义 schema | 指定 custom_schema_path 参数 |
| 公司内部使用 | 后续切换到 internal profile |

---

## 适配器职责

1. **定位 schema 文件**：找到对应版本的外部 schema 文件
2. **支持 schema 合并**：加载默认 schema 并合并 custom override
3. **透传调用**：将用户请求路由到故障模式生成能力
4. **版本维护**：当外部 schema 更新时，只需更新 external-schema.csv，本适配器无需改动
