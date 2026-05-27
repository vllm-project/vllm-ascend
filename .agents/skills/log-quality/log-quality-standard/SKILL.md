---
name: log-quality-standard
description: 日志标准规范的适配器层。根据外部标准文档执行日志质量检查，调用 scan / rewrite / write 能力。触发词：日志标准、日志规范、日志质量标准、log standard、log quality、制定日志规范。
version: adaptor
---

# 日志标准规范（适配器）

本 skill 是**适配器层**，不存储日志标准内容本身，而是定位外部标准文档并将其应用于日志质量检查、整改和新写场景。

---

## 触发词

日志标准、日志规范、日志质量标准、log standard、log quality、制定日志规范

---

## 核心设计

```
外部标准文档（Source of Truth）
    ↓
本适配器（加载标准 + 转换口径）
    ↓
scan / rewrite / write 工具（执行检查和整改）
```

- **外部版**：标准内容在 `external-standard.md`，本 skill 是纯 adaptor
- **内部版**（后续）：公司正式日志规范文档作为 Source of Truth，本 skill 适配公司规范转成可执行口径

---

## 标准文档路径

| 版本 | 文档位置 |
|------|---------|
| 外部版 | `~/.hermes/skills/local/log-quality/log-quality-standard/external-standard.md` |

---

## 调用方式

### 查看标准内容

读取 `external-standard.md` 获取完整标准定义，包括：
- 标准 1：分级清晰
- 标准 2：描述充分
- 标准 3：组件归属明确
- 标准 4：防刷屏
- 标准 5：链路追踪
- 标准 6：隐私保护

以及附录中的仓库/组件速查表、ERROR/WARNING 边界判断、好日志自检清单。

### 使用 scan 能力检查存量日志

调用 `log-quality-scan` 或 `log-quality-scan-code` 时，自动加载本标准的检查口径。

### 使用 rewrite 能力整改不合规日志

调用 `log-quality-rewrite` 时，自动加载本标准的整改规则。

### 使用 write 能力为新功能设计日志

调用 `log-quality-write` 时，自动加载本标准的编写规范。

---

## 适配器职责

1. **定位标准文档**：找到对应版本的外部标准文件
2. **转换口径**：将标准文档内容转换为工具可执行的检查规则
3. **透传调用**：将用户请求路由到 scan / rewrite / write 能力
4. **版本维护**：当外部标准更新时，只需更新 external-standard.md，本适配器无需改动
