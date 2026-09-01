---
name: ascendc-docs-search
description: Ascend C 开发资源检索技能。通过本地 API 文档索引、示例代码映射和在线文档兜底搜索定位开发资料，优先查本地、缺失时再查在线。当需要查询 API 用法、示例代码、兼容性信息、官方资料入口或定位文档来源时使用。
---

# Ascend C 开发资源

## 概述

本技能提供"本地优先，在线兜底"的文档搜索能力：
- **本地资源**：1022 个 API 文档、587 个示例代码、实现参考
- **在线搜索**：华为昇腾社区文档搜索（仅在本地资源不足时使用）

## 官方资源路径

| 资源类型 | 路径 | 说明 |
|---------|------|------|
| API 文档 | 通过 `find "$ASC_DEVKIT_DIR/docs/zh/api/" -name "*.md"` 智能搜索 | API 文档（不依赖具体子目录） |
| 高性能模板 | `$ASC_DEVKIT_DIR/examples/00_introduction/01_add/basic_api_memory_allocator_add/` | 双缓冲+流水线标准实现 |
| 各类示例 | `$ASC_DEVKIT_DIR/examples/00_introduction/` | 加法、减法、多输入等 |
| 完整文档 | `$ASC_DEVKIT_DIR/docs/` | 完整开发文档 |
| Tiling 实现 | `$ASC_DEVKIT_DIR/impl/adv_api/tiling/` | Tiling 参数配置参考 |
| 矢量计算 | `$ASC_DEVKIT_DIR/examples/00_introduction/11_vectoradd/` | 矢量 API 使用 |
| 调试示例 | `$ASC_DEVKIT_DIR/examples/01_utilities/00_printf/printf.asc` | printf 调试方法 |
| HCCL 通信 API 头文件 | `$ASC_DEVKIT_DIR/include/adv_api/hccl/` | HCCL 集合通信头文件（hccl.h、hccl_tiling.h 等），API 文档在 `docs/zh/api/SIMD-API/高阶API/HCCL通信类/` |

## 资料查找优先级

```
0. 先查 [api-index.md](references/api-index.md) 定位 API 所属大类和子目录路径
         ↓
1. $ASC_DEVKIT_DIR/docs/zh/api/ 下的所有 .md 文档（按定位的子目录 find 搜索）
         ↓ 找不到
2. $ASC_DEVKIT_DIR/examples/ (示例代码 - 587 个)
         ↓ 找不到
3. $ASC_DEVKIT_DIR/impl/ (实现代码)
         ↓ 找不到
4. 在线搜索（华为昇腾社区）
   使用 scripts/ 中的 Python 脚本
   推荐版本：8.5.0（与当前环境一致）
```

## ⚠️ API 变体搜索指南（重要）

**问题**：Ascend C 存在 **240+ 个带数字后缀的 API 变体**（如 `Add-25.md`），同名 API 的不同变体功能可能完全不同。

**典型案例**：
- `Add.md` - 基础版本
- `Add-25.md` - 变体版本（支持更多功能）

### 强制搜索步骤

1. **列出所有变体**：
   ```bash
   # 搜索某个 API 的所有变体（将 APIName 替换为实际 API 名称）
   find "$ASC_DEVKIT_DIR/docs/zh/api/" -name "${APIName}*.md"

   # 示例：
   # APIName.md        ← 基础版本
   # APIName-25.md     ← 变体版本
   # APIName-91.md     ← 变体版本
   ```

2. **逐一确认功能**：每个变体的函数签名、参数、功能可能完全不同

### 变体命名规律

| 后缀 | 含义 | 示例 |
|------|------|------|
| 无后缀 | 基础版本 | `Add.md` |
| `-数字` | 变体版本（数字无语义，功能可能完全不同） | `Add-25.md` |

### 变体检测命令

```bash
# 查找某个 API 的所有变体（强制，将 APIName 替换为实际名称）
find "$ASC_DEVKIT_DIR/docs/zh/api/" -name "${APIName}*.md"

# 在所有变体中搜索特定关键词
grep -rl "关键词" "$ASC_DEVKIT_DIR/docs/zh/api/" --include="*.md"
```

## 环境兼容性

**当前环境**：A3 服务器，CANN 8.5.0

查阅资料时必须确认 API/方法适用于当前环境。

## Codex Explore 子代理使用

**何时使用**：
1. 查找 API 文档或实现示例
2. 了解某个技术点的实现方式
3. 搜索类似算子的参考实现
4. 遇到错误时查找相关解决方案
5. 方案走通后探索更优实现

**使用方式**：
```
在任务可独立、范围明确且子代理可用时，调用一个只读 Codex 子代理执行 Explore
提供明确的搜索目标和范围，要求只返回证据与路径、不修改文件
```

**示例提示词**：
- "搜索 $ASC_DEVKIT_DIR 中 Exp API 的使用示例"
- "查找双缓冲的实现参考"
- "搜索类似的三角函数算子实现"

## 示例代码索引

| 示例名称 | 路径 | 用途 |
|---------|------|------|
| 高性能模板 | `$ASC_DEVKIT_DIR/examples/00_introduction/01_add/basic_api_memory_allocator_add/` | 双缓冲+流水线 |
| 多输入加法 | `$ASC_DEVKIT_DIR/examples/00_introduction/04_addn/addn.asc` | 多输入处理 |
| 减法算子 | `$ASC_DEVKIT_DIR/examples/00_introduction/07_sub/sub_custom.asc` | 减法实现 |
| 调试打印 | `$ASC_DEVKIT_DIR/examples/01_utilities/00_printf/printf.asc` | printf 调试 |
| 断言使用 | `$ASC_DEVKIT_DIR/examples/01_utilities/01_assert/assert.asc` | 断言示例 |
| 库函数 | `$ASC_DEVKIT_DIR/examples/03_libraries/00_math/addcdiv/addcdiv_custom.asc` | 库函数使用 |
| 矢量计算 | `$ASC_DEVKIT_DIR/examples/00_introduction/11_vectoradd/vector_add_custom.asc` | 矢量 API |

## 在线搜索

**适用情况**：
- 本地 $ASC_DEVKIT_DIR 文档未覆盖相关 API
- 需要更详细的官方说明或最新版本信息
- 本地文档版本过旧或不完整

**快速搜索**：
```bash
# 基础搜索（推荐使用中文关键词）
python .agents/skills/ascendc-docs-search/scripts/ascend_search_client.py "Ascend C 临时内存申请" --max_results 5

# 搜索 API 文档
python .agents/skills/ascendc-docs-search/scripts/ascend_search_client.py "AscendC::Add 接口原型" --max_results 8

# 带版本过滤（推荐使用 8.5.0）
python .agents/skills/ascendc-docs-search/scripts/ascend_search_client.py "Ascend C API" --version "8.5.0"
```

**获取详细内容**：
```bash
python .agents/skills/ascendc-docs-search/scripts/ascend_content_fetcher.py <URL>
```

**参数说明**：
- `keyword`：搜索关键词（必需），建议使用中文
- `--max_results`：返回结果数量（1-10，默认 10）
- `--lang`：语言设置（zh/en，默认 zh）
- `--version`：版本过滤字符串（如 "8.5.0"）
- `--doc_type`：文档类型（DOC/API，默认 DOC）

**依赖安装**：
```bash
pip install -r .agents/skills/ascendc-docs-search/requirements.txt
```

## 参考资料

- [API 文档索引](references/api-index.md)
- [示例代码目录](references/example-catalog.md)
- [环境兼容性表](references/compatibility.md)
