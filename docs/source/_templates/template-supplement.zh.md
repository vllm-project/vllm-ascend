# 模板补充

<p align="center">
  <a href="template-supplement.md"><b>English</b></a> | <a href="template-supplement.zh.md"><b>中文</b></a>
</p>

> **说明**：本文档为《模型部署技术文档模板》的补充参考手册，旨在帮助文档编写者理解 **Sphinx + MyST-Parser**和 **MkDocs + Material** 两种框架之间的语法差异,避免实际写作中因框架差异出现渲染异常等问题。

## 1 框架概述

### 1.1 版本对应关系

| 框架 | vLLM-Ascend 版本 | 配置文件 | 状态 |
|------|------------------|----------|------|
| Sphinx + MyST-Parser | v0.23.0 及更早 | `docs/source/conf.py` | 旧版 |
| MkDocs + Material | v0.24.0 及之后 | `mkdocs.yml` | 新版（当前使用） |

### 1.2 为什么需要这份参考手册

vLLM-Ascend 在 v0.24.0 版本切换到 MkDocs 框架后，两种框架在 Markdown 解析、锚点生成、扩展语法等方面存在显著差异，本文档整理了这些差异示例以供参考。这些差异会导致：

- 中文文档锚点异常（如 `#5-在线服务部署` 变成 `#5`）
- Tab 切换、提示框等扩展语法需要重写
- 自定义锚点语法不兼容
- 版本占位符等渲染不成功

## 2 语法差异对照表

|       功能         |         MkDocs + Material            |              Sphinx + MyST-Parser       |
| ------------------ | ------------------------------------ | --------------------------------------- |
| **标签页**         | `=== "标签名"`                        | `::::{tab-item} 标签名` + `:::` 闭合    |
| **标签页组同步**   | 默认同步（同名标签自动联动）            | `:sync-group: 组名` + `:sync: 键名`     |
| **版本占位符**     | `{{ vllm_ascend_version }}`           | `\|vllm_ascend_version\|`              |
| **提示框（Note）** | `!!! note "标题"`                     | `:::{note}` ... `:::`                  |
| **警告框**         | `!!! warning "标题"`                  | `:::{warning}` ... `:::` / `{caution}` |
| **Jinja 模板转义** | `{% raw %}` ... `{% endraw %}`        | 无需转义                               |

## 3 标签页

### 3.1 MkDocs + Material

使用 `=== "标签名"` 语法，同级标签页**自动共享同步组**（即同名标签在不同 Tab 组之间会自动联动）。

**示例：**

```markdown
    === "A3 series"

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
    docker run ...
    ```

    === "A2 series"

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
    docker run ...
    ```
```

**渲染效果**：两个并列标签页，点击切换时，页面中所有名为 "A3 series" 或 "A2 series" 的标签页会同步切换。

### 3.2 Sphinx + MyST-Parser

使用 `{tab-set}` 和 `{tab-item}` 指令，需要显式声明同步组（`:sync-group:`）和同步键（`:sync:`）。

**示例：**

```markdown
    :::::{tab-set}
    :sync-group: install

    ::::{tab-item} A3 series
    :sync: A3

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
    docker run ...
    ```

    ::::

    ::::{tab-item} A2 series
    :sync: A2

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
    docker run ...
    ```

    ::::
    :::::
```

**语法说明：**

| 元素 | 含义 |
|------|------|
| `:::::{tab-set}` | 声明一个 Tab 组（5 个冒号表示组容器） |
| `:sync-group: install` | 同步组名称，同名的 `tab-set` 会联动 |
| `::::{tab-item}` | 声明一个标签页（4 个冒号表示子项） |
| `:sync: A3` | 同步键，用于跨组匹配相同内容的标签页 |
| `::::` / `:::::` | 闭合标签（冒号数量需与开启时匹配） |

### 3.3 语法差异

| 旧版语法 | 新版语法 | 注意事项 |
|----------|----------|----------|
| `:::::{tab-set}` + `:sync-group:` | 不需要声明，同级 `===` 自动同步 | 新版同级标签页默认同步 |
| `::::{tab-item} 名称` | `=== "名称"` | 名称需完全一致才能跨组联动 |
| `:sync: 键名` | 不需要 | 新版通过标签名称文本匹配 |
| `::::` / `:::::` 闭合 | 无闭合标签，靠缩进控制 | 新版需确保正确缩进 |

> ⚠️ *常见错误**：
> - 缩进不一致会导致 MkDocs 无法识别标签页内容
> - 标签名称中的空格、大小写需完全一致，否则跨组同步失效

## 4 占位符

### 4.1 MkDocs + Material

```markdown
export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
```

### 4.2 Sphinx + MyST-Parser

```markdown
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
```

## 5 Jinja 语法转义

> **适用场景**：文档中需要展示包含 `{{ }}` 双花括号的 Jinja 模板示例代码时，两种框架的处理方式不同。

### 5.1 问题背景

在 vLLM-Ascend 的文档中，有时需要展示包含 Jinja 模板语法的代码示例（如 prompt 模板、RAG 评测脚本等）。这些示例中本身包含 `{{ variable }}` 语法。
MkDocs 使用 Jinja2 作为模板引擎（通过 `mkdocs-material` 主题），文档正文中的 `{{ }}` 会被视为模板变量进行渲染，导致原本希望展示的 Jinja 示例代码出现异常。

| 框架 | 对 `{{ }}` 的处理 | 风险 |
|------|-------------------|------|
| Sphinx + MyST-Parser | 不解析，原样保留 | 无 |
| MkDocs + Material |解析为模板变量，尝试替换 | 渲染异常，模板代码被破坏 |

### 5.2 解决方案：使用 `{% raw %}` 块

> **适用分支**：`main`（latest 版本）及 v0.24.0 之后的所有版本

使用 `{% raw %}` ... `{% endraw %}` 块包裹 Jinja 模板代码，告诉 MkDocs 的模板引擎**不对块内内容进行任何解析**，原样输出。

**示例：**

```markdown
    ```jinja
      {% raw %}
      <|im_start|>system
      Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
      <|im_start|>user
      <Instruct>: {{
          messages
          | selectattr("role", "eq", "system")
          | map(attribute="content")
          | first
          | default("Given a search query, retrieve relevant candidates that answer the query.")
      }}<Query>:{{
          messages
          | selectattr("role", "eq", "query")
          | map(attribute="content")
          | first
      }}
      <Document>:{{
          messages
          | selectattr("role", "eq", "document")
          | map(attribute="content")
          | first
      }}<|im_end|>
      <|im_start|>assistant

    {% endraw %}
    ```
```

### 5.3 Sphinx（v0.23.0 及更早）：无需转义

> **适用分支**：`v0.23.0`、`v0.18.0` 等历史版本分支

Sphinx 使用 Docutils 解析 Markdown，**不包含 Jinja2 模板引擎**，因此文档正文中的 `{{ }}` 不会被特殊处理，直接原样渲染。

## 5 Note / 提示框（Admonition）

> **适用场景**：需要突出显示重要说明、警告、注意事项等信息。

### 5.1 MkDocs + Material

使用 `!!! note "标题"` 语法，内容通过**缩进**控制范围，支持多种类型标识符。

**语法：**

```markdown
!!! note "自定义标题（可选）"
    提示内容行1
    提示内容行2
    提示内容行3
```
**示例：**

```markdown
!!! note "Atlas 300I DUO"
    Atlas 300I DUO uses its platform-specific CANN 9.1.0 package; refer to the 310P table below for its requirements.
```

关键规则：内容必须与 !!! 声明行保持 4 个空格 缩进，且内容之间用空行分隔即可（无需额外缩进）。

### 5.2 Sphinx + MyST-Parser

使用 `::{note}` 指令语法，内容通过 **冒号层级 + 缩进** 控制范围，需要显式闭合。

**基本语法：**

```markdown
:::{note}
内容行1
内容行2
:::

**示例：**
```markdown
:::{note} Atlas 300I DUO
Atlas 300I DUO uses its platform-specific CANN 9.1.0 package; refer to the 310P table below for its requirements.
:::
```
