# 模板补充

<p align="center">
  <a href="template-supplement.md"><b>English</b></a> | <a href="template-supplement.zh.md"><b>中文</b></a>
</p>

> **说明**：本文档为《模型部署技术文档模板》的补充参考手册，旨在帮助文档编写者理解 `Sphinx + MyST-Parser` 和 `MkDocs + Material` 两种框架之间的语法差异,避免实际写作中因框架差异出现渲染异常等问题。

## 1 框架概述

### 1.1 版本对应关系

| 框架 | 适用分支 | 配置文件 |
|------|----------|----------|
| Sphinx + MyST-Parser | `v0.23.0` 及更早版本分支 | `docs/source/conf.py` |
| MkDocs + Material | `main` 及 `v0.23.0` 之后版本分支 | `mkdocs.yml` |

## 2 语法差异对照表

| 功能 | MkDocs + Material | Sphinx + MyST-Parser |
|------|-------------------|----------------------|
| **标签页** | `=== "标签名"` | `::::{tab-item} 标签名` + `:::` 闭合 |
| **标签页组同步** | 已启用 `content.tabs.link`，同名标签自动联动 | `:sync-group: 组名` + `:sync: 键名` |
| **版本占位符** | `{{ vllm_ascend_version }}` | `\|vllm_ascend_version\|` |
| **提示框（Note）** | `!!! note "标题"` | `:::{note}` ... `:::` |
| **警告框** | `!!! warning "标题"` | `:::{warning}` ... `:::` / `{caution}` |
| **Jinja 模板转义** | `{% raw %}` ... `{% endraw %}` | 无需转义 |
| **中文锚点生成** | 过滤中文字符<br>`## 5. 在线部署` → `#5` | 保留中文字符<br>`## 5. 在线部署` → `#5.-在线部署` |

## 3 文档导航配置

### 3.1 框架差异

| / | Sphinx + MyST-Parser（v0.23.0 及更早） | MkDocs + Material（main 分支） |
|------|----------------------------------------|-------------------------------|
| **导航定义** | 各源文件中的 `toctree` 指令 | `mkdocs.yml` 中的 `nav` 字段 |
| **未纳入导航** | 构建警告，无法通过导航访问 | 构建警告，仍可独立访问，但导航中不显示 |
| **导航标题来源** | `toctree` 中的 `:caption:` 或页面 H1 | `nav_titles.py` Hook 动态映射 |

### 3.2 mkdocs.yml 中的 nav 配置

在 `mkdocs.yml` 的 `nav` 字段中定义导航结构：

```yaml
nav:
  - index.md
  - installation.md
  - tutorials/models/DeepSeek-V3.2.md
```

**注意**：nav 配置主要影响文档在导航栏中的显示

### 3.3 nav_titles.py：导航标题映射

（`docs/hooks/nav_titles.py`）是一个 MkDocs Hook，根据 `DOCS_LANG` 环境变量动态映射中英文标题：

```python
TITLES = {
    "index.md": {"en": "Home", "zh": "首页"},
    "installation.md": {"en": "Installation", "zh": "安装"},
}
```

通过 mkdocs.yml 的 hooks 字段注册：

```yaml
hooks:
  - docs/hooks/nav_titles.py
```

### 3.4 新文档添加检查清单

- 将文档文件放入相应目录
- 在 mkdocs.yml 的 nav 中添加文件路径（否则文档不会出现在导航栏中）
- 在 nav_titles.py 的 TITLES 中添加标题映射（否则中文文档可能无法正常呈现）

## 4 标签页

### 4.1 MkDocs + Material

使用 `=== "标签名"` 语法。当前 vLLM-Ascend 文档已启用 `content.tabs.link`，同名标签在不同 Tab 组之间会自动联动。

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

### 4.2 Sphinx + MyST-Parser

使用 `{tab-set}` 和 `{tab-item}` 指令，声明同步组（`:sync-group:`）和同步键（`:sync:`）。

**示例：**

```markdown
    :::::{tab-set}
    :sync-group: install

    ::::{tab-item} A3 series
    :sync: A3

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
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

### 4.3 语法差异

| Sphinx语法 | MkDocs语法 | MkDocs注意事项 |
|----------|----------|----------|
| `:::::{tab-set}` + `:sync-group:` | 不需要声明，同级 `===` 自动同步 | 需在 mkdocs.yml 中启用 `content.tabs.link` 才可实现跨组联动 |
| `::::{tab-item} 名称` | `=== "名称"` | 名称需完全一致才能跨组联动 |
| `:sync: 键名` | 不需要 | 通过标签名称文本匹配 |
| `::::` / `:::::` 闭合 | 无闭合标签，靠缩进控制 | 需确保正确缩进 |

⚠️**提示**：

> - 缩进不一致会导致 MkDocs 无法识别标签页内容
> - 标签名称中的空格、大小写需完全一致，否则跨组同步失效

## 5 占位符

### 5.1 MkDocs + Material

**示例：**

```markdown
export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
```

### 5.2 Sphinx + MyST-Parser

**示例：**

```markdown
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
```

## 6 Note / 提示框

> **适用场景**：需要突出显示重要说明、警告、注意事项等信息。

### 6.1 MkDocs + Material

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

**关键规则**：内容必须与 `!!!` 声明行保持 **4 个空格** 缩进，且内容之间用空行分隔即可（无需额外缩进）。

### 6.2 Sphinx + MyST-Parser

使用 `::{note}` 指令语法，内容通过 **冒号层级 + 缩进** 控制范围，需要显式闭合。

**基本语法：**

```markdown
:::{note}
内容行1
内容行2
:::
```

**示例：**

```markdown
:::{note} Atlas 300I DUO
Atlas 300I DUO uses its platform-specific CANN 9.1.0 package; refer to the 310P table below for its requirements.
:::
```

### 6.3 提示

> - 缩进不足（少于 4 空格）会使内容跳出提示框
> - 若提示框内包含代码块，代码块需额外缩进 4 空格（共 8 空格）

## 7 Jinja 模板转义

### 7.1 问题背景

在 vLLM-Ascend 的文档中，有时需要展示包含 Jinja 模板语法的代码示例（如 prompt 模板、RAG 评测脚本等），两种框架的处理方式不同。这些示例中本身包含 `{{ variable }}` 语法。
MkDocs 使用 Jinja2 作为模板引擎，文档正文中的 `{{ }}` 会被视为模板变量进行渲染，导致原本希望展示的 Jinja 示例代码出现异常。

| 框架 | 对 `{{ }}` 的处理 | 风险 |
|------|-------------------|------|
| Sphinx + MyST-Parser | 不解析，原样保留 | 无 |
| MkDocs + Material | 解析为模板变量，尝试替换 | 渲染异常，模板代码被破坏 |

### 7.2 解决方案

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

### 7.3 Sphinx（v0.23.0 及更早）：无需转义

> **适用分支**：`v0.23.0`、`v0.18.0` 等历史版本分支

Sphinx 使用 Docutils 解析 Markdown，**不包含 Jinja2 模板引擎**，因此文档正文中的 `{{ }}` 不会被特殊处理，直接原样渲染。

## 8 锚点异常

### 8.1 问题背景

该问题**仅出现在中文文档中**。
vLLM Ascend 社区的中文文档通过 PO 文件 + gettext 工具链从英文源文件翻译生成。中文版本会继承英文源文件的锚点 ID 结构。
在 Sphinx 框架下，英文源文件中若未显式指定锚点，中文翻译会保留中文字符生成锚点（如 `#5-在线服务部署`）。在 MkDocs 框架中，MkDocs 会自动过滤锚点中的非 ASCII 字符，导致同一中文标题生成的锚点 ID 与 Sphinx 不一致（如 `#5-在线服务部署` → `#5`）。

### 8.2 锚点生成规则对比

| 框架 | 锚点生成规则 | 示例标题 | 生成的锚点 |
|------|-------------|----------|-----------|
| Sphinx + MyST-Parser | 保留中文字符 | `## 5. 在线服务部署` | `#5-在线服务部署` |
| MkDocs + Material | 过滤中文字符，仅保留数字、字母和连字符 | `## 5. 在线服务部署` | `#5` |

### 8.3 解决方案

**方案一：手动指定锚点（推荐）**

**示例：**

在英文 Markdown 源文件中为标题指定锚点 ID：

```text
## 5. Online Serving {: #5-online-serving }
```

中文翻译文件（通过 PO/gettext 生成）会自动继承该锚点 ID，生成对应的中文标题锚点：

```text
## 5. 在线服务部署 {: #5-online-serving }
```

引用时使用自定义锚点：

```text
请参见[在线服务部署](#5-online-serving)
```

**方案二：使用 HTML 锚点标签**

**示例：**
略

### 8.4 提示

- 锚点 ID 建议仅使用 ASCII 字符（字母、数字、连字符），确保在两种框架下均能正确解析
- 锚点 ID 在英文源文件中指定后，无需在中文文件中重复指定
- 跨文件引用时，统一使用 ASCII 锚点 ID：
