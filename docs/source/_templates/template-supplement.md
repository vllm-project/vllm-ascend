# Template Supplement

<p align="center">
  <a href="template-supplement.md"><b>English</b></a> | <a href="template-supplement.zh.md"><b>中文</b></a>
</p>

> **Description**: This document serves as a supplementary reference manual for the *Model Deployment Technical Documentation Template*. It is designed to help documentation writers understand the syntax differences between the `Sphinx + MyST-Parser` and `MkDocs + Material` frameworks, avoiding rendering anomalies and other issues caused by framework discrepancies during actual writing.

## 1 Framework Overview

### 1.1 Version Mapping

| Framework | Applicable Branches | Configuration File |
|-----------|---------------------|-------------------|
| Sphinx + MyST-Parser | `v0.23.0` and earlier branches | `docs/source/conf.py` |
| MkDocs + Material | `main` and `v0.23.0` and later branches | `mkdocs.yml` |

## 2 Syntax Comparison Table

| Feature | MkDocs + Material | Sphinx + MyST-Parser |
|---------|-------------------|----------------------|
| **Tabs** | `=== "Tab Label"` | `::::{tab-item} Tab Label` + `:::` closing |
| **Tab Group Synchronization** | `content.tabs.link` enabled; same-name tabs automatically synchronized | `:sync-group: group_name` + `:sync: key_name` |
| **Version Placeholder** | `{{ vllm_ascend_version }}` | `\|vllm_ascend_version\|` |
| **Note / Admonition** | `!!! note "Title"` | `:::{note}` ... `:::` |
| **Warning Box** | `!!! warning "Title"` | `:::{warning}` ... `:::` / `{caution}` |
| **Jinja Template Escaping** | `{% raw %}` ... `{% endraw %}` | No escaping required |
| **Chinese Anchor Generation** | Filters Chinese characters<br>`## 5. 在线部署` → `#5` | Preserves Chinese characters<br>`## 5. 在线部署` → `#5.-在线部署` |

## 3 Document Navigation Configuration

### 3.1 Framework Differences

| / | Sphinx + MyST-Parser (v0.23.0 and earlier) | MkDocs + Material (main branch) |
|------|----------------------------------------|-------------------------------|
| **Navigation Definition** | `toctree` directive in source files | `nav` field in `mkdocs.yml` |
| **Not Included in Navigation** | Build warning, inaccessible via navigation | Build warning, still independently accessible but not displayed in navigation |
| **Navigation Title Source** | `:caption:` in `toctree` or page H1 | Dynamically mapped by `nav_titles.py` Hook |

### 3.2 nav Configuration in mkdocs.yml

Define the navigation structure in the `nav` field of `mkdocs.yml`:

```yaml
nav:
  - index.md
  - installation.md
  - tutorials/models/DeepSeek-V3.2.md
```

**Note**: The `nav` configuration primarily affects whether documents appear in the navigation bar.

### 3.3 nav_titles.py: Navigation Title Mapping

（`docs/hooks/nav_titles.py`）is a MkDocs Hook that dynamically maps bilingual titles based on the `DOCS_LANG` environment variable:

```python
TITLES = {
    "index.md": {"en": "Home", "zh": "首页"},
    "installation.md": {"en": "Installation", "zh": "安装"},
}
```

Registered via the hooks field in mkdocs.yml:

```yaml
hooks:
  - docs/hooks/nav_titles.py
```

### 3.4 New Document Addition Checklist

- Place the document file in the appropriate directory
- Add the file path to nav in mkdocs.yml (otherwise the document will not appear in the navigation bar)
- Add the title mapping to TITLES in nav_titles.py (otherwise the Chinese document may not render correctly)

## 4 Tabs

### 4.1 MkDocs + Material

Uses the `=== "Tab Label"` syntax. The vLLM-Ascend documentation currently has `content.tabs.link` enabled, allowing same-name tabs to automatically synchronize across different tab groups.

**Example:**

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

**Rendering Effect**: For two side-by-side tab groups, when a user clicks on either "A3 series" or "A2 series," all tab groups on the page with identically named tabs will synchronize to the same selection.

### 4.2 Sphinx + MyST-Parser

Uses the `{tab-set}` and `{tab-item}` directives, requiring explicit declaration of synchronization groups (`:sync-group:`) and synchronization keys (`:sync:`).

**Example:**

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

**Syntax Description:**

| Element | Meaning |
|---------|---------|
| `:::::{tab-set}` | Declares a tab group (5 colons denote the group container) |
| `:sync-group: install` | Synchronization group name; `tab-set` instances with the same name will synchronize |
| `::::{tab-item}` | Declares a tab item (4 colons denote a child item) |
| `:sync: A3` | Synchronization key used to match tabs with identical content across groups |
| `::::` / `:::::` | Closing tags (colon count must match the opening tags) |

### 4.3 Syntax Differences

| Sphinx Syntax | MkDocs Syntax | MkDocs Notes |
|---------------|---------------|--------------|
| `:::::{tab-set}` + `:sync-group:` | Declaration not required; sibling `===` automatically synchronize | `content.tabs.link` must be enabled in `mkdocs.yml` for cross-group synchronization |
| `::::{tab-item} Name` | `=== "Name"` | Names must be exactly identical for cross-group synchronization |
| `:sync: key_name` | Not required | Uses tab label text matching |
| `::::` / `:::::` closing | No closing tags; uses indentation control | Proper indentation must be ensured |

⚠️**Notes**:

> - Inconsistent indentation may cause MkDocs to fail to recognize tab content
> - Spaces and capitalization in tab labels must be exactly identical, otherwise cross-group synchronization will fail

## 5 Placeholders

### 5.1 MkDocs + Material

**Example:**

```markdown
export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
```

### 5.2 Sphinx + MyST-Parser

**Example:**

```markdown
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
```

## 6 Note / Admonition

> **Use Case**: Used to highlight important notes, warnings, precautions, and other information.

### 6.1 MkDocs + Material

Uses the `!!! note "Title"` syntax, with content scope controlled by **indentation**, supporting multiple admonition types.

**Syntax:**

```markdown
!!! note "Custom Title (Optional)"
    Content line 1
    Content line 2
    Content line 3
```

**Example:**

```markdown
!!! note "Atlas 300I DUO"
    Atlas 300I DUO uses its platform-specific CANN 9.1.0 package; refer to the 310P table below for its requirements.
```

**Key Rule**: Content must be indented `4 spaces` from the !!! declaration line, and content sections can be separated by blank lines (no additional indentation required).

### 6.2 Sphinx + MyST-Parser

Uses the `::{note}` directive syntax, with content scope controlled by colon hierarchy + indentation, requiring explicit closing.

**Basic Syntax:**

```markdown
:::{note}
Content line 1
Content line 2
:::
```

**Example:**

```markdown
:::{note} Atlas 300I DUO
Atlas 300I DUO uses its platform-specific CANN 9.1.0 package; refer to the 310P table below for its requirements.
:::
```

### 6.3 Notes

> - Insufficient indentation (fewer than 4 spaces) will cause content to break out of the admonition box.
> - If the admonition contains a code block, the code block requires an additional 4 spaces of indentation (8 spaces total).

## 7 Jinja Template Escaping

### 7.1 Background

In vLLM-Ascend documentation, it is sometimes necessary to display code examples containing Jinja template syntax (such as prompt templates, RAG evaluation scripts, etc.). The two frameworks handle this differently, as these examples inherently contain `{{ variable }}`yntax.

MkDocs uses Jinja2 as its template engine. The `{{ }}` in the document body will be interpreted as template variables and rendered, causing the intended Jinja example code to display incorrectly.

| Framework | Handling of `{{ }}` | Risk |
|------|-------------------|------|
| Sphinx + MyST-Parser | Not parsed; preserved as-is | None |
| MkDocs + Material | Parsed as template variables; attempted substitution | Rendering anomalies; template code corrupted |

### 7.2 Solution

> **Applicable Branches**：`main` (latest version) and `v0.24.0` and later versions

Wrap Jinja template code with `{% raw %} ... {% endraw %}` blocks to instruct the MkDocs template engine **not to parse any content within the block**, outputting it as-is.

**Example:**

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

### 7.3 Sphinx (v0.23.0 and earlier): No Escaping Required

> **Applicable Branches**: `v0.23.0`, `v0.18.0` and other historical branches

Sphinx uses Docutils to parse Markdown and **does not include the Jinja2 template engine**, so `{{ }}` in the document body is not processed and is rendered as-is.

### 8.1 Background

This issue **only occurs in Chinese documentation**.

The vLLM Ascend community's Chinese documentation is translated from English source files via PO files + gettext toolchain. The Chinese version inherits the anchor ID structure from the English source files.

In the Sphinx framework, if no anchor is explicitly specified in the English source file, the Chinese translation preserves Chinese characters when generating anchors (e.g., `#5-在线服务部署`). In the MkDocs framework, MkDocs automatically filters out non-ASCII characters from anchors, causing Chinese titles to generate anchor IDs inconsistent with Sphinx (e.g., `#5-在线服务部署` → `#5`).

### 8.2 Anchor Generation Rules Comparison

| Framework | Anchor Generation Rule | Example Heading | Generated Anchor |
|-----------|------------------------|-----------------|------------------|
| Sphinx + MyST-Parser | Preserves Chinese characters | `## 5. 在线服务部署` | `#5-在线服务部署` |
| MkDocs + Material | Filters Chinese characters; only digits, letters, and hyphens are preserved | `## 5. 在线服务部署` | `#5` |

### 8.3 Solution

**Option 1: Manually Specify Anchor IDs (Recommended)**

**Example:**

Specify an anchor ID for the heading in the English Markdown source file:

```text
## 5. Online Serving {: #5-online-serving }
```

The Chinese translation file (generated via PO/gettext) will automatically inherit this anchor ID, generating the corresponding Chinese heading anchor:

```text
## 5. 在线服务部署 {: #5-online-serving }
```

When referencing, use the custom anchor:

```text
Please refer to [Online Serving](#5-online-serving)
请参见[在线服务部署](#5-online-serving)
```

**Option 2: Use HTML Anchor Tags**

**Example:**
Omitted

### 8.4 Notes

- Anchor IDs should use only ASCII characters (letters, digits, hyphens) to ensure correct parsing under both frameworks.
- Once an anchor ID is specified in the English source file, it does not need to be re-specified in the Chinese file.
- When referencing across files, consistently use ASCII anchor IDs.
