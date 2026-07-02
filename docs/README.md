# vLLM Ascend Plugin documents

Live doc: <https://docs.vllm.ai/projects/ascend>

## Build the docs

The documentation uses [MkDocs](https://www.mkdocs.org/) with the
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

### Prerequisites

```bash
# Install documentation dependencies.
pip install -r requirements-docs.txt
```

### Build and serve (English)

```bash
# Serve docs locally with live reload.
make serve

# Or build to site/.
make build

# Open the docs with your browser
python -m http.server -d site/
```

### Build and serve (Chinese)

Chinese docs are generated from `.po` translation files in
`docs/source/locale/zh_CN/LC_MESSAGES/`.

```bash
# Generate Chinese markdown files from .po files.
make gen-zh

# Serve Chinese docs locally.
make serve-zh

# Or build to site/zh/.
make build-zh
```

### Migration from Sphinx

If you are migrating markdown files from the old Sphinx/MyST format,
run the migration script:

```bash
make migrate
```

This converts:

- MyST toctree → removed (nav is in `mkdocs.yml`)
- MyST admonitions → MkDocs admonition syntax
- MyST tab-set/tab-item → MkDocs Material tabbed syntax
- MyST code-block → standard fenced code blocks
- MyST variable substitution `|var|` → `{{ var }}` macro syntax

### Version variables

Version variables (e.g., `{{ vllm_ascend_version }}`) are defined in the
`extra` section of `mkdocs.yml` and substituted at build time by the
`mkdocs-macros` plugin.

To update versions, edit the `extra` section in `mkdocs.yml`.
