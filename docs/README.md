# vLLM Ascend Plugin documents

Live doc: <https://docs.vllm.ai/projects/ascend>

## Build the docs

The documentation uses [MkDocs](https://www.mkdocs.org/) with the
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

### Prerequisites

Run all commands in this guide from the repository root inside a virtual
environment.

```bash
python -m venv .venv
source .venv/bin/activate
# Install documentation dependencies.
python -m pip install -r docs/requirements-docs.txt
```

### Build and serve (English)

```bash
# Serve docs locally with live reload.
make -f docs/Makefile serve
# Open http://127.0.0.1:8000/projects/ascend/en/latest/

# Or build to site/.
make -f docs/Makefile build

# Serve the built static files.
python -m http.server -d site/
# Open http://127.0.0.1:8000/
```

### Build and serve (Chinese)

Chinese docs are generated from `.po` translation files in
`docs/source/locale/zh_CN/LC_MESSAGES/`. The Chinese serve and build targets
generate the translated Markdown files automatically.

```bash
# Serve Chinese docs locally.
make -f docs/Makefile serve-zh

# Or build to site/zh/.
make -f docs/Makefile build-zh

# Or only generate Chinese Markdown files without serving or building.
make -f docs/Makefile gen-zh
```

### Page revision dates

Both sites show the latest Git author date among the English page and its
included Markdown fragments, in the Asia/Shanghai time zone. Renames and
formatting changes count as updates; Chinese translation updates do not.

Revision dates are enabled on Read the Docs and disabled by default for local
builds and previews. To enable them locally, set `extra.show_revision_date: true`
in `mkdocs.yml`. This requires committed sources and complete Git history;
run `git fetch --unshallow` if needed. Read the Docs fetches history automatically.

### Dependencies and automated translation

- `docs/requirements-docs.txt`: build either site, including Chinese Markdown
  generation from existing `.po` files.
- `docs/requirements-translate.txt`: run automated translation. Install these
  dependencies in a separate virtual environment.

Validate dependency changes with Python 3.12, matching Read the Docs.

### Migration from Sphinx

If you are migrating markdown files from the old Sphinx/MyST format, conversion follows these rules:

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
