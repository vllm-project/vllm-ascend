# Documentation Authoring

This page explains how to write model tutorial docs that stay in sync with YAML test cases checked by
`tools/check_docs_yaml_sync.py`.

## What The Checker Scans

The `check-docs-yaml-sync` pre-commit hook currently runs on files that match:

```text
docs/source/tutorials/models/*.md
```

The CLI itself can lint any markdown path that you pass in, but the hook only targets model tutorial
docs.

Within a checked document, every MyST block whose opening fence matches ` ```{test} <lexer> ` is
treated as a sync block. This means:

- A checked file must contain at least one `{test}` block, or the linter reports an error.
- Every `{test}` block in that file must include all required sync metadata.
- If you want to show a shell example that should not be linted, use a normal fenced block such as
  ` ```bash ` instead of ` ```{test} bash `.
- If the whole document cannot be mapped to YAML test cases, add the markdown path to
  `[tool.check_docs_yaml_sync].exclude` in `pyproject.toml`.

## Required Metadata

Each sync block must provide all three metadata fields immediately after the opening fence and before
the first blank line or script content:

````md
```{test} bash
:sync-yaml: path/to/config.yaml
:sync-target: some.path
:sync-class: env

...
```
````

Rules:

- `:sync-yaml:` is a repository-relative path to an existing YAML file.
- `:sync-target:` uses dotted keys and bracket access, for example
  `test_cases[0].envs` or `test_cases[0]["server_cmd"]`.
- Multiple `:sync-target:` values are allowed only by separating targets with spaces.
- `:sync-class:` must be one of the supported classes: `env` or `cmd`.

## How To Write `env` Blocks

Use `:sync-class: env` when the YAML target resolves to a mapping of environment variables.

- `env` blocks must use exactly one `:sync-target:`.
- The YAML target must resolve to a mapping such as `test_cases[0].envs`.
- The checker only reads lines written as `export KEY=value`.
- Blank lines and full-line comments are ignored.
- Inline comments after the value are allowed, for example
  `export SERVER_PORT=DEFAULT_PORT  # Replace with a real port.`
- Non-`export` lines are ignored.
- After quote normalization, the variable names and values must match the YAML mapping.
- Order does not matter, but the final set of `KEY=value` entries must match YAML exactly.

Example:

````md
```{test} bash
:sync-yaml: tests/e2e/nightly/single_node/models/configs/Kimi-K2-Thinking.yaml
:sync-target: test_cases[0].envs
:sync-class: env

export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export OMP_PROC_BIND=false
export HCCL_OP_EXPANSION_MODE=AIV
export SERVER_PORT=DEFAULT_PORT  # Replace DEFAULT_PORT with the actual port.
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
```
````

## How To Write `cmd` Blocks

Use `:sync-class: cmd` when the document contains a `vllm serve` launch command.

- The document block must parse as a valid `vllm serve` command.
- The third token must be the model name.
- After the model, only options and option values are allowed.
- Duplicate option tokens are rejected.
- Inline shell comments are not stripped unless the whole line starts with `#`.
- Backslash-continued multi-line commands are supported.
- Command option order does not matter, but the parsed option set must match YAML exactly.

`cmd` sync targets have two modes:

- One target: the YAML target must resolve to a full command fragment that already includes
  `vllm serve <model> ...`, either as a string or as a flat scalar list.
- Multiple targets: the checker prepends `vllm serve` automatically, then appends each resolved
  fragment in the order listed in `:sync-target:`. A common pattern is
  `test_cases[0].model test_cases[0].server_cmd`.

Example:

````md
```{test} bash
:sync-yaml: tests/e2e/nightly/single_node/models/configs/Kimi-K2-Thinking.yaml
:sync-target: test_cases[0].model test_cases[0].server_cmd
:sync-class: cmd

vllm serve "moonshotai/Kimi-K2-Thinking" \
  --tensor-parallel-size 16 \
  --port $SERVER_PORT \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 12 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --enable-expert-parallel \
  --no-enable-prefix-caching
```
````

## When A Document Should Be Excluded

Exclusions are file-level only. If a markdown file under `docs/source/tutorials/models/` cannot be
mapped cleanly to YAML test cases, add its repository-relative path to
`[tool.check_docs_yaml_sync].exclude` in `pyproject.toml`.

Use this sparingly. Prefer keeping tutorial commands and env vars synchronized with executable test
cases whenever possible.

## Local Validation

Run the sync check directly on one or more markdown files:

```bash
python3 tools/check_docs_yaml_sync.py docs/source/tutorials/models/<your_doc>.md
```

Run the full formatting and lint workflow before pushing:

```bash
bash format.sh
```
