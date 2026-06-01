# Documentation writing guide

## Guide to Writing Model Tutorial Documentation

`docs/source/_templates/Model-Deployment-Tutorial-Template.md` is a template for writing model deployment tutorials. You can copy and modify it to create new docs.

## Testable documentation code block generation (``model-code``)

- For **documentation authors**: how to insert testable command blocks into docs
- For **developers**: how to add a new converter

Built-in supported `converter_tag` values:

| converter_tag | Renders | YAML source |
| --- | --- | --- |
| `single_node` | One `vllm serve` script for a single node | `test_cases[case_index]` |
| `multi_node` | One host's `vllm serve` script | `deployment[host_index]` |
| `external_dp_template` | One external-DP node's env exports + `vllm serve` command | `templates[host_index]` (+ top-level `model`) |
| `external_dp_launch` | One `launch_online_dp.py` line per node | `config[]` |
| `external_dp_proxy` | The load-balance proxy launch command | `config[]` + `routing` |

### For authors: add a block

:::{important}
By default, the generator scans only `.md` files under `docs/source/tutorials/models/` and produces artifacts.
If you put ``model-code`` blocks in other directories, Sphinx builds will not automatically generate the corresponding scripts.
:::

#### Single node (`single_node`)

##### Template 1: minimal (metadata only)

````md
```{model-code}
:block_name: your_unique_block_name
:converter_tag: single_node
:test_case_path: tests/e2e/nightly/single_node/models/configs/your_model.yaml
```
````

##### Template 2: with text (use `{{ generated }}` placeholder)

````md
```{model-code}
:block_name: your_unique_block_name
:converter_tag: single_node
:test_case_path: tests/e2e/nightly/single_node/models/configs/your_model.yaml
:case_index: 0

# You can add any extra content here, e.g. code, explanations, or comments.
{{ generated }}
```
````

##### Options

| Option | Required | Default | Description |
| --- | --- | --- | --- |
| `block_name` | Yes | None | Block name; must be unique within the current document |
| `converter_tag` | Yes | None | Must be `single_node` |
| `test_case_path` | Yes | None | Repository-relative path that stays within the repo (no `..` escape); file must exist |
| `case_index` | No | `0` | Use `test_cases[case_index]` from the YAML as the rendering source |

##### YAML reference

See existing files under `tests/e2e/nightly/single_node/models/configs/`.

`single_node` reads `test_cases[case_index]`. Common fields include:

- `model`: model name (ultimately renders `vllm serve <model> ...`)
- `envs`: rendered as `export ...` (scalar values)
- `server_cmd`: arguments appended to `vllm serve <model>` (shell string or token list)
- `server_cmd_extra` (optional): extra appended arguments

#### Multi node (`multi_node`)

##### Template 1: minimal (metadata only)

````md
```{model-code}
:block_name: your_unique_block_name_0
:converter_tag: multi_node
:test_case_path: tests/e2e/nightly/multi_node/config/your_model.yaml
:host_index: 0
```
````

````md
```{model-code}
:block_name: your_unique_block_name_1
:converter_tag: multi_node
:test_case_path: tests/e2e/nightly/multi_node/config/your_model.yaml
:host_index: 1
```
````

##### Template 2: with text (use `{{ generated }}` placeholder)

````md
```{model-code}
:block_name: your_unique_block_name_0
:converter_tag: multi_node
:test_case_path: tests/e2e/nightly/multi_node/config/your_model.yaml
:host_index: 0

# You can add any extra content here, e.g. code, explanations, or comments.
{{ generated }}
```
````

````md
```{model-code}
:block_name: your_unique_block_name_1
:converter_tag: multi_node
:test_case_path: tests/e2e/nightly/multi_node/config/your_model.yaml
:host_index: 1

# You can add any extra content here, e.g. code, explanations, or comments.
{{ generated }}
```
````

##### Options

| Option | Required | Default | Description |
| --- | --- | --- | --- |
| `block_name` | Yes | None | Block name; must be unique within the current document |
| `converter_tag` | Yes | None | Must be `multi_node` |
| `test_case_path` | Yes | None | Repository-relative path that stays within the repo (no `..` escape); file must exist |
| `host_index` | Yes | None | Use `deployment[host_index]` from the YAML as the rendering source |

##### YAML reference

See existing files under `tests/e2e/nightly/multi_node/config/`.

`multi_node` reads `deployment[host_index]`. Common fields include:

- `envs`: rendered as `export ...` (scalar values)
- `server_cmd`: a complete command (must start with `vllm serve <model>`; shell multi-line string or token list)

#### External data parallel (`external_dp_template` / `external_dp_launch` / `external_dp_proxy`)

These three converters read one **shared** external-DP YAML (see
`tests/e2e/nightly/multi_node/external_dp/config/`) and each render a different
part of the deployment. They are tightly coupled to that schema by design.

The shared YAML provides:

- `model`: model name (top level)
- `config`: a list of per-node settings (`port_start`, `dp_rpc_port`, `dp_size`,
  `dp_size_local`, `dp_rank_start`, `tp_size`, `dp_address`, ...)
- `routing`: `type` plus `groups` (e.g. `prefiller` / `decoder` lists of `config` indices)
- `templates`: a list of per-node `envs` and `server_cmd_template` entries

`server_cmd_template` uses braced `${VAR}` placeholders that
`external_dp_template` rewrites to the positional shell parameters consumed by
`run_dp_template.sh`:

| `${VAR}` | Positional |
| --- | --- |
| `${VISIBLE_DEVICES}` | `$1` |
| `${PORT}` | `$2` |
| `${DP_SIZE}` | `$3` |
| `${DP_RANK}` | `$4` |
| `${DP_ADDRESS}` | `$5` |
| `${DP_RPC_PORT}` | `$6` |
| `${TP_SIZE}` | `$7` |

Unbraced references such as `$SERVER_PORT` and unknown braced variables are left
untouched so they remain live shell expansions.

##### Templates

````md
```{model-code}
:block_name: your_unique_block_name_prefill_node0
:converter_tag: external_dp_template
:test_case_path: tests/e2e/nightly/multi_node/external_dp/config/your_model.yaml
:host_index: 0
```
````

`external_dp_launch` (one `python launch_online_dp.py ...` line per `config` node)
and `external_dp_proxy` (the load-balance proxy command, e.g.
`python load_balance_proxy_server_example.py ...`) read the whole cluster, so they
take **no** index option:

````md
```{model-code}
:block_name: your_unique_block_name_launch
:converter_tag: external_dp_launch
:test_case_path: tests/e2e/nightly/multi_node/external_dp/config/your_model.yaml
```
````

````md
```{model-code}
:block_name: your_unique_block_name_proxy
:converter_tag: external_dp_proxy
:test_case_path: tests/e2e/nightly/multi_node/external_dp/config/your_model.yaml
```
````

##### Options

| Option | Required | Default | Description |
| --- | --- | --- | --- |
| `block_name` | Yes | None | Block name; must be unique within the current document |
| `converter_tag` | Yes | None | One of `external_dp_template`, `external_dp_launch`, `external_dp_proxy` |
| `test_case_path` | Yes | None | Repository-relative path that stays within the repo (no `..` escape); file must exist |
| `host_index` | `external_dp_template` only | None | Use `templates[host_index]` from the YAML as the rendering source |

:::{note}
`external_dp_proxy` currently supports only `routing.type: disaggregated_prefill`
and reads its `routing.groups.prefiller` / `routing.groups.decoder` node lists.
:::

### Local debugging and generation

#### Generate only (without building the full site)

```bash
# Generate all model-code artifacts under docs/source/tutorials/models/
python3 tools/docs_codegen/cli.py

# Generate artifacts for a single document
python3 tools/docs_codegen/cli.py --doc docs/source/tutorials/models/Kimi-K2-Thinking.md

# Generate a single block and print it (no files written)
python3 tools/docs_codegen/cli.py \
  --block docs/source/tutorials/models/Kimi-K2-Thinking.md::kimi_k2_thinking_single_node \
  --dry-run --stdout
```

By default, artifacts are written to: `docs/_build/doc_codegen/<doc_stem>/<block_name>.sh`.

:::{note}
After the script is generated, please make sure to check whether the generated content is runnable, especially key parts such as environment variables and command-line parameters.
:::

#### Concrete YAML-to-shell example

The following `model-code` block reads the first test case from
`tests/e2e/nightly/single_node/models/configs/Kimi-K2-Thinking.yaml`:

````md
```{model-code}
:block_name: kimi_k2_thinking_single_node
:converter_tag: single_node
:test_case_path: tests/e2e/nightly/single_node/models/configs/Kimi-K2-Thinking.yaml
```
````

The YAML fields read by the converter look like this:

```yaml
test_cases:
  - name: "Kimi-K2-Thinking-TP16-Case"
    model: "moonshotai/Kimi-K2-Thinking"
    envs:
      HCCL_BUFFSIZE: "1024"
      TASK_QUEUE_ENABLE: "1"
      OMP_PROC_BIND: "false"
      HCCL_OP_EXPANSION_MODE: "AIV"
      PYTORCH_NPU_ALLOC_CONF: "expandable_segments:True"
      SERVER_PORT: "DEFAULT_PORT"
    server_cmd:
      - "--tensor-parallel-size"
      - "16"
      - "--port"
      - "$SERVER_PORT"
      - "--max-model-len"
      - "8192"
      - "--max-num-batched-tokens"
      - "8192"
      - "--max-num-seqs"
      - "12"
      - "--gpu-memory-utilization"
      - "0.9"
      - "--trust-remote-code"
      - "--enable-expert-parallel"
      - "--no-enable-prefix-caching"
```

Run the block in dry-run mode to see the generated shell without writing files:

```bash
python3 tools/docs_codegen/cli.py \
  --block docs/source/tutorials/models/Kimi-K2-Thinking.md::kimi_k2_thinking_single_node \
  --dry-run --stdout
```

The first line is the artifact path. The remaining lines are the generated shell
content:

```bash
# docs/_build/doc_codegen/Kimi-K2-Thinking/kimi_k2_thinking_single_node.sh
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export OMP_PROC_BIND=false
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SERVER_PORT=8000

vllm serve moonshotai/Kimi-K2-Thinking \
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

In this example, `envs` is rendered as `export` lines, `model` becomes
`vllm serve <model>`, and `server_cmd` is appended as formatted command-line
arguments. `SERVER_PORT: "DEFAULT_PORT"` is resolved to the default single-node
port `8000`.

#### Build the site & preview locally

```bash
# Install documentation build dependencies
python3 -m pip install -r docs/requirements-docs.txt

# (Optional) Clean previous builds
make -C docs clean

# Build the English site
make -C docs html

# (Optional) Build the Chinese site
make -C docs intl

# Preview locally
python3 -m http.server -d docs/_build/html 8000

# Then open in a browser:
# http://localhost:8000
```

### For developers: add a new converter

The goal of adding a converter is to make `converter_tag: <name>` render a given YAML structure into a script (`GeneratedScript`).

#### What to change

1. In `tools/docs_codegen/converters.py`:

   - Add a `BaseConverter` subclass that implements `convert(loaded_yaml, *, block) -> GeneratedScript`
   - Give the converter a unique `name` (the value used by `converter_tag` in docs)
   - Register it in `build_default_converters()`
   - Reuse the shared validation/rendering helpers in `tools/docs_codegen/utils.py`
     (`require_yaml_mapping`, `require_mapping`, `require_scalar_mapping`,
     `require_indexed_mapping`, `parse_command_tokens`, `render_cli_command`, ...)
     rather than re-validating the YAML shape inline

2. If your converter needs new directive options (e.g. `:foo_index:`):

   - Add the option name to `MODEL_CODE_OPTION_NAMES` in `tools/docs_codegen/scanner.py`
   - Add the option name to `ModelCodeDirective.option_spec` in `tools/docs_codegen/sphinx_extension.py`

3. Add a real example snippet in any model doc (recommended under `docs/source/tutorials/models/`) and point it to a YAML file that exists (recommended under `tests/`).

4. Minimal validation via CLI:

   - `python3 tools/docs_codegen/cli.py --doc <your_doc>` or `--block <doc>::<block_name>`
