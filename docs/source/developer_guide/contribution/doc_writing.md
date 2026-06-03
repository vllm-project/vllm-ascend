# Doc writing guide

## Guide to Writing Model Tutorial Doc

`docs/source/_templates/Model-Deployment-Tutorial-Template.md` is a template for writing model deployment tutorials. You can copy and modify it to create new docs.

## Testable doc code block generation (``model-code``)

- For **documentation authors**: how to insert testable command blocks into docs
- For **developers**: how to add a new converter

Built-in supported `converter_tag` values:

| converter_tag | Renders | YAML source |
| --- | --- | --- |
| `single_node` | A single node's env exports + `vllm serve` script | `test_cases[case_index]` |
| `multi_node` | One host's env exports + `vllm serve` script | `deployment[host_index]` |
| `external_dp_template` | One external-DP node's env exports + `vllm serve` command | `templates[host_index]` |
| `external_dp_launch` | One `launch_online_dp.py` line per node | `config[]` |
| `external_dp_proxy` | The load-balance proxy launch command | `config[]` + `routing` |

### For authors: add a block

:::{important}
By default, the generator scans only `.md` files under `docs/source/tutorials/models/` and produces artifacts.
If you put ``model-code`` blocks in other directories, Sphinx builds will not automatically generate the corresponding scripts.
:::

All ``model-code`` blocks need:

| Option | Required | Description |
| --- | --- | --- |
| `block_name` | Yes | Block name; must be unique within the current document |
| `converter_tag` | Yes | Selects one of the built-in converters |
| `test_case_path` | Yes | Repository-relative YAML path that stays within the repo; the file must exist |

Use the body of the block to add shell wrapper lines such as `set -eux`. Always
place the `{{ generated }}` placeholder where the converter output should be
inserted.

#### converter_tag: `single_node`

`single_node` reads one item from `test_cases`. The optional `case_index`
metadata selects the item; when omitted, it defaults to `0`.

Only the fields read by this converter are expanded below. Other test metadata
can be left in the YAML and is ignored by this converter.

```yaml
test_cases:
  - name: qwen3-8b-single
    model: Qwen/Qwen3-8B
    envs:
      HCCL_BUFFSIZE: "1024"
      SERVER_PORT: DEFAULT_PORT
    server_cmd:
      - --tensor-parallel-size
      - "1"
      - --port
      - $SERVER_PORT
      - --trust-remote-code
    server_cmd_extra:
      - --enable-expert-parallel
    benchmarks: ...
```

`envs` is rendered as `export` lines. `SERVER_PORT: DEFAULT_PORT` is resolved
to the default single-node port `8000`. `model` becomes `vllm serve <model>`,
and `server_cmd` plus optional `server_cmd_extra` become command arguments.
Both command fields can be either a shell string or a flat token list.

Write the doc block like this:

````md
```{model-code}
:block_name: qwen3_8b_single_node
:converter_tag: single_node
:test_case_path: tests/e2e/nightly/single_node/models/configs/your_model.yaml
:case_index: 0

set -eux
{{ generated }}
```
````

Generated shell script:

```bash
set -eux
export HCCL_BUFFSIZE=1024
export SERVER_PORT=8000

vllm serve Qwen/Qwen3-8B \
  --tensor-parallel-size 1 \
  --port $SERVER_PORT \
  --trust-remote-code \
  --enable-expert-parallel
```

#### converter_tag: `multi_node`

`multi_node` reads one item from `deployment`. The required `host_index`
metadata selects which host to render.

```yaml
deployment:
  - envs:
      SERVER_PORT: "8000"
    server_cmd: >
      vllm serve Qwen/Qwen3-235B-A22B
      --host 0.0.0.0
      --port $SERVER_PORT
      --data-parallel-size 2
      --tensor-parallel-size 8
      --data-parallel-address $LOCAL_IP
  - envs:
      SERVER_PORT: "8000"
    server_cmd: >
      vllm serve Qwen/Qwen3-235B-A22B
      --headless
      --port $SERVER_PORT
      --data-parallel-size 2
      --tensor-parallel-size 8
      --data-parallel-start-rank 1
      --data-parallel-address $MASTER_IP
benchmarks: ...
```

`server_cmd` must be a complete command starting with `vllm serve <model>`.
It can be written as a shell string or a flat token list.

Write the doc block like this:

````md
```{model-code}
:block_name: qwen3_235b_worker_1
:converter_tag: multi_node
:test_case_path: tests/e2e/nightly/multi_node/internal_dp/config/your_model.yaml
:host_index: 1

set -eux
{{ generated }}
```
````

Generated shell script for `host_index: 1`:

```bash
set -eux
export MASTER_IP=192.168.1.10
export SERVER_PORT=8000

vllm serve Qwen/Qwen3-235B-A22B \
  --headless \
  --port $SERVER_PORT \
  --data-parallel-size 2 \
  --tensor-parallel-size 8 \
  --data-parallel-start-rank 1 \
  --data-parallel-address $MASTER_IP
```

#### converter_tag: `external_dp_template`

`external_dp_template` reads one item from `templates`. The required
`host_index` metadata selects which template to render. The top-level `model`
field is also required because the converter builds `vllm serve <model>`.

```yaml
model: Eco-Tech/GLM-Test
templates:
  - node_index: 0
    envs:
      HCCL_BUFFSIZE: "1024"
      ASCEND_RT_VISIBLE_DEVICES: "${VISIBLE_DEVICES}"
    server_cmd_template:
      - --host
      - 0.0.0.0
      - --port
      - ${PORT}
      - --data-parallel-size
      - ${DP_SIZE}
      - --data-parallel-rank
      - ${DP_RANK}
      - --data-parallel-address
      - ${DP_ADDRESS}
      - --data-parallel-rpc-port
      - ${DP_RPC_PORT}
      - --tensor-parallel-size
      - ${TP_SIZE}
      - --trust-remote-code
config: ...
routing: ...
```

Known braced template variables are rewritten to the positional shell arguments
that `run_dp_template.sh` receives from `launch_online_dp.py`:

| Template variable | Rendered positional |
| --- | --- |
| `${VISIBLE_DEVICES}` | `$1` |
| `${PORT}` | `$2` |
| `${DP_SIZE}` | `$3` |
| `${DP_RANK}` | `$4` |
| `${DP_ADDRESS}` | `$5` |
| `${DP_RPC_PORT}` | `$6` |
| `${TP_SIZE}` | `$7` |

Unknown braced variables and unbraced shell references such as `$SERVER_PORT`
are left unchanged.

Write the doc block like this:

````md
```{model-code}
:block_name: glm_external_dp_template_node0
:converter_tag: external_dp_template
:test_case_path: tests/e2e/nightly/multi_node/external_dp/config/your_model.yaml
:host_index: 0

set -eux
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

{{ generated }}
```
````

Generated shell script for `host_index: 0`:

```bash
set -eux
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export HCCL_BUFFSIZE=1024
export ASCEND_RT_VISIBLE_DEVICES=$1

vllm serve Eco-Tech/GLM-Test \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --trust-remote-code
```

#### converter_tag: `external_dp_launch`

`external_dp_launch` reads the full `config` list and renders one
`launch_online_dp.py` command per node. It does not take an index option.

```yaml
config:
  - node_index: 0
    port_start: 7100
    dp_rpc_port: 12321
    dp_size: 2
    dp_size_local: 2
    dp_rank_start: 0
    tp_size: 8
    dp_address: "${NODE_0_IP}"
  - node_index: 1
    port_start: 7200
    dp_rpc_port: 12321
    dp_size: 4
    dp_size_local: 4
    dp_rank_start: 0
    tp_size: 4
    dp_address: "${NODE_1_IP}"
templates: ...
routing: ...
```

Write the doc block like this:

````md
```{model-code}
:block_name: glm_external_dp_launch
:converter_tag: external_dp_launch
:test_case_path: tests/e2e/nightly/multi_node/external_dp/config/your_model.yaml

set -eux
{{ generated }}
```
````

Generated shell script:

```bash
set -eux
python launch_online_dp.py --dp-size 2 --tp-size 8 --dp-size-local 2 --dp-rank-start 0 --dp-address ${NODE_0_IP} --dp-rpc-port 12321 --vllm-start-port 7100

python launch_online_dp.py --dp-size 4 --tp-size 4 --dp-size-local 4 --dp-rank-start 0 --dp-address ${NODE_1_IP} --dp-rpc-port 12321 --vllm-start-port 7200
```

#### converter_tag: `external_dp_proxy`

`external_dp_proxy` reads `config` and `routing`. It renders the
`load_balance_proxy_server_example.py` command for `routing.type:
disaggregated_prefill`. It does not take an index option.

```yaml
routing:
  type: disaggregated_prefill
  groups:
    prefiller: [0]
    decoder: [1]
config:
  - node_index: 0
    port_start: 7100
    dp_size_local: 2
    dp_rpc_port: 12321
    dp_size: 2
    dp_rank_start: 0
    tp_size: 8
    dp_address: "${NODE_0_IP}"
  - node_index: 1
    port_start: 7200
    dp_size_local: 4
    dp_rpc_port: 12321
    dp_size: 4
    dp_rank_start: 0
    tp_size: 4
    dp_address: "${NODE_1_IP}"
templates: ...
```

`routing.groups.prefiller` and `routing.groups.decoder` contain indices into
`config`. Each referenced node expands to `dp_size_local` host and port entries.
The proxy itself is rendered on `${NODE_0_IP}:1999`.

Write the doc block like this:

````md
```{model-code}
:block_name: glm_external_dp_proxy
:converter_tag: external_dp_proxy
:test_case_path: tests/e2e/nightly/multi_node/external_dp/config/your_model.yaml

set -eux
{{ generated }}
```
````

Generated shell script:

```bash
set -eux
python load_balance_proxy_server_example.py \
  --host ${NODE_0_IP} \
  --port 1999 \
  --prefiller-hosts \
    ${NODE_0_IP} \
    ${NODE_0_IP} \
  --prefiller-ports \
    7100 \
    7101 \
  --decoder-hosts \
    ${NODE_1_IP} \
    ${NODE_1_IP} \
    ${NODE_1_IP} \
    ${NODE_1_IP} \
  --decoder-ports \
    7200 \
    7201 \
    7202 \
    7203
```

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

A converter turns one loaded YAML file plus one parsed `ModelCodeBlock` into a
`GeneratedScript`. The current pipeline is:

1. `BlockScanner` parses ``model-code`` fences and accepts only options listed
   in `MODEL_CODE_OPTION_NAMES`.
2. `YamlLoader` loads `test_case_path`.
3. `get_converter()` looks up `block.converter_tag` from
   `build_default_converters()`.
4. The selected converter returns `GeneratedScript(content=..., language="shell")`.
5. `GeneratorService` replaces `{{ generated }}` in the block body, validates
   that the final script is non-empty, and writes
   `docs/_build/doc_codegen/<doc_stem>/<block_name>.sh`.

To add a converter:

1. In `tools/docs_codegen/converters.py`, add a `BaseConverter` subclass with a
   unique `name`. That name is the value authors put in `:converter_tag:`.
2. Implement `convert(self, loaded_yaml, *, block) -> GeneratedScript`. Use
   `make_docs_codegen_error(..., block=block)` for user-facing validation
   errors so the CLI and Sphinx output include document context.
3. Reuse helpers from `tools/docs_codegen/utils.py`, such as
   `require_mapping`, `require_mapping_list`, `require_scalar_mapping`,
   `require_indexed_mapping`, `require_node_field`, `parse_command_tokens`,
   `substitute_template_positionals`, and `render_cli_command`.
4. Register the converter in `build_default_converters()`. If it is not
   registered, `get_converter()` will reject the new `converter_tag`.
5. If the converter needs new directive metadata, add the option name to
   `MODEL_CODE_OPTION_NAMES` in `tools/docs_codegen/scanner.py` and to
   `ModelCodeDirective.option_spec` in
   `tools/docs_codegen/sphinx_extension.py`. Read the option with
   `block.get_option("<option_name>")`.
6. Add or update tests in `tests/ut/tools/test_docs_codegen.py`. Cover the
   successful render path, required option validation, YAML shape validation,
   and any CLI/Sphinx scanner behavior affected by new metadata.
7. Add a real ``model-code`` example in a model tutorial, preferably under
   `docs/source/tutorials/models/`, and point it to an existing YAML file under
   `tests/`.
8. Validate with the CLI:

   ```bash
   python3 tools/docs_codegen/cli.py --doc <your_doc> --dry-run
   python3 tools/docs_codegen/cli.py --block <your_doc>::<block_name> --dry-run --stdout
   ```

If a converter should render something other than shell, set
`GeneratedScript.language` accordingly so Sphinx can highlight the generated
literal block correctly.
