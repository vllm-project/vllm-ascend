# Documentation writing guide

## Guide to Writing Model Tutorial Documentation

`docs/source/_templates/Model-Deployment-Tutorial-Template.md` is a template for writing model deployment tutorials. You can copy and modify it to create new docs.

## Testable documentation code block generation (``model-code``)

- For **documentation authors**: how to insert testable command blocks into docs
- For **developers**: how to add a new converter

Built-in supported `converter_tag` values:

| converter_tag |
| --- |
| `single_node` |
| `multi_node` |

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

# You can add any extra content here, e.g. code, explanations, or comments.
{{ generated }}
```
````

##### Options

| Option | Required | Default | Description |
| --- | --- | --- | --- |
| `block_name` | Yes | None | Block name; must be unique within the current document |
| `converter_tag` | Yes | None | Must be `single_node` |
| `test_case_path` | Yes | None | Repository-relative path; file must exist |
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
:block_name: your_unique_block_name
:converter_tag: multi_node
:test_case_path: tests/e2e/nightly/multi_node/config/your_model.yaml
:host_index: 0
```
````

##### Template 2: with text (use `{{ generated }}` placeholder)

````md
```{model-code}
:block_name: your_unique_block_name
:converter_tag: multi_node
:test_case_path: tests/e2e/nightly/multi_node/config/your_model.yaml
:host_index: 0

# You can add any extra content here, e.g. code, explanations, or comments.
{{ generated }}
```
````

##### Options

| Option | Required | Default | Description |
| --- | --- | --- | --- |
| `block_name` | Yes | None | Block name; must be unique within the current document |
| `converter_tag` | Yes | None | Must be `multi_node` |
| `test_case_path` | Yes | None | Repository-relative path; file must exist |
| `host_index` | Yes | None | Use `deployment[host_index]` from the YAML as the rendering source |

##### YAML reference

See existing files under `tests/e2e/nightly/multi_node/config/`.

`multi_node` reads `deployment[host_index]`. Common fields include:

- `envs`: rendered as `export ...` (scalar values)
- `server_cmd`: a complete command (must start with `vllm serve <model>`; shell multi-line string or token list)

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

#### Build the site & preview locally

```bash
# Install documentation build dependencies
python3 -m pip install -r docs/requirements-docs.txt

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

2. If your converter needs new directive options (e.g. `:foo_index:`):

   - Add the option name to `MODEL_CODE_OPTION_NAMES` in `tools/docs_codegen/scanner.py`
   - Add the option name to `ModelCodeDirective.option_spec` in `tools/docs_codegen/sphinx_extension.py`

3. Add a real example snippet in any model doc (recommended under `docs/source/tutorials/models/`) and point it to a YAML file that exists (recommended under `tests/`).

4. Minimal validation via CLI:

   - `python3 tools/docs_codegen/cli.py --doc <your_doc>` or `--block <doc>::<block_name>`
