# vLLM-Ascend Single-Node E2E Test Developer Guide

This document is intended to help developers understand the architecture of the single-node E2E (End-to-End) testing framework in `vllm-ascend`, how to run test scripts, and how to add custom testing functionality by writing YAML configuration files and extending the code.

## 1. Test Architecture Overview

To achieve high readability, extensibility, and decoupling of configuration from code, the single-node E2E test adopts a **"YAML-driven + Dispatcher"** architectural structure.

It consists of the following core components:

* **Configuration Parser (`single_node_config.py`)**: Responsible for reading `models/configs/*.yaml` files and parsing them into a strongly-typed `@dataclass` (`SingleNodeConfig`) via `SingleNodeConfigLoader`, while handling regex replacement for environment variables.
* **Service Manager Framework (`test_single_node.py` and `conftest.py`)**: Based on the `service_mode` (`openai` or `epd`), it utilizes context managers to safely start/stop server processes.
* **Test Function Dispatcher (`TEST_HANDLERS` Registry)**: Specific test logic is encapsulated into independent functions and registered in the global `TEST_HANDLERS` dictionary.
* **Performance Benchmarking (`_run_benchmarks`)**: Calls `aisbench` for performance and TTFT testing based on the `benchmarks` parameters in the YAML.

### 1.1 Key Files and Responsibilities

* `tests/e2e/nightly/single_node/models/scripts/single_node_config.py`
    * Defines `SingleNodeConfig` and `SingleNodeConfigLoader`
    * Loads YAML from `tests/e2e/nightly/single_node/models/configs/<CONFIG_YAML_PATH>`
    * Auto-assigns ports when `envs` contains `DEFAULT_PORT` / missing values
    * Expands `$VAR` / `${VAR}` placeholders inside commands via `_expand_values`

* `tests/e2e/nightly/single_node/models/scripts/test_single_node.py`
    * Declares `configs = SingleNodeConfigLoader.from_yaml_cases()` (loaded at import time)
    * `pytest.mark.parametrize("config", configs, ids=[config.name for config in configs])` runs one test per YAML case
        * Controls server lifecycle via context managers
        * Dispatches `test_content` to functions registered in `TEST_HANDLERS`
        * Runs `aisbench` and optional benchmark assertions

### 1.2 End-to-End Flow (High Level)

```txt
pytest starts
  |
  v
import tests/e2e/nightly/single_node/models/scripts/test_single_node.py
  |
  v
configs = SingleNodeConfigLoader.from_yaml_cases()
  |
  v
pytest parametrize("config", configs)  # one config == one test case
  |
  v
test_single_node(config)
  |
  +-----------------------------------------------+
  | Start service (depends on service_mode)       |
  |                                               |
  |  openai: start one vLLM OpenAI-compatible     |
  |         service process                       |
  |  epd:   start (encode service + decode/PD     |
  |         service) + start proxy process        |
  +-----------------------------------------------+
  |
  v
Run test phases (test_content)
  |
  v
Optional benchmarks (if benchmarks is configured)
  |
  v
Shutdown all started processes

Notes:
- One YAML file may contain multiple test_cases; pytest will run them one by one.
- The framework is "YAML-driven": changes are typically done by editing YAML rather than editing Python code.
```

### 1.3 Function Call Relationships (Dispatcher)

`test_content` is a list of “phases”. Each phase maps to one handler function.

```txt
For each test_case:

  test_content (list of phases)
        |
        v
    [Dispatcher]
        |
        +--> phase "completion"      -> send completion request(s)
        |
        +--> phase "chat_completion" -> send chat completion request(s)
        |
        +--> phase "image"           -> send multimodal image request(s)
        |
        \--> (extendable) add your own phase by registering a new handler

After phases:
  if benchmarks is configured -> run aisbench

Notes:
- The dispatcher only controls "what to run"; service lifecycle is controlled by the service manager.
- Phases are intentionally small & composable so you can reuse them across YAML cases.
```

## 2. Running and Debugging Steps

### 2.1 Dependencies

Ensure you are in an NPU environment and have installed `pytest`, `pyyaml`, `openai`, and `aisbench`.

### 2.2 Local Execution

The framework uses the `CONFIG_YAML_PATH` environment variable to specify the configuration file.

```bash
# Switch to the project root directory
cd /vllm-workspace/vllm-ascend

# Run a specific yaml test
export CONFIG_YAML_PATH="Qwen3-32B.yaml"
pytest -sv tests/e2e/nightly/single_node/models/scripts/test_single_node.py
```

### 2.3 Tips for Debugging

* Only run a subset of cases: `pytest -sv ... -k <keyword>` (matches case names in the report output)
* Stop on first failure: `pytest -sv ... -x`
* Keep server logs visible: use `-s` (already included in `-sv`) and increase log verbosity via standard Python logging configuration if needed.

## 3. How to Write YAML Configuration Files

### 3.1 File Location and Selection Rules

* YAML files live under: `tests/e2e/nightly/single_node/models/configs/`
* Selected by env var: `CONFIG_YAML_PATH=<YourConfig>.yaml`
* If not set, the loader uses `SingleNodeConfigLoader.DEFAULT_CONFIG_NAME`

### 3.2 Field Descriptions

| Field Name       | Type       | Required | Default Value   | Description                                                         |
| :--------------- | :--------- | :------- | :-------------- | :------------------------------------------------------------------ |
| `test_cases`     | list       | **Yes** | -                | List of test case objects                                           |
| `name`           | string     | **Yes** | -                | Human-readable case ID shown in pytest output and logs              |
| `model`          | string     | **Yes** | -                | Model name or local path                                            |
| `service_mode`   | string     | No      | `openai`         | Service mode: `openai` or `epd` (disaggregated)                     |
| `envs`           | map        | **Yes** | `{}`             | Environment variables for the server process                        |
| `server_cmd`     | list       | Cond.   | `[]`             | vLLM startup arguments (Required for non-EPD)                       |
| `server_cmd_extra` | list     | No      | `[]`             | Extra vLLM startup arguments appended after `server_cmd`            |
| `prompts`        | list       | No      | built-in default | Prompts for completion/chat tests                                   |
| `api_keyword_args` | map      | No      | built-in default | OpenAI API keyword args (e.g., `max_tokens`, sampling params)       |
| `test_content`   | list       | No      | `["completion"]` | Test phases: `completion`, `chat_completion`, `image`  etc.         |
| `benchmarks`     | map        | No      | `{}`             | Configuration for `aisbench` performance verification               |
| `epd_server_cmds`| list[list] | Cond.   | `[]`             | (EPD Only) Command arrays for starting dual Encode/Decode processes |
| `epd_proxy_args` | list       | Cond.   | `[]`             | (EPD Only) Startup arguments for the EPD routing gateway            |

**Notes / Behaviors**

* `name` is mandatory and must be a non-empty string.
    * It is used directly as pytest case id (e.g., `test_single_node[DeepSeek-R1-0528-W8A8-single]`).
    * It is also printed in `[single-node][START]` marker for log navigation.

* `envs` (ports): the config object recognizes these keys: `SERVER_PORT`, `ENCODE_PORT`, `PD_PORT`, `PROXY_PORT`.
    * If a port key is missing or set to `DEFAULT_PORT`, it will be automatically filled with an available open port.
    * `$SERVER_PORT` / `${SERVER_PORT}` placeholders in commands will be expanded using `envs`.

* `server_cmd` vs `server_cmd_extra`:
    * YAML can define `server_cmd_extra` to append additional args after `server_cmd`.
    * The loader merges them into a single `server_cmd` list.

* Extra fields:
    * Any non-standard fields in a case are stored in `config.extra_config`.
    * This is how extension configs are passed through without changing the dataclass.

### 3.3 YAML Examples

#### Single-Case (similar to DeepSeek-R1-W8A8-HBM)

```yaml
test_cases:
  - name: "<your-case-name>"
    model: "<model-repo-or-local-path>"

    # Optional: The default values are as follows
    prompts:
      - "San Francisco is a"
    api_keyword_args:
      max_tokens: 10

    envs:
      SERVER_PORT: "DEFAULT_PORT"
      # Add only what you need.

    server_cmd:
      - "--port"
      - "$SERVER_PORT"
      # plus your vLLM serve args...
    
    # Optional: omit -> defaults to ["completion"]
    test_content:
      - "chat_completion"

    # Optional: leave empty if you don't run aisbench
    benchmarks:
```

#### Multi-Case + Shared Anchors

```yaml
_envs: &envs
  SERVER_PORT: "DEFAULT_PORT"
  # shared envs...

_server_cmd: &server_cmd
  - "--port"
  - "$SERVER_PORT"
  # shared vLLM serve args...

_benchmarks: &benchmarks
  perf:
    case_type: performance
    dataset_path: vllm-ascend/GSM8K-in3500-bs400
    request_conf: vllm_api_stream_chat
    dataset_conf: gsm8k/gsm8k_gen_0_shot_cot_str_perf
    num_prompts: 400
    max_out_len: 1500
    batch_size: 1000
    baseline: 1
    threshold: 0.97

test_cases:
  - name: "case-a"
    model: "<model>"
    envs:
      <<: *envs
      DYNAMIC_EPLB: "true"
      # private envs...
    server_cmd: *server_cmd
    server_cmd_extra:
      - "--enforce-eager"
    benchmarks:

  - name: "case-b"
    model: "<model>"
    envs:
      <<: *envs
    server_cmd: *server_cmd
    benchmarks:
      <<: *benchmarks_acc
```

#### EPD / Disaggregated Case

```yaml
test_cases:
  - name: "<your-epd-case>"
    model: "<model>"
    service_mode: "epd"
    envs:
      ENCODE_PORT: "DEFAULT_PORT"
      PD_PORT: "DEFAULT_PORT"
      PROXY_PORT: "DEFAULT_PORT"

    epd_server_cmds:
      - ["--port", "$ENCODE_PORT", "--model", "<encode-model>"]
      - ["--port", "$PD_PORT", "--model", "<decode-model>"]

    epd_proxy_args:
      - "--host"
      - "127.0.0.1"
      - "--port"
      - "$PROXY_PORT"
      - "--encode-servers-urls"
      - "http://localhost:$ENCODE_PORT"
      - "--decode-servers-urls"
      - "http://localhost:$PD_PORT"
      - "--prefill-servers-urls"
      - "disable"

    test_content:
      - "chat_completion"
```

## 4. How to Add Custom Tests (Extension)

### Step 1: Write your test logic in `test_single_node.py`

```python
async def run_video_test(config: SingleNodeConfig, server: 'RemoteOpenAIServer | DisaggEpdProxy') -> None:
    client = server.get_async_client()
    # Your custom logic here...
```

### Step 2: Register your function in `TEST_HANDLERS`

```python
TEST_HANDLERS = {
    "completion": run_completion_test,
    "video": run_video_test,  # Registered!
}
```

### Step 3: Enable in YAML

```yaml
    test_content:
      - "completion"
      - "video"
```

## 5. Checklist (Before Submitting a New YAML)

* `test_cases` exists and is a list
* Each case contains required fields for its `service_mode`
    * Common required: `name`, `model`, `envs`
    * `openai`: `server_cmd`
    * `epd`: `epd_server_cmds`, `epd_proxy_args`
* Port envs are set to `DEFAULT_PORT` (or to explicit free ports)
* If using `benchmarks`, ensure each benchmark case includes required aisbench fields (e.g., `case_type`, `dataset_path`, `request_conf`, `dataset_conf`, `max_out_len`, `batch_size`)
