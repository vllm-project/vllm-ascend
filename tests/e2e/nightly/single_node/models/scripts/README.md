# vLLM-Ascend Single-Node E2E Test Developer Guide

This document is intended to help developers understand the architecture of the single-node E2E (End-to-End) testing framework in `vllm-ascend`, how to run test scripts, and how to add custom testing functionality by writing YAML configuration files and extending the code.

---

## 1. Test Architecture Overview

To achieve high readability, extensibility, and decoupling of configuration from code, the single-node E2E test adopts a **"YAML-driven + Dispatcher"** architectural structure.

It consists of the following core components:

* **Configuration Parser (`single_node_config.py`)**: Responsible for reading `models/models_yaml/*.yaml` files and parsing them into a strongly-typed `@dataclass` (`SingleNodeConfig`) via `SingleNodeConfigLoader`, while handling regex replacement for environment variables.
* **Service Manager Framework (`test_single_node.py` and `conftest.py`)**: Based on the `service_mode` (`openai` or `epd`), it utilizes context managers to safely start/stop server processes.
* **Test Function Dispatcher (`TEST_HANDLERS` Registry)**: Specific test logic is encapsulated into independent functions and registered in the global `TEST_HANDLERS` dictionary.
* **Performance Benchmarking (`_run_benchmarks`)**: Calls `aisbench` for performance and TTFT testing based on the `benchmarks` parameters in the YAML.

---

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

---

## 3. How to Write YAML Configuration Files

### 3.1 Field Descriptions

| Field Name       | Type       | Required | Default Value   | Description                                                         |
| :--------------- | :--------- | :------- | :-------------- | :------------------------------------------------------------------ |
| `model`          | string     | **Yes** | -                | Model name or local path                                            |
| `service_mode`   | string     | No      | `openai`         | Service mode: `openai` or `epd` (disaggregated)                     |
| `envs`           | map        | No      | `{}`             | Environment variables for the server process                        |
| `server_cmd`     | list       | Cond.   | `[]`             | vLLM startup arguments (Required for non-EPD)                       |
| `test_content`   | list       | No      | `["completion"]` | Test phases: `completion`, `chat_completion`, `image`  etc.         |
| `benchmarks`     | map        | No      | `{}`             | Configuration for `aisbench` performance verification               |
| `epd_server_cmds`| list[list] | Cond.   | `[]`             | (EPD Only) Command arrays for starting dual Encode/Decode processes |
| `epd_proxy_args` | list       | Cond.   | `[]`             | (EPD Only) Startup arguments for the EPD routing gateway            |

### 3.2 Benchmarking and Metric Assertions (benchmark_comparisons)

```yaml
    test_content:
      - "benchmark_comparisons"

    benchmark_comparisons_args:
      - metric: "TTFT"
        baseline: "baserun"
        target: "fastrun"
        ratio: 0.8
        operator: "<"
```

### 3.3 OpenAI Standard Mode Example

```yaml
test_cases:
  - name: "Qwen-32B-Standard-Test"
    model: "Qwen/Qwen2.5-32B-Instruct"
    service_mode: "openai"
    server_cmd:
      - "--tensor-parallel-size"
      - "4"
      - "--port"
      - "$SERVER_PORT"
    test_content:
      - "chat_completion"
```

---

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
