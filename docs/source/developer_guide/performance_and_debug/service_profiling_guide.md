# Service Profiling Guide

In inference service processes, we sometimes need to monitor the internal execution flow of the inference service framework to identify performance issues. By collecting start and end timestamps of key processes, identifying critical functions or iterations, recording key events, and capturing diverse types of information, we can quickly pinpoint performance bottlenecks.

This guide walks you through collecting performance data for the vllm-ascend service framework and operators. It covers the full workflow from preparation and collection to analysis and visualization, helping you quickly get started with the profiling tool.

## Quick Start

### 0 Installation

Install the `msserviceprofiler` package using pip:

```bash
pip install msserviceprofiler==1.2.2
```

### 1 Preparation
Before starting the service, set the environment variable `SERVICE_PROF_CONFIG_PATH` to point to the profiling configuration file, and set the environment variable `PROFILING_SYMBOLS_PATH` to specify the YAML configuration file for the symbols that need to be imported. After that, start the vLLM service according to your deployment method.

```bash
cd ${path_to_store_profiling_files}
# Set environment variable
export SERVICE_PROF_CONFIG_PATH=ms_service_profiler_config.json
export PROFILING_SYMBOLS_PATH=service_profiling_symbols.yaml

# Start vLLM service
vllm serve Qwen/Qwen2.5-0.5B-Instruct &
```

The file `ms_service_profiler_config.json` is the profiling configuration. If it does not exist at the specified path, a default configuration will be generated automatically. If needed, you can customize it in advance according to the instructions in the `Profiling Configuration File` section below.

`service_profiling_symbols.yaml` is the configuration file containing the profiling points to be imported. You can choose **not** to set the `PROFILING_SYMBOLS_PATH` environment variable, in which case the default configuration file will be used. If the file does not exist at the path you specified, likewise, the system will generate a configuration file at your specified path for future configuration. You can customize it according to the instructions in the `Symbols Configuration File` section below.

### 2 Enable Profiling
To enable the performance data collection switch, change the `enable` field from `0` to `1` in the configuration file `ms_service_profiler_config.json`. This can be accomplished by executing the following sed command:

```bash
sed -i 's/"enable":\s*0/"enable": 1/' ./ms_service_profiler_config.json
```

### 3 Send Requests
Choose a request-sending method that suits your actual profiling needs:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json"  \
    -d '{
         "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "prompt": "Beijing is a",
        "max_tokens": 5,
        "temperature": 0
}' | python3 -m json.tool
```

### 4 Analyze Data

```bash
# xxxx-xxxx is the directory automatically created based on vLLM startup time
cd /root/.ms_server_profiler/xxxx-xxxx

# Analyze data
msserviceprofiler analyze --input-path=./ --output-path output
```

### 5 View Results

After analysis, the `output` directory will contain:
- `chrome_tracing.json`: Chrome tracing format data, which can be opened in [MindStudio Insight](https://www.hiascend.com/document/detail/zh/mindstudio/81RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html).
- `profiler.db`: Performance data in database format.
- `request.csv`: Request-related data.
- `request_summary.csv`: Overall request metrics.
- `kvcache.csv`: KV Cache-related data.
- `batch.csv`: Batch scheduling-related data.
- `batch_summary.csv`: Overall batch scheduling metrics.
- `service_summary.csv`: Overall service-level metrics.

---

## Appendix
(profiling-configuration-file)=
### 1 Profiling Configuration File

The profiling configuration file controls profiling parameters and behavior.

#### File Format

The configuration is in JSON format. Main parameters:

| Parameter | Description | Required |
|:------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----:|
| enable | Switch for profiling: <br />0: disable<br />1: enable<br />Default: 0 | Yes |
| prof_dir | Directory to store collected performance data. <br />Default: $HOME/.ms_service_profiler | No |
| profiler_level | Data collection level. Default is "INFO" (normal level). | No |
| host_system_usage_freq | Sampling frequency of host CPU and memory metrics. Disabled by default. Range: integer 1–50, unit: Hz (times per second). Set to -1 to disable. <br />Note: Enabling this may consume significant memory. | No |
| npu_memory_usage_freq | Sampling frequency of NPU memory utilization. Disabled by default. Range: integer 1–50, unit: Hz (times per second). Set to -1 to disable. <br />Note: Enabling this may consume significant memory. | No |
| acl_task_time | Switch to collect operator dispatch latency and execution latency: <br />0: disable (default; 0 or invalid values mean disabled).<br />1: enable; calls `aclprofCreateConfig` with `ACL_PROF_TASK_TIME_L0`.<br />2: enable MSPTI-based data dumping; uses MSPTI for profiling and requires: `export LD_PRELOAD=$ASCEND_TOOLKIT_HOME/lib64/libmspti.so` | No |
| acl_prof_task_time_level | Level and duration for profiling: <br />L0: collect operator dispatch and execution latency only; lower overhead (no operator basic info).<br />L1: collect AscendCL interface performance (host–device and inter-device sync/async memory copy latencies), plus operator dispatch, execution, and basic info for comprehensive analysis.<br />time: profiling duration, integer 1–999, in seconds.<br />If unset, defaults to L0 until program exit; invalid values fall back to defaults.<br />Level and duration can be combined, e.g., `"acl_prof_task_time_level": "L1,10"`. | No |
| api_filter | Filter to select API performance data to dump. For example, specifying "matmul" dumps all API data whose `name` contains "matmul". String, case-sensitive; use "；" to separate multiple targets. Empty means dump all. <br />Effective only when `acl_task_time` is 2. | No |
| kernel_filter | Filter to select kernel performance data to dump. For example, specifying "matmul" dumps all kernel data whose `name` contains "matmul". String, case-sensitive; use "；" to separate multiple targets. Empty means dump all. <br />Effective only when `acl_task_time` is 2. | No |
| timelimit | Profiling duration for the service. The process stops automatically after this time. Range: integer 0–7200, unit: seconds. Default 0 means unlimited. | No |
| domain | Limit profiling to the specified domains to reduce data volume. String, separated by semicolons, case-sensitive, e.g., "Request; KVCache".<br />Empty means all available domains.<br />Available domains: Request, KVCache, ModelExecute, BatchSchedule, Communication.<br />Note: If the selected domains are incomplete, analysis output may show warnings due to missing data. See [Reference Table 1](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/devaids/Profiling/mindieprofiling_0009.html#ZH-CN_TOPIC_0000002370256365__table1985410131831). | No |

#### Example Configuration

```json
{
  "enable": 1,
  "prof_dir": "vllm_prof",
  "profiler_level": "INFO",
  "acl_task_time": 0,
  "acl_prof_task_time_level": "",
  "timelimit": 0
}
```

---

(symbols-configuration-file)=
### 2 Symbols Configuration File

The symbols configuration file defines which functions/methods to profile and supports flexible configuration with custom attribute collection.

#### 2.1 File Name and Loading

- Default load path:`~/.config/vllm_ascend/service_profiling_symbols.MAJOR.MINOR.PATCH.yaml`( According to the installed version of vllm )

If you need to customize the profiling points, it is highly recommended to copy a profiling configuration file to your working directory using the `PROFILING_SYMBOLS_PATH` environment variable.

#### 2.2 Field Descriptions

| Field | Description | Example |
|:-----:|:-----|:-----|
| symbol | Python import path + attribute chain | `"vllm.v1.core.kv_cache_manager:KVCacheManager.free"` |
| handler | Handler type | `"timer"` (default) or `"pkg.mod:func"` (custom) |
| domain | Domain tag | `"KVCache"`, `"ModelExecute"` |
| name | Event name | `"EngineCoreExecute"` |
| min_version | Upper version constraint | `"0.9.1"` |
| max_version | Lower version constraint | `"0.11.0"` |
| attributes | Custom attribute collection | Only support for `"timer"` handler. See the section below |

#### 2.3 Examples

- Example 1: Custom handler

```yaml
- symbol: vllm.v1.core.kv_cache_manager:KVCacheManager.free
  handler: vllm_profiler.config.custom_handler_example:kvcache_manager_free_example_handler
  domain: Example
  name: example_custom
```

- Example 2: Default timer

```yaml
- symbol: vllm.v1.engine.core:EngineCore.execute_model
  domain: ModelExecute
  name: EngineCoreExecute
```

- Example 3: Version constraint

```yaml
- symbol: vllm.v1.executor.abstract:Executor.execute_model
  min_version: "0.9.1"
  # No handler specified -> default timer
```

#### 2.4 Custom Attribute Collection

The `attributes` field supports flexible custom attribute collection and allows operations and transformations on function arguments and return values.

##### Basic Syntax

- Argument access: use the parameter name directly, e.g., `input_ids`
- Return value access: use the `return` keyword
- Pipeline operations: use `|` to chain multiple operations
- Attribute access: use `attr` to access object attributes

##### Example

```yaml
- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner.execute_model
  name: ModelRunnerExecuteModel
  domain: ModelExecute
  attributes:
  - name: device
    expr: args[0] | attr device | str
  - name: dp
    expr: args[0] | attr dp_rank | str
  - name: batch_size
    expr: args[0] | attr input_batch | attr _req_ids | len
```

##### Expression Notes

1. `len(input_ids)`: get the length of parameter `input_ids`.
2. `len(return) | str`: get the length of the return value and convert to string (equivalent to `str(len(return))`).
3. `return[0] | attr input_ids | len`: get the length of the `input_ids` attribute of the first element in the return value.

##### Supported Expression Types

- Basic operations: `len()`, `str()`, `int()`, `float()`
- Index access: `return[0]`, `return['key']`
- Attribute access: `return | attr attr_name`
- Pipeline composition: chain operations with `|`

##### Advanced Examples

```yaml
attributes:
  # Get tensor shape
  - name: tensor_shape
    expr: input_tensor | attr shape | str
  
  # Get specific value from a dict
  - name: batch_size
    expr: kwargs['batch_size']
  
  # Conditional expression (requires custom handler support)
  - name: is_training_mode
    expr: training | bool
  
  # Complex data processing
  - name: processed_data_len
    expr: data | attr items | len | str
```

#### 2.5 Custom Handler

When `handler` specifies a custom function, it must match the following signature:

```python
def custom_handler(original_func, this, *args, **kwargs):
    """
    Custom handler
    
    Args:
        original_func: the original function object
        this: the bound object (for methods)
        *args: positional arguments
        **kwargs: keyword arguments
    
    Returns:
        processing result
    """
    # Custom logic
    pass
```

If the custom handler fails to import, the system will automatically fall back to the default timer mode.
