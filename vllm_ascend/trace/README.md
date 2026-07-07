# vLLM Ascend Trace

This directory provides the lightweight timeline tracing used to analyze
vLLM-Ascend disaggregated prefill/decode requests.

## Enable Trace

Trace is disabled by default. Enable it with:

```bash
export VLLM_ASCEND_TRACE=1
```

The legacy `OMNI_NPU_VLLM_PATCHES` switch is not used by vLLM-Ascend trace.
It can remain in an old launch script, but it does not enable this trace path.

## Environment Variables

Required:

```bash
export VLLM_ASCEND_TRACE=1
export ROLE=prefill        # or decode
```

Optional but recommended:

```bash
export TRACE_OUTPUT_DIRECTORY=/tmp/vllm_ascend_trace
export PROFILING_NAMELIST=/path/to/omnilogger_namelist_vllm_023.yml
```

If `PROFILING_NAMELIST` is not set, vLLM-Ascend uses the packaged default:

```text
vllm_ascend/trace/omnilogger_namelist_vllm_023.yml
```

If `PROFILING_NAMELIST` is set, the custom YAML is loaded first and the
packaged default YAML is loaded afterwards. A target that has already been
wrapped is skipped, so duplicate wrapping is avoided.

## Example: Prefill Node

```bash
export VLLM_ASCEND_TRACE=1
export ROLE=prefill
export TRACE_OUTPUT_DIRECTORY=/tmp/vllm_ascend_trace/prefill
export PROFILING_NAMELIST=/path/to/vllm_ascend/trace/omnilogger_namelist_vllm_023.yml

vllm serve ...
```

## Example: Decode Node

```bash
export VLLM_ASCEND_TRACE=1
export ROLE=decode
export TRACE_OUTPUT_DIRECTORY=/tmp/vllm_ascend_trace/decode
export PROFILING_NAMELIST=/path/to/vllm_ascend/trace/omnilogger_namelist_vllm_023.yml

vllm serve ...
```

## YAML Format

The YAML format follows the original omni trace style:

```yaml
type: marker
targets:
  - module: "vllm.v1.core.sched.scheduler:Scheduler"
    trace_scope: platform
    function_name: schedule
    entry_operation: |
      ...
    exit_operation: |
      ...
```

Supported fields:

- `type`: currently `marker` and `timer` are supported.
- `targets[].module`: module path, optionally followed by `:ClassName`.
- `targets[].function_name`: function or method to wrap.
- `targets[].entry_operation`: Python snippet executed before the function.
- `targets[].exit_operation`: Python snippet executed after the function.
- `targets[].trace_scope`: optional `platform` or `worker`.

If `trace_scope` is missing, vLLM-Ascend falls back to module-prefix filtering
for compatibility with old YAML files.

## Patch Scope

`platform` trace targets cover API server, OpenAI serving, engine core,
scheduler, KV cache manager, and request status.

`worker` trace targets cover NPU model runner and MooncakeHybrid KV transfer.

If a target module, class, or function is not available in the current runtime,
the target is skipped with a warning and service startup continues.

## Output

Trace logs are written under `TRACE_OUTPUT_DIRECTORY`, one log file per
process/thread:

```text
log_pid_<pid>_tid_<tid>.log
```

Action records keep the original report format:

```text
<<<Action: <action>; Timestamp:<seconds>; RequestID:<request_id>; Role:<role>_<ip>
```

Engine-step records use:

```text
profile: ...
profile_mainmodel: ...
profile_mtpmodel: ...
```

## Parse Logs

The parser is located at:

```text
tools/omni_trace/parse_logs.py
```

It is an optional log-analysis tool. The analysis environment needs `pandas`
and `openpyxl` installed because the report is written as `.xlsx`.

Run:

```bash
python tools/omni_trace/parse_logs.py /path/to/all_pd_logs_directory
```

Reports are generated in the input log directory:

```text
time_analysis.xlsx
engine_step.xlsx
```
