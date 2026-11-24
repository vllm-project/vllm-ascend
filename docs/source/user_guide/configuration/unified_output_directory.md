# Unified Output Directory

vLLM Ascend and its supporting components generate logs, caches, and diagnostic files in different locations. Organizing these files under a common directory simplifies troubleshooting and operational maintenance by making them easier to locate, collect, and manage.

This section explains how to configure these paths in an existing deployment, using `/root/ascend/log` as the common root directory inside the inference container. Before starting the service, create the required directories or parent directories for target files, and ensure that each process has write access to its output location.

## Environment variables

Set these variables in the existing launcher before it starts the relevant process. For example, add `export XDG_CACHE_HOME=/root/ascend/log/cache` before `vllm serve`. For a deployment that already defines the service environment, set the same name and value there.

| Component | Output | Environment variable | Example path |
| --- | --- | --- | --- |
| **vLLM** | Disk cache | `XDG_CACHE_HOME` | `/root/ascend/log/cache` |
| | Configuration root | `XDG_CONFIG_HOME` | `/root/ascend/log/config` |
| | FX graph dump directory override | `VLLM_DEBUG_DUMP_PATH` | `/root/ascend/log/debug/fx` |
| **Ascend / CANN** | [Working directory](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/maintenref/envvar/envref_07_0007.html), including logs | `ASCEND_WORK_PATH` | `/root/ascend/log/cann/work` |
| | [Cache directory](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/maintenref/envvar/envref_07_0006.html) | `ASCEND_CACHE_PATH` | `/root/ascend/log/cann/cache` |
| **PyTorch / torch_npu** | [Inductor compilation cache](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_configuration_tutorial.html) | `TORCHINDUCTOR_CACHE_DIR` | `/root/ascend/log/cann/torchinductor` |
| **Triton Ascend** | [Base directory](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/environment_variable_and_compiler_options_reference.md) | `TRITON_HOME` | `/root/ascend/log/triton` |
| | Compilation cache override | `TRITON_CACHE_DIR` | `/root/ascend/log/cache/triton` |

With the XDG settings above, vLLM uses `/root/ascend/log/cache/vllm` and `/root/ascend/log/config/vllm`. If your deployment already sets `VLLM_CACHE_ROOT` or `VLLM_CONFIG_ROOT`, update those values to the corresponding paths. Explicit feature cache paths such as `VLLM_ASSETS_CACHE` also need to point under the common root if you use them. See [vLLM environment variables](https://docs.vllm.ai/en/latest/configuration/env_vars/).

## vLLM configuration

Add the fields below to the JSON objects passed to the listed flags in your existing `vllm serve` command. Merge fields that share a flag into one object and keep the other fields already in that object. If you use a YAML file with `--config`, omit the leading `--` from the flag name and put the fields under that YAML key. For example, put `ascend_log_path` and `runtime_config_path` under `additional-config`.

| Output | vLLM flag | Field | Set it to |
| --- | --- | --- | --- |
| Service log files | `--additional-config` | `ascend_log_path` | `/root/ascend/log/vllm_ascend` |
| Watchdog stack dumps | `--watchdog-config` | `dump_dir` | `/root/ascend/log/watchdog` |
| Runtime Guard configuration file | `--additional-config` | `runtime_config_path` | `/root/ascend/log/runtime/config/runtime_config.json` |
| [Runtime Guard incident reports](../feature_guide/runtime_guard.md) | `--additional-config` | `runtime_report_dir` | `/root/ascend/log/runtime/report` |
| [Runtime Guard KV cache dumps](../feature_guide/runtime_guard.md) | `--additional-config` | `runtime_dump_dir` | `/root/ascend/log/runtime/kv_cache` |
| [msprobe capture configuration file](../../developer_guide/performance_and_debug/msprobe_guide.md) | `--additional-config` | `dump_config_path` | `/root/ascend/log/msprobe/config.json` |
| FX graph dumps | `--compilation-config` | `debug_dump_path` | `/root/ascend/log/debug/fx` |
| [Ascend PyTorch profiler traces](../../developer_guide/performance_and_debug/service_profiling_guide.md#ascend-pytorch-profiler) | `--profiler-config` | `profiler`, `torch_profiler_dir` | `"torch"`, `"/root/ascend/log/profiler/torch"`, respectively |

- **Watchdog** works on builds that list `--watchdog-config` in `vllm serve --help=all`. Setting `dump_dir` enables stack dumps on timeout. See the [vLLM CLI reference](https://docs.vllm.ai/en/latest/cli/serve/#watchdog-config).
- **msprobe** writes capture results to `dump_path` in the msprobe JSON file. Set that field to `/root/ascend/log/msprobe/dump`.
- **FX graph dumps** follow `VLLM_DEBUG_DUMP_PATH` if that environment variable is set.
- **Runtime Guard** derives its report and KV dump directories from the configuration file location. With the base settings above, reports go under `/root/ascend/log/runtime/report` and KV dumps under its `kv_cache` subdirectory when capture is enabled. Omit both directory overrides unless you need a different layout.

After starting with the recommended settings, look for `vllm_ascend_*.log` in the service log directory and for the Runtime Guard JSON file. Diagnostic files appear when the corresponding feature produces them, for example after a Watchdog timeout, a Runtime Guard report action, or the end of a profiling capture.

## External component logs

This section covers logging for components used in disaggregated serving and KV pooling.

### MemCache and MemFabric logs

| Output | Default location |
| --- | --- |
| MemCache MetaService | `/var/log/memcache_hybrid/logs/mmc-meta.log` (service) and `/var/log/memcache_hybrid/logs/mmc-meta-audit.log` (audit). [MemCache configuration](https://gitcode.com/Ascend/memcache/blob/master/docs/zh/memcache_config.md) |
| MemFabric runtime messages | Standard output of the process using MemFabric. [MemFabric logger](https://gitcode.com/Ascend/memfabric_hybrid/blob/master/src/util/csrc/mf_out_logger.h) |
| MemFabric ptracer (when enabled) | `/var/log/memfabric_hybrid/ptracer_<pid>.dat`. [Ptracer guide](https://gitcode.com/Ascend/memfabric_hybrid/blob/master/docs/zh/wiki/developer_guide/debugging/ptracer.md) |

### Mooncake logs

| Process | Where to set the log directory |
| --- | --- |
| Transfer Engine for P/D disaggregation or KV pooling | Before `vllm serve`, export `MC_LOG_DIR=/root/ascend/log/mooncake/transfer` in each worker launcher (both P and D for disaggregation). The directory must already exist and be writable by the worker, or the engine logs to stderr. [Transfer Engine source](https://github.com/kvcache-ai/Mooncake/blob/main/mooncake-transfer-engine/src/config.cpp) |
| Mooncake Store master for KV pooling | Create `/root/ascend/log/mooncake/master` in the master container and add `--log_dir=/root/ascend/log/mooncake/master` to the existing `mooncake_master` command. [Store master logging](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/deployment/mooncake-store-deployment-guide.md#logging) |

## Minimal example

Replace `/path/to/model` with your model directory. When updating an existing launcher, merge the two fields below into its `--additional-config` object. Include the Watchdog argument if your build supports it and you want timeout stack dumps.

```bash
export XDG_CACHE_HOME=/root/ascend/log/cache
export XDG_CONFIG_HOME=/root/ascend/log/config

# Optional: collect Ascend component files and the torch_npu compilation cache.
export ASCEND_WORK_PATH=/root/ascend/log/cann/work
export ASCEND_CACHE_PATH=/root/ascend/log/cann/cache
export TORCHINDUCTOR_CACHE_DIR=/root/ascend/log/cann/torchinductor

vllm serve /path/to/model \
  --watchdog-config '{"dump_dir": "/root/ascend/log/watchdog"}' \
  --additional-config '{
    "ascend_log_path": "/root/ascend/log/vllm_ascend",
    "runtime_config_path": "/root/ascend/log/runtime/config/runtime_config.json"
  }'
```
