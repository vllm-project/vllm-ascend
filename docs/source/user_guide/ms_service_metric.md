# MS Service Metric

MS Service Metric collects vLLM Ascend runtime metrics through function hooks and exposes them in Prometheus format. Install `ms_service_metric` separately before using this integration; vLLM Ascend itself does not require it to run inference.

For installation, configuration syntax, and complete usage, see the [MS Service Metric documentation](https://gitcode.com/Ascend/msserviceprofiler/blob/master/ms_service_metric/README.md).

## Basic Usage

Enable metric collection before starting the service:

```bash
export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

ms-service-metric on
vllm serve <model-path>
```

`PROMETHEUS_MULTIPROC_DIR` stores the Prometheus multiprocess metric files. Use a separate directory for each service instance. Before restarting a service for validation, remove stale files only after confirming that no running service is using the directory.

After sending inference requests, query the vLLM metrics endpoint:

```bash
curl http://127.0.0.1:8000/metrics
```

The output can be scraped by Prometheus and visualized in Grafana. Use `ms-service-metric off` to disable collection.

## Metric Level

Ascend-owned handlers record a `MetricSpec` only when its `level` is less than or equal to the active level. Hooks stay installed for the life of the process. The next sample in that process uses the new level.

| Value | Name | Recorded metrics |
| --- | --- | --- |
| `0` | `off` | None from Ascend-owned handlers |
| `1` | `basic` | Default. Metrics shipped today, including EPLB hotness |
| `2` | `detail` | `basic`, plus specs declared at detail |
| `3` | `debug` | Every declared level |

Set it in `runtime_config.json`, the same file as `ascend_log`:

```json
"metrics": {
  "level": "basic"
}
```

`off`, `basic`, `detail`, `debug`, and `0` through `3` are accepted. The stored value is the name. With `runtime_config_hot_reload=true`, the next apply writes that name into the process-local level: startup, the API / EngineCore file poll, and the worker wave-head cascade. Each process keeps its own integer. An invalid level rejects that load and leaves the previous level in place.

Change the level on a running server:

```bash
curl -X POST "http://127.0.0.1:8000/set_metrics_level?level=detail"
```

The API server sends a utility RPC to every EngineCore, the same path as `POST /sleep`. Each EngineCore sets its own level (scheduler metrics run there) and broadcasts `set_metrics_level` to its workers, including pipeline stages that do not run the wave-head reload. The command also writes `metrics.level` into an existing `runtime_config.json`, so the next hot-reload keeps this value. A restart rebuilds that file from defaults and `additional_config.runtime_config`; put the level there when it should survive a restart. The response is `{"status": "ok", "level": "detail"}`. An unknown level returns HTTP 400.

This level applies to handlers that record through `vllm_ascend.observability.ms_metrics.handlers.common`. Metrics emitted by `ms_service_metric.provider_handlers` (scheduler, model-runner phase, memory, and EPLB duration) follow the `ms-service-metric` on/off switch. Lowering the level stops new samples. Series already created in the Prometheus registry remain until the process exits.

Declare `level` on a new expensive metric. The default is `1` (`basic`):

```python
MetricSpec(
    "example:per_layer:value",
    MetricType.GAUGE,
    ("rank", "layer"),
    level=2,
)
```

## Custom Metric Configuration

Set `MS_SERVICE_METRIC_CONFIG_PATH` before starting vLLM. It can point to either one YAML file or a directory containing multiple YAML files:

```bash
# Load one YAML file.
export MS_SERVICE_METRIC_CONFIG_PATH=/data/custom_metrics/custom.yaml

# Or load all first-level .yaml and .yml files in one directory.
export MS_SERVICE_METRIC_CONFIG_PATH=/data/custom_metrics

ms-service-metric on
vllm serve <model-path>
```

Directory mode loads first-level `.yaml` and `.yml` files in filename order and does not scan subdirectories recursively. The configured directory is also the root for external Handler modules referenced by `module:function` in these YAML files. Users are responsible for maintaining their custom YAML files and Handler implementations.

Run `ms-service-metric restart` after changing YAML configuration. Restart the vLLM service after changing Handler Python code because imported Python modules are not reloaded dynamically.

## Adding a Metric Point

vLLM Ascend-owned metric configurations are stored as YAML files in `vllm_ascend/observability/ms_metrics/config/`, one YAML file per feature. All YAML files in this directory are loaded automatically.

1. Identify the target function and add its fully qualified `module:Class.method` name as `symbol`.
2. Reuse a stable handler from `ms_service_metric.provider_handlers` when it provides the required data processing.
3. For Ascend-specific processing, add a handler module `vllm_ascend/observability/ms_metrics/handlers/<feature>.py`, re-export the handler in `handlers/__init__.py`, and reference it as `vllm_ascend.observability.ms_metrics.handlers:function_name`. Reuse the shared helpers in `handlers/common.py` instead of re-implementing them: declare each metric as a `MetricSpec` (name, type, label names, and optional `level`) and pass the specs to `register_metrics`. Record samples with `record_metric`, and skip expensive collection when `metric_enabled(spec)` is false. Wrap recording code in `safe_recording` for failure isolation. `common.py` is the only module that imports `ms_service_metric`, so feature handlers take `MetricType` and the helpers from there.
4. Add the metric name and Prometheus type under `metrics`, then update `tests/ut/observability/test_ms_metrics_provider.py`.

Example:

```yaml
- symbol: vllm_ascend.worker.v1.model_runner:NPUModelRunner.execute_model
  handler: ms_service_metric.provider_handlers:model_runner_phase_handler
  metrics:
    - name: executor:model_runner_execute_model:duration
      type: histogram
```

Keep configurations and handlers aligned with the vLLM Ascend implementation. A missing symbol disables only the affected metric and must not affect inference.
