# Batch-Job-Aware Scheduler

Batch-Job-Aware Scheduler is a specialized scheduler designed for **offline batch inference** scenarios where throughput and hardware utilisation are the primary goals. It is particularly effective when processing multiple batch jobs concurrently, each with a distinct set of requests.

> **Note:** This scheduler **does not implement starvation prevention**. It is intended solely for **batch processing scenarios** or **online inference scenarios where request waiting time is not a concern**. If requests have strict latency or fairness requirements, this scheduler may not be suitable.

## Overview

The scheduler implements three key strategies to improve throughput:

1. **LPT (Longest Processing Time first) scheduling**: Prioritises longer tasks first and schedules shorter ones to fill gaps, particularly during the decode step. This improves the average number of tokens computed per scheduling round.

2. **KV cache reservation**: Estimates and reserves KV cache budget in advance for running requests, reducing preemption overhead.

3. **Job-aware request grouping**: Groups requests by **job name** (extracted from the request ID), assigns each job its own request bucket, and dynamically adjusts job scheduling order based on KV cache availability:
   - When available tokens > threshold (default 4096): prioritise long decode jobs
   - When available tokens ≤ threshold: prioritise short decode jobs

The scheduler is implemented by extending the **waiting queue** with **dynamic priority scheduling**, while inheriting most features from the original scheduler class, including chunked prefill, async scheduling, and more.

## How It Works

### Job Name Extraction

The scheduler identifies which job a request belongs to via a `#job_name[${JOB_NAME}]#` tag embedded in the request ID. For example:

```python
request_id = "req_001#job_name[my_batch_job]#"
```

Requests without a job name tag are grouped under the `__default__` job.

### Decode Length Estimation

Each job's decode length is predicted using a pure **EWMA (Exponentially Weighted Moving Average)** estimator:

- **No samples yet**: Returns a cold-start default value (128 tokens).
- **First observation**: EWMA is initialised to the observed decode length.
- **Subsequent observations**: EWMA is updated incrementally, giving more weight to recent data while smoothing out fluctuations.

This approach is simple, responsive, and avoids the complexity of multi-phase estimation while producing stable predictions even with limited data.

This enables the scheduler to distinguish between "long decode" jobs (which benefit from being scheduled first when resources are abundant) and "short decode" jobs (which are prioritised when resources are scarce).

## Getting Started

### Prerequisites

- **vLLM v1 engine** is required (the batch-job-aware scheduler is built on the v1 scheduling framework).
- **Ascend NPU** with sufficient memory for the target model(s).

### Enabling the Feature

The batch-job-aware scheduler is enabled via the `additional_config` parameter. It is currently only supported in **offline batch** mode.

```bash
python -m vllm.entrypoints.openai.run_batch \
    --model /path/to/model \
    -i /path/to/input.jsonl \
    -o /path/to/output.jsonl \
    --additional-config '{"batch_job_sched_config": {"enabled": true}}'
```

### Request ID Format

To take advantage of job-aware scheduling, embed the `#job_name[...]#` tag into the request ID inside your batch input file:

```jsonl
# In your batch input file (e.g., input.jsonl):
{"custom_id": "#job_name[job_A]#req_001", "method": "POST", "url": "/v1/chat/completions", "body": {"request_id": "#job_name[job_A]#req_001", "messages": [{"role": "user", "content": "Hello"}], "n": 1}}
{"custom_id": "#job_name[job_A]#req_002", "method": "POST", "url": "/v1/chat/completions", "body": {"request_id": "#job_name[job_A]#req_002", "messages": [{"role": "user", "content": "What is AI?"}], "n": 1}}
{"custom_id": "#job_name[job_B]#req_003", "method": "POST", "url": "/v1/chat/completions", "body": {"request_id": "#job_name[job_B]#req_003", "messages": [{"role": "user", "content": "Explain quantum computing"}], "n": 1}}
```

Requests without a job name tag will be grouped under the default job and still benefit from the scheduler's KV cache reservation and LPT scheduling.

## Configuration Parameters

All parameters are nested under `batch_job_sched_config` in the `additional_config` dictionary.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Enable the batch-job-aware scheduler |
| `max_jobs` | int | `20` | Maximum number of tracked jobs |
| `reserve_margin_blocks` | int | `2` | Extra block margin added to the KV cache reserve as safety buffer |
| `reserve_max_blocks` | int | `8` | Maximum number of blocks that can be reserved |
| `low_available_tokens_threshold` | int | `4096` | Threshold for prioritising long vs short decode jobs |
| `short_decode_token_threshold` | int | `32` | Threshold for classifying a job as "short decode" |

## Usage

### Basic Offline Batch

```bash
python -m vllm.entrypoints.openai.run_batch \
    --model /path/to/model \
    -i /path/to/input.jsonl \
    -o /path/to/output.jsonl \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --additional-config '{"batch_job_sched_config": {"enabled": true}}'
```

### Offline Batch with Custom Configuration

```bash
python -m vllm.entrypoints.openai.run_batch \
    --model /path/to/model \
    -i /path/to/input.jsonl \
    -o /path/to/output.jsonl \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --additional-config '{
        "batch_job_sched_config": {
            "enabled": true,
            "max_jobs": 10,
            "reserve_margin_blocks": 4,
            "reserve_max_blocks": 12,
            "low_available_tokens_threshold": 2048,
            "short_decode_token_threshold": 32
        }
    }'
```

### Using via Python API

```python
from vllm import LLM

llm = LLM(
    model="/path/to/model",
    max_model_len=4096,
    gpu_memory_utilization=0.9,
    additional_config={
        "batch_job_sched_config": {
            "enabled": True,
        },
    },
)
```

## Best Practices

1. **Encode job names in request IDs**: Use the `#job_name[${JOB_NAME}]#` prefix in your request IDs to help the scheduler group and prioritise requests effectively.

2. **Adjust `low_available_tokens_threshold`**: If your workload is consistently long-decode-heavy, consider lowering this threshold to keep long jobs prioritised. For mixed workloads, keep the default.
