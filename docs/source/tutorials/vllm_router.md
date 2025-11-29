# VLLM Router

A high-performance request routing system for vLLM deployments, providing advanced load balancing and specialized routing for modern LLM serving architectures.

## Overview
- Unified control plane for registering, monitoring, and orchestrating prefill, decode, and regular workers across heterogeneous model fleets.
- Data plane that routes requests across HTTP, PD (prefill/decode), gRPC, and OpenAI-compatible backends with shared reliability features.
- Multi-model inference gateway mode (`--enable-igw`) that runs several routers at once and applies per-model policies.
- Built-in reliability primitives: retries with exponential backoff, circuit breakers, token-bucket rate limiting, and queuing.
- First-class observability with structured logging and Prometheus metrics.

## Installation

### Prerequisites

**Rust and Cargo:**

```bash
# Install rustup (Rust installer and version manager)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Reload shell environment
source "$HOME/.cargo/env"

# Verify installation
rustc --version
cargo --version
```

**Python with pip installed**

### Installation & Basic Usage

#### Rust Binary

```bash
# Build Rust components
cargo build --release
```

#### Python Package

```bash
pip install setuptools-rust wheel build
python -m build
pip install dist/*.whl
```

## Quick Start

### Standard Data Parallelism Routing

In DP-aware mode, the vLLM router automatically selects the appropriate data-parallel rank to handle each request according to specified Load balancing policy.

- **Rust binary**

    ```bash
    # Launch router with data parallelism (8 replicas per worker URL)
    # When data-parallel-size > 1, the router automatically creates DP-aware workers
    ./target/release/vllm-router \
        --worker-urls http://0.0.0.0:8000 \
        --policy consistent_hash \
        --data-parallel-size 8
        
    # Alternative: using cargo run
    cargo run --release -- \
        --worker-urls http://0.0.0.0:8000 \
        --policy consistent_hash \
        --data-parallel-size 8
    ```

- **Python launcher**
  
  ```bash
    # Alternative: using python launcher
    vllm-router \
    --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy cache_aware
    ```

### Prefill/Decode Disaggregation (PD)

- **Rust binary**

    ```bash
    ./target/release/vllm-router \
    --pd-disaggregation \
    --prefill http://prefill1:30001 9001 \
    --prefill http://prefill2:30002 \
    --decode http://decode1:30011 \
    --decode http://decode2:30012 \
    --policy cache_aware \
    --prefill-policy cache_aware \
    --decode-policy power_of_two
    ```

- **Python launcher**
  
  ```bash
    vllm-router \
    --pd-disaggregation \
    --prefill http://prefill1:30001 9001 \
    --prefill http://prefill2:30002 \
    --decode http://decode1:30011 \
    --decode http://decode2:30012 \
    --policy cache_aware
    ```

Prefill entries accept an optional bootstrap port. PD mode merges prefill metadata with decode outputs and streams results back to the client.

### Multi-Model Inference Gateway

Enable IGW mode to route multiple models through a vllm router while applying per-model policies:

```bash
./target/release/vllm-router \
  --enable-igw \
  --policy cache_aware \
  --max-concurrent-requests 512

# Register workers dynamically
curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{
        "url": "http://worker-a:8000",
        "model_id": "qwen3-8b",
        "priority": 10,
        "labels": {"tier": "gold"}
      }'

# Add another worker with a different model/policy hint
curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{
        "url": "http://worker-b:8000",
        "model_id": "llama3",
        "priority": 20,
        "labels": {"policy": "power_of_two", "tier": "silver"}
      }'

# Inspect registered workers
curl http://localhost:30000/workers
```

Sample response (http workers):

```json
{
  "workers": [
    {"id":"http://0.0.0.0:31378","url":"http://0.0.0.0:31378","model_id":"qwen3-8b","priority":50,"cost":1.0,"worker_type":"regular","is_healthy":true,"load":0,"connection_mode":"Http"},
    {"id":"http://0.0.0.0:34881","url":"http://0.0.0.0:34881","model_id":"llama3","priority":50,"cost":1.0,"worker_type":"regular","is_healthy":true,"load":0,"connection_mode":"Http"}
  ],
  "total": 2,
  "stats": {
    "prefill_count": 0,
    "decode_count": 0,
    "regular_count": 2
  }
}
```

Add more workers with the same API; include optional `labels` (for per-model policies) or `tokenizer_path` / `reasoning_parser` / `tool_parser` fields as needed. `/workers/{url}` exposes queued job status while background jobs finalize registration.

| Method   | Path             | Description                                                                                                                                               |
|----------|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `POST`   | `/workers`       | Queue worker registration (prefill/decode/regular). Body matches `WorkerConfigRequest`. Returns `202 Accepted` while the job queue processes the request. |
| `GET`    | `/workers`       | List workers with health, load, policy metadata, and queued job status.                                                                                   |
| `GET`    | `/workers/{url}` | Inspect a specific worker or job queue entry.                                                                                                             |
| `DELETE` | `/workers/{url}` | Queue worker removal.                                                                                                                                     |
| `POST`   | `/flush_cache`   | Trigger cache flush across HTTP workers with success/failure breakdown.                                                                                   |
| `GET`    | `/get_loads`     | Sample current load reported by each worker.                                                                                                              |

## Configuration

### Logging

Enable structured logging with optional file output:

```python
from vllm_router import Router

# Console logging (default)
router = Router(worker_urls=["http://worker1:8000", "http://worker2:8000"])

# File logging enabled
router = Router(
    worker_urls=["http://worker1:8000", "http://worker2:8000"],
    log_dir="./logs"  # Daily log files created here
)
```

Set log level with `--log-level` flag ([documentation](https://docs.vllm.ai/backend/server_arguments.html#logging)).

### Metrics

Prometheus metrics endpoint available at `127.0.0.1:29000` by default.

```bash
# Custom metrics configuration
vllm-router \
    --worker-urls http://localhost:8080 http://localhost:8081 \
    --prometheus-host 0.0.0.0 \
    --prometheus-port 9000
```

### Retries and Circuit Breakers

#### Retry Configuration
Retries are enabled by default with exponential backoff and jitter:

```bash
vllm-router \
  --worker-urls http://localhost:8080 http://localhost:8081 \
  --retry-max-retries 3 \
  --retry-initial-backoff-ms 100 \
  --retry-max-backoff-ms 10000 \
  --retry-backoff-multiplier 2.0 \
  --retry-jitter-factor 0.1
```

#### Circuit Breaker Configuration
Circuit breakers protect workers and provide automatic recovery:

```bash
vllm-router \
  --worker-urls http://localhost:8080 http://localhost:8081 \
  --cb-failure-threshold 5 \
  --cb-success-threshold 2 \
  --cb-timeout-duration-secs 30 \
  --cb-window-duration-secs 60
```

**Circuit Breaker State Machine:**
- `Closed` → `Open` after N consecutive failures (failure-threshold)
- `Open` → `HalfOpen` after timeout (timeout-duration-secs)
- `HalfOpen` → `Closed` after M consecutive successes (success-threshold)
- Any failure in `HalfOpen` reopens immediately

**Retry Policy:** Retries on HTTP status codes 408/429/500/502/503/504, with backoff/jitter between attempts.

### Request ID Tracking

Track requests across distributed systems with configurable headers:

```bash
# Use custom request ID headers
vllm-router \
    --worker-urls http://localhost:8080 \
    --request-id-headers x-trace-id x-request-id
```

**Default headers:** `x-request-id`, `x-correlation-id`, `x-trace-id`, `request-id`

## Advanced Features

### Kubernetes Service Discovery

Automatic worker discovery and management in Kubernetes environments.

#### Basic Service Discovery

- `--service-discovery`: Enable Kubernetes service discovery
- `--service-discovery-port`: Port for worker URLs (default: 8000)
- `--service-discovery-namespace`: Kubernetes namespace to watch
- `--selector`: Label selectors for regular mode (format: `key1=value1 key2=value2`)

```bash
vllm-router \
    --service-discovery \
    --selector app=vllm-worker role=inference \
    --service-discovery-namespace default
```

#### RBAC Configuration

**Namespace-scoped (recommended):**

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: vllm-router
  namespace: vllm-system
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: vllm-system
  name: vllm-router
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: vllm-router
  namespace: vllm-system
subjects:
- kind: ServiceAccount
  name: vllm-router
  namespace: vllm-system
roleRef:
  kind: Role
  name: vllm-router
  apiGroup: rbac.authorization.k8s.io
```
