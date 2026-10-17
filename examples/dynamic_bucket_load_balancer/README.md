# Dynamic Bucket Load Balancer

A dynamic bucketing-based hybrid load balance proxy for [vLLM](https://github.com/vllm-project/vllm).

The proxy fronts multiple vLLM backend servers and distributes
OpenAI-compatible requests across them. It can run in two modes:

- **Plain load balancing** (default): for each request, estimate a load score
  and forward it to the least-loaded backend instance.
- **Dynamic bucket load balancing** (`--enable-dynamic-bucket`): split the
  backend pool into a **short-request group** and a **long-request group**, route
  requests to a group by their length, and dynamically rebalance across groups
  based on the load gap and length affinity.

## Files

- `dynamic_bucket_load_balancer.py` — the core algorithm (pure standard library).
  Buckets requests by length, then dynamically adjusts bucket assignment using
  bucket load and length affinity.
- `hybrid_proxy_server.py` — the FastAPI proxy server that uses the algorithm to
  route requests to the backend servers.

## How It Works

1. **Static bucketing by length.** Each request is first mapped to its *standard
   bucket* by request length. With dynamic bucketing enabled the proxy uses two
   buckets: short `[0, --server-group-threshold)` and long
   `[--server-group-threshold, --max-request-tokens)`.

2. **Server groups.** The ordered backend list is split into the same number of
   groups as buckets, **in order**: the first instances form the short group, the
   last instances form the long group. With 4 backends and 2 buckets, backends 0
   and 1 serve the short bucket, backends 2 and 3 serve the long bucket.
   - > **Tip:** configure the first two instances for short sequences and the
     > last two for long sequences (e.g. smaller `max-model-len` / KV cache for
     > the short group, larger for the long group) to get the best throughput.

3. **Dynamic rebalancing.** For a new request, the balancer looks at neighbor
   buckets with a lighter load and computes a redirect probability
   `(load-gap probability) × (length-affinity factor)`. If it exceeds the
   threshold (`0.12`), the request is redirected to the neighbor bucket. This
   means a large load gap is suppressed when the request length is far from the
   neighbor bucket, while a modest gap can still trigger a redirect when the
   length is close to the boundary.

4. **Within a group**, the least-loaded server (smallest active token count) is
   picked via a min-heap, the load is accumulated for the duration of the
   request, and released when streaming completes.

## Prerequisites

- Python 3.10+
- Install dependencies:

  ```bash
  pip install "fastapi<0.124.0" httpx uvicorn
  ```

## Step 1: Start Your Backend Servers

Start at least two vLLM servers, each as a separate process on its own port. The
proxy also works with a single backend, but load balancing is only meaningful
with two or more.

```bash
vllm serve --host 0.0.0.0 --port 8100 ...   # vLLM Server 0
vllm serve --host 0.0.0.0 --port 8101 ...   # vLLM Server 1
```

## Step 2: Start the Proxy Server

From `examples/dynamic_bucket_load_balancer/`, point the proxy at each backend
with `--server-hosts` / `--server-ports`:

```bash
python hybrid_proxy_server.py \
  --host 0.0.0.0 --port 8000 \
  --server-hosts 127.0.0.1 127.0.0.1 \
  --server-ports 8100 8101
```

This starts the proxy on port 8000 and load balances across the two backends.

### Enable Dynamic Bucket Load Balancing

Add `--enable-dynamic-bucket` to split the pool into short/long groups. The
server count must be `>= 2` so each bucket has at least one instance. With 4
servers the first two form the short group and the last two the long group:

```bash
python hybrid_proxy_server.py \
  --host 0.0.0.0 --port 8000 \
  --server-hosts 127.0.0.1 127.0.0.1 127.0.0.1 127.0.0.1 \
  --server-ports 8100 8101 8102 8103 \
  --enable-dynamic-bucket \
  --server-group-threshold 32768
```

## Step 3: Send a Request to the Proxy

Send OpenAI-compatible requests to the proxy. For example:

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "your-model",
        "prompt": "The quick brown fox jumps over the lazy dog",
        "max_tokens": 16
      }'
```

Or for chat completions:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "your-model",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 16
      }'
```

## Step 4: Health Check

Check that the proxy is running and how many backends it fronts:

```bash
curl http://localhost:8000/healthcheck
```

Returns a JSON object, e.g.:

```json
{"status": "ok", "server_instances": 2}
```

## Configuration

| Argument | Default | Description |
| --- | --- | --- |
| `--host` | `localhost` | Proxy listen host. |
| `--port` | `8000` | Proxy listen port. |
| `--server-hosts` | `localhost` | Hosts of the backend vLLM servers (one per server, in order). |
| `--server-ports` | `8001` | Ports of the backend vLLM servers (one per server, in order). |
| `--enable-dynamic-bucket` | `False` | Enable dynamic bucket load balancing. |
| `--server-group-threshold` | `32768` | Length boundary between the short and long buckets. |
| `--max-request-tokens` | `131072` | Upper bound of the long bucket (max request length). |
| `--max-retries` | `3` | Max retries for a backend HTTP request. |
| `--retry-delay` | `0.001` | Base delay (seconds) for exponential backoff retries. |

The number of `--server-hosts` must equal the number of `--server-ports`.
