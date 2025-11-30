This file provides a elastic proxy demo to support elastic scaling for P/D instances based on KV pool.

We can launch multiple vllm instances (2 for prefill and 2 for decode), and
launch this proxy demo through:

```shell
export ADMIN_API_KEY=YOUR_ADMIN_API_KEY
python3 examples/elastic_scaling/elastic_proxy.py  \
   --model $model_name  \
   --prefill localhost:8100 localhost:8101   \
   --decode localhost:8200 localhost:8201   \
   --port 8000
```

### Support API routes
* `/v1/completions`: get completions request response.
* `/v1/chat/completions`: get chat request response.
* `/status`: get the supported prefill nodes and decode nodes list.
* `/instances/add`: add prefill nodes or decode nodes to the list.

examples:
```shell
# /v1/completions
curl -X POST http://0.0.0.0:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{"model": "'$model_name'", "max_tokens": 50, "prompt": "hello"}'

# /v1/chat/completions
curl -X POST http://0.0.0.0:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{"model": "'$model_name'", "max_tokens": 50,
    "messages": [{
        "role": "user",
        "content": "hello"
    }]}'

# /status
curl -X POST http://0.0.0.0:8000/status

# /instance/add
curl -X POST http://0.0.0.0:8000/instances/add \
-H "Content-Type: application/json" \
-H "X-Api-Key: YOUR_ADMIN_API_KEY" \
-d '{"type": "prefill", "instance": "0.0.0.0:8100"}'
```

### Support functions

* Support adding prefill nodes or decode nodes at any time. 
  - If prefill or decode server has been deployed, proxy can add nodes when the proxy is deployed.
  - If prefill or decode server deployed after the proxy deployed, server can use `/instances/add` API to join the proxy server. The prefill server or decode server sends a signal to the proxy server, and the proxy server will check the status of the node util the node is available.
* Support removing nodes when the prefill or decode server failed more than a certain number of times.
* Support elastic scaling. When the current node is unavailable, the proxy server will schedule to the next available node.