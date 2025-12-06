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

After the proxy is deployed:
```text
INFO: Started server process [xxxxx]
INFO: Waiting for application startup.
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:xxxx
```

### Support API routes
* `/v1/completions`: get completions request response.
* `/v1/chat/completions`: get chat request response.
* `/status`: get the supported prefill nodes and decode nodes list.
* `/instances/add`: add prefill nodes or decode nodes to the list.
* `/instances/remove`: remove prefill nodes or decode nodes from the list.

Examples:
#### get request response
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
```

#### get server status
```shell
# /status
curl -X POST http://0.0.0.0:8000/status
```
The response:
```text
{"prefill_node_count":x,"decode_node_count":x,"prefill_nodes":[xx.xx.xx.xx:xxxx],"decode_nodes":[xx.xx.xx.xx:xxxx]}
```

#### add nodes to the server
```shell
# /instance/add
curl -X POST http://0.0.0.0:8000/instances/add \
-H "Content-Type: application/json" \
-H "X-Api-Key: YOUR_ADMIN_API_KEY" \
-d '{"type": "prefill", "instance": "0.0.0.0:8100"}'
```
* Case 1: If the node is not available, the server will waiting for the node to be available:
```text
INFO: Verifying xx.xx.xx.xx:xxxx ...
ERROR: Cannot connect to host xx.xx.xx.xx:xxxx ...
INFO: Waiting for prefill_instance xx.xx.xx.xx:xxxx to start.
INFO: Verifying xx.xx.xx.xx:xxxx ...
...
```
The response:
```text
{"message":"Waiting for prefill_instance xx.xx.xx.xx:xxxx to start."}
```
* Case 2: If the node is available, try to add the node to the server:
```text
INFO: Verifying xx.xx.xx.xx:xxxx ...
INFO: Instance: xx.xx.xx.xx:xxxx could be added.
INFO: Added xx.xx.xx.xx:xxxx to prefill_instances. prefill node counts: x, decode node counts: x
```
If the node has been added to the server before:
```text
INFO: prefill_instance xx.xx.xx.xx:xxxx already exists.
```
The response:
```text
{"message":"Added xx.xx.xx.xx:xxxx to prefill_instances."}
```

#### remove nodes from the server
```shell
# /instance/remove
curl -X POST http://0.0.0.0:8000/instances/remove \
-H "Content-Type: application/json" \
-H "X-Api-Key: YOUR_ADMIN_API_KEY" \
-d '{"type": "prefill", "instance": "0.0.0.0:8100"}'
```
After the node is removed:
```text
INFO: Removed xx.xx.xx.xx:xxxx from prefill_instances. prefill node counts: x, decode node counts: x
```
The response:
```text
{"message":"Removed xx.xx.xx.xx:xxxx from prefill_instances."}
```
If the node is not in the corresponding nodes list:
```text
{"message": f"Instance xx.xx.xx.xx:xxxx is not in the prefill_instances."}
```

### Support functions

* Support adding prefill nodes or decode nodes at any time. 
  - If prefill or decode server has been deployed, proxy can add nodes when the proxy is deployed.
  - If prefill or decode server deployed after the proxy deployed, server can use `/instances/add` API to join the proxy server. The prefill server or decode server sends a signal to the proxy server, and the proxy server will check the status of the node util the node is available.
* Support removing nodes for the following two situations:
  - Support removing nodes when the prefill or decode server failed more than a certain number of times.
  - Support using `/instances/remove` API to delete the node from the proxy server.
* Support elastic scaling. 
  - When the current node is unavailable, the proxy server will schedule to the next available node.