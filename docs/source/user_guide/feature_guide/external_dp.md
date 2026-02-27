# External DP

For larger-scale deployments especially, it can make sense to handle the orchestration and load balancing of data parallel ranks externally.

In this case, it's more convenient to treat each DP rank like a separate vLLM deployment, with its own endpoint, and have an external router balance HTTP requests between them, making use of appropriate real-time telemetry from each server for routing decisions.

## Getting Start

The functionality of [external DP](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/?h=external#external-load-balancing) is already natively supported by vLLM. In vllm-ascend we provide two enhanced functionalities:

1. A launch script that helps to launch multiple vLLM instances in one command.
2. A request-length-aware load-balance proxy for external DP.

This tutorial will introduce the usage of them.

### Prerequisites

- Python 3.10+
- Install dependencies needed by load-balance proxy server:

```shell
pip install fastapi httpx uvicorn
```

## Starting External DP Servers

First, you need to have at least two vLLM servers running in data parallel. These can be mock servers or actual vLLM servers. Note that this proxy also works with only one vLLM server running, but will fall back to direct request forwarding which is meaningless.

You can start external vLLM DP servers one-by-one manually or using the launch script in `examples/external_online_dp`. For scenarios of large DP size across multiple nodes, we recommend using our launch script for convenience.

### Manually Launch

```shell
# This example shows how to manually launch a vLLM service with DP size 2 in one node.
vllm serve --host 0.0.0.0 --port 8100 --data-parallel-size 2 --data-parallel-rank 0 ... # vLLM DP0
vllm serve --host 0.0.0.0 --port 8101 --data-parallel-size 2 --data-parallel-rank 1 ... # vLLM DP1
```

### Use Launch Script

Firstly, you need to modify the `examples/external_online_dp/run_dp_template.sh` according to your vLLM configuration. Then you can use `examples/external_online_dp/launch_online_dp.py` to launch multiple vLLM instances in one command on each node. It will internally call `examples/external_online_dp/run_dp_template.sh` for each DP rank with proper DP-related parameters.

An example of running external DP in one single node:

```shell
cd examples/external_online_dp
# running DP4 TP4 in a node with 16 NPUs
python launch_online_dp.py --dp-size 4 --tp-size 4 --dp-size-local 4 --dp-rank-start 0 --dp-address x.x.x.x --dp-rpc-port 12342
```

An example of running external DP in two nodes:

```shell
cd examples/external_online_dp
# running DP4 TP4 in two nodes with 8 NPUs each
# Node 0 holds DP0 DP1 and node 1 holds DP2 DP3
# Here x.x.x.x:12342 is served as the common data parallel RPC address

# On node 0:
python launch_online_dp.py --dp-size 4 --tp-size 4 --dp-size-local 2 --dp-rank-start 0 --dp-address x.x.x.x --dp-rpc-port 12342

# On node 1:
python launch_online_dp.py --dp-size 4 --tp-size 4 --dp-size-local 2 --dp-rank-start 2 --dp-address x.x.x.x --dp-rpc-port 12342
```

## Starting Load-balance Proxy Server

After all vLLM DP instances are launched, you can now launch the load-balance proxy server, which serves as an entrypoint for coming requests and load-balances them between vLLM DP instances.

The proxy server has the following features:

- Load balances requests to multiple vLLM servers based on request length.
- Supports OpenAI-compatible `/v1/completions` and `/v1/chat/completions` endpoints.
- Streams responses from backend servers to clients.

To run the proxy server, you need to specify the host and port for each vLLM DP Instance:

```shell
# For example, we have already started two DP instances in single node:
# python launch_online_dp.py --dp-size 2 --tp-size 8 --dp-size-local 2 --dp-rank-start 0 --dp-address x.x.x.x --dp-rpc-port 12342
# By default, launch_online_dp.py will launch vLLM instances from starting port 9000,
# so the vLLM ports for DP0 and DP1 are 9000 and 9001 separately.
# Then you can start the load-balance proxy server by:
cd examples/external_online_dp
python dp_load_balance_proxy_server.py \
    --host 0.0.0.0 --port 8000 \
    --dp-hosts 127.0.0.1 127.0.0.1 \
    --dp-ports 9000 9001 \
```

After this, you can directly send requests to the proxy server and run DP with external load balancing.
