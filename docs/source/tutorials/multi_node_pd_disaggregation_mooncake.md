# Prefill-Decode Disaggregation Mooncake Verification (Qwen)

## Getting Start

vLLM-Ascend now supports prefill-decode (PD) disaggregation with EP (Expert Parallel) options. This guide takes one-by-one steps to verify these features with constrained resources.

Take the Qwen3-235B model as an example, use 4 Atlas 800T A3 servers to deploy the "2P1D" architecture. Assume the ip of the prefiller server is 192.0.0.1 (prefill 1) and 192.0.0.2 (prefill 2), and the decoder servers are 192.0.0.3 (decoder 1) and 192.0.0.4 (decoder 2). On each server, use 8 NPUs 16 chips to deploy one service instance.

## Verify Multi-Node Communication Environment

### Physical Layer Requirements

- The physical machines must be located on the same WLAN, with network connectivity.
- All NPUs must be interconnected. Intra-node connectivity is via HCCS, and inter-node connectivity is via RDMA.

### Verification Process

1. Single Node Verification:

Execute the following commands on each node in sequence. The results must all be `success` and the status must be `UP`:

```bash
# Check the remote switch ports
for i in {0..15}; do hccn_tool -i $i -lldp -g | grep Ifname; done 
# Get the link status of the Ethernet ports (UP or DOWN)
for i in {0..15}; do hccn_tool -i $i -link -g ; done
# Check the network health status
for i in {0..15}; do hccn_tool -i $i -net_health -g ; done
# View the network detected IP configuration
for i in {0..15}; do hccn_tool -i $i -netdetect -g ; done
# View gateway configuration
for i in {0..15}; do hccn_tool -i $i -gateway -g ; done
```

2. Check NPU network configuration:

```bash
# Ensure that the hccn.conf file exists in the environment. If using Docker, mount it into the container.
cat /etc/hccn.conf
```

3. Get NPU IP Addresses

```bash
for i in {0..15}; do hccn_tool -i $i -ip -g | grep ipaddr; done
```

4. Cross-Node PING Test

```bash
# Execute on the target node (replace 'x.x.x.x' with actual npu ip address)
for i in {0..15}; do hccn_tool -i $i -ping -g address x.x.x.x;done
```

## Install Mooncake

Mooncake is the serving platform for Kimi, a leading LLM service provided by Moonshot AI. First, we need to obtain the Mooncake project. Refer to the following command:

```shell
git clone -b pooling_async_memecpy_v1 https://github.com/AscendTransport/Mooncake
```

Update and install Python.

```shell
apt-get update
apt-get install python3
```

Install the relevant dependencies. The installation of Go is not required.

```shell
cd Mooncake
bash dependencies.sh -y
```

Install mpi

```shell
apt purge mpich libmpich-dev -y
apt purge openmpi-bin -y
apt purge openmpi-bin libopenmpi-dev -y
apt install mpich libmpich-dev -y
export CPATH=/usr/lib/aarch64-linux-gnu/mpich/include/:$CPATH
export CPATH=/usr/lib/aarch64-linux-gnu/openmpi/lib:$CPATH
```

Compile and install

```shell
mkdir build
cd build
cmake ..
make -j
make install
cp mooncake-transfer-engine/src/transport/ascend_transport/hccl_transport/ascend_transport_c/libascend_transport_mem.so /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/
cp mooncake-transfer-engine/src/libtransfer_engine.so /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/
```

## Prefiller/Decoder Deployment

We can run the following scripts to launch a server on the prefiller/decoder node, respectively. Please note that each P/D node will occupy ports ranging from kv_port to kv_port + num_chips to initialize socket listeners. To avoid any issues, port conflicts should be prevented. Additionally, ensure that each node's engine_id is uniquely assigned to avoid conflicts.

### Layerwise

:::::{tab-set}

::::{tab-item} Prefiller node 1

```shell
unset ftp_proxy
unset https_proxy
unset http_proxy
export HCCL_IF_IP=192.0.0.1
export GLOO_SOCKET_IFNAME="eth0"  # network card name
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export ASCEND_AGGREGATE_ENABLE=1  # enable aggregated transmission
export ASCEND_TRANSPORT_PRINT=0  # print ascend transport logs
export ACL_OP_INIT_MODE=1  # acl op initialization mode to prevent device id acquisition failure
export ASCEND_A3_ENABLE=1  # enable hccs transmission for A3; set to 0 for A2
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH

vllm serve /model/Qwen3-235B-A22B-W8A8 \
  --host 0.0.0.0 \
  --port 8004 \
  --api-server-count 1 \
  --data-parallel-size 2 \
  --data-parallel-size-local 2 \
  --data-parallel-address 192.0.0.1 \
  --data-parallel-rpc-port 13389 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --seed 1024 \
  --enforce-eager \
  --distributed-executor-backend mp \
  --served-model-name qwen3-moe \
  --max-model-len 32768 \
  --max-num-batched-tokens 32768 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeLayerwiseConnector",
  "kv_role": "kv_producer",
  "kv_port": "30000",
  "engine_id": "0",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_layerwise_connector",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 32,
                    "tp_size": 1
             }
      }
  }'
```

::::

::::{tab-item} Prefiller node 2

```shell
unset ftp_proxy
unset https_proxy
unset http_proxy
export HCCL_IF_IP=192.0.0.2
export GLOO_SOCKET_IFNAME="eth0"  # network card name
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=0
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH

vllm serve /model/Qwen3-235B-A22B-W8A8 \
  --host 0.0.0.0 \
  --port 8004 \
  --api-server-count 1 \
  --data-parallel-size 2 \
  --data-parallel-size-local 2 \
  --data-parallel-address 192.0.0.2 \
  --data-parallel-rpc-port 13389 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --seed 1024 \
  --enforce-eager \
  --distributed-executor-backend mp \
  --served-model-name qwen3-moe \
  --max-model-len 32768 \
  --max-num-batched-tokens 32768 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeLayerwiseConnector",
  "kv_role": "kv_producer",
  "kv_port": "30100",
  "engine_id": "1",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_layerwise_connector",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 32,
                    "tp_size": 1
             }
      }
  }'
```

::::

::::{tab-item} Decoder node 1 (master node)

```shell
unset ftp_proxy
unset https_proxy
unset http_proxy
export HCCL_IF_IP=192.0.0.3
export GLOO_SOCKET_IFNAME="eth0"  # network card name
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=2048
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=0
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH

vllm serve /model/Qwen3-235B-A22B-W8A8 \
  --host 0.0.0.0 \
  --port 8004 \
  --api-server-count 1 \
  --data-parallel-size 32 \
  --data-parallel-size-local 16 \
  --data-parallel-address 192.0.0.3 \
  --data-parallel-rpc-port 5964  \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --seed 1024 \
  --distributed-executor-backend mp \
  --served-model-name qwen3-moe \
  --max-model-len 32768 \
  --max-num-batched-tokens 512 \
  --max-num_seqs 16 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.9 \
  --compilation-config '{"cudagraph_capture_sizes":[16]}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeLayerwiseConnector",
  "kv_role": "kv_consumer",
  "kv_port": "30200",
  "engine_id": "2",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_layerwise_connector",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 32,
                    "tp_size": 1
             }
      }
  }'
```

::::

::::{tab-item} Decoder node 2 (primary node)

```shell
unset ftp_proxy
unset https_proxy
unset http_proxy
export HCCL_IF_IP=192.0.0.4
export GLOO_SOCKET_IFNAME="eth0"  # network card name
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=2048
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=0
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH

vllm serve /model/Qwen3-235B-A22B-W8A8 \
  --host 0.0.0.0 \
  --port 8004 \
  --headless \
  --data-parallel-size 32 \
  --data-parallel-size-local 16 \
  --data-parallel-start-rank 16 \
  --data-parallel-address 192.0.0.3 \
  --data-parallel-rpc-port 5964  \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --seed 1024 \
  --distributed-executor-backend mp \
  --served-model-name qwen3-moe \
  --max-model-len 32768 \
  --max-num-batched-tokens 512 \
  --max-num_seqs 16 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.9 \
  --compilation-config '{"cudagraph_capture_sizes":[16]}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeLayerwiseConnector",
  "kv_role": "kv_consumer",
  "kv_port": "30200",
  "engine_id": "2",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_layerwise_connector",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 32,
                    "tp_size": 1
             }
      }
  }'
```

::::

:::::

### Non-layerwise

:::::{tab-set}

::::{tab-item} Prefiller node 1

```shell
unset ftp_proxy
unset https_proxy
unset http_proxy
export HCCL_IF_IP=192.0.0.1
export GLOO_SOCKET_IFNAME="eth0"  # network card name
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=0
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH

vllm serve /model/Qwen3-235B-A22B-W8A8 \
  --host 0.0.0.0 \
  --port 8004 \
  --api-server-count 1 \
  --data-parallel-size 2 \
  --data-parallel-size-local 2 \
  --data-parallel-address 192.0.0.1 \
  --data-parallel-rpc-port 13389 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --seed 1024 \
  --enforce-eager \
  --distributed-executor-backend mp \
  --served-model-name qwen3-moe \
  --max-model-len 32768 \
  --max-num-batched-tokens 32768 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnector",
  "kv_role": "kv_producer",
  "kv_port": "30000",
  "engine_id": "0",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 32,
                    "tp_size": 1
             }
      }
  }'
```

::::

::::{tab-item} Prefiller node 2

```shell
unset ftp_proxy
unset https_proxy
unset http_proxy
export HCCL_IF_IP=192.0.0.2
export GLOO_SOCKET_IFNAME="eth0"  # network card name
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=0
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH

vllm serve /model/Qwen3-235B-A22B-W8A8 \
  --host 0.0.0.0 \
  --port 8004 \
  --api-server-count 1 \
  --data-parallel-size 2 \
  --data-parallel-size-local 2 \
  --data-parallel-address 192.0.0.2 \
  --data-parallel-rpc-port 13389 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --seed 1024 \
  --enforce-eager \
  --distributed-executor-backend mp \
  --served-model-name qwen3-moe \
  --max-model-len 32768 \
  --max-num-batched-tokens 32768 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnector",
  "kv_role": "kv_producer",
  "kv_port": "30100",
  "engine_id": "1",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 32,
                    "tp_size": 1
             }
      }
  }'
```

::::

::::{tab-item} Decoder node 1 (master Node)

```shell
unset ftp_proxy
unset https_proxy
unset http_proxy
export HCCL_IF_IP=192.0.0.3
export GLOO_SOCKET_IFNAME="eth0"  # network card name
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=2048
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=0
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH

vllm serve /model/Qwen3-235B-A22B-W8A8 \
  --host 0.0.0.0 \
  --port 8004 \
  --api-server-count 1 \
  --data-parallel-size 32 \
  --data-parallel-size-local 16 \
  --data-parallel-address 192.0.0.3 \
  --data-parallel-rpc-port 5964  \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --seed 1024 \
  --distributed-executor-backend mp \
  --served-model-name qwen3-moe \
  --max-model-len 32768 \
  --max-num-batched-tokens 512 \
  --max-num_seqs 16 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.9 \
  --compilation-config '{"cudagraph_capture_sizes":[16]}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnector",
  "kv_role": "kv_consumer",
  "kv_port": "30200",
  "engine_id": "2",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 32,
                    "tp_size": 1
             }
      }
  }'
```

::::

::::{tab-item} Decoder node 2 (primary Node)

```shell
unset ftp_proxy
unset https_proxy
unset http_proxy
export HCCL_IF_IP=192.0.0.4
export GLOO_SOCKET_IFNAME="eth0"  # network card name
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=2048
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=0
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH

vllm serve /model/Qwen3-235B-A22B-W8A8 \
  --host 0.0.0.0 \
  --port 8004 \
  --headless \
  --data-parallel-size 32 \
  --data-parallel-size-local 16 \
  --data-parallel-start-rank 16 \
  --data-parallel-address 192.0.0.3 \
  --data-parallel-rpc-port 5964  \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --seed 1024 \
  --distributed-executor-backend mp \
  --served-model-name qwen3-moe \
  --max-model-len 32768 \
  --max-num-batched-tokens 512 \
  --max-num_seqs 16 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.9 \
  --compilation-config '{"cudagraph_capture_sizes":[16]}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnector",
  "kv_role": "kv_consumer",
  "kv_port": "30200",
  "engine_id": "2",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 32,
                    "tp_size": 1
             }
      }
  }'
```

::::

:::::

## Example Proxy for Deployment

Run a proxy server on the same node with the prefiller service instance. You can get the proxy program in the repository's examples: [load\_balance\_proxy\_layerwise\_server\_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py) or [load\_balance\_proxy\_server\_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

:::::{tab-set}

::::{tab-item} Layerwise

```shell
python load_balance_proxy_layerwise_server_example.py \
    --host 192.0.0.1 \
    --port 8080 \
    --prefiller-hosts 192.0.0.1 192.0.0.2\
    --prefiller-port 8004 8004\
    --decoder-hosts 192.0.0.3\
    --decoder-ports 8004
```

::::

::::{tab-item} Non-layerwise

```shell
python load_balance_proxy_server_example.py \
    --host 192.0.0.1 \
    --port 8080 \
    --prefiller-hosts 192.0.0.1 192.0.0.2\
    --prefiller-port 8004 8004\
    --decoder-hosts 192.0.0.3\
    --decoder-ports 8004
```

::::

:::::

## Verification

Check service health using the proxy server endpoint.

```shell
curl http://192.0.0.1:8080/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3-moe",
        "prompt": "Who are you?",
        "max_tokens": 100,
        "temperature": 0
    }'
```
