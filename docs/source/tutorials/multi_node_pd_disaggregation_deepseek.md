# Prefill-Decode Disaggregation Verification (DeepSeek)

## Getting Start

vLLM-Ascend now supports prefill-decode (PD) disaggregation with EP (Expert Parallel) options. This guide take one-by-one steps to verify these features with constrained resources.

Take the `DeepSeek-V3.2-Exp-W8A8` model as an example, use vllm-ascend v0.11.0rc0 (with vLLM v0.11.0) on 2 Atlas 800T A3 servers to deploy the "1P1D" architecture. Assume the ip of the prefiller server is 172.22.0.218, and the decoder servers are 172.22.0.155. On each server, use 16 NPUs to deploy one service instance.

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
# View NPU network configuration
cat /etc/hccn.conf
```

2. Get NPU IP Addresses

```bash
for i in {0..15}; do hccn_tool -i $i -ip -g;done
```

3. Cross-Node PING Test

```bash
# Execute on the target node (replace 'x.x.x.x' with actual npu ip address)
for i in {0..15}; do hccn_tool -i $i -ping -g address x.x.x.x;done
```

## Generate Ranktable

The rank table is a JSON file that specifies the mapping of Ascend NPU ranks to nodes. For more details please refer to the [vllm-ascend examples](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/README.md). Execute the following commands for reference.

```shell
# Run the script on each node synchronously
# The ip address and network card name can get from ifconfig for each node
cd vllm-ascend/examples/disaggregate_prefill_v1/
bash gen_ranktable.sh --ips 172.22.0.218 172.22.0.155 \
  --npus-per-node 16 --network-card-name enp23s0f3 --prefill-device-cnt 16 --decode-device-cnt 16
```

Rank table will generated at /vllm-workspace/vllm-ascend/examples/disaggregate_prefill_v1/ranktable.json

|Parameter  | meaning |
| --- | --- |
| --ips | Each node's local ip (prefiller nodes should be front of decoder nodes) |
| --npus-per-node | Each node's npu clips  |
| --network-card-name | The physical machines' NIC |
|--prefill-device-cnt  | Npu clips used for prefill |
|--decode-device-cnt |Npu clips used for decode |

## Prefiller / Decoder Deployment

We can run the following scripts to launch a server on the prefiller/decoder node respectively.

:::::{tab-set}

::::{tab-item} Prefiller node

```shell
#!/bin/bash

export DISAGGREGATED_PREFILL_RANK_TABLE_PATH="/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/ranktable.json"
nic_name="enp23s0f3"
local_ip="172.22.0.218"
export HCCL_IF_IP=$local_ip # node ip
export GLOO_SOCKET_IFNAME=$nic_name  # network card name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100

vllm serve /root/.cache/weights/dpsk_w8a8 \
--host 0.0.0.0 \
--port 8000 \
--tensor-parallel-size 16 \
--seed 1024 \
--quantization ascend \
--served-model-name deepseek_v3.2 \
--max-num-seqs 16 \
--max-model-len 32768 \
--max-num-batched-tokens 32768 \
--enable-expert-parallel \
--trust-remote-code \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.92 \
--additional-config '{"torchair_graph_config":{"enabled":true,"graph_batch_sizes":[16]}}'
--kv-transfer-config  \
  '{"kv_connector": "LLMDataDistCMgrConnector",
    "kv_buffer_device": "npu",
    "kv_role": "kv_producer",
    "kv_parallel_size": 1,
    "kv_port": "20001",
    "engine_id": "0",
    "kv_connector_module_path": "vllm_ascend.distributed.llmdatadist_c_mgr_connector"
  }'
```

::::

::::{tab-item} Decoder node

```shell
#!/bin/bash

export DISAGGREGATED_PREFILL_RANK_TABLE_PATH="/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/ranktable.json"
nic_name="enp23s0f3"
local_ip="172.22.0.155"
export HCCL_IF_IP=$local_ip # node ip
export GLOO_SOCKET_IFNAME=$nic_name  # network card name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100

vllm serve /root/.cache/weights/dpsk_w8a8 \
--host 0.0.0.0 \
--port 8000 \
--tensor-parallel-size 16 \
--seed 1024 \
--quantization ascend \
--served-model-name deepseek_v3.2 \
--max-num-seqs 16 \
--max-model-len 32768 \
--max-num-batched-tokens 32768 \
--enable-expert-parallel \
--trust-remote-code \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.9 \
--kv-transfer-config  \
  '{"kv_connector": "LLMDataDistCMgrConnector",
  "kv_buffer_device": "npu",
  "kv_role": "kv_consumer",
  "kv_parallel_size": 1,
  "kv_port": "20001",
  "engine_id": "0",
  "kv_connector_module_path": "vllm_ascend.distributed.llmdatadist_c_mgr_connector"
  }'
```

::::

:::::

## Example proxy for Deployment

Run a proxy server on the same node with prefiller service instance. You can get the proxy program in the repository's examples: [load\_balance\_proxy\_server\_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

```shell
python load_balance_proxy_server_example.py \
    --host 172.22.0.218 \
    --port 8080 \
    --prefiller-hosts 172.22.0.218 \
    --prefiller-port 8000 \
    --decoder-hosts 172.22.0.155 \
    --decoder-ports 8000
```

## Verification

Check service health using the proxy server endpoint.

```shell
curl http://172.22.0.218:8080/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek_v3.2",
        "prompt": "Who are you?",
        "max_tokens": 100,
        "temperature": 0
    }'
```
