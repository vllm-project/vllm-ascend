# Prefill-Decode Disaggregation Verification (Qwen)

## Getting Start

vLLM-Ascend now supports prefill-decode (PD) disaggregation with EP (Expert Parallel) options. This guide take one-by-one steps to verify these features with constrained resources.

Take the Qwen3-30B-A3B model as an example, use vllm-ascend v0.10.1rc1 (with vLLM v0.10.1.1) on 3 Atlas 800T A2 servers to deploy the "1P2D" architecture. Assume the ip of the prefiller server is 192.0.0.1, and the decoder servers are 192.0.0.2 (decoder 1) and 192.0.0.3 (decoder 2). On each server, use 2 NPUs to deploy one service instance.

## Verify Multi-Node Communication Environment

### Physical Layer Requirements

- The physical machines must be located on the same WLAN, with network connectivity.
- All NPUs must be interconnected. Intra-node connectivity is via HCCS, and inter-node connectivity is via RDMA.

### Verification Process

1. Single Node Verification:

Execute the following commands on each node in sequence. The results must all be `success` and the status must be `UP`:

```bash
# Check the remote switch ports
for i in {0..7}; do hccn_tool -i $i -lldp -g | grep Ifname; done
# Get the link status of the Ethernet ports (UP or DOWN)
for i in {0..7}; do hccn_tool -i $i -link -g ; done
# Check the network health status
for i in {0..7}; do hccn_tool -i $i -net_health -g ; done
# View the network detected IP configuration
for i in {0..7}; do hccn_tool -i $i -netdetect -g ; done
# View gateway configuration
for i in {0..7}; do hccn_tool -i $i -gateway -g ; done
# View NPU network configuration
cat /etc/hccn.conf
```

2. Get NPU IP Addresses

```bash
for i in {0..7}; do hccn_tool -i $i -ip -g;done
```

3. Cross-Node PING Test

```bash
# Execute on the target node (replace 'x.x.x.x' with actual npu ip address)
for i in {0..7}; do hccn_tool -i $i -ping -g address x.x.x.x;done
```

## Generate Ranktable

The rank table is a JSON file that specifies the mapping of Ascend NPU ranks to nodes. For more details please refer to the [vllm-ascend examples](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/README.md). Execute the following commands for reference.

```shell
cd vllm-ascend/examples/disaggregate_prefill_v1/
bash gen_ranktable.sh --ips <prefiller_node1_local_ip> <prefiller_node2_local_ip> <decoder_node1_local_ip> <decoder_node2_local_ip> \
  --npus-per-node  <npu_clips> --network-card-name <nic_name> --prefill-device-cnt <prefiller_npu_clips> --decode-device-cnt <decode_npu_clips> \
  [--local-device-ids <id_1>,<id_2>,<id_3>...]
```

Assume that we use device 0,1 on the prefiller server node and device 6,7 on both of the decoder server nodes. Take the following commands as an example. (`--local-device-ids` is necessary if you specify certain NPU devices on the local server.)

```shell
# On the prefiller node
cd vllm-ascend/examples/disaggregate_prefill_v1/
bash gen_ranktable.sh --ips 192.0.0.1 192.0.0.2 192.0.0.3 \
  --npus-per-node  2 --network-card-name eth0 --prefill-device-cnt 2 --decode-device-cnt 4 --local-device-ids 0,1

# On the decoder 1
cd vllm-ascend/examples/disaggregate_prefill_v1/
bash gen_ranktable.sh --ips 192.0.0.1 192.0.0.2 192.0.0.3 \
  --npus-per-node  2 --network-card-name eth0 --prefill-device-cnt 2 --decode-device-cnt 4 --local-device-ids 6,7

# On the decoder 2
cd vllm-ascend/examples/disaggregate_prefill_v1/
bash gen_ranktable.sh --ips 192.0.0.1 192.0.0.2 192.0.0.3 \
  --npus-per-node  2 --network-card-name eth0 --prefill-device-cnt 2 --decode-device-cnt 4 --local-device-ids 6,7
```

Rank table will generated at /vllm-workspace/vllm-ascend/examples/disaggregate_prefill_v1/ranktable.json

|Parameter  | meaning |
| --- | --- |
| --ips | Each node's local ip (prefiller nodes should be front of decoder nodes) |
| --npus-per-node | Each node's npu clips  |
| --network-card-name | The physical machines' NIC |
|--prefill-device-cnt  | Npu clips used for prefill |
|--decode-device-cnt |Npu clips used for decode |
|--local-device-ids |Optional. No need if using all devices on the local node. |

## Prefiller / Decoder Deployment

We can run the following scripts to launch a server on the prefiller/decoder node respectively.

:::::{tab-set}

::::{tab-item} Prefiller node

```shell
export HCCL_IF_IP=192.0.0.1 # node ip
export GLOO_SOCKET_IFNAME="eth0"  # network card name
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export DISAGGREGATED_PREFILL_RANK_TABLE_PATH="/path/to/your/generated/ranktable.json"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1

vllm serve /model/Qwen3-30B-A3B  \
  --host 0.0.0.0 \
  --port 13700 \
  --tensor-parallel-size 2 \
  --no-enable-prefix-caching \
  --seed 1024 \
  --served-model-name qwen3-moe \
  --max-model-len 6144  \
  --max-num-batched-tokens 6144  \
  --trust-remote-code \
  --gpu-memory-utilization 0.9  \
  --enable-expert-parallel \
  --kv-transfer-config  \
  '{"kv_connector": "LLMDataDistCMgrConnector",
    "kv_buffer_device": "npu",
    "kv_role": "kv_producer",
    "kv_parallel_size": 1,
    "kv_port": "20001",
    "engine_id": "0",
    "kv_connector_module_path": "vllm_ascend.distributed.llmdatadist_c_mgr_connector"
  }'  \
  --additional-config \
  '{"torchair_graph_config": {"enabled":false,              "enable_multistream_shared_expert":false}, "ascend_scheduler_config":{"enabled":true, "enable_chunked_prefill":false}}' \
  --enforce-eager
```

::::

::::{tab-item} Decoder node 1

```shell
export HCCL_IF_IP=192.0.0.2  # node ip
export GLOO_SOCKET_IFNAME="eth0"  # network card name
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export DISAGGREGATED_PREFILL_RANK_TABLE_PATH="/path/to/your/generated/ranktable.json"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1

vllm serve /model/Qwen3-30B-A3B  \
  --host 0.0.0.0 \
  --port 13700 \
  --no-enable-prefix-caching \
  --tensor-parallel-size 2 \
  --seed 1024 \
  --served-model-name qwen3-moe \
  --max-model-len 6144  \
  --max-num-batched-tokens 6144  \
  --trust-remote-code \
  --gpu-memory-utilization 0.9  \
  --enable-expert-parallel \
  --kv-transfer-config  \
  '{"kv_connector": "LLMDataDistCMgrConnector",
  "kv_buffer_device": "npu",
  "kv_role": "kv_consumer",
  "kv_parallel_size": 1,
  "kv_port": "20001",
  "engine_id": "0",
  "kv_connector_module_path": "vllm_ascend.distributed.llmdatadist_c_mgr_connector"
  }'  \
  --additional-config \
  '{"torchair_graph_config": {"enabled":false, "enable_multistream_shared_expert":false}, "ascend_scheduler_config":{"enabled":true, "enable_chunked_prefill":false}}'
```

::::

::::{tab-item} Decoder node 2

```shell
export HCCL_IF_IP=192.0.0.3  # node ip
export GLOO_SOCKET_IFNAME="eth0"  # network card name
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export DISAGGREGATED_PREFILL_RANK_TABLE_PATH="/path/to/your/generated/ranktable.json"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1

vllm serve /model/Qwen3-30B-A3B  \
  --host 0.0.0.0 \
  --port 13700 \
  --no-enable-prefix-caching \
  --tensor-parallel-size 2 \
  --seed 1024 \
  --served-model-name qwen3-moe \
  --max-model-len 6144  \
  --max-num-batched-tokens 6144  \
  --trust-remote-code \
  --gpu-memory-utilization 0.9  \
  --enable-expert-parallel \
  --kv-transfer-config  \
  '{"kv_connector": "LLMDataDistCMgrConnector",
  "kv_buffer_device": "npu",
  "kv_role": "kv_consumer",
  "kv_parallel_size": 1,
  "kv_port": "20001",
  "engine_id": "0",
  "kv_connector_module_path": "vllm_ascend.distributed.llmdatadist_c_mgr_connector"
  }'  \
  --additional-config \
  '{"torchair_graph_config": {"enabled":false, "enable_multistream_shared_expert":false}, "ascend_scheduler_config":{"enabled":true, "enable_chunked_prefill":false}}'
```

::::

:::::

## Example proxy for Deployment

Run a proxy server on the same node with prefiller service instance. You can get the proxy program in the repository's examples: [load\_balance\_proxy\_server\_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

```shell
python load_balance_proxy_server_example.py \
    --host 192.0.0.1 \
    --port 8080 \
    --prefiller-hosts 192.0.0.1 \
    --prefiller-port 13700 \
    --decoder-hosts 192.0.0.2 192.0.0.3 \
    --decoder-ports 13700 13700
```

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
