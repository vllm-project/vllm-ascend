# Prefill-Decode Disaggregation Llmdatadist Verification (Qwen2.5-VL)

## Getting Start

vLLM-Ascend now supports prefill-decode (PD) disaggregation. This guide takes one-by-one steps to verify these features with constrained resources.

Using the Qwen2.5-VL-7B-Instruct model as an example, use vllm-ascend v0.11.0rc1 (with vLLM v0.11.0) on 1 Atlas 800T A2 server to deploy the "1P1D" architecture. Assume the IP address is 192.0.0.1.

## Verify Communication Environment

### Verification Process

1. Single Node Verification:

Execute the following commands in sequence. The results must all be `success` and the status must be `UP`:

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

## Generate Ranktable

The rank table is a JSON file that specifies the mapping of Ascend NPU ranks to nodes. For more details, please refer to the [vllm-ascend examples](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/README.md). Execute the following commands for reference.

```shell
cd vllm-ascend/examples/disaggregate_prefill_v1/
bash gen_ranktable.sh --ips 192.0.0.1 \
  --npus-per-node  2 --network-card-name eth0 --prefill-device-cnt 1 --decode-device-cnt 1
```

If you want to run "2P1D", please set npus-per-node to 3 and prefill-device-cnt to 2. The rank table will be generated at /vllm-workspace/vllm-ascend/examples/disaggregate_prefill_v1/ranktable.json

|Parameter  | Meaning |
| --- | --- |
| --ips | Each node's local IP address (prefiller nodes should be in front of decoder nodes) |
| --npus-per-node | Each node's NPU clips |
| --network-card-name | The physical machines' NIC |
|--prefill-device-cnt  | NPU clips used for prefill |
|--decode-device-cnt |NPU clips used for decode |

## Prefiller/Decoder Deployment

We can run the following scripts to launch a server on the prefiller/decoder NPU, respectively.

:::::{tab-set}

::::{tab-item} Prefiller

```shell
export ASCEND_RT_VISIBLE_DEVICES=0
export HCCL_IF_IP=192.0.0.1 # node ip
export GLOO_SOCKET_IFNAME="eth0"  # network card name
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export DISAGGREGATED_PREFILL_RANK_TABLE_PATH="/path/to/your/generated/ranktable.json"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_ASCEND_LLMDD_RPC_PORT=5959

vllm serve /model/Qwen2.5-VL-7B-Instruct  \
  --host 0.0.0.0 \
  --port 13700 \
  --tensor-parallel-size 1 \
  --no-enable-prefix-caching \
  --seed 1024 \
  --served-model-name qwen25vl \
  --max-model-len 40000  \
  --max-num-batched-tokens 40000  \
  --trust-remote-code \
  --gpu-memory-utilization 0.9  \
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

::::{tab-item} Decoder

```shell
export ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_IF_IP=192.0.0.1  # node ip
export GLOO_SOCKET_IFNAME="eth0"  # network card name
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export DISAGGREGATED_PREFILL_RANK_TABLE_PATH="/path/to/your/generated/ranktable.json"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_ASCEND_LLMDD_RPC_PORT=5979

vllm serve /model/Qwen2.5-VL-7B-Instruct  \
  --host 0.0.0.0 \
  --port 13701 \
  --no-enable-prefix-caching \
  --tensor-parallel-size 1 \
  --seed 1024 \
  --served-model-name qwen25vl \
  --max-model-len 40000  \
  --max-num-batched-tokens 40000  \
  --trust-remote-code \
  --gpu-memory-utilization 0.9  \
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

If you want to run "2P1D", please set ASCEND_RT_VISIBLE_DEVICES, VLLM_ASCEND_LLMDD_RPC_PORT and port to different values for each P process.

## Example Proxy for Deployment

Run a proxy server on the same node with the prefiller service instance. You can get the proxy program in the repository's examples: [load\_balance\_proxy\_server\_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

```shell
python load_balance_proxy_server_example.py \
    --host 192.0.0.1 \
    --port 8080 \
    --prefiller-hosts 192.0.0.1 \
    --prefiller-port 13700 \
    --decoder-hosts 192.0.0.1 \
    --decoder-ports 13701
```

|Parameter  | Meaning |
| --- | --- |
| --port | Port of proxy |
| --prefiller-port | All ports of prefill |
| --decoder-ports | All ports of decoder |

## Verification

Check service health using the proxy server endpoint.

```shell
curl http://192.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen25vl",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
                {"type": "text", "text": "What is the text in the illustrate?"}
            ]}
            ],
        "max_tokens": 100,
        "temperature": 0
    }'
```
