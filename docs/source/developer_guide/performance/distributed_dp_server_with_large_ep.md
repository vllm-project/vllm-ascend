# Distributed DP Server With Large EP (DeepSeek)

## Getting Start

vLLM-Ascend now supports prefill-decode (PD) disaggregation in the large **Expert  Parallelism (EP)** scenario. To achieve better performance，the distributed DP server is applied in vLLM-Ascend. In the PD separation scenario, different optimization strategies can be implemented based on the distinct characteristics of PD nodes, thereby enabling more flexible model deployment.

## Verify Multi-Node Communication Environment

### Physical Layer Requirements:

- The physical machines must be located on the same WLAN, with network connectivity.
- All NPUs are connected with optical modules, and the connection status must be normal.

### Verification Process:

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

### NPU Interconnect Verification:

#### 1. Get NPU IP Addresses

```bash
for i in {0..15}; do hccn_tool -i $i -vnic -g;done
```

#### 2. Get superpodid and SDID

```bash
for i in {0..7}; do npu-smi info -t spod-info -i $i -c 0;npu-smi info -t spod-info -i $i -c 1;done
```

#### 3. Cross-Node PING Test

```bash
# Execute on the target node (replace with actual IP)
for i in {0..15}; do hccn_tool -i $i -hccs_ping -g address x.x.x.x;done
```

## Generate Ranktable

You need to generate a ranktable to make  mulit nodes to communicate with each other. For more details please refer to the [vllm-ascend examples](https://github.com/vllm-project/vllm-ascend/blob/v0.9.1-dev/examples/disaggregate_prefill_v1/README.md). Execute the following commands for reference.

```shell
cd vllm-ascend/examples/disaggregate_prefill_v1/
bash gen_ranktable.sh --ips prefiller_node1_local_ip prefiller_node2_local_ip decoder_node1_local_ip decoder_node2_local_ip \
  --npus-per-node  npu_clips --network-card-name nic_name --prefill-device-cnt prefiller_npu_clips --decode-device-cnt decode_npu_clips
```

|Parameter  | meaning |
| --- | --- |
| --ips | Each node's local ip (prefiller nodes should be front of decoder nodes) |
| --npus-per-node | Each node's npu clips  |
| --network-card-name | The physical machines' NIC |
|--prefill-device-cnt  | Npu clips used for prefill |
|--decode-device-cnt |Npu clips used for decode |

## Use the Distributed DP Server

Execute the following commands to use the distributed DP server. (We recommend using this feature on the v0.9.1-dev branch)

```python
import multiprocessing
import os
import sys
dp_size = "total number of DP workers for decode/prefill"
dp_size_local = "number of DP workers on the current node"
dp_rank_start = "starting DP rank for the current node"
dp_ip = "master node ip"
dp_port = "port used for communication"
engine_port = "the starting port for all DP groups on the current node"
template_path = "./run_dp_template.sh"
if not os.path.exists(template_path):
  print(f"Template file {template_path} does not exist.")
  sys.exit(1)
def run_command(dp_rank_local, dp_rank, engine_port_):
  command = f"bash ./run_dp_template.sh {dp_size} {dp_ip} {dp_port} {dp_rank_local} {dp_rank} {engine_port_} {dp_size_local}"
  os.system(command)
processes = []
for i in range(dp_size_local):
  dp_rank = dp_rank_start + i
  dp_rank_local = i
  engine_port_ = engine_port + i
  process = multiprocessing.Process(target=run_command, args=(dp_rank_local, dp_rank, engine_port_))
  processes.append(process)
  process.start()
for process in processes:
  process.join()
```

Note that the prefiller nodes and the decoder nodes may have differenet configurations. You can use the following shell script for configuring the prefiller and decoder nodes respectively.

```shell
# run_dp_template.sh
#!/bin/sh

# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip
nic_name="xxxx"
local_ip="xxxx"

# basic configuration for HCCL and connection
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export HCCL_BUFFSIZE=256
export DISAGGREGATED_PREFILL_RANK_TABLE_PATH='ranktable you generate'

# obtain parameters from distributed DP server
export VLLM_DP_SIZE=$1
export VLLM_DP_MASTER_IP=$2
export VLLM_DP_MASTER_PORT=$3
export VLLM_DP_RANK_LOCAL=$4
export VLLM_DP_RANK=$5
export VLLM_DP_SIZE_LOCAL=$7

#pytorch_npu settings and vllm settings
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_USE_V1=1

# enable the distributed DP server 
export VLLM_WORKER_MULTIPROC_METHOD="fork"
export VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED=1

# The w8a8 weight can obtained from https://www.modelscope.cn/models/vllm-ascend/DeepSeek-R1-W8A8
# "--additional-config" is used to enable characteristics from vllm-ascend
vllm serve /root/.cache/ds_r1 \
    --host 0.0.0.0 \
    --port $6 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name deepseek_r1 \
    --max-model-len 17000 \
    --max-num-batched-tokens 16384 \
    --trust-remote-code \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.9 \
    --quantization ascend \
    --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
    --kv-transfer-config \
        '{"kv_connector": "LLMDataDistCMgrConnector",
        "kv_buffer_device": "npu",
        "kv_role": "kv_consumer",
        "kv_parallel_size": "1",
        "kv_port": "20001",
        "engine_id": "0",
        "kv_connector_module_path": "vllm_ascend.distributed.llmdatadist_c_mgr_connector"
        }' \
    --additional-config '{"ascend_scheduler_config":{"enabled":true},"torchair_graph_config":{"enabled":true}}'
```

In the PD separation scenario, we provide a recommended optimized configuration.

- **prefiller node**

1. set HCCL_BUFFSIZE=256
2. add '--enforce-eager' command to 'vllm serve'
3. Take '--additional-config' as follow

```shell
--additional-config '{"ascend_scheduler_config":{"enabled":false}, "torchair_graph_config":{"enabled":false},"enable_weight_nz_layout":true,"enable_prefill_optimizations":true}'
```

- **decoder node**

1. set HCCL_BUFFSIZE=1024
2. Take '--additional-config' as follow

```shell
--additional-config '{"ascend_scheduler_config":{"enabled":false}, "torchair_graph_config":{"enabled":true,"enable_multistream_mla":true,"enable_multistream_moe":true,"graph_batch_sizes":[28], "enable_super_kernel":true, "use_cached_graph":true},"enable_weight_nz_layout":true}'
```

<br>

'--additional-config'  Parameter Introduction:

- **"torchair_graph_config"：** The config options for torchair graph mode.
- **"ascend_scheduler_config"：** The config options for ascend scheduler.
- **"enable_weight_nz_layout"：** Whether to convert quantized weights to NZ format to accelerate matrix multiplication.
- **"enable_prefill_optimizations"：** Whether to enable DeepSeek models' prefill optimizations.
  <br>

"torchair_graph_config" Parameter Introduction:

- **"enable_multistream_mla"：** Whether to put vector ops of MLA to another stream. This option only takes effects on models using MLA.
- **"enable_multistream_moe"：** Whether to enable multistream shared expert. This option only takes effects on DeepSeek moe models.
- **"graph_batch_sizes"：**  The batch size for torchair graph cache.
- **"enable_super_kernel"：** Whether to enable super kernel.
- **"use_cached_graph"：** Whether to use cached graph

## Toy proxy for Distributed DP Server

In the PD separation scenario, we need a proxy to distribute requests. Execute the following commands to enable the toy proxy:

```shell
python load_balance_proxy_server_example.py \
  --port "proxy port" \
  --host 0.0.0.0 \
  --prefiller-hosts \
    prefiller node1 local ip \
    prefiller node2 local ip \
  --prefiller-ports  \
    engine_port engine_port \
  --decoder-hosts \
    decoder node1 local ip  \
    decoder node1 local ip  \
    decoder node2 local ip  \
    decoder node2 local ip  \
  --decoder-ports  \
    engine_port ...  \ # Increase by dp_size_local e.g. 9000 9001
    engine_port ...  \ # Increase by dp_size_local e.g. 9000 9001
```

:::{note}
Each node local ip should repeat the same times as its '**dp_size_local**', at the same time, each node has the same number of ports as '**dp_size_local**', and their ports increase sequentially starting from '**engine_port**'.
:::

You can get the proxy program in the repository's examples, [load\_balance\_proxy\_server\_example.py](https://github.com/vllm-project/vllm-ascend/blob/v0.9.1-dev/examples/disaggregate_prefill_v1/load_balance_proxy_server_example.py)

## Recommended Configuration

For example，if the average input length is 3.5k, and the output length is 1.1k, the context length is 16k, the max length of the input dataset is 7K. In this scenario, we give a recommended configuration for distributed DP server with high EP. Here we use 4 nodes for prefill and 4 nodes for decode.
<br>

| node     | DP | TP | EP | max-model-len | max-num-batched-tokens | max-num-seqs |  gpu-memory-utilization |
|----------|----|----|----|---------------|------------------------|--------------|-----------|
| prefill  | 2  |  8 | 16 |     17000     |         16384          |      4       |    0.9    |
| decode   | 64 |  1 | 64 |     17000     |          256           |      28      |    0.9    |

## FAQ

### 1. Prefiller nodes need to warmup

Since the computation of some NPU operators requires several rounds of warm-up to achieve best performance, we recommend preheating the service with some requests before conducting performance tests to achieve the best end-to-end throughput.
