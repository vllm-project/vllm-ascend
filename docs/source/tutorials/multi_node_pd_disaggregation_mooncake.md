# Prefill-Decode Disaggregation Mooncake Verification (Deepseek)

## Getting Start

vLLM-Ascend now supports prefill-decode (PD) disaggregation with EP (Expert Parallel) options. This guide take one-by-one steps to verify these features with constrained resources.

Take the Deepseek-r1-w8a8 model as an example, use 4 Atlas 800T A3 servers to deploy the "2P1D" architecture. Assume the ip of the prefiller server is 192.0.0.1 (prefill 1) and 192.0.0.2 (prefill 2), and the decoder servers are 192.0.0.3 (decoder 1) and 192.0.0.4 (decoder 2). On each server, use 8 NPUs 16 chips to deploy one service instance.

## Verify Multi-Node Communication Environment

### Physical Layer Requirements

- The physical machines must be located on the same WLAN, with network connectivity.
- All NPUs must be interconnected. Intra-node connectivity is via HCCS, and inter-node connectivity is via RDMA.

### Verification Process


Execute the following commands on each node in sequence. The results must all be `success` and the status must be `UP`:

:::::{tab-set}
::::{tab-item} A3

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
Ensure that the hccn.conf file exists in the environment. If using Docker, mount it into the container.

```bash
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

::::

::::{tab-item} A2

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
```

2. Check NPU network configuration:
Ensure that the hccn.conf file exists in the environment. If using Docker, mount it into the container.

```bash
cat /etc/hccn.conf
```

3. Get NPU IP Addresses

```bash
for i in {0..7}; do hccn_tool -i $i -ip -g;done
```

4. Cross-Node PING Test

```bash
# Execute on the target node (replace 'x.x.x.x' with actual npu ip address)
for i in {0..7}; do hccn_tool -i $i -ping -g address x.x.x.x;done
```

::::

:::::

## Install Mooncake

Mooncake is the serving platform for Kimi, a leading LLM service provided by Moonshot AI. First, we need to obtain the Mooncake project. Refer to the following command:

```shell
git clone -b v0.3.7.post2 --depth 1 https://github.com/kvcache-ai/Mooncake.git
```

(Optional) Replace go install url if the network is poor

```shell
cd Mooncake
sed -i 's|https://go.dev/dl/|https://golang.google.cn/dl/|g' dependencies.sh
```

Install mpi

```shell
apt-get install mpich libmpich-dev -y
```

Install the relevant dependencies. The installation of Go is not required.

```shell
bash dependencies.sh -y
```

Compile and install

```shell
mkdir build
cd build
cmake .. -DUSE_ASCEND_DIRECT=ON
make -j
make install
TODO cp to /usr/local
```

## Prefiller/Decoder Deployment

We can run the following scripts to launch a server on the prefiller/decoder node, respectively. Please note that each P/D node will occupy ports ranging from kv_port to kv_port + num_chips to initialize socket listeners. To avoid any issues, port conflicts should be prevented. Additionally, ensure that each node's engine_id is uniquely assigned to avoid conflicts.

### launch_online_dp.py
```python
import argparse
import multiprocessing
import os
import subprocess
import sys
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dp-size",
        type=int,
        required=True,
        help="Data parallel size."
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size."
    )
    parser.add_argument(
        "--dp-size-local",
        type=int,
        default=-1,
        help="Local data parallel size."
    )
    parser.add_argument(
        "--dp-rank-start",
        type=int,
        default=0,
        help="Starting rank for data parallel."
    )
    parser.add_argument(
        "--dp-address",
        type=str,
        required=True,
        help="IP address for data parallel master node."
    )
    parser.add_argument(
        "--dp-rpc-port",
        type=str,
        default=12345,
        help="Port for data parallel master node."
    )
    parser.add_argument(
        "--vllm-start-port",
        type=int,
        default=9000,
        help="Starting port for the engine."
    )
    return parser.parse_args()
args = parse_args()
dp_size = args.dp_size
tp_size = args.tp_size
dp_size_local = args.dp_size_local
if dp_size_local == -1:
    dp_size_local = dp_size
dp_rank_start = args.dp_rank_start
dp_address = args.dp_address
dp_rpc_port = args.dp_rpc_port
vllm_start_port = args.vllm_start_port
def run_command(visiable_devices, dp_rank, vllm_engine_port):
    command = [
        "bash",
        "./run_dp_template.sh",
        visiable_devices,
        str(vllm_engine_port),
        str(dp_size),
        str(dp_rank),
        dp_address,
        dp_rpc_port,
        str(tp_size),
    ]
    subprocess.run(command, check=True)
if __name__ == "__main__":
    template_path = "./run_dp_template.sh"
    if not os.path.exists(template_path):
        print(f"Template file {template_path} does not exist.")
        sys.exit(1)
    processes = []
    num_cards = dp_size_local * tp_size
    for i in range(dp_size_local):
        dp_rank = dp_rank_start + i
        vllm_engine_port = vllm_start_port + i
        visiable_devices = ",".join(str(x) for x in range(i * tp_size, (i + 1) * tp_size))
        process = multiprocessing.Process(target=run_command,
                                        args=(visiable_devices, dp_rank,
                                                vllm_engine_port))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
```

### run_dp_template.sh
#### Layerwise

:::::{tab-set}
:sync-group: nodes

::::{tab-item} Prefiller node 1
:sync: prefill node1

```shell
nic_name="eth0"  # network card name
local_ip="192.0.0.1"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_BUFFSIZE=256
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
vllm serve /path_to_weight/DeepSeek-V3.1_w8a8mix_mtp \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name ds_r1 \
  --max-model-len 40000 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 8 \
  --enforce-eager \
  --trust-remote-code \
  --gpu-memory-utilization 0.9  \
  --quantization ascend \
  --no-enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
  --additional-config '{"recompute_scheduler_enable":true,"enable_shared_expert_dp": true}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeLayerwiseConnector",
  "kv_role": "kv_producer",
  "kv_port": "30000",
  "engine_id": "0",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_layerwise_connector",
  "kv_connector_extra_config": {
            "use_ascend_direct": true,
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
:sync: prefill node2

```shell
nic_name="eth0"  # network card name
local_ip="192.0.0.2"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_BUFFSIZE=256
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
vllm serve /path_to_weight/DeepSeek-V3.1_w8a8mix_mtp \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name ds_r1 \
  --max-model-len 40000 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 8 \
  --enforce-eager \
  --trust-remote-code \
  --gpu-memory-utilization 0.9  \
  --quantization ascend \
  --no-enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
  --additional-config '{"recompute_scheduler_enable":true,"enable_shared_expert_dp": true}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeLayerwiseConnector",
  "kv_role": "kv_producer",
  "kv_port": "30100",
  "engine_id": "1",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_layerwise_connector",
  "kv_connector_extra_config": {
            "use_ascend_direct": true,
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

::::{tab-item} Decoder node 1
:sync: decoder node1

```shell
nic_name="eth0"  # network card name
local_ip="192.0.0.3"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_BUFFSIZE=600
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
vllm serve /path_to_weight/DeepSeek-V3.1_w8a8mix_mtp \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name ds_r1 \
  --max-model-len 40000 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 40 \
  --trust-remote-code \
  --gpu-memory-utilization 0.94  \
  --quantization ascend \
  --no-enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert": true,"lm_head_tensor_parallel_size":16}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeLayerwiseConnector",
  "kv_role": "kv_consumer",
  "kv_port": "30200",
  "engine_id": "2",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_layerwise_connector",
  "kv_connector_extra_config": {
            "use_ascend_direct": true,
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

::::

::::{tab-item} Decoder node 2 (primary Node)
:sync: decoder node2

```shell
nic_name="eth0"  # network card name
local_ip="192.0.0.4"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_BUFFSIZE=600
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export ASCEND_BUFFER_POOL=4:8
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
vllm serve /path_to_weight/DeepSeek-V3.1_w8a8mix_mtp \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name ds_r1 \
  --max-model-len 40000 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 40 \
  --trust-remote-code \
  --gpu-memory-utilization 0.94  \
  --quantization ascend \
  --no-enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert": true,"lm_head_tensor_parallel_size":16}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeLayerwiseConnector",
  "kv_role": "kv_consumer",
  "kv_port": "30200",
  "engine_id": "2",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_layerwise_connector",
  "kv_connector_extra_config": {
            "use_ascend_direct": true,
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


#### Non-layerwise

:::::{tab-set}
:sync-group: nodes

::::{tab-item} Prefiller node 1
:sync: prefill node1

```shell
nic_name="eth0"  # network card name
local_ip="192.0.0.1"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_BUFFSIZE=256
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export ASCEND_BUFFER_POOL=4:8
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
vllm serve /path_to_weight/DeepSeek-V3.1_w8a8mix_mtp \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name ds_r1 \
  --max-model-len 40000 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 8 \
  --enforce-eager \
  --trust-remote-code \
  --gpu-memory-utilization 0.9  \
  --quantization ascend \
  --no-enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
  --additional-config '{"recompute_scheduler_enable":true,"enable_shared_expert_dp": true}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnector",
  "kv_role": "kv_producer",
  "kv_port": "30000",
  "engine_id": "0",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
  "kv_connector_extra_config": {
            "use_ascend_direct": true,
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
:sync: prefill node2

```shell
nic_name="eth0"  # network card name
local_ip="192.0.0.2"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_BUFFSIZE=256
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export ASCEND_BUFFER_POOL=4:8
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
vllm serve /path_to_weight/DeepSeek-V3.1_w8a8mix_mtp \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name ds_r1 \
  --max-model-len 40000 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 8 \
  --enforce-eager \
  --trust-remote-code \
  --gpu-memory-utilization 0.9  \
  --quantization ascend \
  --no-enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
  --additional-config '{"recompute_scheduler_enable":true,"enable_shared_expert_dp": true}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnector",
  "kv_role": "kv_producer",
  "kv_port": "30100",
  "engine_id": "1",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
  "kv_connector_extra_config": {
            "use_ascend_direct": true,
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

::::{tab-item} Decoder node 1
:sync: decoder node1

```shell
nic_name="eth0"  # network card name
local_ip="192.0.0.3"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_BUFFSIZE=600
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export ASCEND_BUFFER_POOL=4:8
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
vllm serve /path_to_weight/DeepSeek-V3.1_w8a8mix_mtp \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name ds_r1 \
  --max-model-len 40000 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 40 \
  --trust-remote-code \
  --gpu-memory-utilization 0.94  \
  --quantization ascend \
  --no-enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert": true,"lm_head_tensor_parallel_size":16}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnector",
  "kv_role": "kv_consumer",
  "kv_port": "30200",
  "engine_id": "2",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
  "kv_connector_extra_config": {
            "use_ascend_direct": true,
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

::::

::::{tab-item} Decoder node 2 (primary Node)
:sync: decoder node2

```shell
nic_name="eth0"  # network card name
local_ip="192.0.0.4"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_BUFFSIZE=600
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export ASCEND_BUFFER_POOL=4:8
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
vllm serve /path_to_weight/DeepSeek-V3.1_w8a8mix_mtp \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name ds_r1 \
  --max-model-len 40000 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 40 \
  --trust-remote-code \
  --gpu-memory-utilization 0.94  \
  --quantization ascend \
  --no-enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert": true,"lm_head_tensor_parallel_size":16}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnector",
  "kv_role": "kv_consumer",
  "kv_port": "30200",
  "engine_id": "2",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
  "kv_connector_extra_config": {
            "use_ascend_direct": true,
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

### Start the service
```bash
# on 190.0.0.1
python launch_online_dp.py --dp-size 2 --tp-size 8 --dp-size-local 2 --dp-rank-start 0 --dp-address 190.0.0.1 --dp-rpc-port 12321 --vllm-start-port 7100
# on 190.0.0.2
python launch_online_dp.py --dp-size 2 --tp-size 8 --dp-size-local 2 --dp-rank-start 0 --dp-address 190.0.0.2 --dp-rpc-port 12321 --vllm-start-port 7100
# on 190.0.0.3
python launch_online_dp.py --dp-size 32 --tp-size 1 --dp-size-local 16 --dp-rank-start 0 --dp-address 190.0.0.3 --dp-rpc-port 12321 --vllm-start-port 7100
# on 190.0.0.4
python launch_online_dp.py --dp-size 32 --tp-size 1 --dp-size-local 16 --dp-rank-start 16 --dp-address 190.0.0.3 --dp-rpc-port 12321 --vllm-start-port 7100
```

## Example Proxy for Deployment

Run a proxy server on the same node with the prefiller service instance. You can get the proxy program in the repository's examples: [load\_balance\_proxy\_layerwise\_server\_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py) or [load\_balance\_proxy\_server\_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

:::::{tab-set}

::::{tab-item} Layerwise

```shell
python load_balance_proxy_layerwise_server_example.py \
    --host 192.0.0.1 \
  --port 1999 \
  --host 192.0.0.1 \
  --prefiller-hosts \
    192.0.0.1 \
    192.0.0.1 \
    192.0.0.2 \
    192.0.0.2 \
  --prefiller-ports  \
    7100 7101 7100 7101 \
  --decoder-hosts \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
  --decoder-ports  \
    7100 7101 7102 7103 7104 7105 7106 7107 7108 7109 7110 7111 7112 7113 7114 7115\
    7100 7101 7102 7103 7104 7105 7106 7107 7108 7109 7110 7111 7112 7113 7114 7115\
```

::::

::::{tab-item} Non-layerwise

```shell
python load_balance_proxy_server_example.py \
  --port 1999 \
  --host 192.0.0.1 \
  --prefiller-hosts \
    192.0.0.1 \
    192.0.0.1 \
    192.0.0.2 \
    192.0.0.2 \
  --prefiller-ports  \
    7100 7101 7100 7101 \
  --decoder-hosts \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.3  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
    192.0.0.4  \
  --decoder-ports  \
    7100 7101 7102 7103 7104 7105 7106 7107 7108 7109 7110 7111 7112 7113 7114 7115\
    7100 7101 7102 7103 7104 7105 7106 7107 7108 7109 7110 7111 7112 7113 7114 7115\
```

::::

:::::

Start the service
```bash
  bash server.sh
```
|Parameter  | meaning |
| --- | --- |
| --port | Proxy service Port |
| --host | Proxy service Host IP|
| --prefiller-hosts | Hosts of prefiller nodes |
| --prefiller-hosts-num | Number of repetitions for prefiller node hosts |
| --prefiller-ports | Ports of prefiller nodes |
| --prefiller-ports-inc | Number of increments for prefiller node ports |
| --decoder-hosts | Hosts of decoder nodes |
| --decoder-hosts-num | Number of repetitions for decoder node hosts |
| --decoder-ports | Ports of decoder nodes |
| --decoder-ports-inc | Number of increments for decoder node ports |
You can get the proxy program in the repository's examples, [load\_balance\_proxy\_server\_example.py](https://github.com/vllm-project/vllm-ascend/blob/v0.9.1-dev/examples/disaggregate_prefill_v1/load_balance_proxy_server_example.py)

## Benchmark

We recommend use aisbench tool to assess performance. [aisbench](https://gitee.com/aisbench/benchmark) Execute the following commands to install aisbench

```shell
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark/
pip3 install -e ./
```

You need to canncel the http proxy before assessing performance, as following

```shell
# unset proxy
unset http_proxy
unset https_proxy
```

- You can place your datasets in the dir: `benchmark/ais_bench/datasets`
- You can change the configurationin the dir :`benchmark/ais_bench/benchmark/configs/models/vllm_api` Take the ``vllm_api_stream_chat.py`` for examples

```python
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="/root/.cache/ds_r1",
        model="dsr1",
        request_rate = 28,
        retry = 2,
        host_ip = "192.0.0.1", # Proxy service host IP
        host_port = 8000,  # Proxy service Port
        max_out_len = 10,
        batch_size=1536,
        trust_remote_code=True,
        generation_kwargs = dict(
            temperature = 0,
            seed = 1024,
            ignore_eos=False,
        )
    )
]
```

- Take gsm8k dataset for example, execute the following commands  to assess performance.

```shell
ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_0_shot_cot_str_perf  --debug  --mode perf
```

- For more details for commands and parameters for aisbench, refer to  [aisbench](https://gitee.com/aisbench/benchmark)


## FAQ

### 1. Prefiller nodes need to warmup

Since the computation of some NPU operators requires several rounds of warm-up to achieve best performance, we recommend preheating the service with some requests before conducting performance tests to achieve the best end-to-end throughput.


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