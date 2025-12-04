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

## Run with Docker
Start a Docker container on each node.

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:|vllm_ascend_version|
export NAME=vllm-ascend

# Run the container using the defined variables
# Note: If you are running bridge network with docker, please expose available ports for multiple nodes communication in advance
docker run --rm \
--name $NAME \
--net=host \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci4 \
--device /dev/davinci5 \
--device /dev/davinci6 \
--device /dev/davinci7 \
--device /dev/davinci8 \
--device /dev/davinci9 \
--device /dev/davinci10 \
--device /dev/davinci11 \
--device /dev/davinci12 \
--device /dev/davinci13 \
--device /dev/davinci14 \
--device /dev/davinci15 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /mnt/sfs_turbo/.cache:/root/.cache \
-it $IMAGE bash
```

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
```

## Prefiller/Decoder Deployment

We can run the following scripts to launch a server on the prefiller/decoder node, respectively. Please note that each P/D node will occupy ports ranging from kv_port to kv_port + num_chips to initialize socket listeners. To avoid any issues, port conflicts should be prevented. Additionally, ensure that each node's engine_id is uniquely assigned to avoid conflicts.

### launch_online_dp.py
Use `launch_online_dp.py` to launch external dp vllm servers.
[launch\_online\_dp.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/external_online_dp/launch_online_dp.py)

### run_dp_template.sh
Modify `run_dp_template.py` on each node.
[run\_dp\_template.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/external_online_dp/run_dp_template.sh)

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
export HCCL_BUFFSIZE=256
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
vllm serve /path_to_weight/DeepSeek-r1_w8a8_mtp \
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
export HCCL_BUFFSIZE=256
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
vllm serve /path_to_weight/DeepSeek-r1_w8a8_mtp \
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
vllm serve /path_to_weight/DeepSeek-r1_w8a8_mtp \
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
  --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert": true,"lm_head_tensor_parallel_size":16}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
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

::::

::::{tab-item} Decoder node 2
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
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
vllm serve /path_to_weight/DeepSeek-r1_w8a8_mtp \
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
  --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert": true,"lm_head_tensor_parallel_size":16}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
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
export HCCL_BUFFSIZE=256
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
vllm serve /path_to_weight/DeepSeek-r1_w8a8_mtp \
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
export HCCL_BUFFSIZE=256
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
vllm serve /path_to_weight/DeepSeek-r1_w8a8_mtp \
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
vllm serve /path_to_weight/DeepSeek-r1_w8a8_mtp \
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
  --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert": true,"lm_head_tensor_parallel_size":16}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
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

::::

::::{tab-item} Decoder node 2
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
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
vllm serve /path_to_weight/DeepSeek-r1_w8a8_mtp \
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
  --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert": true,"lm_head_tensor_parallel_size":16}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
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

Run a proxy server on the same node where your prefiller service instance is deployed. You can find the proxy implementation in the repository's examples directory.

We provide two different proxy implementations with distinct request routing behaviors:

- **`load_balance_proxy_layerwise_server_example.py`**: Requests are first routed to the D nodes, which then forward to the P nodes as needed.This proxy is designed for use with the MooncakeLayerwiseConnector.[load\_balance\_proxy\_layerwise\_server\_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py)

- **`load_balance_proxy_server_example.py`**: Requests are first routed to the P nodes, which then forward to the D nodes for subsequent processing.This proxy is designed for use with the MooncakeConnector.[load\_balance\_proxy\_server\_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

:::::{tab-set}

::::{tab-item} Layerwise

```shell
python load_balance_proxy_layerwise_server_example.py \
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

|Parameter  | meaning |
| --- | --- |
| --port | Proxy service Port |
| --host | Proxy service Host IP|
| --prefiller-hosts | Hosts of prefiller nodes |
| --prefiller-ports | Ports of prefiller nodes |
| --decoder-hosts | Hosts of decoder nodes |
| --decoder-ports | Ports of decoder nodes |

You can get the proxy program in the repository's examples, [load\_balance\_proxy\_server\_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

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
        request_rate = 14,
        retry = 2,
        host_ip = "192.0.0.1", # Proxy service host IP
        host_port = 8000,  # Proxy service Port
        max_out_len = 10,
        batch_size=768,
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
