# Multi-Node (DeepSeek V3.2)

:::{note}
Only machines with aarch64 is supported currently, x86 is coming soon. This guide take A3 as the example.
:::

## Verify Multi-Node Communication Environment

### Physical Layer Requirements:

- The physical machines must be located on the same WLAN, with network connectivity.
- All NPUs are connected with optical modules, and the connection status must be normal.

### Verification Process:

Execute the following commands on each node in sequence. The results must all be `success` and the status must be `UP`:

:::::{tab-set}
::::{tab-item} A2 series

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

::::
::::{tab-item} A3 series

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

::::
:::::

### NPU Interconnect Verification:
#### 1. Get NPU IP Addresses
:::::{tab-set}
::::{tab-item} A2 series

```bash
for i in {0..7}; do hccn_tool -i $i -ip -g | grep ipaddr; done
```

::::
::::{tab-item} A3 series

```bash
for i in {0..15}; do hccn_tool -i $i -ip -g | grep ipaddr; done
```

::::
:::::

#### 2. Cross-Node PING Test

```bash
# Execute on the target node (replace with actual IP)
hccn_tool -i 0 -ping -g address 10.20.0.20
```

## Deploy DeepSeek-V3.2-Exp with vLLM-Ascend:

Currently, we provide a all-in-one image (include CANN 8.2RC1 + [SparseFlashAttention/LightningIndexer](https://gitcode.com/cann/cann-recipes-infer/tree/master/ops/ascendc) + [MLAPO](https://github.com/vllm-project/vllm-ascend/pull/3226)). You can also build your own image refer to [link](https://github.com/vllm-project/vllm-ascend/issues/3278).

- `DeepSeek-V3.2-Exp`: requreid 2 Atlas 800 A3(64G*16) nodes or 4 Atlas 800 A2(64G*8). [Model weight link](https://modelers.cn/models/Modelers_Park/DeepSeek-V3.2-Exp-BF16)
- `DeepSeek-V3.2-Exp-w8a8`: requreid 1 Atlas 800 A3(64G*16) node or 2 Atlas 800 A2(64G*8). [Model weight link](https://modelers.cn/models/Modelers_Park/DeepSeek-V3.2-Exp-w8a8)

Run the following command to start the container in each node(This guide suppose you have download the weight to /root/.cache already):

:::::{tab-set}
::::{tab-item} A2 series

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
# export IMAGE=quay.io/ascend/vllm-ascend:v0.11.0rc0-deepseek-v3.2-exp
export IMAGE=quay.nju.edu.cn/ascend/vllm-ascend:v0.11.0rc0-deepseek-v3.2-exp
export NAME=vllm-ascend

# Run the container using the defined variables
# Note if you are running bridge network with docker, Please expose available ports
# for multiple nodes communication in advance
docker run --rm \
--name $NAME \
--net=host \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci4 \
--device /dev/davinci5 \
--device /dev/davinci6 \
--device /dev/davinci7 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-it $IMAGE bash
```

::::
::::{tab-item} A3 series

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
# openEuler:
# export IMAGE=quay.io/ascend/vllm-ascend:v0.11.0rc0-a3-openeuler-deepseek-v3.2-exp
# Ubuntu:
# export IMAGE=quay.io/ascend/vllm-ascend:v0.11.0rc0-a3-deepseek-v3.2-exp
export IMAGE=quay.nju.edu.cn/ascend/vllm-ascend:v0.11.0rc0-a3-deepseek-v3.2-exp
export NAME=vllm-ascend

# Run the container using the defined variables
# Note if you are running bridge network with docker, Please expose available ports
# for multiple nodes communication in advance
docker run --rm \
--name $NAME \
--net=host \
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
-v /root/.cache:/root/.cache \
-it $IMAGE bash
```

::::
:::::

:::::{tab-set}
::::{tab-item} DeepSeek-V3.2-Exp A3 series

Run the following scripts on two nodes respectively

:::{note}
Before launch the inference server, ensure the following environment variables are set for multi node communication
:::

**node0**

```shell
#!/bin/sh

# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip
nic_name="xxxx"
local_ip="xxxx"

export VLLM_USE_MODELSCOPE=True
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export HCCL_BUFFSIZE=1024

vllm serve /root/.cache/Modelers_Park/DeepSeek-V3.2-Exp \
--host 0.0.0.0 \
--port 8000 \
--data-parallel-size 2 \
--data-parallel-size-local 1 \
--data-parallel-address $local_ip \
--data-parallel-rpc-port 13389 \
--tensor-parallel-size 16 \
--seed 1024 \
--served-model-name deepseek_v3.2 \
--enable-expert-parallel \
--max-num-seqs 16 \
--max-model-len 17450 \
--max-num-batched-tokens 17450 \
--trust-remote-code \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.9 \
--additional-config '{"ascend_scheduler_config":{"enabled":true},"torchair_graph_config":{"enabled":true,"graph_batch_sizes":[16]}}'
```

**node1**

```shell
#!/bin/sh

nic_name="xxx"
local_ip="xxx"

export VLLM_USE_MODELSCOPE=True
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export HCCL_BUFFSIZE=1024

vllm serve /root/.cache/Modelers_Park/DeepSeek-V3.2-Exp \
--host 0.0.0.0 \
--port 8000 \
--headless \
--data-parallel-size 2 \
--data-parallel-size-local 1 \
--data-parallel-start-rank 1 \
--data-parallel-address <node0_ip> \
--data-parallel-rpc-port 13389 \
--tensor-parallel-size 16 \
--seed 1024 \
--served-model-name deepseek_v3.2 \
--max-num-seqs 16 \
--max-model-len 17450 \
--max-num-batched-tokens 17450 \
--enable-expert-parallel \
--trust-remote-code \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.92 \
--additional-config '{"ascend_scheduler_config":{"enabled":true},"torchair_graph_config":{"enabled":true,"graph_batch_sizes":[16]}}'
```

::::

::::{tab-item} DeepSeek-V3.2-Exp-W8A8 A3 series

```shell
#!/bin/sh
export VLLM_USE_MODELSCOPE=true

vllm serve vllm-ascend/DeepSeek-V3.2-Exp-W8A8 \
--host 0.0.0.0 \
--port 8000 \
--tensor-parallel-size 16 \
--seed 1024 \
--quantization ascend \
--served-model-name deepseek_v3.2 \
--max-num-seqs 16 \
--max-model-len 17450 \
--max-num-batched-tokens 17450 \
--enable-expert-parallel \
--trust-remote-code \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.92 \
--additional-config '{"ascend_scheduler_config":{"enabled":true},"torchair_graph_config":{"enabled":true,"graph_batch_sizes":[16]}}'
```

::::
::::{tab-item} DeepSeek-V3.2-Exp-W8A8 A2 series

Run the following scripts on two nodes respectively

**node0**

```shell
#!/bin/sh

# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip
nic_name="xxxx"
local_ip="xxxx"

export VLLM_USE_MODELSCOPE=True
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

vllm serve vllm-ascend/DeepSeek-V3.2-Exp-W8A8 \
--host 0.0.0.0 \
--port 8000 \
--data-parallel-size 2 \
--data-parallel-size-local 1 \
--data-parallel-address $local_ip \
--data-parallel-rpc-port 13389 \
--tensor-parallel-size 8 \
--seed 1024 \
--served-model-name deepseek_v3.2 \
--enable-expert-parallel \
--max-num-seqs 16 \
--max-model-len 17450 \
--max-num-batched-tokens 17450 \
--trust-remote-code \
--quantization ascend \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.9 \
--additional-config '{"ascend_scheduler_config":{"enabled":true},"torchair_graph_config":{"enabled":true,"graph_batch_sizes":[16]}}'
```

**node1**

```shell
#!/bin/sh

nic_name="xxx"
local_ip="xxx"

export VLLM_USE_MODELSCOPE=True
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

vllm serve vllm-ascend/DeepSeek-V3.2-Exp-W8A8 \
--host 0.0.0.0 \
--port 8000 \
--headless \
--data-parallel-size 2 \
--data-parallel-size-local 1 \
--data-parallel-start-rank 1 \
--data-parallel-address <node0_ip> \
--data-parallel-rpc-port 13389 \
--tensor-parallel-size 8 \
--seed 1024 \
--served-model-name deepseek_v3.2 \
--max-num-seqs 16 \
--max-model-len 17450 \
--max-num-batched-tokens 17450 \
--enable-expert-parallel \
--trust-remote-code \
--quantization ascend \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.92 \
--additional-config '{"ascend_scheduler_config":{"enabled":true},"torchair_graph_config":{"enabled":true,"graph_batch_sizes":[16]}}'
```

::::
:::::

Once your server is started, you can query the model with input prompts:

```shell
curl http://<node0_ip>:<port>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek_v3.2",
        "prompt": "The future of AI is",
        "max_tokens": 50,
        "temperature": 0
    }'
```
