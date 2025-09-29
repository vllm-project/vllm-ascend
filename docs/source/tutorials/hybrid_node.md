# Hybrid-Node (DeepSeek)

:::{note}
Only machines with Atlas 800 A3 and aarch64 is supported currently, A2 and x86 is coming soon.
:::

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
for i in {0..15}; do hccn_tool -i $i -ip -g | grep ipaddr; done
```

#### 2. Cross-Node PING Test

```bash
# Execute on the target node (replace with actual IP)
hccn_tool -i 0 -ping -g address 10.20.0.20
```

:::::{tab-set}
::::{tab-item} Run with docker

Assume you have two Atlas 800 A3(64G*16) nodes, and want to deploy the `DeepSeek-V3.2-Exp` and `DeepSeek-V3.2-Exp-w8a8` quantitative model across multi-node.

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
<!-- TODO: CHANGE THE IMAGE TAG -->
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:v0.11.0rc0-A3-DSv3.2
export NAME=vllm-ascend

# Run the container using the defined variables
# Note if you are running bridge network with docker, Please expose available ports for multiple nodes communication in advance
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
-v /mnt/sfs_turbo/.cache:/root/.cache \
-it $IMAGE bash
```

::::

::::{tab-item} Install step by step

Install the package `custom-ops` to make the kernels available.

1. Download and install the `CANN-custom_ops-linux.aarch64.run` and source env for it:

```bash
wget https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/CANN-custom_ops-linux.aarch64.run
./CANN-custom_ops-linux.aarch64.run --quiet --install-path=/usr/local/Ascend/ascend-toolkit/latest/opp
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
```

2. Download and install the `custom_ops-1.0-cp311-cp311-linux_aarch64.whl`:

```bash
wget https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/custom_ops-1.0-cp311-cp311-linux_aarch64.whl
pip install custom_ops-1.0-cp311-cp311-linux_aarch64.whl
```

TODO: MLAPO ...

::::
:::::

:::::{tab-set}
::::{tab-item} DeepSeek-V3.2-Exp

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

vllm serve deepseek-ai/DeepSeek-V3.2-Exp \
--host 0.0.0.0 \
--port 8004 \
--data-parallel-size 2 \
--data-parallel-size-local 1 \
--data-parallel-address $local_ip \
--data-parallel-rpc-port 13389 \
--tensor-parallel-size 16 \
--seed 1024 \
--served-model-name deepseek_v3.2 \
--enable-expert-parallel \
--max-num-seqs 16 \
--max-model-len 32768 \
--max-num-batched-tokens 32768 \
--trust-remote-code \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.9 \
--additional-config '{"ascend_scheduler_config":{"enabled":true},"torchair_graph_config":{"enabled":true}}'
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

vllm serve deepseek-ai/DeepSeek-V3.2-Exp \
--host 0.0.0.0 \
--port 8004 \
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
--max-model-len 32768 \
--max-num-batched-tokens 32768 \
--enable-expert-parallel \
--trust-remote-code \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.92 \
--additional-config '{"ascend_scheduler_config":{"enabled":true},"torchair_graph_config":{"enabled":true}}'
```

::::

::::{tab-item} DeepSeek-V3.2-Exp-W8A8

```shell
#!/bin/sh

vllm serve vllm-ascend/DeepSeek-V3.2-Exp-W8A8 \
--host 0.0.0.0 \
--port 8004 \
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
--additional-config '{"ascend_scheduler_config":{"enabled":true},"torchair_graph_config":{"enabled":true}}'
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
