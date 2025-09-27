# Multi-Node-Ray (Qwen/Qwen3-235B-A22B)

Multi-node inference is suitable for the scenarios that the model cannot be deployed on a single machine. In such cases, the model can be distributed using tensor parallelism or pipeline parallelism. The specific parallelism strategies will be covered in the following sections. To successfully deploy multi-node inference, the following three steps need to be completed:

* **Verify Multi-Node Communication Environment**
* **Set Up and Start the Ray Cluster**
* **Start the Online Inference Service on multinode**

## Verify Multi-Node Communication Environment

### Physical Layer Requirements:

* The physical machines must be located on the same LAN, with network connectivity.
* All NPUs are connected with optical modules, and the connection status must be normal.

### Verification Process:

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

### NPU Interconnect Verification:
#### 1. Get NPU IP Addresses

```bash
for i in {0..7}; do hccn_tool -i $i -ip -g | grep ipaddr; done
```

#### 2. Cross-Node PING Test

```bash
# Execute on the target node (replace with actual IP)
hccn_tool -i 0 -ping -g address 10.20.0.20
```

## Set Up and Start the Ray Cluster
### Setting Up the Basic Container
To ensure a consistent execution environment across all nodes, including the model path and Python environment, it is recommended to use Docker images.

For setting up a multi-node inference cluster with Ray, **containerized deployment** is the preferred approach. Containers should be started on both the master and worker nodes, with the `--net=host` option to enable proper network connectivity.

Below is the example container setup command, which should be executed on **all nodes** :

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
export IMAGE=quay.nju.edu.cn/ascend/vllm-ascend:|vllm_ascend_version|
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
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /path/to/shared/cache:/root/.cache \ # IMPORTANT: This must be a shared directory accessible by all nodes
-it $IMAGE bash
```

### Start Ray Cluster
After setting up the containers and installing vllm-ascend on each node, follow the steps below to start the Ray cluster and execute inference tasks.

Choose one machine as the head node and the others as worker nodes. Before proceeding, use `ip addr` to check your `nic_name` (network interface name).

Set the `ASCEND_RT_VISIBLE_DEVICES` environment variable to specify the NPU devices to use. For Ray versions above 2.1, also set the `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES` variable to avoid device recognition issues.

Below are the commands for the head and worker nodes:

**Head node**:

:::{note}
When starting a Ray cluster for multi-node inference, the environment variables on each node must be set **before** starting the Ray cluster for them to take effect.
Updating the environment variables requires restarting the Ray cluster.
:::

```shell
# Head node
export HCCL_IF_IP={local_ip}
export GLOO_SOCKET_IFNAME={nic_name}
export TP_SOCKET_IFNAME={nic_name}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ray start --head
```

**Worker node**:

:::{note}
When starting a Ray cluster for multi-node inference, the environment variables on each node must be set **before** starting the Ray cluster for them to take effect. Updating the environment variables requires restarting the Ray cluster.
:::

```shell
# Worker node
export HCCL_IF_IP={local_ip}
export GLOO_SOCKET_IFNAME={nic_name}
export TP_SOCKET_IFNAME={nic_name}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ray start --address='{head_node_ip}:6379' --node-ip-address={local_ip}
```

Once the cluster is started on multiple nodes, execute `ray status` and `ray list nodes` to verify the Ray cluster's status. You should see the correct number of nodes and NPUs listed.

## Start the Online Inference Service on multinode scenario
In the container, you can use vLLM as if all NPUs were on a single node. vLLM will utilize NPU resources across all nodes in the Ray cluster.

**You only need to run the vllm command on one node.**

To set up parallelism, the common practice is to set the `tensor-parallel-size` to the number of NPUs per node, and the `pipeline-parallel-size` to the number of nodes.

For example, with 16 NPUs across 2 nodes (8 NPUs per node), set the tensor parallel size to 8 and the pipeline parallel size to 2:

```shell
vllm serve Qwen/Qwen3-235B-A22B \
  --distributed-executor-backend ray \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --seed 1024 \
  --max-model-len 8192  \
  --max-num-seqs 25 \
  --served-model-name qwen \
  --trust-remote-code \
  --gpu-memory-utilization 0.9
```

Alternatively, if you want to use only tensor parallelism, set the tensor parallel size to the total number of NPUs in the cluster. For example, with 16 NPUs across 2 nodes, set the tensor parallel size to 16:

```shell
vllm serve Qwen/Qwen3-235B-A22B \
  --distributed-executor-backend ray \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --seed 1024 \
  --max-model-len 8192  \
  --max-num-seqs 25 \
  --served-model-name qwen \
  --trust-remote-code \
  --gpu-memory-utilization 0.9
```

Once your server is started, you can query the model with input prompts:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen",
        "prompt": "tell me how to sleep well",
        "max_tokens": 100,
        "temperature": 0
    }'
```
