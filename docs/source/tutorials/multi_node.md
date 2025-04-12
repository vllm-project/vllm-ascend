# Multi-Node (DeepSeek)

Multi-node inference is suitable for scenarios where the model cannot be deployed on a single NPU. In such cases, the model can be distributed using tensor parallelism and pipeline parallelism. The specific parallelism strategies will be covered in the following sections. To successfully deploy multi-node inference, the following three steps need to be completed:

* **Verify Multi-Node Communication Environment** 
* **Set Up and Start the Ray Cluster**
* **Start the Online Inference Service on multinode**


## Verify Multi-Node Communication Environment

### Physical Layer Requirements:

- The physical machines must be located on the same WLAN, with network connectivity.
- All NPUs are connected with optical modules, and the connection status must be normal.

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



```shell
# Define the image and container name
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
export NAME=vllm-ascend

# Run the container using the defined variables
docker run --rm \
--name $NAME \
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
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-p 8000:8000 \
-it $IMAGE bash
```

### Start Ray Cluster
After setting up the containers and installing vllm-ascend on each node, follow the steps below to start the Ray cluster and execute inference tasks.

Choose one machine as the head node and the others as worker nodes. Before proceeding, use `ip addr` to check your `nic_name` (network interface name).

Set the `ASCEND_RT_VISIBLE_DEVICES` environment variable to specify the NPU devices to use. For Ray versions above 2.1, also set the `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES` variable to avoid device recognition issues. The `--num-gpus` parameter defines the number of NPUs to be used on each node.

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
ray start --head --num-gpus=8
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
ray start --address='{head_node_ip}:{port_num}' --num-gpus=8 --node-ip-address={local_ip}
```
:::{tip}
Before starting the Ray cluster, set the `export ASCEND_PROCESS_LOG_PATH={plog_save_path}` environment variable on each node to redirect the Ascend plog, which helps in debugging issues during multi-node execution.
:::


Once the cluster is started on multiple nodes, execute `ray status` and `ray list nodes` to verify the Ray cluster's status. You should see the correct number of nodes and NPUs listed.


## Start the Online Inference Service on multinode
In the container, you can use vLLM as if all NPUs were on a single node. vLLM will utilize NPU resources across all nodes in the Ray cluster. You only need to run the vllm command on one node. 

To set up parallelism, the common practice is to set the `tensor-parallel-size` to the number of NPUs per node, and the `pipeline-parallel-size` to the number of nodes.

For example, with 16 NPUs across 2 nodes (8 NPUs per node), set the tensor parallel size to 8 and the pipeline parallel size to 2:

```shell
python -m vllm.entrypoints.openai.api_server \
       --model="Deepseek/DeepSeek-V2-Lite-Chat" \
       --trust-remote-code \
       --enforce-eager \
       --distributed_executor_backend "ray" \
       --tensor-parallel-size 8 \
       --pipeline-parallel-size 2 \
       --disable-frontend-multiprocessing \
       --port {port_num}
```
:::{note}
Pipeline parallelism currently requires AsyncLLMEngine, hence the `--disable-frontend-multiprocessing`  is set.
:::

Alternatively, if you want to use only tensor parallelism, set the tensor parallel size to the total number of NPUs in the cluster. For example, with 16 NPUs across 2 nodes, set the tensor parallel size to 16:
```shell
python -m vllm.entrypoints.openai.api_server \
       --model="Deepseek/DeepSeek-V2-Lite-Chat" \
       --trust-remote-code \
       --distributed_executor_backend "ray" \
       --enforce-eager \
       --tensor-parallel-size 16 \
       --port {port_num}
```

:::{note}
If you're running DeepSeek V3/R1, please remove `quantization_config` section in `config.json` file since it's not supported by vllm-ascend currently.
:::

Once your server is started, you can query the model with input prompts:

```shell
curl -X POST http://127.0.0.1:{prot_num}/v1/completions  \
     -H "Content-Type: application/json" \
     -d '{
         "model": "Deepseek/DeepSeek-V2-Lite-Chat",
         "prompt": "The future of AI is",
         "max_tokens": 24
     }'
```

If you query the server successfully, you can see the info shown below (client):

```
{"id":"cmpl-6dfb5a8d8be54d748f0783285dd52303","object":"text_completion","created":1739957835,"model":"/home/data/DeepSeek-V2-Lite-Chat/","choices":[{"index":0,"text":" heavily influenced by neuroscience and cognitiveGuionistes. The goalochondria is to combine the efforts of researchers, technologists,","logprobs":null,"finish_reason":"length","stop_reason":null,"prompt_logprobs":null}],"usage":{"prompt_tokens":6,"total_tokens":30,"completion_tokens":24,"prompt_tokens_details":null}}
```

Logs of the vllm server:

```
INFO:     127.0.0.1:59384 - "POST /v1/completions HTTP/1.1" 200 OK
INFO 02-19 17:37:35 metrics.py:453 Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 1.9 tokens/s, Running: 0 reqs, Swapped: 0 reqs, Pending: 0 reqs, NPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
```
