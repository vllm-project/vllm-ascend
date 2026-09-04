# GLM-5.3-Flash

## 1 Introduction

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) is the first natively multimodal model in the GLM-5 series. Built on a hybrid architecture that combines sparse and linear attention for the first time in the GLM series, it adopts Manifold-Constrained Hyper-Connections (mHC) and is trained on a 30T-token multimodal pre-training corpus. With 320B total parameters and only 18B active parameters, it outperforms GLM-5.2 across benchmarks and real-world workloads at one-tenth the price, while approaching Claude Opus 4.8 on coding and agentic benchmarks. GLM-5.3-Flash also supports controlling the thinking budget through the `reasoning_effort` parameter (`low`, `high`, `max`).

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

- `GLM-5.3-Flash-w8a8-mxfp8 (Ascend950DT mxfp8 Quantized)`: requires 1 Ascend950DT (96GB × 8) node.[Download model weight](https://www.modelscope.cn/models/Eco-Tech/GLM-5.3-Flash-w8a8-mxfp8).
- `GLM-5.3-Flash-w8a8`: requires 1 Atlas 800 A3 (128GB × 8) node.[Download model weight](https://modelers.cn/models/Eco-Tech/GLM-5.3-Flash-w8a8).
- `GLM-5.3-Flash-w8a8`: requires 2 Atlas 800 A2 (64GB × 16) nodes.[Download model weight](https://www.modelscope.cn/models/Eco-Tech/GLM-5.3-Flash-w8a8).

- You can use [msmodelslim](https://gitcode.com/Ascend/msmodelslim) to quantize the model directly.

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

## 4 Installation

### 4.1 Docker Image Installation

=== "Ascend950DT series"

    Start the docker image on each node.

    ```shell
    export IMAGE=quay.io/ascend/vllm-ascend:glm-5.3-flash-a5-openeuler
    export NAME=vllm-ascend

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
    --device /dev/davinci_manager \
    --device /dev/hisi_hdc \
    --device /dev/ummu \
    --device /dev/uburma \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json \
    -v /etc/hixlep/:/etc/hixlep/ \
    -v /root/.cache:/root/.cache \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/bin/urma_admin:/usr/bin/urma_admin \
    -v /lib/route.conf:/lib/route.conf \
    -itd $IMAGE bash
    ```

=== "A3 series"

    Start the docker image on each node.

    ```shell

    export IMAGE=quay.io/ascend/vllm-ascend:glm-5.3-flash-a3
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
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
    ```

=== "A2 series"

    Start the docker image on each node.

    ```shell

    export IMAGE=quay.io/ascend/vllm-ascend:glm-5.3-flash
    export NAME=vllm-ascend

    docker run --rm \
    --name $NAME \
    --net=host \
    --shm-size=500g \
    --privileged \
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
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /etc/hccn.conf:/etc/hccn.conf:ro \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
    ```

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

=== "Ascend950DT series"

    - Quantized model `GLM-5.3-Flash-w8a8-mxfp8` can be deployed on 1 Ascend950DT (96GB × 8) .

    Run the following script to execute online inference.

    ```shell

    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_BUFFSIZE=1024

    vllm serve Eco-Tech/GLM-5.3-Flash-w8a8-mxfp8 \
      --host 0.0.0.0 \
      --port 8011 \
      --data-parallel-size 1 \
      --tensor-parallel-size 8 \
      --enable-expert-parallel \
      --seed 1024 \
      --quantization ascend \
      --served-model-name glm \
      --max-num-seqs 32 \
      --max-model-len 132096 \
      --max-num-batched-tokens 8192 \
      --trust-remote-code \
      --gpu-memory-utilization 0.9 \
      --limit-mm-per-prompt '{"image": 1, "video": 0}' \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,16,32,64,96,128]}' \
      --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'
    ```

=== "Atlas 800 A3 series"

    - Quantized model `GLM-5.3-Flash-w8a8` can be deployed on 1 A3 (64GB × 16) .

    Run the following script to execute online inference.

    ```shell
    #!/bin/sh

    source /usr/local/Ascend/cann-9.1.0/opp/vendors/custom_transformer/bin/set_env.bash
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_BUFFSIZE=400

    vllm serve Eco-Tech/GLM-5.3-Flash-w8a8   \
      --host 0.0.0.0 \
      --port 8077 \
      --max-model-len 133120  \
      --data-parallel-size 1 \
      --tensor-parallel-size 16 \
      --enable-expert-parallel \
      --seed 1024 \
      --served-model-name glm \
      --safetensors-load-strategy prefetch \
      --max-num-seqs 32 \
      --max-num-batched-tokens 8192 \
      --trust-remote-code \
      --quantization ascend \
      --limit-mm-per-prompt '{"image": 1, "video": 0}' \
      --gpu-memory-utilization 0.85 \
      --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}' \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,16,32,64,96,128]}' \
      --api-server-count 1
    ```

#### Key Parameter Descriptions

Only the key parameters specific to this model/scenario are described below. `max-model-len` and `max-num-seqs` need to be set according to the actual usage scenario.

**Model-specific parameters:**

- `--data-parallel-size 1`: Runs a single DP rank. `--tensor-parallel-size` is 8 on Ascend950DT and 16 on Atlas 800 A3. This layout is recommended to balance memory capacity and compute efficiency for the w8a8 weights.
- `--enable-expert-parallel`: Must be enabled for the MoE architecture of GLM-5.3-Flash.
- `--quantization ascend`: Enables Ascend quantization for the w8a8 quantized weights.
- `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`: Enables graph capture for the decode phase only, improving decode performance by reducing kernel launch overhead.
- `--limit-mm-per-prompt '{"image": 1, "video": 0}'`: For text-only deployment, --limit-mm-per-prompt can be omitted. For multimodal deployment, configure this parameter according to the actual request shape. For example, use --limit-mm-per-prompt '{"image":2,"video":0}' for two-image requests, and use --limit-mm-per-prompt '{"image":0,"video":1}' for one-video requests.
- `--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'`: Enables Multi-Token Prediction (MTP) speculative decoding with the DeepSeek-style MTP draft head of GLM-5.3-Flash. `num_speculative_tokens` (3-5) controls how many tokens are speculated per step; `enforce_eager: true` is required because GLM-5.3-Flash does not support graph-mode speculative decoding.

### 5.2 Multi-Node Deployment

=== "A2 series"

    - Quantized model `GLM-5.3-Flash-w8a8` can be deployed on 2 Atlas 800 A2 (64GB × 8) nodes with DP2 across the two nodes (one DP rank per node) and TP8 inside each node.

    Run the following scripts on two nodes respectively.

    **node 0**

    ```shell
    # this obtained through ifconfig
    # nic_name is the network interface name corresponding to local_ip of the current node
    nic_name="xxxx"
    local_ip="xx.xx.xx.1"

    # The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
    node0_ip="xx.xx.xx.1"

    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_OP_EXPANSION_MODE=AIV
    export HCCL_BUFFSIZE=1024
    export VLLM_RPC_TIMEOUT=3600000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export HCCL_EXEC_TIMEOUT=3600
    export HCCL_CONNECT_TIMEOUT=1200
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export HCCL_IF_IP=$local_ip
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

    vllm serve /path/to/GLM-5.3-Flash-w8a8 \
        --host 0.0.0.0 \
        --port 8077 \
        --max-model-len 133120 \
        --data-parallel-size 2 \
        --data-parallel-size-local 1 \
        --data-parallel-rank 0 \
        --data-parallel-address $node0_ip \
        --data-parallel-rpc-port 12321 \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name glm \
        --safetensors-load-strategy prefetch \
        --max-num-seqs 128 \
        --max-num-batched-tokens 8192 \
        --trust-remote-code \
        --quantization ascend \
        --limit-mm-per-prompt '{"image":1,"video":0}' \
        --gpu-memory-utilization 0.85 \
        --speculative-config '{"num_speculative_tokens":2,"method":"deepseek_mtp","enforce_eager":true}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64,96,128,256,384]}' \
        --api-server-count 1
    ```

    **node 1**

    ```shell
    # this obtained through ifconfig
    # nic_name is the network interface name corresponding to local_ip of the current node
    nic_name="xxxx"
    local_ip="xx.xx.xx.2"

    # The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
    node0_ip="xx.xx.xx.1"

    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_OP_EXPANSION_MODE=AIV
    export HCCL_BUFFSIZE=1024
    export VLLM_RPC_TIMEOUT=3600000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export HCCL_EXEC_TIMEOUT=3600
    export HCCL_CONNECT_TIMEOUT=1200
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export HCCL_IF_IP=$local_ip
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

    vllm serve /path/to/GLM-5.3-Flash-w8a8 \
        --host 0.0.0.0 \
        --port 8077 \
        --max-model-len 133120 \
        --data-parallel-size 2 \
        --data-parallel-size-local 1 \
        --data-parallel-rank 1 \
        --data-parallel-address $node0_ip \
        --data-parallel-rpc-port 12321 \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name glm \
        --safetensors-load-strategy prefetch \
        --max-num-seqs 128 \
        --max-num-batched-tokens 8192 \
        --trust-remote-code \
        --quantization ascend \
        --limit-mm-per-prompt '{"image":1,"video":0}' \
        --gpu-memory-utilization 0.85 \
        --speculative-config '{"num_speculative_tokens":2,"method":"deepseek_mtp","enforce_eager":true}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64,96,128,256,384]}' \
        --api-server-count 1
    ```

#### Key Parameter Descriptions

**Multi-node network and data parallel configuration:**

- `HCCL_IF_IP`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, `HCCL_SOCKET_IFNAME`: Network interface configuration for multi-node communication. Set `nic_name` to the network interface name (obtained via `ifconfig`) and `local_ip` to the current node's IP address. These must be correctly configured on each node for successful multi-node communication.
- `--data-parallel-size 2 --data-parallel-size-local 1`: Runs two DP ranks across the two nodes, one rank per node; each rank uses TP8 within its node.
- `--data-parallel-rank`: DP rank of the current node. Node 0 uses `0`, node 1 uses `1`.
- `--data-parallel-address`: IP address of the data parallel master node (node 0). Must match the `local_ip` of the master node.
- `--data-parallel-rpc-port 12321`: RPC port for data parallel master communication. Must be the same across all nodes.

## 6 Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl http://<node0_ip>:<port>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "glm",
        "prompt": "The future of AI is",
        "max_completion_tokens": 50,
    }'
```

Expected Result:
The expected result of this request is a JSON payload containing the model’s generated text in a text_completion format.

```json
{
  "id": "cmpl-123abc",
  "object": "text_completion",
  "created": 1725444000,
  "model": "glm",
  "choices": [
    {
      "text": " incredibly promising, with rapid advancements in machine learning and autonomous systems.",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 15,
    "total_tokens": 20
  }
}
```

## 7 Accuracy Evaluation

Here are two accuracy evaluation methods.

### 7.1 Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result.

### 7.2 Using Language Model Evaluation Harness

Not tested yet.

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

## 9 FAQ

- **Q: How to enable function calling for GLM-5.3-Flash?**

  A: Please add following configurations in vLLM startup command

  ```shell
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  ```
