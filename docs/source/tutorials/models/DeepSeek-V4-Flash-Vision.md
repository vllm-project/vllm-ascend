# DeepSeek-V4-Flash-Vision-Exp (Experimental)

## 1 Introduction

DeepSeek-V4-Flash-Vision-Exp is a multimodal mixture-of-experts model in the
DeepSeek-V4 family. It combines the DeepSeek-V4 language model with a vision
encoder and aligner, and accepts text, single-image, and multi-image requests
through the OpenAI-compatible chat API.

Support on vLLM Ascend is experimental and is available on the `main` branch
with the matching vLLM 0.27.x revision. This guide documents Ascend W8A8
deployment on one Atlas 800 A3 server or two Atlas 800 A2 servers. Currently,
only colocated deployment is supported: Prefill and Decode run in the same
service. Do not use Prefill-Decode disaggregation for this model.

## 2 Supported Features

Refer to the [Supported Models](../../user_guide/support_matrix/supported_models.md)
for the complete support matrix and the
[Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration.

The A3 configuration in this guide has been validated with W8A8 weights,
TP4/DP4/EP16, automatic prefix caching, and `FULL_DECODE_ONLY` ACL Graph. The A2
configuration preserves the same global TP4/DP4/EP16 topology across two
servers. BF16 serving, DSA context parallelism, FlashComm1, and Prefill-Decode
disaggregation are not covered by this guide.

## 3 Prerequisites

### 3.1 Model Weights and Hardware

| Model | Download | Hardware requirements |
| --- | --- | --- |
| DeepSeek-V4-Flash-Vision-Exp-w8a8-QuaRot | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-Vision-Exp-w8a8-QuaRot) | One Atlas 800 A3 server (128GB × 8 NPUs) or two Atlas 800 A2 servers (64GB × 8 NPUs each) |
| DeepSeek-V4-Flash-Vision-Exp | [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp) | Original model weights; use the ModelScope W8A8 QuaRot checkpoint for the deployment in this guide |

The launch command below uses the public Ascend W8A8 QuaRot checkpoint from
ModelScope. The checkpoint includes the ModelSlim quantization description and
the multimodal `bias_vl` tensors required by vLLM Ascend.

Download or prepare the checkpoint in a directory of your choice and record
the absolute path. The examples use `<YOUR_MODEL_PATH>`; replace it with that
path, for example
`/data/weights/DeepSeek-V4-Flash-Vision-Exp-w8a8-QuaRot`.

### 3.2 Software

Use the published image for the target hardware. Each image contains matching
vLLM and vLLM Ascend revisions:

```text
A3: quay.io/ascend/vllm-ascend:DeepSeek-V4-Flash-Vision-Exp-a3
A2: quay.io/ascend/vllm-ascend:DeepSeek-V4-Flash-Vision-Exp
```

Do not replace either package inside the container independently. Mixing other
vLLM and vLLM Ascend revisions is not supported.

## 4 Installation

### 4.1 A3 Container

Pull the A3 image and start the container as follows. The A3 server has 8 NPUs,
exposed as 16 logical devices (`davinci0` through `davinci15`) in the container.

```shell
export IMAGE=quay.io/ascend/vllm-ascend:DeepSeek-V4-Flash-Vision-Exp-a3
export MODEL_ROOT="/data/weights"

docker pull "$IMAGE"

docker run --rm -it \
  --name dsv4-vision \
  --net=host \
  --shm-size=512g \
  --privileged=true \
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
  -v "$MODEL_ROOT:$MODEL_ROOT" \
  "$IMAGE" bash
```

### 4.2 A2 Containers

Run the following command on both A2 servers. Each server has 8 NPU dies,
exposed as `davinci0` through `davinci7`.

```shell
export IMAGE=quay.io/ascend/vllm-ascend:DeepSeek-V4-Flash-Vision-Exp
export MODEL_ROOT="/data/weights"

docker pull "$IMAGE"

docker run --rm -it \
  --name dsv4-vision \
  --net=host \
  --shm-size=512g \
  --privileged=true \
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
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v "$MODEL_ROOT:$MODEL_ROOT" \
  "$IMAGE" bash
```

### 4.3 Verification and Source Build

Change `MODEL_ROOT` if the checkpoint is stored elsewhere. Keep the same
absolute model path inside and outside the container, and use the same path on
both A2 servers. Verify the installed packages after entering each container:

```shell
python -m pip show vllm vllm-ascend
```

Both packages must be present. To build from source instead, follow the
[software environment installation guide](../../getting_started/installation.md#installation-software-environment)
and build the `main` branch with the matching vLLM revision from
`.github/vllm-main-verified.commit`. Use `Dockerfile.a3` for A3 and the
hardware-specific Dockerfile for A2.

## 5 Online Service Deployment

### 5.1 Single-Node A3 Colocated Deployment

The validated topology uses four data-parallel engines. Each engine uses four
tensor-parallel ranks, while expert parallelism spans all 16 ranks.

```shell
# Replace <YOUR_MODEL_PATH> with the actual path recorded in Section 3.1.
export MODEL_PATH="<YOUR_MODEL_PATH>"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_BUFFSIZE=1024

vllm serve "$MODEL_PATH" \
  --served-model-name dsv4-vision \
  --max-model-len 130000 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.9 \
  --data-parallel-size 4 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --tokenizer-mode deepseek_v4 \
  --quantization ascend \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":32}' \
  --block-size 32 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --port 8900
```

Key parameters:

- `--data-parallel-size 4` and `--tensor-parallel-size 4` consume all 16
  visible logical devices backed by the server's 8 NPUs.
  `--enable-expert-parallel` distributes MoE experts across the ranks.
- `--max-model-len 130000` limits the combined input and output length of one
  request. Reduce it if the service cannot allocate enough KV cache.
- `--max-num-seqs 32` is the maximum number of sequences scheduled by each DP
  engine. Reduce it first if runtime memory pressure is observed.
- `--block-size 32` is required by the validated DeepSeek-V4 prefix-cache and
  sparse-attention configuration.
- `FULL_DECODE_ONLY` captures decode execution while keeping Prefill outside
  the full decode graph.

Wait until all four DP engines finish loading weights and graph capture. A
successful startup includes output similar to:

```text
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

For general startup issues, refer to the [Public FAQ](../../faqs.md).

### 5.2 Two-Node A2 Colocated Deployment

The A2 deployment uses two servers because each server provides 8 NPU dies.
Each server runs two local DP ranks with TP4, consuming all 8 dies. Together,
the two servers form the global DP4/TP4/EP16 topology.

Before starting the service, follow
[Verify Multi-node Communication](../../getting_started/installation.md#installation-multi-node-interconnect).
Use the same model path and DP RPC port on both servers. Set `NIC_NAME` to the
interface associated with `LOCAL_IP`, and make sure Node 1 can reach Node 0 at
`NODE0_IP`.

=== "Node 0"

    ```shell
    # Replace the placeholders with the values for Node 0.
    export MODEL_PATH="<YOUR_MODEL_PATH>"
    export NIC_NAME="<NODE0_NIC_NAME>"
    export LOCAL_IP="<NODE0_IP>"
    export DP_RPC_PORT=13389

    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export HCCL_IF_IP="$LOCAL_IP"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_BUFFSIZE=1024
    export HCCL_INTRA_PCIE_ENABLE=1
    export HCCL_INTRA_ROCE_ENABLE=0

    vllm serve "$MODEL_PATH" \
      --host 0.0.0.0 \
      --port 8900 \
      --served-model-name dsv4-vision \
      --max-model-len 130000 \
      --max-num-batched-tokens 4096 \
      --max-num-seqs 32 \
      --gpu-memory-utilization 0.9 \
      --data-parallel-size 4 \
      --data-parallel-size-local 2 \
      --data-parallel-start-rank 0 \
      --data-parallel-address "$LOCAL_IP" \
      --data-parallel-rpc-port "$DP_RPC_PORT" \
      --tensor-parallel-size 4 \
      --enable-expert-parallel \
      --tokenizer-mode deepseek_v4 \
      --quantization ascend \
      --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":32}' \
      --block-size 32 \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    ```

=== "Node 1"

    ```shell
    # Replace the placeholders with the values for Node 1 and Node 0.
    export MODEL_PATH="<YOUR_MODEL_PATH>"
    export NIC_NAME="<NODE1_NIC_NAME>"
    export LOCAL_IP="<NODE1_IP>"
    export NODE0_IP="<NODE0_IP>"
    export DP_RPC_PORT=13389

    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export HCCL_IF_IP="$LOCAL_IP"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_BUFFSIZE=1024
    export HCCL_INTRA_PCIE_ENABLE=1
    export HCCL_INTRA_ROCE_ENABLE=0

    vllm serve "$MODEL_PATH" \
      --headless \
      --host 0.0.0.0 \
      --port 8900 \
      --served-model-name dsv4-vision \
      --max-model-len 130000 \
      --max-num-batched-tokens 4096 \
      --max-num-seqs 32 \
      --gpu-memory-utilization 0.9 \
      --data-parallel-size 4 \
      --data-parallel-size-local 2 \
      --data-parallel-start-rank 2 \
      --data-parallel-address "$NODE0_IP" \
      --data-parallel-rpc-port "$DP_RPC_PORT" \
      --tensor-parallel-size 4 \
      --enable-expert-parallel \
      --tokenizer-mode deepseek_v4 \
      --quantization ascend \
      --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":32}' \
      --block-size 32 \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    ```

Start Node 0 first, then start Node 1. Only Node 0 exposes the API endpoint;
Node 1 runs as a headless worker. The multi-node DP parameters have the
following meanings:

- `--data-parallel-size 4` is the global DP size across both servers.
- `--data-parallel-size-local 2` creates two DP ranks on each server. Each DP
  rank uses four local dies through TP4, so each A2 server uses all eight dies.
- `--data-parallel-start-rank 0` assigns DP ranks 0 and 1 to Node 0, while
  `--data-parallel-start-rank 2` assigns ranks 2 and 3 to Node 1.
- `--data-parallel-address` points both servers to Node 0, and
  `--data-parallel-rpc-port` must have the same value on both servers.
- `--headless` prevents Node 1 from starting another API server.

Wait until all four DP engines are ready before sending requests to
`http://<NODE0_IP>:8900`.

### 5.3 Prefill-Decode Disaggregation

Prefill-Decode disaggregation is not currently supported for
DeepSeek-V4-Flash-Vision-Exp. Use one of the colocated deployments in Sections
5.1 and 5.2.

## 6 Functional Verification

Set `IMAGE_URL` to an HTTP(S) URL that is reachable from the serving container,
then send a multimodal chat-completions request. For the two-node A2 deployment,
replace `127.0.0.1` with the Node 0 IP address if the request is sent remotely.

```shell
export IMAGE_URL="<YOUR_IMAGE_URL>"

curl -sS -o response.json -w '%{http_code}\n' \
  http://127.0.0.1:8900/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"dsv4-vision\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"${IMAGE_URL}\"}},
        {\"type\": \"text\", \"text\": \"Describe this image.\"}
      ]
    }],
    \"temperature\": 0,
    \"max_tokens\": 256
  }"

jq -e '.choices[0].message.content | length > 0' response.json
```

Expected output:

```text
200
true
```

The generated description is stored in
`response.json` under `.choices[0].message.content`. To use local image files,
mount their parent directory into the container and add
`--allowed-local-media-path <DIRECTORY>` to the serving command.

## 7 Accuracy Evaluation

The merged implementation was validated on OCRBench V1 with 1,000 samples. It
completed without request errors and scored **826/1000 (82.6)** using the W8A8
checkpoint, concurrency 64, temperature 1.0, top-p 0.95, maximum output length
8192, seed 7, and thinking enabled with high reasoning effort.

For reproducing an accuracy evaluation, refer to
[Using AISBench](../../developer_guide/evaluation/using_ais_bench.md). Results
depend on the checkpoint, prompt template, decoding parameters, and dataset
version; record all of them when comparing runs.

## 8 Performance Evaluation

No production performance baseline is published for this experimental model.
Use the [AISBench performance evaluation guide](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation)
with one of the deployments in Section 5, and report image resolution,
input/output lengths, request concurrency, TTFT, TPOT, ITL, and throughput.

## 9 Performance Tuning

The values in Section 5 are the recommended starting point, not globally
optimal settings. Tune `--max-num-seqs`, `--max-num-batched-tokens`, and
`--gpu-memory-utilization` together for the target image sizes and sequence
lengths. Keep the global TP4/DP4/EP topology and `--block-size 32` until an
alternative configuration has been validated.

Refer to the
[performance tuning guide](../../developer_guide/performance_and_debug/optimization_and_tuning.md)
for general tuning methods.

## 10 Limitations

- Colocated serving is documented on either one Atlas 800 A3 server
  (128GB × 8 NPUs) or two Atlas 800 A2 servers (64GB × 8 NPUs each).
- Prefill-Decode disaggregation, DSA context parallelism, and FlashComm1 are
  not supported in the documented configuration.
- BF16 full-model validation and production performance qualification are not
  complete.
- With TP4 ACL Graph, graph capture batch sizes must be multiples of four.
- DSpark uses target-side multimodal Prefill followed by text-only speculative
  Decode; the draft model does not consume image embeddings or propose tokens
  inside image spans.
