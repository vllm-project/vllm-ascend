# Kimi-K3

## 1 Introduction

Kimi-K3 is a multimodal mixture-of-experts model that combines Kimi Delta
Attention (KDA), gated Multi-head Latent Attention (MLA), attention residuals,
SiTU activations, and latent MoE layers.

This tutorial covers model preparation, installation, online serving, and
accuracy and performance evaluation for the full W4A8 checkpoint on Atlas A3.
This tutorial is based on **vLLM-Ascend main with vLLM 0.27.1**. Use a
main-branch build with Kimi-K3 support and its matching vLLM revision.

## 2 Supported Features

Refer to the [Supported Models](../../user_guide/support_matrix/supported_models.md)
for the model support matrix and the
[Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration.

This guide covers A3 W4A8 text and multimodal serving, TP/DP/EP, Prefix Cache,
`FULL_DECODE_ONLY` ACL Graph, and DSpark with a matching draft checkpoint.

## 3 Prerequisites

### 3.1 Model Weights and Hardware

| Model | Download | Purpose | Requirements |
| --- | --- | --- | --- |
| Eco-Tech/Kimi-K3-w4a8 | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Kimi-K3-w4a8) | Full 93-layer, 896-expert W4A8 target | About 1.49 TB of weight storage; the reference deployment uses four Atlas 800 A3 nodes with 16 logical NPUs per node (DP4/TP16/EP64) |
| RadixArk/Kimi-K3-DSpark | [ModelScope](https://www.modelscope.cn/models/RadixArk/Kimi-K3-DSpark) / [Hugging Face](https://huggingface.co/RadixArk/Kimi-K3-DSpark) | Optional GQA draft | Set `num_speculative_tokens` to `7` |
| Inferact/Kimi-K3-DSpark | [ModelScope](https://www.modelscope.cn/models/Inferact/Kimi-K3-DSpark) / [Hugging Face](https://huggingface.co/Inferact/Kimi-K3-DSpark) | Optional MLA draft | Set `num_speculative_tokens` to `7` |
| Inferact/Kimi-K3-DSpark-Block5 | [ModelScope](https://www.modelscope.cn/models/Inferact/Kimi-K3-DSpark-Block5) / [Hugging Face](https://huggingface.co/Inferact/Kimi-K3-DSpark-Block5) | Optional MLA draft with five-token blocks | Set `num_speculative_tokens` to `5` |

Download weights, tokenizer, and processor files before starting the service.
Mount them at identical paths on every node, for example under `/path/to/models`.
The weight size is a storage requirement, not an estimate of runtime NPU memory;
KV cache, activations, communication buffers, and the optional draft also need
memory.

### 3.2 Verify Multi-Node Communication

Before launching the four-node service, follow
[Verify Multi-Node Communication](../../installation.md#verify-multi-node-communication).
Use a reachable address and the correct communication interface on each node.
All nodes must use the same model revision and compatible software stack.

## 4 Installation

The Kimi-K3 installation and deployment instructions in this tutorial apply
only to the Atlas A3 series.

### 4.1 Docker Image Installation

Follow the [installation requirements](../../installation.md#requirements) for
the host driver and firmware. Start the A3 container below on every node.

The example uses the main-branch nightly image for the vLLM 0.27-based stack.
Check that the image includes Kimi-K3 support and the matching vLLM version.
Pin its digest for reproducible deployments.
To build an image from the selected source revision, follow Section 4.2.

```shell
export IMAGE=quay.io/ascend/vllm-ascend:nightly-main-a3
export MODEL_ROOT="/path/to/models"

docker run --rm -it \
  --name vllm-kimi-k3 \
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
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v "$MODEL_ROOT:$MODEL_ROOT" \
  "$IMAGE" bash
```

Verify the installed revisions before starting the service:

```shell
python -m pip show vllm vllm-ascend
```

### 4.2 Source Code Installation

To build the A3 image from source, run the following on an A3 build host.
The Dockerfile installs the matching CANN/PyTorch/TorchNPU stack and compiles
the Ascend operators; `VLLM_COMMIT` selects the upstream revision verified for
the chosen vLLM-Ascend checkout.

```shell
git clone --branch main https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git submodule update --init --recursive
docker build -f Dockerfile.a3 \
  --build-arg VLLM_COMMIT="$(cat .github/vllm-main-verified.commit)" \
  -t vllm-ascend-kimi-k3:main .
export IMAGE=vllm-ascend-kimi-k3:main
```

Use the resulting image on every node with the container command in Section 4.1,
replacing its `IMAGE` value. Record the source revision and image digest.
For an installation outside Docker, follow the
[source installation guide](../../installation.md#set-up-using-python), using
the selected main checkout and its verified vLLM commit instead of the older
release versions in the generic examples.

## 5 Online Service Deployment

### 5.1 Four-Node Online Deployment

The full checkpoint uses four Atlas 800 A3 nodes in a DP4/TP16/EP64 mixed
deployment. Start Node 0 first. Each worker owns one global DP rank and joins
Node 0 through the DP RPC address.

Set these variables on every node:

```shell
export MODEL_PATH="<KIMI_K3_FULL_W4A8_PATH>"
export TOKENIZER_PATH="<KIMI_K3_TOKENIZER_PATH>"
export LOCAL_IP="<CURRENT_NODE_IP>"
export NODE0_IP="<NODE0_IP>"
export NIC_NAME="<CURRENT_NODE_NIC>"
export SERVICE_PORT=8000
export RPC_PORT=13345
export DP_SIZE=4
export TP_SIZE=16

export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export HCCL_BUFFSIZE=1024
export HCCL_BUFFSIZE_EP=2048
export HCCL_IF_IP=$LOCAL_IP
export HCCL_INTRA_ROCE_ENABLE=0
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=$NIC_NAME
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export GLOO_SOCKET_IFNAME=$NIC_NAME
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OPENBLAS_NUM_THREADS=1
```

Run on Node 0:

```shell
vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port $SERVICE_PORT \
  --served-model-name kimi-k3 \
  --tokenizer "$TOKENIZER_PATH" \
  --tokenizer-mode kimi_k3 \
  --quantization ascend \
  --safetensors-load-strategy lazy \
  --tensor-parallel-size $TP_SIZE \
  --data-parallel-size $DP_SIZE \
  --data-parallel-size-local 1 \
  --data-parallel-address $LOCAL_IP \
  --data-parallel-rpc-port $RPC_PORT \
  --enable-expert-parallel \
  --enable-prefix-caching \
  --max-model-len 133120 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --enable-auto-tool-choice \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

Run on Nodes 1 through 3. Set `DP_START_RANK` to 1, 2, or 3 respectively:

```shell
export DP_START_RANK="<1_OR_2_OR_3>"

vllm serve "$MODEL_PATH" \
  --headless \
  --host 0.0.0.0 \
  --port $SERVICE_PORT \
  --served-model-name kimi-k3 \
  --tokenizer "$TOKENIZER_PATH" \
  --tokenizer-mode kimi_k3 \
  --quantization ascend \
  --safetensors-load-strategy lazy \
  --tensor-parallel-size $TP_SIZE \
  --data-parallel-size $DP_SIZE \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank $DP_START_RANK \
  --data-parallel-address $NODE0_IP \
  --data-parallel-rpc-port $RPC_PORT \
  --enable-expert-parallel \
  --enable-prefix-caching \
  --max-model-len 133120 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --enable-auto-tool-choice \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

Key parameter descriptions:

- `--data-parallel-size 4` and `--data-parallel-size-local 1` create one DP
  rank per node. `--tensor-parallel-size 16` uses all 16 local logical NPUs;
  `--enable-expert-parallel` distributes experts across the 64-rank EP group.
- `--data-parallel-start-rank` identifies each worker's global DP rank.
  `--headless` workers join Node 0 without serving a separate HTTP endpoint.
- `--max-model-len` bounds input plus output tokens;
  `--max-num-batched-tokens` bounds tokens scheduled per iteration, and
  `--max-num-seqs` bounds concurrent sequences per engine.
- `--gpu-memory-utilization` controls the model executor's device-memory
  budget. Check capacity with the actual checkpoint before increasing sequence
  length, concurrency, or adding a draft.
- `--tokenizer-mode kimi_k3`, `--reasoning-parser kimi_k3`, and
  `--tool-call-parser kimi_k3` select the Kimi-K3 request/response format.
  `--enable-auto-tool-choice` enables automatic tool selection.
- `FULL_DECODE_ONLY` captures decode execution; `--enable-prefix-caching`
  enables reuse of cached prefixes.

Wait for all ranks to finish weight loading and graph compilation before
running the request in Section 6. For general startup issues, see the
[Public FAQ](../../faqs.md).

### 5.2 Speculative Decoding with DSpark

Choose a GQA or MLA draft from the [model weights](#31-model-weights-and-hardware)
listed in Section 3.1 and download it to the same path on every node.
The following configuration uses seven speculative tokens for
`RadixArk/Kimi-K3-DSpark` or `Inferact/Kimi-K3-DSpark`. For
`Inferact/Kimi-K3-DSpark-Block5`, select that checkpoint's path and change
`num_speculative_tokens` to `5`.

Add the same speculative configuration to the Node 0 and headless worker commands:

```shell
--speculative-config \
'{
  "method": "dspark",
  "model": "<DSPARK_DRAFT_PATH>",
  "num_speculative_tokens": 7,
  "draft_tensor_parallel_size": 16,
  "max_model_len": 4096,
  "draft_sample_method": "greedy",
  "enforce_eager": true
}'
```

The example proposes seven draft tokens in one query block, followed by
target-model verification.
Set `draft_tensor_parallel_size` to the topology used to shard the draft model.

## 6 Functional Verification

Once all four nodes have completed model loading and graph compilation, send
a request to the Node 0 service from a host in the serving network.

```shell
curl "http://$NODE0_IP:8000/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kimi-k3",
    "messages": [
      {"role": "user", "content": "Explain why prefix caching helps repeated long prompts."}
    ],
    "temperature": 0,
    "max_tokens": 128
  }'
```

Expected result: HTTP 200 with a JSON response containing a non-null choice and
generated text. Generation may stop before `max_tokens` when a stop token is
reached.

For a multimodal smoke test, replace the message content with an image and a
text instruction:

```json
{
  "role": "user",
  "content": [
    {"type": "image_url", "image_url": {"url": "<IMAGE_URL_OR_DATA_URL>"}},
    {"type": "text", "text": "Describe the image."}
  ]
}
```

For automatic tool selection, keep `--enable-auto-tool-choice` and
`--tool-call-parser kimi_k3` in the startup command and include `tools` and
`tool_choice: "auto"` in the chat request.

## 7 Accuracy Evaluation

Use the full 93-layer, 896-expert checkpoint with the four-node service in
Section 5.1.

### Using AISBench

Follow [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) to
configure the `kimi-k3` chat-completions endpoint and run GPQA. Keep the model
and tokenizer revisions, chat rendering, reasoning mode, sampling parameters,
dataset, and evaluator revisions fixed when comparing results. Record completed,
failed, missing, and unparsed samples alongside the score.

### Using Language Model Evaluation Harness

See [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md) for
evaluator setup. Use the same full checkpoint and serving configuration when
comparing results across backends.

## 8 Performance Evaluation

### Using AISBench

Refer to [AISBench performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation)
for configuration and execution instructions.

### Using vLLM Benchmark

After the service in Section 5.1 is ready, run the following from a load generator
in the serving network with the same vLLM version and tokenizer files. Set
`NODE0_IP` and `TOKENIZER_PATH` to the service address and local tokenizer path.

```shell
export NODE0_IP="<NODE0_IP>"
export TOKENIZER_PATH="<KIMI_K3_TOKENIZER_PATH>"

vllm bench serve \
  --backend openai-chat \
  --base-url "http://$NODE0_IP:8000" \
  --endpoint /v1/chat/completions \
  --model kimi-k3 \
  --tokenizer "$TOKENIZER_PATH" \
  --tokenizer-mode kimi_k3 \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 128 \
  --random-range-ratio 0.0 \
  --num-prompts 100 \
  --max-concurrency 4 \
  --request-rate inf \
  --ignore-eos \
  --save-result \
  --result-dir ./benchmark-results
```

This example measures 100 requests with at most four in flight, using random
1024-token inputs and 128-token outputs. The chat format can add input tokens;
`--ignore-eos` prevents early EOS from shortening the requested output length.
`--request-rate inf` sends requests as soon as concurrency slots are available.
Check that all 100 measured requests succeed and review throughput, TTFT, and
TPOT in the console and saved JSON results. Adjust lengths and concurrency for
the intended workload; this example is not a peak-performance configuration.
See the [vLLM benchmark tools](https://docs.vllm.ai/en/latest/benchmarking/)
for additional options.

Record the checkpoint revision, topology, graph mode, Prefix Cache setting,
input/output lengths, concurrency, completed requests, and error count together
with throughput and latency. Report cold-prefix and prefix-hit workloads
separately. With DSpark, also record the draft checkpoint and speculative-token
count.

## 9 Performance Tuning

### 9.1 Reference Configuration

The following is the configuration used by the full-checkpoint commands above,
not a claim of optimal throughput or latency for every workload.

| Deployment | Nodes | Total logical NPUs | TP per DP rank | Global DP / EP | Max Num Seqs per Engine | Max Num Batched Tokens | Max Model Len | Graph Mode |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| Full W4A8, mixed prefill/decode | 4 A3 | 64 | 16 | 4 / 64 | 16 | 8192 | 133120 | `FULL_DECODE_ONLY` |

Optional DSpark uses draft TP16 and draft `max_model_len=4096`, as shown in
Section 5.2. Use seven speculative tokens for the RadixArk GQA draft or the
Inferact seven-token MLA draft, and five for `Inferact/Kimi-K3-DSpark-Block5`.
Account for its extra memory before enabling it.

### 9.2 Tuning Guidelines

| Goal | Parameters to evaluate | Check before adopting a change |
| --- | --- | --- |
| Higher throughput | Increase `--max-num-seqs` or `--max-num-batched-tokens` gradually | Memory headroom, completed requests, and tail latency |
| Lower latency | Compare the baseline with a matching DSpark draft | End-to-end latency, draft acceptance, and accuracy on the same workload |
| Longer context | Adjust `--max-model-len` together with concurrency and the token budget | Target/draft compatibility and available hybrid KV-cache capacity |
| Repeated prompts | Enable Prefix Cache and compare cold and cached requests separately | Cache-hit metrics and output correctness |

Refer to the [public performance tuning guide](../../developer_guide/performance_and_debug/optimization_and_tuning.md)
and [feature matrix](../../user_guide/support_matrix/feature_matrix.md) for
general tuning methods.

## 10 FAQ

For common environment, installation, and general parameter issues, refer to
the [Public FAQ](../../faqs.md). This section covers Kimi-K3-specific questions.

### Which checkpoint should I use with DSpark?

Use a draft that matches the target architecture, tokenizer, and quantization
variant. Section 3.1 lists the RadixArk GQA draft and both Inferact MLA drafts,
together with their `num_speculative_tokens` values. For QuaRot checkpoints,
retain the checkpoint's quantization metadata and bundled rotation tensors on
every node.
