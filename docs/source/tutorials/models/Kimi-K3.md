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
[Verify Multi-Node Communication](../../getting_started/installation.md#installation-multi-node-interconnect).
Use a reachable address and the correct communication interface on each node.
All nodes must use the same model revision and compatible software stack.

## 4 Installation

The Kimi-K3 installation and deployment instructions in this tutorial apply
only to the Atlas A3 series.

### 4.1 Docker Image Installation

Follow the [installation requirements](../../getting_started/installation.md#installation-requirements) for
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
[source installation guide](../../getting_started/installation.md#installation-existing-cann-install), using
the selected main checkout and its verified vLLM commit instead of the older
release versions in the generic examples.

### 4.3 Model Runner V2 Target-Only Validation

The Model Runner V2 gates below target vLLM `v0.27.1` at commit
`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`. Verify the source checkout before
building the runtime image:

```shell
test "$(git -C /vllm-workspace/vllm rev-parse HEAD)" = \
  "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac"
```

Set the following environment variable before starting a target-only MRV2
functional test:

```shell
export VLLM_USE_V2_MODEL_RUNNER=1
```

Omit `--speculative-config` from the launch command. The bounded functional
configuration uses `--max-model-len 2048`, `--max-num-seqs 4`,
`--max-num-batched-tokens 512`, `--mamba-cache-mode align`, and prefix caching.
Use `--enforce-eager` for the eager baseline, or
`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` for ACL Graph.

The target-only regression starts separate eager and ACL Graph engines with the
same seed and requires identical token IDs for block-boundary, exact-prefix and
partial-prefix requests. The automated ACL Graph test installs a worker-local
counter around `ModelAclGraphManager.run_fullgraph` and requires every worker
to replay at least once; merely configuring or capturing `FULL_DECODE_ONLY` is
not accepted as graph evidence.

Run the metadata, NPU zeroing, and target functional gates from the repository
root:

```shell
pytest -sv tests/ut/worker/test_model_runner_v2_mamba.py
pytest -sv tests/ut/worker/test_kv_block_zeroer.py
pytest -sv tests/e2e/pull_request/one_card/model_runner_v2/test_kv_block_zeroer_npu.py
pytest -sv tests/e2e/pull_request/four_card/test_kimi_k3.py::test_k3_mrv2_without_draft
python benchmarks/k3_mrv2_kv_zeroer.py --layers 24 --blocks-per-step 4
```

The functional test uses a reduced dummy target to exercise the runtime shape
and cache lifecycle. It does not establish checkpoint loading or model
accuracy. Before declaring the target gate complete, repeat eager and ACL Graph
parity with the full real checkpoint, record the vLLM/vLLM Ascend commits,
image digest, CANN and torch-npu versions, and retain the graph replay log and
worker replay count together with the zeroer benchmark result.

This target-only scope does not establish long-context capacity, P/D support,
or MRV2 DSpARK correctness. Keep using MRV1 for production DSpARK until the
separate draft gates are completed.

### 4.4 Model Runner V2 MLA DSpARK Eager Validation

The MLA eager gate keeps both the target and draft in eager mode. The MLA draft
declares a raw-prefix-sum auxiliary-hidden-state contract when it is loaded;
the K3 target validates the requested layer boundaries and hidden width before
selecting that stream. The speculator then validates the aggregate runtime
shape, dtype, and device without reading NPU tensor values, and delegates
context/query preparation, Markov sampling, and state commit/rollback to the
upstream MRV2 `DSparkSpeculator`.

Run the contract tests and the four-card dummy functional comparison:

```shell
pytest -sv tests/ut/models/test_dspark_aux.py
pytest -sv tests/ut/spec_decode/test_dspark_speculator.py
pytest -sv tests/e2e/pull_request/four_card/test_kimi_k3.py::test_k3_mrv2_mla_dspark_eager
```

The functional test starts independent target-only and MLA DSpARK eager
engines with the same seed. It compares exact token IDs across block boundaries
and cold/repeated/reset requests with prefix caching enabled, and requires the
draft-token metric to be non-zero so speculative decoding cannot be bypassed
silently. The target-only oracle must report a real prefix-cache hit. vLLM
0.27.1 does not report a reusable prefix-cache hit for the same request while
DSpARK is enabled, so the draft run validates deterministic output but does not
claim draft + prefix-cache state reuse.

This reduced dummy test does not validate checkpoint loading, QuaRot, draft
acceptance patterns, or model accuracy. Before accepting the MLA eager gate,
repeat it with the full target and MLA draft checkpoints and retain evidence
for zero, partial, and full acceptance, block crossing, abort/request reuse,
and deterministic parity with target-only eager. Prefix-cache state reuse while
DSpARK is enabled also remains a separate follow-up gate. GQA DSpARK and draft
ACL Graph remain separate follow-up scopes. Multimodal draft inputs are also
deferred for this bring-up because vLLM 0.27.1 keeps DFlash multimodal capability
disabled.

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

export HCCL_IF_IP=$LOCAL_IP
export GLOO_SOCKET_IFNAME=$NIC_NAME
export TP_SOCKET_IFNAME=$NIC_NAME
export HCCL_SOCKET_IFNAME=$NIC_NAME
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export HCCL_BUFFSIZE_EP=2048
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OPENBLAS_NUM_THREADS=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
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
