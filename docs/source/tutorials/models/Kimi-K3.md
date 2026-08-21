# Kimi K3

## 1 Introduction

Kimi K3 is a multimodal mixture-of-experts model that combines Kimi Delta
Attention (KDA), gated Multi-head Latent Attention (MLA), attention residuals,
SiTU activations, and latent MoE layers. This guide describes the W4A8
deployment validated by the Kimi K3 integration for the vLLM 0.27-based
vLLM-Ascend branch.

The full W4A8 checkpoint is approximately 1.49 TB. Download it from
[ModelScope](https://www.modelscope.cn/models/sgl-npu/Kimi-K3-W4A8), or make it
available from shared storage at the same path on every serving node.

## 2 Supported and Validated Features

The integration contains the following model paths. The table distinguishes
runtime validation from implementation-only coverage.

| Capability | Status in this integration |
| --- | --- |
| Text and multimodal serving | Validated on Atlas A3 |
| TP16 with expert parallelism | Validated on one and multiple nodes |
| `FULL_DECODE_ONLY` ACL Graph | Validated |
| Prefix Cache and hybrid KDA/MLA state | Validated |
| Prefill-Decode disaggregation | Validated on two nodes |
| MTP and DSpark adapters | Implemented; use a matching draft checkpoint and validate it separately |
| Atlas A5 SiTU MX quantization | Cross-built; target-hardware execution is still required |

Refer to the [supported features](../../user_guide/support_matrix/supported_features.md)
for the project-wide feature matrix.

## 3 Choosing a Checkpoint

Kimi K3 checkpoints serve different validation purposes. A reduced or dummy
checkpoint must not be used to report GPQA or other semantic accuracy.

| Checkpoint | Typical storage | What it can validate | What it cannot validate |
| --- | ---: | --- | --- |
| Full 93-layer, 896-expert W4A8 | About 1.49 TB | Deployment and benchmark accuracy | Not applicable |
| Full 93-layer, 16-expert derivative | About 113 GB | Single-node integration and long-context execution | Full-model semantics and full expert routing |
| Five-layer, 16-expert W4A8 derivative | About 12.1 GiB | Real quantized loading, KDA/MLA/MoE execution, graph and cache parity | Full-depth behavior and benchmark accuracy |
| Five-layer, 16-expert dummy | Configuration only | Model construction, TP16/EP, graph replay, cache-state parity, finite outputs | Real weight loading, W4A8 numerics, or semantic accuracy |

The nightly test uses the last option so it does not depend on a large model
cache. Its configuration preserves the production hidden size, head geometry,
mixed KDA/MLA layout, attention residuals, and top-16 expert routing. It reduces
only the layer and expert counts, then initializes BF16 dummy weights.

For a storage-limited real-weight nightly artifact, derive the five-layer,
16-expert checkpoint from the W4A8 checkpoint as follows:

1. Keep tokenizer and configuration metadata.
1. Keep all non-expert tensors needed by the first five layers.
1. Keep routed experts 0 through 15 in those layers and rewrite the Safetensors
   index without renaming tensors.
1. Set `num_hidden_layers`, `num_experts`, and `num_experts_per_token` to 5,
   16, and 16 respectively. Keep KDA layers 1, 2, 3, and 5, and MLA layer 4.
1. Load the result with `--quantization ascend` and record deterministic token
   and log-probability parity. Do not create a task-accuracy baseline from it.

:::{note}
The reduced checkpoint is a CI fixture, not a model release. Keep the source
checkpoint revision and a manifest of retained tensors with the artifact so
that it can be reproduced when the quantized weights change.
:::

## 4 Installation

Use an Atlas A3 image containing the vLLM version pinned by this release and
the matching vLLM-Ascend build. Mount the checkpoint at the same path on all
nodes.

```shell
export IMAGE=<VLLM_ASCEND_A3_IMAGE>
export MODEL_ROOT=<HOST_MODEL_ROOT>

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
  -v "$MODEL_ROOT:$MODEL_ROOT" \
  "$IMAGE" bash
```

Verify the installed revisions before starting the service:

```shell
python -c "import vllm, vllm_ascend; print(vllm.__version__, vllm_ascend.__version__)"
```

## 5 Single-Node Functional Deployment

Use a full-depth 16-expert derivative for single-node functional testing. It
preserves every transformer layer but is not semantically equivalent to the
full 896-expert checkpoint.

```shell
export MODEL_PATH=<KIMI_K3_93_LAYER_16_EXPERT_PATH>
export TOKENIZER_PATH=<KIMI_K3_TOKENIZER_PATH>
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=1024
export HCCL_BUFFSIZE_EP=2048
export OMP_PROC_BIND=false
export OPENBLAS_NUM_THREADS=1

vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name kimi-k3 \
  --tokenizer "$TOKENIZER_PATH" \
  --quantization ascend \
  --safetensors-load-strategy lazy \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --enable-prefix-caching \
  --max-model-len 133120 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

Lower `--max-model-len` and `--max-num-batched-tokens` for a short smoke test.
The larger values above require a capacity check with the exact checkpoint and
runner memory configuration.

## 6 Four-Node Full-Checkpoint Deployment

The full checkpoint uses four Atlas 800 A3 nodes in a DP4/TP16/EP64 mixed
deployment. Start Node 0 first. Each worker owns one global DP rank and joins
Node 0 through the DP RPC address.

Set these variables on every node:

```shell
export MODEL_PATH=<KIMI_K3_FULL_W4A8_PATH>
export TOKENIZER_PATH=<KIMI_K3_TOKENIZER_PATH>
export LOCAL_IP=<CURRENT_NODE_IP>
export NODE0_IP=<NODE0_IP>
export NIC_NAME=<CURRENT_NODE_NIC>
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
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

Run on Nodes 1 through 3. Set `DP_START_RANK` to 1, 2, or 3 respectively:

```shell
export DP_START_RANK=<1_OR_2_OR_3>

vllm serve "$MODEL_PATH" \
  --headless \
  --host 0.0.0.0 \
  --port $SERVICE_PORT \
  --served-model-name kimi-k3 \
  --tokenizer "$TOKENIZER_PATH" \
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
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

For Prefill-Decode disaggregation, use the same model, tokenizer,
parallelism, and parser settings on both sides, then follow the
[Mooncake deployment guide](../features/pd_disaggregation_mooncake_multi_node.md).

## 7 Functional Verification

Run request generation inside the serving environment or its trusted service
network. Avoid sending a large benchmark load across a developer workstation
or VPN.

```shell
curl http://<NODE0_IP>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kimi-k3",
    "messages": [
      {"role": "user", "content": "Explain why prefix caching helps repeated long prompts."}
    ],
    "temperature": 0,
    "max_tokens": 128,
    "logprobs": true
  }'
```

The response must be HTTP 200, contain one non-null choice, and contain the
requested number of finite token log probabilities unless generation reaches a
configured stop token. Repeat an identical prompt and confirm that Prefix Cache
metrics increase without changing deterministic output tokens.

## 8 Accuracy and Nightly Validation

Use the following validation ladder:

1. The storage-free nightly guard loads the committed five-layer, 16-expert
   configuration with dummy BF16 weights on one 16-NPU A3 node. In one model
   instance it compares cold prefill, a Prefix Cache hit that leaves exactly
   one token to prefill, and a post-reset cold run. It requires identical token
   IDs, close and finite chosen-token log probabilities, complete outputs, and
   `FULL_DECODE_ONLY` replay.
1. A hosted five-layer, 16-expert W4A8 fixture should run the same parity case
   with real weights. This adds quantized loader and W4A8 numerical coverage
   while staying small enough for a nightly worker.
1. Run GPQA with the full 93-layer, 896-expert checkpoint on the four-node
   deployment. This is the semantic accuracy gate and cannot be replaced by
   either reduced fixture.

Keep the model revision, tokenizer, chat rendering, reasoning mode, sampling
parameters, dataset revision, and evaluator revision fixed when comparing
GPQA results. Record completed, failed, missing, and unparsed samples in
addition to the final score. Refer to [AISBench](../../developer_guide/evaluation/using_ais_bench.md)
or [lm_eval](../../developer_guide/evaluation/using_lm_eval.md) for evaluator
setup.
