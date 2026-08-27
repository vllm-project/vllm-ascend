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
| GQA and MLA DSpark adapters | Supported with a matching draft checkpoint; see Section 6 |
| MTP adapter | Implemented; validate the target and draft checkpoint pair separately |
| Atlas A5 SiTU MX quantization | Cross-built; target-hardware execution is still required |

Refer to the [supported features](../../user_guide/support_matrix/supported_features.md)
for the project-wide feature matrix.

## 3 Choosing a Checkpoint

Kimi K3 checkpoints serve different validation purposes. A reduced
checkpoint must not be used to report GPQA or other semantic accuracy.

| Checkpoint | Typical storage | What it can validate | What it cannot validate |
| --- | ---: | --- | --- |
| Full 93-layer, 896-expert W4A8 | About 1.49 TB | Deployment and benchmark accuracy | Not applicable |
| Full 93-layer, 16-expert derivative | About 113 GB | Single-node integration and long-context execution | Full-model semantics and full expert routing |
| Five-layer, 16-expert W4A8 derivative | About 12.1 GiB | Real quantized loading, KDA/MLA/MoE execution, graph and cache parity | Full-depth behavior and benchmark accuracy |

For a storage-limited real-weight validation checkpoint, derive the five-layer,
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
The reduced checkpoint is a validation artifact, not a model release. Keep the source
checkpoint revision and a manifest of retained tensors with the artifact so
that it can be reproduced when the quantized weights change.
:::

## 4 Installation

Use an Atlas A3 image containing the vLLM version pinned by this release and
the matching vLLM-Ascend build. Mount the checkpoint at the same path on all
nodes.

For multi-node serving, first follow the
[multi-node communication check](../../installation.md#verify-multi-node-communication).
The commands below cover the A3 configurations validated for this integration.
The A2 deployment from the vLLM-Ascend 0.23 guide is not carried forward as a
validated main-branch configuration until it is rerun with the current runtime.

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

### 6.1 Enabling DSpark

For the GQA path, a public draft checkpoint is
[`RadixArk/Kimi-K3-DSpark`](https://huggingface.co/RadixArk/Kimi-K3-DSpark).
For the MLA path, use a matching Kimi K3 MLA draft checkpoint. Add the same
speculative configuration to the Node 0 and headless worker commands:

```shell
--speculative-config \
'{
  "method": "dspark",
  "model": "<MATCHING_GQA_OR_MLA_DRAFT_PATH>",
  "num_speculative_tokens": 7,
  "draft_tensor_parallel_size": 16,
  "max_model_len": 4096,
  "draft_sample_method": "greedy",
  "enforce_eager": true
}'
```

The example uses seven draft tokens, so each proposal cycle contains seven
draft steps followed by one target-model verification step. Set
`draft_tensor_parallel_size` to the topology used to shard the draft model.

K3 cache grouping is derived from the target and draft layer layouts rather
than a hard-coded TP16 layout. The grouping contracts cover TP8 and TP16, but
the full-checkpoint deployment documented above was validated with TP16. For a
different TP size, first verify that both checkpoints' head and hidden
dimensions are divisible by that TP size, then rerun the functional and
accuracy checks in Sections 8 and 9.

## 7 Two-Node Prefill-Decode Deployment

The functional Prefill-Decode (P/D) check uses one 16-NPU A3 Prefill node and
one 16-NPU A3 Decode node. Both engines use TP16/EP and the same checkpoint,
tokenizer, KDA/MLA cache layout, and model revision. Install Mooncake and check
the data-plane network as described in the
[multi-node Mooncake guide](../features/pd_disaggregation_mooncake_multi_node.md).

Use these K3-specific settings in addition to the common model and environment
arguments from Section 5:

| Setting | Prefill | Decode |
| --- | --- | --- |
| Parallelism | TP16/EP | TP16/EP |
| Execution mode | `--enforce-eager` | `FULL_DECODE_ONLY` |
| Hybrid state layout | `--mamba-cache-mode align` | `--mamba-cache-mode align` |
| KV role | `kv_producer` | `kv_consumer` |
| Prefix Cache | Enabled | Enabled |

Add the following arguments to the Prefill service. Select a `kv_port` outside
Mooncake's reserved AscendDirectTransport range; for a 16-NPU node, use a port
of at least 36000.

```shell
--enforce-eager \
--mamba-cache-mode align \
--kv-transfer-config \
'{
  "kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_producer",
  "kv_port": "<PREFILL_KV_PORT>",
  "kv_connector_extra_config": {
    "prefill": {"dp_size": 1, "tp_size": 16},
    "decode": {"dp_size": 1, "tp_size": 16}
  }
}'
```

Add the following arguments to the Decode service:

```shell
--mamba-cache-mode align \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
--kv-transfer-config \
'{
  "kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_consumer",
  "kv_port": "<DECODE_KV_PORT>",
  "kv_connector_extra_config": {
    "prefill": {"dp_size": 1, "tp_size": 16},
    "decode": {"dp_size": 1, "tp_size": 16}
  }
}'
```

Start the standard Mooncake proxy with the Prefill and Decode endpoints, then
send requests to the proxy rather than directly to an engine:

```shell
python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --host 0.0.0.0 \
  --port 9000 \
  --prefiller-hosts <PREFILL_HOST> \
  --prefiller-ports <PREFILL_SERVICE_PORT> \
  --decoder-hosts <DECODE_HOST> \
  --decoder-ports <DECODE_SERVICE_PORT>
```

The proxy CLI may evolve with the shared P/D implementation. Treat the linked
Mooncake guide and `--help` output from the checked-out revision as
authoritative for proxy-only arguments. Do not change the K3 model, tokenizer,
TP size, or hybrid cache mode between the two engines.

## 8 Functional Verification

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

Run concurrency and benchmark requests from a host inside the trusted serving
network. A developer workstation or VPN should be used only for bounded smoke
requests.

## 9 Accuracy Validation

Use the following validation ladder:

1. Use a reduced W4A8 checkpoint for execution validation. Compare cold
   prefill, a Prefix Cache hit that leaves exactly one token to prefill, and
   a post-reset cold run. Check deterministic output tokens, close and finite
   chosen-token log probabilities, complete outputs, and `FULL_DECODE_ONLY`
   replay.
1. Run GPQA with the full 93-layer, 896-expert checkpoint on the four-node
   deployment. This is the semantic accuracy gate and cannot be replaced by
   a reduced checkpoint.

Keep the model revision, tokenizer, chat rendering, reasoning mode, sampling
parameters, dataset revision, and evaluator revision fixed when comparing
GPQA results. Record completed, failed, missing, and unparsed samples in
addition to the final score. Refer to [AISBench](../../developer_guide/evaluation/using_ais_bench.md)
or [lm_eval](../../developer_guide/evaluation/using_lm_eval.md) for evaluator
setup.

## 10 Performance Evaluation

Use [AISBench](../../developer_guide/evaluation/using_ais_bench.md) or the
[vLLM benchmark tools](https://docs.vllm.ai/en/latest/benchmarking/) from a
server-side load-generator environment. Record the checkpoint revision,
topology, graph mode, Prefix Cache setting, input/output lengths, concurrency,
completed requests, and error count together with throughput and latency.

Reduced checkpoints are useful for execution and scaling comparisons, but
their throughput is not representative of the full 896-expert model.

## 11 FAQ

### The service returns an incomplete or null choice

First check every rank log for a worker abort, a non-finite tensor, or a graph
replay failure. Then repeat the same deterministic request with log
probabilities and verify that every generated token has a finite chosen-token
log probability. An HTTP 200 response alone is not a sufficient pass condition.

### A Prefix Cache hit fails only at block-size plus one token

Use a prompt that leaves exactly one uncached token after a full cached block.
Compare its output tokens and chosen-token log probabilities against a cold
prefill and a post-reset cold run in the same model instance. This exercises
the one-token prefill classification without conflating it with decode.

### P/D starts but requests hang

Verify that both engines use `--mamba-cache-mode align`, the same model
revision and TP size, complementary Mooncake roles, reachable non-overlapping
KV ports, and topology values matching the actual Prefill and Decode groups.
Check both engine logs and the proxy response; engine health alone does not
prove hybrid KDA/MLA state transfer.
