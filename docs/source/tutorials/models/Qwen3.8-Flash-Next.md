# Qwen3.8-Flash-Next

## 1 Introduction

Qwen3.8-Flash-Next is an experimental multimodal MoE model with 125B language-model parameters (6B activated), a 51B n-gram embedding, and a 4B MTP head. Its `Qwen4ExpForConditionalGeneration` architecture combines Gated DeltaNet and Qwen Sparse Attention (QSA), four-stream Gated Residual connections, Position Learning Enhancement (PLE), vision input, and a native 262,144-token context window.

Support on Ascend is experimental. For the v0.26.0 release stack, the plugin
vendors the Qwen4Exp model and weight loader from upstream vLLM PRs 53896 and
53899, then replaces CUDA-only QSA, PLE, and HyperConnection glue with
NPU-compatible implementations. Run the real-weight and accuracy gates on the
target hardware before production use.

## 2 Supported Features

| Feature | Status | Notes |
|---|---|---|
| Text generation | Experimental | The v0.26 adapter contains the required Qwen4Exp backport |
| Image and video input | Experimental | Reuses the Qwen3-VL processor and vision encoder |
| Expert parallelism | Experimental | The model has 512 routed experts |
| ACLGraph | Unverified | Start with graph mode; use eager mode only for isolation |
| FlashComm1 | Unverified | Deprecated in current vLLM Ascend; do not use for the initial gate |
| MTP | Experimental | Register `Qwen4ExpMTP` through the Ascend adapter |
| PLE CPU offload | Unsupported | Upstream implementation uses CUDA IPC stream semaphores |

Refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) and [Feature Guide](../../user_guide/feature_guide/index.md) for general platform options.

## 3 Environment Preparation

The BF16 checkpoint contains approximately 176B parameters including the n-gram embedding and MTP weights. TP8 is the initial single-node validation configuration because 24 query heads are divisible by 8; confirm actual memory use on the selected machine.

=== "Atlas 800 A3"

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:latest-a3
    docker run --rm -it \
      --shm-size=1g \
      --name vllm-ascend-qwen38-flash-next \
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
      -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
      -v /models:/models \
      -p 8000:8000 \
      "$IMAGE" bash
    ```

=== "Atlas 800 A2"

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:latest
    docker run --rm -it \
      --shm-size=1g \
      --name vllm-ascend-qwen38-flash-next \
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
      -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
      -v /models:/models \
      -p 8000:8000 \
      "$IMAGE" bash
    ```

Install the paired vLLM v0.26.0 build, then install the vLLM Ascend adapter from
source. The Qwen4Exp model sources required by this release are contained in the
adapter; a separately patched vLLM checkout is not required. Do not upgrade
`transformers` independently from the versions pinned by the two projects.

```bash
cd /vllm-workspace/vllm
pip install -e .
cd /vllm-workspace/vllm-ascend
pip install -e .
python -c "import vllm, vllm_ascend; print(vllm.__file__); print(vllm_ascend.__file__)"
```

## 4 Deployment

First validate architecture and API wiring with dummy weights:

```bash
cd /workspace
HCCL_OP_EXPANSION_MODE=AIV \
vllm serve /models/Qwen3.8-Flash-Next \
  --served-model-name qwen3.8-flash-next \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --load-format dummy \
  --port 8000
```

Dummy weights do not validate the PLE embedding shards, QSA cache updates, quantization scales, or checkpoint key mapping. Repeat the command without `--load-format dummy` for the mandatory real-weight gate.

The real-weight MTP and graph-mode verification configuration is:

```bash
mkdir -p /home/w00804037/Qwen3.8-flash/Logs

vllm serve /models/Qwen3.8-Flash-Next \
  --host 0.0.0.0 \
  --port 8010 \
  --served-model-name qwen3.8-flash-next \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --enable-expert-parallel \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 4096 \
  --async-scheduling \
  --no-enable-prefix-caching \
  --speculative-config '{"method":"qwen4_exp_mtp","num_speculative_tokens":3}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"ascend_compilation_config":{"fuse_norm_quant":false}}' \
  > /home/w00804037/Qwen3.8-flash/Logs/serve-mtp-full-decode.log 2>&1
```

Do not specify `cudagraph_capture_sizes`; this configuration deliberately uses
the platform defaults. `fuse_norm_quant` is disabled because CANN versions that
do not provide `aclnnAddRmsNormBias` cannot compile that fusion. This does not
disable `FULL_DECODE_ONLY` graph capture.

If graph capture fails, add `--enforce-eager` to isolate the failure. Do not enable `VLLM_PLE_CPU_OFFLOAD`; the upstream offload transport is CUDA-only.

If eager mode still fails inside TorchDynamo while isolating a portable QSA
operator, retry once with `TORCHDYNAMO_DISABLE=1` before reporting the first
failing operator and stack trace. This is a diagnostic fallback, not the
supported performance configuration.

## 5 Functional Verification

Check readiness and then require a non-empty generated response:

```bash
curl -sf http://127.0.0.1:8000/v1/models

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-flash-next",
    "messages": [{"role": "user", "content": "Say hi."}],
    "temperature": 0,
    "max_tokens": 16
  }'
```

For multimodal verification, send one image request using the same OpenAI-compatible endpoint and require HTTP 200 with non-empty `choices`. A successful startup log without a successful request is not a pass.

## 6 Accuracy Evaluation

Run the model evaluation suite only after real-weight text and multimodal smoke tests pass:

```bash
pytest -sv tests/e2e/models/test_lm_eval_correctness.py \
  --config tests/e2e/models/configs/Qwen3.8-Flash-Next.yaml \
  --tp-size 8
```

Add the YAML baseline after recording the real Ascend result. Do not use the model-card score as an Ascend accuracy result because the evaluation harness and runtime differ.

## 7 Performance

The Torch QSA fallback prioritizes functional portability and has not been performance-tuned. After correctness validation, benchmark representative text and multimodal workloads:

```bash
vllm bench serve \
  --model Qwen/Qwen3.8-Flash-Next \
  --served-model-name qwen3.8-flash-next \
  --dataset-name random \
  --random-input-len 2048 \
  --random-output-len 256 \
  --num-prompts 200 \
  --request-rate 1
```

Report the theoretical context limit (262,144 tokens) separately from the largest context that passes real-weight inference on the tested hardware. The initial capacity gate is 131,072 tokens with `max-num-seqs=16`.

## 8 Validation Status

The real-weight eager baseline was validated on 16 Ascend NPUs with CANN 9.1.0,
TP8, PP2, and expert parallelism. `/health`, `/v1/models`, and a two-token text
completion succeeded and the service remained healthy.

The MTP3, asynchronous-scheduling, and `FULL_DECODE_ONLY` validation was then
used to identify and fix these integration failures:

- unsupported generic `mtp` method: use `qwen4_exp_mtp` and normalize the draft config;
- unavailable `aclnnAddRmsNormBias`: set `fuse_norm_quant=false`;
- missing multimodal arguments on `embed_input_ids`: accept the runner signature;
- invalid dynamic specialization of `query_start_loc`: keep its graph shape static;
- PP2 MTP feedback width of 2560 instead of four HC streams: use
  `hidden_size * hc_count` and treat the local last-stage drafter as its first
  logical stage.

The graph-mode service and GPQA accuracy gates have not passed yet. The current
logs under `/home/w00804037/Qwen3.8-flash/Logs` end during engine initialization,
so no GPQA score is reported by this PR. Rerun both gates after deploying the
integrated patch; do not interpret the eager smoke test as graph-mode or
accuracy validation.
