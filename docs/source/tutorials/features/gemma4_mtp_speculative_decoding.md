# Gemma4 MTP Speculative Decoding

## **Introduction**

Gemma4 MTP (Multi-Token Prediction) is a speculative decoding technique that accelerates Gemma4 inference by drafting multiple tokens per decode step with a lightweight assistant model, then verifying them against the target model in a single forward pass. On vllm-ascend, the Gemma4 MTP drafter reuses upstream vLLM's `Gemma4Proposer` for model-generic behavior and adds the Ascend-specific execution: cross-model KV sharing, per-KV-group attention metadata for the heterogeneous Gemma4 cache layout (sliding-window layers with `head_dim=256` and full-attention layers with `head_dim=512`), query-only RoPE for the draft (K/V are read from the target model's cache), and eager centroids sampling.

This tutorial describes how to deploy and verify Gemma4 MTP speculative inference on Atlas A5 (Ascend 950) hardware with `vllm-ascend`. The configuration below deploys the `google/gemma-4-31B-it` model with its `google/gemma-4-31B-it-assistant` MTP checkpoint and `num_speculative_tokens=3` on a TP=1 layout:

* Acceptance rate depends on the workload and prompting style; use the verification method below to measure it on your own traffic.
* Acceptance stays stable as concurrency scales on data-parallel deployments (one TP=1 replica per card).

## **Download vllm-ascend Image**

This tutorial uses the official image, version `{{ vllm_ascend_version }}`. Use the following command to download:

```bash
docker pull quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
```

## **Run with Docker**

Container startup command:

```bash
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
export NAME=vllm-ascend

# Run the container using the defined variables
# This test uses one Atlas A5 (Ascend 950) NPU card.
docker run --rm \
--name $NAME \
--net=host \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:\
/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-it $IMAGE bash
```

## **Start the vLLM Service with MTP**

```bash
vllm serve google/gemma-4-31B-it \
  --served-model-name gemma4-31b-it-mtp \
  --tensor-parallel-size 1 \
  --speculative-config '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":3}' \
  --max-model-len 5500 --max-num-seqs 16 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --trust-remote-code --host 0.0.0.0 --port 8845
```

Notes on the configuration:

* `--speculative-config` selects the MTP drafter and points `model` at the assistant checkpoint; `num_speculative_tokens=3` drafts three tokens per step.
* The service is ready when the log prints `Gemma4 MTP: propagated KV-sharing target to 4/4 draft layers`, which confirms the drafter reads K/V from the target model's cache. A lower number indicates the draft would fall back to its own (empty) cache and acceptance collapses.
* Data-parallel serving (`--data-parallel-size N`, one TP=1 replica per card) is supported with the same speculative configuration.

## **Verify the Acceptance Rate**

The per-position acceptance rate is exposed on the `/metrics` endpoint as `vllm:spec_decode_num_accepted_tokens_per_pos` (one series per draft position) over `vllm:spec_decode_num_drafts`. Scrape the deltas before and after a greedy batch and divide:

```python
rate[pos] = (accepted_per_pos_delta[pos]) / drafts_delta
```

Acceptance typically decreases with the draft position (`pos0 > pos1 > pos2`) and varies by workload. A collapse at `pos1`/`pos2` while `pos0` stays intact is not a workload effect — it typically indicates a draft attention metadata wiring issue.

## **Limitations**

* Supported on Ascend 950 (A5).
* The draft model's centroids-based sampling runs eagerly (no ACL graph capture for the centroids head); decode graphs are captured as usual with `FULL_DECODE_ONLY`.
* Both the target checkpoint and the assistant checkpoint must be available; the assistant is resolved through the same loader as the target model.
