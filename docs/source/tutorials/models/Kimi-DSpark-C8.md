# Kimi DSpark C8 with QuantFlashMLA

This integration uses the existing CANN `quant_flash_mla_metadata` and
`quant_flash_mla` interfaces for compatible Kimi dense-MLA draft decode on
Ascend A5. It builds on [PR #21569](https://github.com/vllm-project/vllm-ascend/pull/21569)
and includes follow-up fixes for real weight loading, query quantization and
cache writing. It does not add a native operator.

## Configuration and scope

Use Model Runner V2, a compatible BF16-compute target/draft pair, and a draft
checkpoint containing the attention quantization recipe and scales. The draft's
linear projections may remain floating point: C8 cache storage does not imply
quantized linear weights.

A minimal eager configuration, with paths replaced for the deployment, is:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve /path/to/target \
  --dtype bfloat16 \
  --quantization ascend \
  --tensor-parallel-size 2 \
  --kv-cache-dtype fp8 \
  --enforce-eager \
  --speculative-config '{"method":"dspark","model":"/path/to/draft-c8","quantization":"ascend","num_speculative_tokens":7,"draft_sample_method":"greedy","rejection_sample_method":"standard","enable_adaptive_verification":false}'
```

The example does not require a `--block-size` override. Quantized writer staging
uses the backend's supported kernel page size, independently of the manager's
provisional block size. Existing floating-writer page limits are preserved.

Both QuantFlashMLA interfaces must be available with the expected ABI. An image
tag, installation-directory name, or Python wheel version alone does not
identify those interfaces. Check the selected attention backend; unavailable
operators or unsupported geometry do not exercise this integration.

PCP/DCP greater than one remains excluded from QuantFlashMLA selection. This
change does not implement the scale propagation, current-token FP8 cache or
LSE combination required for that distributed path. Ordinary DSpark/DCP support
is a separate path and is not removed.

Real prefill attention retains the existing FIA path. Draft context-KV writes
may produce FP8 cache through the writer, but this is not a QuantFlashMLA
prefill-attention call. Multi-token speculative decode remains decode.

## Integration contracts

- Resolve the draft quantization configuration and local recipe layer IDs under
  the draft loader context, without changing target ownership or the recipe.
- Keep unquantized draft cache storage in the implementation's floating dtype,
  even when the target requests FP8 storage.
- Classify both projection weight methods during fused postprocessing; reject
  incompatible pairs instead of inventing a `weight_scale`.
- Supply persistent FP32 RoPE and latent scale vectors to the KV writer. Refresh
  their values in place on weight reload.
- Scatter quantized staging rows into the persistent cache using real slots.
  Preserve optional unquantized BF16 returns for their existing consumers.
- For the full-quant backend, quantize the complete query in the cache's units:
  `[latent_query, rope_query / kv_descale]`. Use the returned per-token/head
  descale for both components. Keep the other MLA preprocessing path unchanged.
- Preserve the existing raw-cache layout, metadata and execution extension
  points. No new environment variable or inference-time CPU inspection is added.

## Validation provenance

**This main-rebased Draft must be validated at its own revision.** The evidence
below is historical evidence for an isolated R5 source snapshot, not proof that
all subsequent main dependencies or the new image combination have passed.

Historical sources were PR #21569 head
`51a96c8180c33f249fd77f64f71201e5f7180044` plus the follow-up fixes, paired with
vLLM `24420b7a341af37a9bc3186e6fc6062512e0aa58`. The tested MLA SHA256 was
`3d606c53c2dff5948395f4316c2ba3aaa94470940ec7146e41ed99f4c05b4452`.
The fixed image ID was
`sha256:04c836f810f42a03639e7831a1b1fedc51f1375cb55f9c2626d21cd0b70e5c7a`.
Its Python framework sources were overlaid; native libraries and dependencies
were retained from that image. An image release label is not the source revision.

### Historical single-node evidence

A 93-layer reduced target with eight experts/top-k eight was tested on Ascend
950 DT, TP2, with a five-layer DSpark draft. Runs used standard greedy
verification, fixed seed, and no entropy, synthetic or adaptive acceptance.
All model configurations omitted user `block_size`.

| Test | Observed result | Limit |
| --- | --- | --- |
| Writer workspace CPU/mock tests | 17 passed | Not native execution |
| Native writer boundaries | Eight cases completed, including expected over-capacity rejection | Empty, page-boundary, nonsequential-slot and unwritten-row checks |
| Eager target-only/BF16/C8 | 64 requests per state, all paired final token sequences equal | 30 unique inputs; repetition is not extra independent data |
| Cache lifecycle | 33 requests per state, actual peak four live requests, no observed live-block collision | Controlled cache budget; no forced preemption |
| Decode-only graph | Ten requests per state matched its eager reference; actual target/draft capture and replay observed | Intermediate values were not inspected on every replay |

For the 24 ordinary unique inputs, BF16 accepted 2479/4522 draft tokens
(54.8209%) and C8 accepted 2465/4592 (53.6803%). Six paired per-request ARs
improved, twelve were unchanged and six decreased. Artificial page-boundary
inputs are reported separately and must not be pooled into representative AR.
Counts include verification before final output-length trimming.

The target-only baseline itself produced whitespace-only text for five of the
24 ordinary inputs and repetitive text for others. Inputs were frozen raw token
IDs, not chat-template requests. Matching these outputs is evidence about
execution/cache consistency, not answer-quality acceptance. No production AR,
full-expert model quality or performance claim follows from these results.

### Historical single-node PD evidence

Separate Prefill TP2 and Decode TP2 instances, with DCP/PCP one, used the same
frozen sources and target through MooncakePullConnector/Ascend Direct. Three
states completed eleven requests each. Two core raw completions matched a
no-PD baseline token for token, and actual transfer parameters plus native
receive completion were recorded. Boundary requests had transfer evidence,
not paired no-PD token baselines. A separate two-request simultaneous-scheduling
check completed for BF16/C8. The online containers additionally installed
`orjson==3.11.4`; this was not part of the unchanged historical offline image.

The raw completions were repetitive. Chat requests returned HTTP 400 because
the tokenizer had no chat template; those were retained as failures. No
multi-node, PD+DCP, graph-PD or service-performance acceptance is claimed.

## Tests and remaining checks

Focused tests cover API detection, descale forwarding, draft recipe ownership,
projection methods, joint query quantization, cache specs and staging geometry.
The tensor-view test checks shape, dtype, shared address and values rather than
requiring Python object identity. Cache-spec fixtures explicitly configure
unrelated sparse-offload/HiSparse settings. Floating staging tests preserve the
current main PROLOG page limit.

Before leaving Draft, run the affected pytest suites with the main-pinned vLLM
revision, native writer and QuantFlashMLA checks, and the default-configuration
three-state model comparison on the exact proposed revision and native package.
Also complete repository format/CI checks. DCP adaptation, a new operator
package's device compatibility, random-sampling distribution tests, long
contexts, full-expert acceptance rates and uninstrumented performance remain
separate validation work. Raw model data and deployment logs are not included
in this repository.
