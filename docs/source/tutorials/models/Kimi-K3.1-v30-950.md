# Kimi K3.1 on v30 and 950DT

This guide accompanies the local backport of [PR #20878](https://github.com/vllm-project/vllm-ascend/pull/20878).
It targets 950DT (A5), with Model Runner V2. The original integration guide is
[Kimi-K3.1-Pro.md](Kimi-K3.1-Pro.md); its historical accuracy results and large PD
deployments are not validation of this v30 backport. 950PR is outside this recipe.

## Fixed source and runtime

| Component | Revision / requirement |
| --- | --- |
| vLLM | Local v30, `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| vLLM Ascend | Local v30, `24675566a41d35fa3ec01ceaf743789c03bf35ec`, plus this backport |
| Imported implementation | `8e07a81fc13d5a9c8e715738135fe09d0e5c12b0` |
| Container recipe | This checkout's `Dockerfile.a5` |
| CANN base | `quay.io/ascend/cann:9.2.1-950-ubuntu22.04-py3.12` |
| Transformer operators | The 950 ops-transformer 9.3.0 package pinned in `Dockerfile.a5` |
| Device build target | `SOC_VERSION=ascend950dt_9582` |
| Other dependencies | Use the torch, torch_npu, Triton Ascend and FLA versions installed by that Dockerfile |

Do not pair this backport with an arbitrary newer vLLM main. Do not copy only
Python files into an old wheel: QLI, SFA, metadata and AttnRes native schemas and
kernels change together. Build custom kernels from the modified checkout.

On the 950 host, copy both source trees, then build from the modified Ascend tree:

```bash
docker build -f Dockerfile.a5 \
  --build-arg VLLM_TAG=v0.30.0 \
  --build-arg VLLM_COMMIT=ced6857afa0ea7b2e3f0846a62e1394e90f15607 \
  --build-arg VLLM_VERSION_OVERRIDE=0.30.0 \
  --build-arg SOC_VERSION=ascend950dt_9582 \
  --build-arg COMPILE_CUSTOM_KERNELS=1 \
  -t vllm-ascend:k31-v30-950 .
```

The explicit version override uses vLLM's existing build setting: fetching a
single commit without tags otherwise lets setuptools-scm infer a version that
does not identify the v30 release. This affects plugin compatibility selection
and the launcher's version preflight.

Follow the original guide's 950 container device and driver mounts. Mount the
complete checkpoint read-only and a separate directory for logs. Check `npu-smi
info` inside the container. Driver/firmware must match the installed CANN stack.
The build above compiles this local checkout; a generic v30 image does not contain
the backport. Keep the build log and image digest with acceptance results.

## Checkpoint and memory

Use the **950DT W4A8_MXFP** ModelSlim checkpoint, for example the original guide's
`Kimi-K3.1-Pro-w4a8-mxfp`. A3 W4A8 weights and a GPU FP8 checkpoint are different
formats. Keep `config.json`, tokenizer assets, `quant_model_description.json`,
the safetensors index and every referenced shard together.

The first run uses the complete target model on eight visible devices. Confirm
that the full checkpoint fits their aggregate HBM with room for runtime buffers
and caches. Reducing context/sequence limits reduces cache memory, not weight
memory. If full weights do not fit, provision a larger topology and validate it
separately; removing experts/layers is not full-model acceptance.

## First launch

Run from the installed backport checkout inside the container:

```bash
python examples/kimi_k31_950.py \
  --model /models/Kimi-K3.1-Pro-w4a8-mxfp --check-only

python examples/kimi_k31_950.py \
  --model /models/Kimi-K3.1-Pro-w4a8-mxfp 2>&1 | tee /logs/k31-v30-950.log
```

`--dry-run` validates the checkpoint manifest and prints the command without
loading NPU libraries. `--check-only` also verifies the vLLM version, visible
950 devices, Ascend build family, critical native schemas and NoPE prolog import.
It does not allocate the full model or prove that all weights are loadable.

Defaults are TP8 / PP1 / PCP1 / DCP1, EP, eager execution, FP8 main/indexer KV,
1536-token blocks, 4096-token context, 2048 batched tokens and one active
sequence. MRV2 is mandatory. DSpark, PD, RecoverSSM and optional MoE overlaps are
not enabled. The local listener defaults to `127.0.0.1:8000`.

The launcher starts in language-only mode. After text acceptance, rerun with
`--multimodal` and test a known image. Both modes explicitly disable device
normalization because this PR's vision wrapper uses CPU-normalized pixels.

## Generation acceptance

In a second container terminal:

```bash
curl --fail http://127.0.0.1:8000/health
python examples/kimi_k31_smoke.py --url http://127.0.0.1:8000 \
  --output /logs/k31-v30-smoke.json
```

Acceptance requires a ready service after complete weight loading, positive
completion token counts and nonempty output for repeated requests, with no NPU,
shape, illegal-memory, missing-weight or graph errors in the server log. The
smoke script records raw responses; manually review output quality and compare
greedy outputs with a trusted K3.1 run using the same full checkpoint. A nonempty
response alone is not an accuracy benchmark.

## Incremental validation

1. Run operator tests after rebuilding: QLI v2/metadata, quantized SFA native
   heads, AttnRes cooperative, KDA state copy and output gate. These require NPU.
2. Pass eager full-model text acceptance, then image acceptance.
3. Enable graph execution and compare outputs with eager on identical prompts.
4. Validate DCP8 with matching block/interleave geometry; compare with DCP1.
5. Add a matching DSpark checkpoint, then validate accepted-token state and
   target/draft cache separation. Do not use the target checkpoint as the draft.
6. Validate PCP/PP, PD, prefix reuse and long context separately, using the
   source guide's topology only after checking every v30 flag and memory budget.

The code includes the PR's DCP/PCP and DSpark integration. CPU checks do not
establish their collective ordering, kernel numerics or hardware stability.
Record versions, full checkpoint identity, device HBM, topology, command,
startup log, raw responses, precision comparison and memory use for each stage.

## Troubleshooting

- Unknown `kimi_k31`: verify the installed Ascend checkout and plugin loading.
- Native schema missing/stale: rebuild the custom extension; inspect which
  `vllm_ascend` directory Python imports.
- Missing `cann_ops_transformer.mla_prolog`: install the Dockerfile's matching
  transformer operator package; NoPE must preserve the learned tail dimension.
- Cache geometry errors: keep main/indexer FP8, 1536 blocks, MRV2 and the hybrid
  cache manager. Do not use `--disable-hybrid-kv-cache-manager` for K3.1.
- Out of memory during weight loading: verify full weight capacity before
  changing cache limits. Out of memory during profiling can additionally require
  reducing batched tokens or memory utilization.

## Backport lifecycle

`patch_kimi_k31_config.py` supplies the absent v30 HF config registration. Remove
it when the paired upstream release provides equivalent config validation and
K3.1 identity through `with_hf_config`. Ascend cache specs carry a non-comparing
`is_draft` field because v30 lacks that metadata; remove this compatibility field
after adopting the upstream equivalent. Cache admission uses v30's existing
`kv_cache_utils` hooks, retaining its two-argument admission API and DCP block
resolution. In v30, pool accounting accepts only groups, so the paired planner
marks its indexer specs explicitly; worker projection retains global geometry
while counting only local layers. The main-only cache builder is not imported
by the v30 path. The existing release warmups run before the added JIT registry
compiles KDA state-copy, output-gate, indexer and DCP kernels.
