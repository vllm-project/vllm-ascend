# Generalized nano MTP operator provenance

This port starts from vLLM-Ascend upstream main
`b49962987e89b850586f1819ce8f85daa85a0f81`, targeting image vLLM 0.28.0.
The import carries only the generalized operator families:

| Operator | Source | Pinned revision |
| --- | --- | --- |
| MTP LIM, including Q=1 | `xwLearnsLLM/nanovllm-DSA-offload:ops_lim_standardization` | `012962af05f06bf1bdd089ca7e7e4357d021682f` |
| MTP copy-SFA, including Q=1 | `xwLearnsLLM/nanovllm-DSA-offload:ops_copysfa_mtp_standardization` | `1518a90dd17592dc3aa96c16869cfde597a250b4` |
| MTP C8 LIM | Previously imported `fused_li_manage_mtp_c8` in `Angazenn/vllm-ascend:kv_offload_sparse` | through `0edbec116f23aa8f343553ee753a3699649308cf`, initially `11fbfda30` |

The nano repository is private. Source files were carried from the prior
vLLM-Ascend integration at `bdbec521587981853a4abffe71529c9dc21af3cc`.
Copyright notices are retained and the CANN license is included in the parent
attention directory. Earlier passing results from that integration do not
establish correctness of this new upstream port.

## Native contract and adaptations

LIM writes contiguous int32 `[B,32768]` request miss lists shared directly with
copy-SFA, plus per-query `[T,1,2048]` selections and miss counts. All sequence
length inputs are contiguous int32 NPU tensors; no operator needs CPU lengths.
Generalized native LIM supports 1–14 query rows per request, while copy-SFA
supports 1–16. The initial serving route limits queries to 1–7 and its
block-aligned hot budget to `[Q_max * 2048,16256]`.

Native copy-SFA supports 8, 16, 32, 64 and 128 attention heads. Serving pads
fewer than eight heads to eight, then compacts the output. MTP C8 LIM is
registered and built for A3, but is deliberately not selected in serving;
its runtime correctness is unverified in this port.

Build adaptations retain vLLM-Ascend operator names, private copy-SFA tiler
helper names, and the internal `FirstFillScatterCopy` dependency. The ordered
ACLNN launcher preserves tensor/workspace lifetimes and binds stream resources
for deferred execution. It does not replace unrelated operator launchers.
Clean the generated native build directory when changing these operator ABIs.
No legacy non-MTP LIM/copy-SFA, non-MTP C8 LIM, or standalone tail-attention
operator is included.

The BF16/FP16 LIM import additionally invalidates state `-3` TopK positions
beyond each query's causal visible-key count. At the pinned revision, a
four-query, four-key request publishes all four source IDs for every query,
even though `npu_lightning_indexer` returns one, two, three and four valid IDs.
The fix uses the existing device-side visible count before publishing both
source and destination outputs. It does not change the offload states `-2`
and `-1`. The C8 kernel logic remains unchanged and unverified at runtime.

## Python integration

Opt in with `sparse_kv_offload_config.fused_op_type="nano"`;
`use_fused_overlap` and nano are mutually exclusive. The existing runner owns
request pool slots and allocation generations in `CpuGpuBuffer`s. Common
attention metadata carries these to the SFA offload builder, which prepares
device sequence lengths, stable-prefix boundaries, cache budgets and tail
geometry once per target/draft step. Attention implementations own persistent
LIM maps, outputs and reusable copy descriptors. Shared-indexer layers refer
to their selection owner's outputs for the current step.

Each draft step has its own derived metadata buffers: the proposer prepares
all draft metadata before executing the first draft, so later one-query
steps must not overwrite its query boundaries or copy descriptors.

MTP index sharing remains enabled when configured by the model. Later draft
steps consume the proposer's compacted logical TopK rows and resolve their
resident slots without rerunning LIM. Resident selections have zero copy
misses; a prefill or invalidated first step bootstraps a compact TopK cache
once. Reused attention consumes exactly that saved selection, without adding
a newly advanced dense tail. Its source-to-slot map, cache budget and request
generation remain coherent for subsequent drafting and the next LIM call.

No separate MTP runtime, graph wrapper, batch wrapper or descriptor class is
introduced. Native inputs use authoritative device lengths after rejection.
Request allocation generations, budget changes and prefix rollback invalidate
residency. Eager fallback/prefill invalidates the overlapping hot arena.

Each request has two circular tail blocks after its hot buffer. RD2H populates
the registered host main cache, so the implementation explicitly restores
prior tail KV with at most two block-contiguous spans per KV component.
Every TP rank writes current query KV directly into its own local tail,
avoiding a race with TP0's current-query D2H. Padding uses private cache rows,
state `-3`, zero misses and a positive dummy cache budget. Real cold/reset
requests use `-2`; valid residency uses `-1`. These states are computed on the
device during capture/replay, rather than freezing capture-time activity.

Short prompts initially retain the existing eager offload path. Eligibility
uses immutable prompt length as a conservative lower bound, with a one-block
and seven-query margin above TopK. This routing policy needs no device-to-host
sequence-length synchronization. A short prompt remains on fallback even if
its generated sequence later becomes long.

Model validation used FULL_DECODE_ONLY with eager MTP3 drafting on Ascend A3,
DP2 TP8, GLM-5.2-w4a8, FlashComm1, shared-expert DP and prefix caching disabled.
Four sequential requests, each with 10,555 input tokens, returned the same
expected 82-token answer. Both DP replicas logged graph replay, and the
model configuration retained `index_share_for_mtp_iteration=true`.

Focused NPU validation passed 14 LIM cases and six metadata cases. After the
MTP reuse change, all six metadata cases and three new reuse cases passed;
the latter compare native copy-SFA with reference attention and exercise
eager execution, inactive-to-active graph replay, request-generation reset
and block-boundary reuse without repeated H2D.

The independent GLM RMSNorm bias-detection prerequisite must be identical in
nano-enabled and nano-disabled baseline comparisons. PD, 128K inputs, GSM8K,
performance comparisons and full-model graph drafting remain unverified
for this port. Runtime artifacts are recorded in the workspace validation
manifests.
