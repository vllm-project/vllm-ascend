# DeepSeek-R1-Distill 310P adaptation notes

## Scope and model routing

DeepSeek-R1-Distill Qwen and Llama are dense text checkpoints. Reuse the
Qwen2/Llama model paths. Multimodal preprocessing, MLA, MoE, expert parallelism,
and MTP are not applicable to these representative checkpoints. Add runtime
patches only after a minimal reproduction shows that an existing path fails.

Use actual 310P hardware for acceptance. A2, A3, and 950DT runs do not establish
310P correctness. Record the number of devices used independently of the board
count; one 310P3 board with two devices is not TP4 evidence.

## Inspect tokenization before model inference

Check `config.json`, `tokenizer_config.json`, and `tokenizer.json` together.
The Distill Llama checkpoint can declare `LlamaTokenizerFast` while shipping
byte-level BPE. Some Transformers 5 versions choose a Llama class that replaces
that pipeline, losing whitespace and even all Chinese tokens. This is an input
compatibility failure; it does not justify changing an NPU attention kernel.

Use `examples/prepare_deepseek_distill_tokenizer.py` for affected local
checkpoints. It creates a separate tokenizer directory, preserves the original
tokenizer definition and chat template, checks token IDs against the raw JSON
pipeline, and refuses incompatible metadata or an existing output directory.
The model's chat template supplies its BOS token; check the rendered input
against the original pipeline rather than adding special tokens speculatively.

An effective debugging prompt is: “Compare the selected tokenizer class, raw
tokenizer.json token IDs, whitespace round-trip, Chinese input, and rendered
chat tokens before changing the model or kernel.” Removing Chinese test inputs
would hide this failure.

## Validate eager, capture, and replay separately

Use FP16 execution for BF16 weights on 310P. Bound context, maximum sequences,
and capture sizes explicitly. Start with `FULL_DECODE_ONLY` and sizes `[1,2,4]`.
Keep common engine arguments identical for eager and graph, and create engines
sequentially so their weight and KV allocations do not overlap.

Disable prefix caching for the replay regression. Generate multiple decode
tokens and alternate batch sizes 4, 1, 3, and 2. The three-request batch exercises
padding to the four-request bucket. Compare generated token IDs, excluding the
prompt; nonempty strings alone are insufficient. Repeat identical inputs to
check persistent-buffer updates.

Capture completion is not replay evidence. The 0.23.0.post1 release logs
`Replaying aclgraph`; current main logs `ACL graph replay is active`. Check the
actual source version and retain worker logs. Token equality between modes
cannot detect a shared tokenizer error or establish answer accuracy.

## Check attention intermediates against a finite reference

Mode parity can pass when both modes produce incorrect answers. In the initial
Qwen-1.5B run, eager and graph agreed but the 32-sample GSM8K evaluation scored
zero with both answer filters. CPU FP16 Transformers also produced nonfinite
logits, so it was unsuitable as a golden reference. Use a finite CPU FP32
reference and record generated token IDs, rather than trusting its backend name.

FP32 profiling found unscaled, unmasked QK scores around 310k in the first
attention layer, exceeding the FP16 limit before the usual attention scale.
An in-process Q/8 and attention-scale-times-eight diagnostic restored all four
32-token probes to the independent FP32 reference; the scale-one negative
control retained the incorrect outputs. Disabling chunked prefill did not fix
the problem.

The 310P backend pre-scales queries for unquantized Qwen2 FP16 and compensates
the attention scale in prefill, decode, and splitfuse paths. Keys and values are
unchanged. The power-of-two factor avoids additional rounding for normal values;
it does not establish a universal bound for arbitrary checkpoints or contexts.
Keep the workaround scoped and validate accuracy, graph replay, and performance
with the actual deployment configuration. Do not apply the diagnostic wrapper
on top of the production fix.

## Preserve measurement boundaries

Choose a complete software compatibility row. The initial release-stack run
used CANN 9.1.0, torch 2.10.0, torch-npu 2.10.0.post4, vLLM 0.23.0, and
vLLM-Ascend 0.23.0.post1. These results must not be relabeled as current-main
hardware validation. Follow the 310P Dockerfile's exclusion of Triton even if
release package metadata still declares it; report that dependency-check
exception explicitly.

Measure streaming TTFT/TPOT separately from offline batch wall time. Record
capture time separately from engine initialization and weight loading. For the
0.23.0 serving benchmark, `--random-range-ratio 0.0` requests fixed lengths;
one permits zero-length inputs. Retain failed attempts and original JSON.

Sample device memory during sustained load. Use the monitor's actual field
names: the tested 310P3 driver exposes `Memory-Usage`, not a separate HBM
counter. A runtime graph-allocation estimate is not total device memory.
Neither a pair of snapshots nor a short generation smoke proves leak freedom.

## Evidence ownership

Qwen-1.5B initially passed three rounds per mode, 12 batches/30 requests per
mode, and 32 generated tokens per request on 310P3 Device 0 TP1. Generated token
IDs matched exactly and real replay was logged. Capture was approximately two
seconds and runtime-reported graph memory was 0.15 GiB. This establishes only
short mode consistency on that release stack; accuracy, performance, larger
checkpoints, and sustained stability require their own evidence.

Existing PR #11130 by @330800awesome contains separate adaptation work and
author-reported measurements. Do not reuse its numeric baselines or validation
claims as results of this patch. Associate any independent submission with
issues #10668 and #9079 and state the incremental scope.

Llama-8B subsequently passed the same three-round, 60-request paired check with
the verified tokenizer directory. The release runtime reported approximately
two seconds for capture and 0.16 GiB of graph memory. Preserve the original
tokenizer failure as a negative control; do not describe the unmodified local
checkpoint path as passing on Transformers 5.5.4.
