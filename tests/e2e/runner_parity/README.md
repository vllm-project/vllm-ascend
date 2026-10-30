# GLM-5.1 paired runner evidence

This collector uses the complete model and one shared server configuration. The
runner selector is the only arm-specific setting. The caller owns sequential
server lifecycle, accelerator exclusivity, immutable image/source/weight records,
and cleanup. Do not run a full quality dataset as a prerequisite.

## Collect

The exact A5 TP8/EP8 configuration is configs/glm51_a5_tp8.json. It records the
frozen source revisions, full checkpoint revision and graph configuration. Both
arms use the same FP8 mapping compatibility fix, now included in this change's
worker patch. Apply this change to the frozen Ascend source when reproducing
the recorded image's FP8 mapping behavior. The image ID records the local
runtime, not a downloadable tag.

    python tests/e2e/runner_parity/serve.py --runner v1 --model /path/to/full/checkpoint

Start the first arm with explicit Model Runner V1 selection, then run:

    python tests/e2e/runner_parity/collect.py --runner v1 --inputs inputs.json --output run/v1 --stage precision

Stop the first arm, verify accelerator resources are released, start V2 with the
same settings, and collect into run/v2 using the same inputs.json. Collect the first performance round explicitly with --stage performance --rounds 1. For alternating
paired rounds, use --stage performance --rounds 1 --round-offset N. Precision and
performance collectors retain completed case IDs when resumed; input and request
hashes are compared before generating a paired report.

Default precision coverage is Chinese/English/code, input lengths 128/2048/32768,
greedy and seeded non-greedy, client concurrency 1/8, generated token IDs/logprobs,
and fixed-history prompt logprobs. Generated outputs are bounded to 32 tokens;
these checks are not a semantic benchmark or a full-dataset accuracy claim.

The numerical comparison executes the exact check_logprobs_close function body
from an explicitly supplied, pinned vLLM tests/models/utils.py. Its source hash is
recorded. The helper checks reciprocal token membership through the first
generated divergence, and exact prompt top-k key sets for common histories.
Missing logprobs, mismatched requests, duplicate cases, or incomplete pairing are
errors. Sampled text equality is reported, not universally assumed.

## Compare and validate

    python tests/e2e/runner_parity/compare.py --v1 run/v1 --v2 run/v2 --upstream-utils /vllm/tests/models/utils.py --output run/comparison.json
    pytest --confcutdir=tests/e2e/runner_parity tests/e2e/runner_parity --parity-root run --upstream-utils /vllm/tests/models/utils.py --expected-rounds 1

Use --expected-rounds 3 when three rounds were actually captured. A separate
serving-only capture explicitly skips the HTTP precision check; pass
--control-root control_run to validate its separate deterministic control.
Ownership records are required: captures with foreign NPU processes must be
rerun before using them for formal serving conclusions.

The focused CPU checks reject missing evidence and deliberately incompatible
distributions. Full-model tests require actual artifacts and are skipped when
their paths are not supplied.

Standalone comparator checks require `pytest` and `regex`. The pinned runtime
also provides `parser_checks.py --model /path/to/checkpoint --output parser.json`
for synthetic GLM reasoning/tool formats with the actual tokenizer. That CPU
check does not replace real-model generation and parser integration coverage.

## Performance scope

Performance uses fixed input token IDs, fixed output lengths, no returned
logprobs, real-request warmup, streamed TTFT/TPOT/E2E and token-count checks.
The stream requests `return_token_ids=True`: TTFT uses the first generated
token-ID chunk, including special tokens with empty visible text. The complete
stream token count must agree with final usage. Invalid streams are rejected
and saved as `stream-error-*.json` when an error directory is supplied. A
capture made with visible-text timing must not be mixed with token-ID timing.
Requests at concurrency 1/8/32 cover input lengths 128/2048/8192, with three
paired rounds planned; report only the rounds actually collected. Raw SSE
events and before/after server metrics remain available.

Client concurrency does not prove actual engine batch size. Inspect the engine
iteration histogram and server logs. HTTP input throughput is a serving metric,
not isolated prefill kernel throughput. Request prefill/decode phase counters
must be labeled by their server-side scope; separate synchronized phase/profile
measurements are required for device-level attribution.

Model-specific parser, MTP, exceptional-path and additional regression coverage
must be recorded separately. This suite does not claim the entire community task
is accepted merely because these first paired checks pass.

## Fixed-group offline control

The HTTP collector creates and saves inputs.json on its first run. Reuse that
file unchanged for offline control; do not substitute approximately equivalent
text. Run each arm in a fresh process/container with the same visible devices:

    python tests/e2e/runner_parity/control.py --runner v1 --model /path/to/full/checkpoint --inputs inputs.json --output control/v1/steps.json --mode graph
    python tests/e2e/runner_parity/control.py --runner v2 --model /path/to/full/checkpoint --inputs inputs.json --output control/v2/steps.json --mode graph
    python tests/e2e/runner_parity/compare_control.py --v1 control/v1/controlled-precision.jsonl --v2 control/v2/controlled-precision.jsonl --upstream-utils /vllm/tests/models/utils.py --output control/comparison.json

Stop the first process and verify NPU release before starting the next. The
control keeps EngineCore in-process and TP workers in separate processes, so all
eight requests enter the queue before stepping. Its four groups cover uniform
128/2048/32768-token inputs and mixed lengths. It records scheduled inputs on all
eight ranks and preserves raw token IDs/logprobs. Metadata capture is separate
from performance measurement. Raw precision comparison alone does not validate
runtime identity or accelerator ownership.

Use `--mode eager` for an explicitly eager control and `--token-budget 16384`
for a separate, same-configuration paired prefill-budget experiment. A
representative MTP control uses `--mode eager --mtp-tokens 3 --features` on both
arms. It also checks stop/cancel/recovery and context boundaries, and requires
positive draft and accepted-token counters. A configured MTP flag alone does
not establish that speculation ran. Use a fresh output directory after an
interruption; retain the incomplete capture separately.

Add `--model-parser-checks` to generate two chat requests with the checkpoint's
own template: a short reasoning response and a weather tool call. The capture
retains the actual prompt/output token IDs and raw text, then applies the pinned
`glm45` reasoning and `glm47` tool parsers. Each arm must finish both requests
and satisfy parsing assertions. This is distinct from `parser_checks.py`'s
synthetic CPU format checks; a configured parser alone is not coverage.

Validate a complete paired feature capture with:

    pytest --confcutdir=tests/e2e/runner_parity tests/e2e/runner_parity/test_features.py --feature-root feature_run --upstream-utils /vllm/tests/models/utils.py

The feature checks require both successful arms, matching runtime/source hashes,
clean ownership windows, actual speculative counters, all seven boundary cases,
and raw model generations with successful reasoning/tool parsing. MTP may accept
different draft lengths in the two arms, so the base control's identical-step
schedule assertion is not applied to this feature capture. The numerical rule,
input/request identity, and runtime checks still apply.
