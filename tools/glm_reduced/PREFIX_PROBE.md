# GLM-5.2 prefix numerical diagnostic

This diagnostic compares a full 78-layer GLM-5.2 W4A8 model's boundary after
decoder layer 7 against an independently loaded eight-layer artifact. The full
reference still executes all 78 layers. It does not substitute the reduced
model as its own reference.

The reference runtime is the user's selected pinned version, not an assertion
that a full-model semantic accuracy evaluation has passed. The source checkpoint
must be verified against the same immutable source descriptor used for reduction.

## Captured values

- Residual-merged hidden state after layer 7.
- The same boundary passed through the loaded model's final RMSNorm.
- Full-vocabulary logits from its normal `compute_logits` / output-head path.
- Candidate's actual final logits, to check the instrumented early-head path.
- Candidate's actual final normalized state, to isolate norm versus head errors.

Hidden and residual tensors are cloned before invoking the potentially in-place
fused normalization. Token-axis TP shards are gathered and the original token
batch shape is preserved through normalization; only then are valid output rows
selected. Selecting a row before normalization produced small discrepancies
against the normal forward in the initial long-prefill self-check. Each TP rank
participates in head collectives, even if that rank returns
no logits. CPU copies and file writes make these instrumented runs unsuitable
for performance measurements.

Both sides use TP8/EP, PP1, batch one, eager execution, no MTP, no prefix caching,
no chunked prefill, and synchronous scheduling. This first diagnostic does not
qualify the original FULL_DECODE_ONLY/MTP nightly configuration.

The existing eight fixed token-ID requests are used, including lengths 2047,
2048 and 2049. Four sampled positions are collected: the last prefill position
and three decode positions. At the sampling boundary, the actual final logits
are saved before substituting a distribution that selects the fixed continuation
`[17, 18, 19, 20]`. Thus both sides use identical decode histories rather than
independent generated histories. These token IDs are test inputs, not answers.

## Commands

`engine.json` must specify the controlled engine arguments accepted by
`run_prefix_probe.py`, including `enforce_eager=true`, `max_num_seqs=1`,
`pipeline_parallel_size=1`, `async_scheduling=false`, and disabled prefix caching
and chunked prefill. No speculative configuration is allowed.

```bash
python -m tools.glm_reduced.run_prefix_probe \
  --role reference --model /verified/full/checkpoint \
  --requests tools/glm_reduced/data/glm52/requests.json \
  --engine-json engine.json --output reference-run \
  --reference-evidence 'Pinned runtime and source audit record'

python -m tools.glm_reduced.run_prefix_probe \
  --role candidate --model /verified/reduced/checkpoint \
  --requests tools/glm_reduced/data/glm52/requests.json \
  --engine-json engine.json --output candidate-run \
  --reference-evidence 'Same pinned runtime and source audit record'

python -m tools.glm_reduced.compare_prefix_runs reference-run candidate-run \
  --output comparison.json
```

Use new output directories. The collector never overwrites or automatically
retries a run. Partial runs retain `STARTED` status and cannot be compared as
successful observations. The comparator validates matching workloads, runtime,
probe source hashes and recorded environment. It reports absolute errors,
RMS error, cosine similarity and argmax agreement. Argmax agreement alone is
not a precision criterion.

The comparison remains `UNASSESSED` until numerical tolerances and reference
qualification are established; it never manufactures an accepted baseline.
If the first comparison is unsatisfactory, preserve artifacts and present a
diagnostic proposal to the user. Do not change tolerances, configurations or
repeat the experiment without the requested user confirmation.
