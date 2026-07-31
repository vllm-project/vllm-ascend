# Interface mapping generator accuracy log

This log records why each generator iteration changed, the boundary case it
handles, and the evidence used to decide whether a result is a source risk or a
generator problem.

## v0.10.0 - classify non-callable field mutations

- Starting commit: `03f03957c`.
- Problem: module fields, class fields, dataclass-field injections, and global
  state swaps entered callable replacement resolution and were reported as
  generator failures.
- Change: index module/class values separately from callables. Existing field
  writes are retained in the main output as `verified/field_mutation`; fields
  added under a negative `hasattr` or field-membership guard are retained as
  `expected/inject_missing_field`.
- Safety boundary: an unguarded missing field is a risk; a dynamic owner or a
  right-hand side that may be callable does not receive field classification.
- Fixed-source effect: six existing field mutations and two guarded field
  injections left generator review without disappearing from the main result.
  Review findings fell from 13 to 5 and generator issues from 8 to 0; relations
  remained 966.
- The first full run incorrectly treated
  `causal_conv1d_update = causal_conv1d_update_cpu` as a data field and hid one
  callable patch. Callable resolution now takes precedence when a symbol has
  both a function definition and a later assignment; a regression fixture
  covers this exact boundary case.
- Reason: these are real downstream dependencies on upstream state, but they
  are not callable signature relations. Keeping a separate verified finding
  preserves variable-change visibility without corrupting the method table.

## v0.9.0 - classify saved and restored original callables

- Starting commit: `ded2d6c6f`.
- Problem: saving an upstream method into a backup attribute and restoring a
  temporarily patched method were reported as unresolved replacement calls.
- Change: preserve provenance for direct callable aliases and literal-name
  `getattr` snapshots. A write back to the exact source target is classified as
  `restore_original`; a same-owner missing backup attribute containing
  `original` is classified as `save_original`.
- Safety boundary: a different owner, multiple possible sources, a dynamic
  attribute name, or a non-backup alias remains review rather than being
  treated as lifecycle evidence.
- Fixed-source effect: two save-original records and the expected temporary
  verifier restore became explained exclusions. The same provenance rule also
  surfaced seven restore assignments that v0.8 silently skipped, matching the
  independent raw patch-site audit. The final result contains 2 save and 8
  restore findings; review findings fell from 16 to 13; generator issues fell
  from 11 to 8. Verified relations remained 966.
- Reason: these assignments describe patch lifecycle, not independent
  downstream implementations, and their identity is statically provable.

## v0.8.0 - resolve typed lazy module exports

- Starting commit: `91f3356f8`.
- Problem: `vllm.platforms.current_platform` is created by module
  `__getattr__`, so the index could not find
  `current_platform.verify_quantization` even though its interface is declared
  as `Platform`.
- Change: bind a lazy export to its annotated class only when the module both
  annotates the exact export name and handles that fixed string in
  `__getattr__`. Patch evidence also retains the source target expression in
  addition to the canonical definition owner.
- Safety boundary: an annotation alone, a dynamic name, or an unresolved type
  does not create an alias.
- Fixed-source effect: the temporary platform verifier patch became one
  verified relation to `Platform.verify_quantization`, retaining
  `current_platform.verify_quantization` as source evidence. Relations
  increased from 965 to 966; findings fell from 33 to 32; generator issues
  fell from 12 to 11. Its restore assignment remains a separate
  lifecycle-classification task.
- Reason: the annotation and literal lazy-export branch jointly prove the
  runtime interface owner; the earlier missing-owner result was a generator
  error.

## v0.7.0 - synthesize provable dataclass constructors

- Starting commit: `92e942be8`.
- Problem: `ModelRunnerOutput.__init__` exists at runtime because the class is
  decorated with `@dataclass`, but the source has no explicit method node.
- Change: synthesize `__init__` for statically resolved dataclasses and derive
  the parameter contract from annotated fields, inherited dataclass fields,
  defaults, `default_factory`, `init=False`, `kw_only`, `KW_ONLY`, and
  `ClassVar` exclusions.
- Safety boundary: dynamic decorator/field options, unresolved external bases,
  or an unprovable dataclass field graph do not produce a synthetic method.
- Fixed-source effect: the `ModelRunnerOutput.__init__` patch moved from a
  missing-member risk to one verified patch relation with a synthesized
  12-parameter constructor contract. Relations increased from 964 to 965;
  findings fell from 34 to 33; upstream risks fell from 14 to 13.
- Reason: Python generates this callable deterministically from the class
  definition; treating it as absent was a generator error.

## v0.6.0 - resolve statically provable wrapper factories

- Starting commit: `944a5b924`.
- Problem: a patch replacement produced by any function call was treated as
  unresolved. This missed `make_load_weights`, `tensor_parallel_wrap`, and
  `_wrap_destroy_distributed_environment`.
- Change: resolve a local/downstream factory and inspect returns in that exact
  function scope without entering nested scopes. Accept one returned nested
  function or lambda, optionally together with an identity return of a factory
  parameter. Propagate the produced callable through a simple local assignment.
- Safety boundary: multiple returned wrappers, non-callable return values, and
  unknown callees remain review findings; the resolver does not execute code.
- Full-source audit exposed one adjacent resolver gap: the public
  `vllm.distributed.destroy_distributed_environment` name comes from
  `from .parallel_state import *`. The index now follows public callable star
  re-exports to the defining symbol instead of reporting the export as missing.
- Fixed-source effect: four findings became three verified patch edges because
  the two destroy-function patch sites use the same returned wrapper and remain
  separate evidence occurrences. Relations increased from 961 to 964;
  findings fell from 38 to 34; generator issues fell from 16 to 12.
- A fast candidate gate avoids analysing ordinary function calls as factories;
  the final full run returned to about 97 seconds after an initial 144-second
  audit run, without changing output hash
  `dc30d3c9d548b568f2518fb3a9e72a2f47788c592329c795ea3f4fa580b4e02c`.
- Reason: these return bindings are directly provable from AST control-flow
  shape and were true generator omissions.

## v0.5.0 - resolve class-body callable aliases

- Starting commit: `f147a936f`.
- Problem: `_method_nodes()` indexed only `def` statements. A valid binding
  such as `get_state_dtype = _310p_get_state_dtype` was therefore absent from
  both override discovery and patch-replacement lookup.
- Change: collect simple class-body callable assignments, including
  `staticmethod`, `classmethod`, and `property` wrappers; materialize them only
  when the right-hand side resolves to a real function or lambda. Class-valued
  data attributes are not promoted to methods.
- Evidence retained: the helper definition line and the class binding line.
- Fixed-source effect: the two patch sites using
  `AscendGatedDeltaNetAttention310.get_state_dtype` collapse into one verified
  monkey-patch edge with two evidence occurrences, and the class binding adds
  one verified override. Relations increased from 959 to 961; findings fell
  from 40 to 38; generator issues fell from 18 to 16.
- Reason: this is statically provable Python binding behavior and was a true
  generator omission, not an upstream compatibility risk.

## v0.4.0 - classify every non-verified candidate

- Baseline commit: `7954d7c2ab35959c450b48aa52dae5401a8d4b4f`.
- Source pair: vLLM `88402a41c4ab272ebbbd33f4a77fbbac0431cbb9`
  and vllm-ascend `81d3450128528be2c343232fcc28220814a15fd6`.
- Before: 959 verified relations and 40 records all labelled `unresolved`.
- Problem: real upstream removals, expected missing-member injection, inactive
  guards, incomplete MRO, field writes, and parser limitations were mixed
  together. A real missing upstream target was absent from the main mapping
  output.
- Change: schema v3 includes candidate findings in the main JSONL and gives
  every finding a `status`, `reason_code`, and `generator_issue` flag.
- After the fixed-source run: 959 verified relations plus 40 findings: 14
  upstream risks, 2 expected injections, 1 inactive branch, and 23 reviews.
  Eighteen findings are still marked as generator work for later iterations.
- Generic rules added in this iteration:
  - a missing inherited base is an upstream risk;
  - a missing patch member under `not hasattr(...)` is an expected injection;
  - a missing patch member under `hasattr(...)` is an inactive branch;
  - a missing member on a known upstream owner is an upstream risk;
  - an unknown patch owner remains a generator review instead of being guessed.
- Reason: later parser fixes must reclassify only genuine generator gaps. They
  must not make real upstream incompatibilities disappear merely to reduce the
  unresolved count.

## v0.3.0 - rollback baseline

- Commit: `7954d7c2ab35959c450b48aa52dae5401a8d4b4f`.
- Added the AST-only patch, inheritance, and verified-override generator.
- Tests: 12 passed; Ruff passed.
- Full fixed-source result: 959 relations and 40 unresolved records; output
  SHA-256 `52b5064257a30dfbf70a47e80061aa2319c60ee2c5e468051d710ce19461952e`.
