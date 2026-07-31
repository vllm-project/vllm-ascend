# Interface mapping generator accuracy log

This log records why each generator iteration changed, the boundary case it
handles, and the evidence used to decide whether a result is a source risk or a
generator problem.

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
