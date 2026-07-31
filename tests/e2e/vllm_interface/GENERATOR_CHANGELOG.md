# Interface mapping generator accuracy log

This log records why each generator iteration changed, the boundary case it
handles, and the evidence used to decide whether a result is a source risk or a
generator problem.

## v0.13.0 - resolve statically exact indirect patch owners

- Starting commit: `0da1dc25c`.
- Problem: two real patch sites were present in source but were not emitted by
  the generator. One helper received the target vLLM module through a parameter;
  the other cached a module returned by literal
  `sys.modules.get("vllm...")`. These were generator omissions, not upstream
  incompatibilities.
- Change: for a private helper parameter, inspect all active-main direct call
  sites and bind the parameter only when every call resolves to the same exact
  vLLM module or class. Track literal `sys.modules.get("vllm...")` and
  `sys.modules["vllm..."]` bindings as exact module provenance.
- Safety boundary: a parameter assignment, an unknown/missing/indirect call,
  conflicting call arguments, or a non-literal `sys.modules` key prevents
  attribution. Release-only `vllm_version_is("...")` calls do not influence
  the main-branch result. Lexical qualified names keep a nested helper that
  shadows a module-level helper separate; a full-source rerun after fixing this
  boundary produced the same intended result.
- Rejected shortcut: collecting helper arguments by parameter name across the
  whole module was not used because unrelated local scopes can reuse the same
  name and would create false owners.
- Fixed-source effect: relations increase from 970 to 971. The only new edge is
  `HunYuanVLProcessingInfo.get_hf_processor`; the literal `sys.modules` site
  adds a second occurrence to the existing `get_kv_cache_coordinator` edge.
  All 970 prior exact edges remain. The 44 findings are byte-identical, so the
  seven real upstream risks are unchanged; review and generator issues remain
  zero.
- Audited pre-version output SHA-256:
  `71d130f83609dd0c70dd4777fac5bd6162aede488ff5e76e9efacebc5158a566`.

## coverage audit v0.1.0 - add an independent candidate backstop

- Starting commit: `afd2f6794`.
- Problem: a zero-review generator result proves that every candidate found by
  the generator was classified, but it cannot prove that the generator did not
  silently miss a source dependency.
- Change: add a second AST scanner that does not import the generator. It
  independently enumerates main-branch patch assignments, direct inheritance,
  and callable overrides, then checks that every source site has exactly one
  relation or finding in the schema-v3/v4 JSONL output. It verifies the pinned
  vLLM and vllm-ascend SHAs and reports missing, conflicting, orphan, and
  generator-review dispositions separately.
- Audit corrections made before this checkpoint: strip generic bases such as
  `Base[T]`, finish alias/re-export collection before deriving class edges, and
  stop override lookup at a downstream method owner before considering a later
  upstream method. These were independent-auditor errors, not generator gaps.
- Fixed-source evidence: the raw first run reported 975 candidates, 22 missing,
  and 65 orphan sites. After correcting only the three audit rules above, it
  reports 1,015 candidates, 6 missing, and 9 orphan sites, with no conflicting
  status and no generator-issue review. The remaining six sites are one
  external Triton object and five complex multiple-inheritance cases; the nine
  orphan sites are exact PyTorch-only overrides. They are intentionally not
  attributed to the generator until the independent scanner has exact external
  indexing and C3 MRO support.
- Tests: all 33 isolated generator/auditor tests pass; Ruff passes for the two
  new Python files.
- Reason: completeness needs an independent, high-recall source inventory. Its
  own false positives and false negatives must be fixed before it can be used
  to justify a generator change.

## coverage audit v0.3.0 - exact external source and C3 site audit

- Starting commit: `c1ba0c59a`.
- Problem: the first independent audit lost Python base order, used DFS instead
  of C3, and did not index the pinned external source. That produced five false
  override candidates, one external Triton false patch, and nine PyTorch-only
  orphan dispositions.
- Change: retain each AST base in source order, resolve aliases and star
  re-exports, and calculate strict C3 over vLLM, vllm-ascend, and explicitly
  supplied external package indexes. An unknown base, alias ambiguity, cycle,
  or failed C3 merge remains incomplete and never selects an owner. Exact
  structural nodes for `abc.ABC`, `typing.Generic`, and `typing.Protocol` are
  modelled without treating arbitrary standard-library or external classes as
  complete.
- Ownership boundary: a value imported through a vLLM module is followed to its
  defining package before it can become a patch candidate. This removes the
  Triton re-export false positive generically rather than by file or symbol
  allowlist. External effective override owners remain auditable candidates.
- Source provenance: the audit now verifies both mapping metadata and the actual
  source input. vLLM and vllm-ascend must be exact Git checkout roots at the
  requested HEAD. Each external package must be an exact Git checkout or a
  manifest snapshot whose complete Python file set and every SHA-256 digest
  match. This also fixes Git output decoding for non-ASCII Windows paths.
- Rejected intermediate result: strict external C3 initially reported 23
  `incomplete_mro` candidates (15 through `abc.ABC`, eight through
  `typing.Generic`). They were auditor modelling gaps, not generator omissions;
  the three exact structural nodes removed them without relaxing other bases.
- Fixed-source result: 1,018 candidates and 1,018 classified sites: 185
  inheritance, 167 patch, and 666 override. Missing, conflicting, orphan, and
  generator-issue review counts are all zero against vLLM `88402a41...`,
  vllm-ascend `81d3450...`, and PyTorch `449b176...`.
- Tests: 12 dedicated auditor tests pass; Ruff check, Ruff format check, and
  `git diff --check` pass.
- Reason: a site can be accepted only after a second implementation reaches the
  same source inventory under the exact dependency versions. Upstream or
  downstream source risk is not rewritten merely to make the audit pass.

## v0.12.0 - resolve exact external inheritance without widening vLLM scope

- Starting commit: `2f0b95b5e`.
- Problem: six candidates could not be decided because the combined MRO
  stopped at `torch.nn.Module`. Guessing a later vLLM owner would be wrong, but
  leaving the exact runtime dependency unindexed made valid vLLM overrides and
  the `MoonViT3dPretrainedModel.to` patch invisible.
- Exact external input: the upstream `vllm-interface` lane installs vLLM with
  `VLLM_TARGET_DEVICE=empty` and then installs vllm-ascend requirements. The
  effective PyTorch pin is therefore vllm-ascend's `torch==2.10.0`, official
  commit `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0`.
- Change: accept optional external package indexes and include the defining
  package in schema v4. The CLI requires an expected external SHA and verifies
  it against either the checkout HEAD or every file digest in a source snapshot
  manifest. The boundary UT now resolves a record from its declared source
  package instead of assuming every definition lives below the vLLM root.
- MRO boundary: exact structural bases `abc.ABC`, `typing.Generic`, and
  `typing.Protocol` are modelled explicitly. Any other unindexed base still
  makes the chain incomplete; a regression test proves that an unknown parent
  inside an indexed external class remains `review/ambiguous_mro`.
- Scope boundary: a patch whose target is a vLLM class remains a vLLM boundary
  even when the patched method is inherited from PyTorch. A downstream method
  whose effective overridden owner is only PyTorch is not added to the vLLM
  relation table; it is retained as `excluded/external_only_override`. When
  PyTorch shadows a later vLLM candidate, that candidate is retained as
  `excluded/external_override_owner`.
- Rejected intermediate result: the first strict implementation treated every
  unindexed standard-library base as opaque. It reduced relations from 966 to
  847 and created 131 reviews, almost all through `abc.ABC`; this was a
  generator regression, not 125 new source breaks. Explicit standard-library
  structural bases restored the previously verified edges without relaxing
  arbitrary external MRO handling.
- Fixed-source effect: 970 relations (192 inheritance, 123 monkey patch, 655
  override) and 44 classified findings. Review and generator issues are both
  zero. The seven real upstream risks are unchanged. Relative to v0.11, the
  only four new relations are the MoonViT `to` patch plus verified
  `set_aux_hidden_state_layers`, `get_attn_backend`, and `get_kv_cache_spec`
  overrides. Nine external-only overrides and two externally shadowed vLLM
  candidates are explicit exclusions.
- Audited output SHA-256:
  `2ebb4f0979eec3e59eaf5d6abee99702a723acadd639e6659a38a27fac36465f`.

## v0.11.0 - separate live injections from stale patch candidates

- Starting commit: `d22bb2aef`.
- Problem: every unguarded assignment to a missing upstream member was labelled
  as the same upstream risk, even though some members are intentionally added
  and used by verified replacement methods while others are no longer read by
  current upstream code.
- Change: build a member-use closure per upstream owner. Verified patch
  replacements are roots; `self.<member>` references reach injected
  replacements transitively. Reachable missing members become
  `expected/inject_missing_member`; unreachable ones remain
  `risk/possible_stale_patch`.
- External boundary: an unreachable missing method on a class with a direct
  external base becomes `review/external_inherited_method`, not an asserted
  vLLM removal.
- Safety boundary: a dead helper merely present in the same module is not
  enough; it must be reachable from a verified patch binding. Incomplete
  external inheritance is still not guessed.
- Fixed-source effect: two `_split_ba_for_tp` occurrences and three MiniMax
  helper injections became expected; the two guarded Qwen properties remain
  expected. Five obsolete Triton/sample patches are now explicit stale risks;
  `MoonViT3dPretrainedModel.to` is external review. Total risk findings fell
  from 13 to 7; expected findings rose from 4 to 9; review findings are the
  five incomplete MRO cases plus this one external method. Generator issues
  remain zero and verified relations remain 966.

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
