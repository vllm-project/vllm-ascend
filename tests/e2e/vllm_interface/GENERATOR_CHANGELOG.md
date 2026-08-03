# Interface mapping generator accuracy log

This log records why each generator iteration changed, the boundary case it
handles, and the evidence used to decide whether a result is a source risk or a
generator problem.

## v0.27.0 - deduplicate descriptor-derived receiver findings

- Red-test checkpoint: `9799e7037`. A classmethod replaced by a zero-argument
  ordinary function produced both `descriptor_kind_mismatch` and
  `invalid_receiver_binding`, even though both findings described the same
  descriptor-protocol change.
- Change: when an exact descriptor-kind mismatch is already reported, suppress
  only its derived receiver-binding finding. Keep a separate finding if an
  independent unknown decorator also changes the runtime callable contract.
- Safety boundary: this does not make the relation compatible and does not
  suppress any descriptor finding. It removes duplicate evidence only when
  the upstream and installed descriptor kinds are both known.
- Reason: one source incompatibility should have one owning finding. Duplicate
  derived findings inflate risk counts and make the generated report harder to
  review without adding evidence.

## v0.26.0 - persist and compare runtime signature contracts (in progress)

- Starting red-test checkpoint: `a1d5b6fa5`. Ten tests failed because the
  runtime signature model introduced in v0.25 was still transient: contracts
  were neither persisted nor compared, receiver binding collapsed invalid and
  unknown cases, and exact signature compatibility was not checked.
- Changes:
  - schema v6 serializes and reloads the upstream, downstream-definition, and
    installed runtime contracts without changing relation endpoint identity;
  - grouping includes the upstream runtime contract, while relation comparison
    reports contract changes separately and ignores the one-time schema-v5 to
    schema-v6 migration when the baseline contains no contract;
  - descriptor and signature allowlists are independent. A source-SHA-pinned
    descriptor classification no longer silently proves that a decorator is
    signature-transparent;
  - receiver binding now keeps a varargs-only method exact, marks a callable
    with no positional receiver slot invalid, and leaves the bound signature
    unknown when the installed descriptor kind is unknown;
  - exact upstream call shapes are bound against the installed downstream
    signature in the compatibility direction. Missing optional keywords,
    renamed keyword-capable parameters, extra downstream requirements, async
    protocol changes, and missing `*args`/`**kwargs` acceptance become
    supplemental `signature_incompatible` risks;
  - known descriptor mismatches remain owned by the descriptor finding instead
    of producing a duplicate derived signature risk;
  - deduplication merges runtime contracts across occurrences. Different
    reachable variants retain the relation but make its installed contract
    unknown and add `conditional_signature_contract` review evidence.
- Safety boundaries:
  - compatibility is evaluated only when both runtime contracts are exact;
  - unknown decorators and unknown descriptors are never guessed;
  - a schema-v5 baseline remains readable and does not create a false contract
    change solely because the new generator can now persist contracts;
  - Findings supplement the verified relation; they do not delete the
    downstream dependency edge.
- Test evidence: all 188 isolated generator and independent-auditor tests pass;
  Ruff and `git diff --check` pass.
- Fixed-source evidence using vLLM
  `88402a41c4ab272ebbbd33f4a77fbbac0431cbb9`, vllm-ascend
  `81d3450128528be2c343232fcc28220814a15fd6`, and PyTorch
  `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0`: 971 relations retain 100%
  endpoint equality with the schema-v5 mapping; all 1,019 independently
  enumerated candidates are classified with no missing or conflicting
  candidate. The 108 signature incompatibilities were manually and
  mechanically audited: 106 reject at least one valid upstream call shape and
  two change positional-argument meaning. They are source risks, not generator
  errors. Twenty unknown signature transforms remain conservative reviews.
- Performance note: this fixed-source run took about 687 seconds, compared with
  about 414 seconds for the historical v0.24 run. Correctness is accepted for
  this checkpoint, but profiling remains necessary before CI use.

## v0.25.0 - runtime signature contracts (in progress)

- Red-test checkpoint: `2e3445e46`.  Five tests first failed only because the
  relation model had no runtime-signature fields.  They cover
  `functools.wraps`, an unknown decorator inside `classmethod`, exact
  source-SHA pinning for `torch.inference_mode`, receiver binding, and an
  unknown decorator on a module-level patch target.
- Problem: the old mapping stored only the syntax-level `def` signature.  That
  is insufficient after a decorator changes the callable object, and it also
  mixes the explicit `self`/`cls` parameter with the interface seen after
  Python binds a method to an instance or class.
- Changes:
  - every verified override and monkey patch now keeps four distinct views:
    source definition, runtime entry, introspection-reported signature, and
    bound-call signature;
  - `functools.wraps(target)` keeps the wrapper's real runtime entry while
    recording the target signature exposed by introspection and the exact
    forwarded target;
  - descriptor binding removes a receiver only for an ordinary method,
    classmethod, or property installed in a class namespace; module and
    instance writes do not borrow class binding semantics;
  - decorator effects are processed from inner to outer.  Builtin descriptor
    decorators, standard transparent decorators, and exact source-SHA-pinned
    adapters stay exact; every other runtime transform fails closed with a
    supplemental `unknown_signature_transform` review;
  - exact decorator references are retained by AST-node identity when patch
    scanning reconstructs local functions.  This fixes a false unknown result
    for module-level `@classmethod` functions later installed on a class;
  - existing descriptor tests now assert descriptor and signature findings
    separately.  A decorator may have a known descriptor kind but still have
    an unknown runtime calling convention, so suppressing one dimension with
    the other would hide a real dependency risk.
- Test evidence: all 177 isolated generator and independent-auditor tests pass;
  Ruff and Ruff formatting pass.  A fixed-source regeneration and independent
  review are still required before this checkpoint is accepted.
- Independent review after the green tests found five release-blocking gaps:
  descriptor-only SHA allowlists do not prove signature transparency;
  receiver binding does not yet distinguish varargs, invalid bindings, and an
  unknown descriptor; contracts are not yet serialized or compared; exact
  upstream and installed signatures are not yet checked directionally for call
  compatibility; and relation deduplication does not merge contract variants.
  This commit is therefore a rollback-safe model scaffold, not an accepted
  generator result.
- Next known gaps: LoRA conditional wrapper protocols, Triton JIT launch
  protocols, `support_torch_compile` class transformations, and serialization
  of the new contracts.  Until those are modeled, the generator must retain
  their unknown/review results rather than claim complete accuracy.

## v0.24.0 - descriptor binding contracts (in progress)

- Red-test checkpoints: `922a810e8` (10 new tests; 9 initially failed and one
  existing-correct module-wrapper control passed) and `2e032820e` (8 focused
  installation tests; the typed lazy instance case initially failed).
  Checkpoint `2f3f910cf` adds eight failing scope/property cases; the ambiguous
  wrapper fixture was then replaced by an exactly provable classmethod alias
  in `83d7f5781`.  Checkpoint `5b38bb57b` adds nine alias/assignment tests;
  eight initially failed and the imported builtin alias control already
  passed.
- Problem: signature-only contracts could not see a change between an
  ordinary method, `property`, `classmethod`, and `staticmethod`.  In the
  pinned source pair this hid ten real override differences, including
  `Platform.num_compute_units`, whose parameter names still match.
- Changes:
  - schema v5 records the upstream definition kind, downstream definition
    kind, and the kind actually installed by a patch; descriptor fields stay
    outside relation identity so a binding change is not misreported as a
    removed/new edge;
  - a known mismatch or unknown/conditional kind keeps the verified Relation
    and adds a supplemental review finding; the independent coverage auditor
    ignores supplemental findings as dispositions, avoiding a false
    `verified + review` conflict;
  - class-body decorators and explicit `property(...)`, `classmethod(...)`,
    and `staticmethod(...)` wrappers are interpreted in Python application
    order.  Unknown outer decorators are not guessed, while an outer proven
    descriptor wrapper remains decisive;
  - patch definition kind and installed kind are separate, preserving cases
    such as an ordinary getter installed with `property(getter)` and
    `staticmethod(lambda ...)`;
  - class, module, and typed-runtime-instance targets are distinguished because
    only writes into a class namespace install descriptors;
  - an instance write no longer erases the descriptor kind of the upstream
    class member.  This preserves a real `classmethod -> instance attribute`
    mismatch, while the established `ordinary -> ordinary instance function`
    binding remains an intentional equivalent case;
  - semantic adapters for `vllm.tracing.instrument` and
    `torch.inference_mode` are enabled only for the exact pinned source SHAs;
    an unregistered source version remains `unknown` instead of inheriting an
    unverified assumption.
  - descriptor names are now resolved from the normal-path binding state at
    definition time.  This respects dead branches, deletion, import aliases,
    later rebinding, a missing `builtins` import, and class-local shadowing;
    conditional classmethod/staticmethod aliases remain explicit variants;
  - property getter, setter, and deleter definitions are retained as three
    separate callable contracts.  The getter remains the read signature after
    a later setter/deleter definition, and an accessor is accepted only when
    the preceding normal-path binding is proved to be a property.
  - class assignment aliases are materialized from every active control-flow
    branch and keep distinct descriptor variants.  A staticmethod read through
    another class becomes an ordinary method when installed, a property stays
    a property, and a bound classmethod object remains unknown instead of
    being copied as a new classmethod;
  - `property(getter)` assignments now seed the same accessor state used by
    `@name.setter`, and patch wrappers resolve the live module/function binding
    rather than trusting the wrapper's spelling or stale import entry.
- First fixed-source run (before class/module/instance refinement): 971
  relations, exactly matching all v0.23 edges, plus 67 supplemental reviews
  (52 unknown and 15 known mismatches).  Manual review proved that the 15
  mismatches are source facts, while most unknowns were analyzer errors caused
  by treating whole-class and module-level replacements as class descriptors.
- Additional fixed-source audit established the expected stable result:
  - all 655 overrides have known kinds and retain ten source mismatches;
  - all 124 patches have an installed-kind distribution of 66 not-applicable,
    50 ordinary, 3 classmethod, 4 staticmethod, and 1 property, retaining five
    source mismatches;
  - three module-level `@classmethod` replacements must remain classmethod when
    installed into a class, while the temporary
    `current_platform.verify_quantization` write is an instance attribute and
    therefore has no installed descriptor kind.
- Refined fixed-source run: the pinned source set generated in 330.4 seconds
  with 971 relations and 60 findings.  All 971 edges exactly match v0.23;
  the only additions are 15 supplemental `descriptor_kind_mismatch` reviews,
  with no unknown descriptor and no generator issue.  Independent source
  review confirmed every one of the 15 as a real binding difference (10
  overrides and 5 patches), not an analyzer error.  The output SHA-256 is
  `727710aa6c8229c71e3064f28f81764dda1f51e8420eef88da49f2cf5cf3d257`.
- Independent coverage audit v0.5 still classifies all 1,019 candidates with
  zero missing, conflicting, or generator-issue records and the same two
  pre-existing auditor-only patch orphans.
- Scope-aware regression run: checkpoint `259df81a8` regenerated the same
  pinned sources in 373.7 seconds.  It retained all 971 exact edges, all 60
  finding dispositions, all 15 real descriptor reviews, and the identical
  output SHA-256 `727710aa6c8229c71e3064f28f81764dda1f51e8420eef88da49f2cf5cf3d257`.
- Alias-variant regression run: checkpoint `2b3eddf10` regenerated the same
  pinned sources in 413.7 seconds with the identical 971 relations, 60
  findings, 15 descriptor reviews, and byte-identical SHA-256.  The two
  class-assignment aliases present in the fixed mapping therefore remain
  stable while the newly covered variants are available for future changes.
- Status: implementation tests pass (172 total).  The fixed mapping is exact
  for the descriptor forms exercised by the pinned source, but broader
  property accessor, wrapper-shadowing, and conditional descriptor red tests
  are still required before this checkpoint can be accepted as the general
  descriptor implementation.

## v0.23.0 - exact presence and missing-super checkpoint

- Previous accepted checkpoint: `10c86dc4a`; initial red-test checkpoint:
  `8079a7892`. Additional red-test checkpoints cover negative `hasattr`,
  wrapper reachability, and builtin `super()` targets.
- Problems fixed:
  - implicit exception scanning walked deferred lambda bodies and skipped
    short-circuit rules, while `assert False` did not produce an exact
    `AssertionError` path;
  - a negative `hasattr` path could still be emitted as a verified patch, and
    `hasattr` did not account for bound values or inherited members;
  - downstream methods and upstream base classes that were callable/classes
    on only some normal paths could be treated as unconditional;
  - same-name conditional class definitions overwrote one another, losing
    callable signature variants;
  - class-body `staticmethod(_impl)` and `classmethod(_impl)` assignments
    exposed `_impl` instead of the installed member, and lost conditional
    source presence; module-level descriptor wrappers were incorrectly
    promoted to callable endpoints;
  - a callable written over a definitely non-callable final member was called
    a field mutation instead of a possible stale interface patch;
  - a downstream `super().same_method(...)` call disappeared when the upstream
    method was removed, while dead calls and methods supplied by `object` could
    be false positives.
- Changes:
  - evaluate expression exceptions only on paths that execute now; model
    assertion success and exact `AssertionError` exits separately;
  - use final binding alternatives and complete MRO lookup for callable,
    class, value, unbound, and `hasattr` presence;
  - aggregate every final same-name class variant and every method binding,
    adding `unbound` for a variant that lacks the member; differing conditional
    base shapes make the MRO incomplete rather than selecting one branch;
  - propagate final kinds and callable signatures through provable class-body
    aliases while keeping the installed endpoint name;
  - report a missing upstream same-name `super()` target only for a reachable
    direct call, a complete MRO, and a method not supplied by `object`.
- Safety boundaries:
  - an incomplete inheritance chain remains review and is never guessed;
  - a genuine missing upstream class, patch target, or direct `super()` target
    remains a non-generator risk;
  - a downstream missing-member injection remains expected and is not turned
    into a verified relation;
  - builtin descriptor wrappers are recognized only in a class namespace with
    one positional argument and no keywords.
- Test evidence: all 137 isolated generator/auditor tests pass. The new
  fixtures cover assertion flow, short-circuit calls, custom exception
  inheritance, negative `hasattr`, conditional class/method presence,
  same-name class variants, assignment-form static/class methods, definite
  `None`, reachable/dead/builtin `super()` calls, wrapper-source alternatives,
  and module-level descriptor wrappers.
- Fixed-source result: vLLM
  `88402a41c4ab272ebbbd33f4a77fbbac0431cbb9`, vllm-ascend
  `81d3450128528be2c343232fcc28220814a15fd6`, and PyTorch
  `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0` generated in 307.8 seconds:
  971 relations and 45 findings (8 risk, 9 expected, 22 excluded, 6 verified),
  with zero review and zero generator issues. Its SHA-256 is
  `49e9b8df8d4abfc21f51d636e6502ba2d47891bbd2fa826aa1b5465dcd55c661`.
  All 673 raw contract records are byte-for-byte identical to the independently
  rerun v0.22 output. The only new finding is the real downstream call from
  `NPUModelRunner.postprocess` to the now-missing
  `GPUModelRunner.postprocess`; four `object.__init__/__repr__` false positives
  were reproduced and removed.
- Independent coverage audit v0.4 classifies all 1,019 candidates with zero
  missing, conflicting, or generator-issue records. It recognizes the new
  missing-super risk as a real candidate and retains only the two pre-existing
  monkey-patch orphans.
- This is a rollback checkpoint, not final completion. Descriptor binding
  kinds and `support_torch_compile` runtime class transformation remain the
  next accuracy stages.

## v0.22.0 - exact NPU platform path checkpoint

- Starting commit: `f2a2bf13a`; rollback checkpoint: `04e6a5cf8`.
- Problem: the callable-alias implementation correctly found two different
  signatures for `causal_conv1d_update`, but it treated
  `current_platform.is_cpu()` as reachable while generating an NPU consumer
  map.  That produced one false review and removed one otherwise verified
  monkey-patch relation.
- Change: the active-main condition evaluator now selects the exact inactive
  result for an argument-free `current_platform.is_cpu()` guard in this
  vllm-ascend/NPU generator.  The alias regression remains independently
  covered under an unknown runtime condition, so the fix cannot pass merely by
  suppressing every alias branch.  One new red test uses deliberately
  incompatible CPU/default signatures and proves that only the NPU-reachable
  signature is emitted.
- Test evidence: all 117 isolated generator/auditor tests pass; Ruff, Ruff
  format, and `git diff --check` pass.
- Fixed-source result: the pinned sources generated in 250.5 seconds with 971
  relations and 44 findings (7 risk, 9 expected, 22 excluded, 6 verified), zero
  review and zero generator issues.  All 971 relations are exact semantic
  matches to the accepted v0.19 baseline, with no missing or new downstream
  endpoint.  The independent coverage audit classified all 1,018 candidates
  with zero missing, conflicting, or generator-issue records and the same two
  known auditor-only orphans.
- This is an accepted rollback checkpoint, not final completion.  Independent
  review has since reproduced unmodelled descriptor kinds, explicit downstream
  override intent after an upstream deletion, conditional class/downstream
  method presence, negative `hasattr`, custom exception inheritance, precise
  expression exceptions, and several lower-frequency namespace flow edges.
  Those are generator work; none is hidden by relabelling it as an upstream
  break.

## v0.21.0 - callable alias checkpoint (rejected)

- Starting commit: `26b471763`; rollback checkpoint: `c78b06086`.
- Problem addressed: final binding labelled imported callable aliases and
  annotated class-body callable aliases as ordinary values.  This produced a
  false conditional-presence review for vLLM's CPU implementation alias and
  changed six PyTorch `Module.forward` exclusions into reviews.
- Change: retain unresolved name assignments as explicit alias bindings;
  resolve them after the repository index is complete; recover their source
  callable variants and signatures; and use those variants at every relation
  consumer.  Two new red tests cover a conditional imported function alias and
  `run: Callable = implementation`; all 116 isolated generator/auditor tests
  pass after the change.
- Fixed-source result: the pinned source set took 260.9 seconds and produced
  970 relations plus 45 findings.  The six PyTorch external-owner findings are
  restored exactly.  The remaining difference is
  `causal_conv1d_update`: the generator now correctly sees that the default and
  CPU aliases have different signatures and reports
  `review/conditional_callable_variants`.
- Rejection reason: vllm-ascend runs on NPU, so the exact
  `current_platform.is_cpu()` branch is inactive for this dependency map.  The
  generic main-path condition evaluator still treats it as reachable.  This is
  a target-platform path-selection defect in the generator, not an upstream
  interface break.  Keep this commit only as a rollback point before adding
  exact runtime-platform guard evaluation.

## v0.20.0 - namespace exception flow checkpoint (rejected)

- Starting commit: `9bd978d14` (the red-test checkpoint); the previous accepted
  generator checkpoint is `29229bb3e`.
- Problem addressed: module/class final binding discarded `value` and
  `unbound` alternatives after retaining callable nodes, and ordinary
  `try/except` treated every handler as reachable from the try-entry state.
  This could silently verify a conditionally non-callable endpoint, walk past a
  `None` MRO shadow, activate an impossible handler signature, or index code
  after an unmatched explicit raise.
- Change: preserve final module/class binding alternatives; execute namespace
  statements as normal and abrupt outcomes; snapshot state at each possible
  raise; route exact exceptions through handlers in source order with
  never/maybe/always matching; run `else` only after normal try completion;
  and apply `finally` independently to normal and abrupt outcomes.  A direct
  endpoint or effective MRO member that is callable only on some normally
  completing paths is now a non-generator
  `review/conditional_callable_presence`, not a verified relation.  Only an
  unbound MRO path may continue to a later owner; a value blocks lookup.
- Test evidence: twelve new fixtures started as nine failures and three
  existing-correct controls.  All twelve now pass; all 114 isolated
  generator/auditor tests pass, as do Ruff, Ruff format, and
  `git diff --check`.
- Fixed-source result: the pinned vLLM, vllm-ascend, and PyTorch sources took
  233.8 seconds and produced 970 relations plus 45 findings.  Compared with the
  accepted v0.19 semantics, 970 relations match exactly, one monkey-patch
  relation disappeared, and six external `forward` findings changed from
  excluded to review.
- Rejection reason: all seven differences are generator alias-classification
  errors, not newly discovered source risks.  The CPU branch aliases
  `causal_conv1d_update` to the imported callable
  `causal_conv1d_update_cpu`, while PyTorch declares
  `Module.forward: Callable = _forward_unimplemented`.  The v0.20 state labels
  both callable aliases as ordinary values.  This checkpoint is retained only
  for rollback and must not be used as an accepted mapping baseline.

## v0.19.0 - final runtime binding checkpoint

- Starting commit: `4b543aaa64daffa373d9ad02ba2ce89e4227c05d`.
- Problem: v0.18 retained every same-name `def` that appeared on a possible
  branch and then selected the first one. That is not Python runtime behavior:
  a later definition, assignment, deletion, or `finally` binding replaces an
  earlier value on that path. The bug selected the first PyTorch `@overload`
  declaration for `Module.to` instead of its final implementation. Adjacent
  review also reproduced lexical-local leakage, a false safe/unsafe `len`
  decision, a partially unresolved exception tuple being treated as fully
  resolved, and known empty `nullcontext`/`suppress` managers being labelled
  opaque.
- Final-binding change: add a deterministic module/class namespace flow that
  follows source order, splits unknown `if` paths, excludes terminal paths,
  applies `finally` to each live path, and records the final effect of
  definitions, simple callable aliases, non-callable assignments, and
  deletion. A later unconditional binding now replaces every older
  conditional binding; same-signature branches collapse to one contract;
  incompatible final signatures remain an explicit
  `review/conditional_callable_variants`.
- Consumer change: module-level callables now retain all final variants, and
  verified override collection checks both upstream and downstream signature
  sets before writing a relation. It no longer picks a conditional signature
  by branch order. A callable overwritten by a field or `del` is no longer
  retained as an old method target.
- Scope and exception change: pre-scan Python function locals without entering
  nested scopes, clear shadowed outer bindings and provenance on function
  entry, and judge the literal `len` optimization from the current execution
  context rather than the repository-wide final symbol table. Exception tuple
  resolution now records whether any member is unknown instead of silently
  dropping it. A known manager set with no suppressed exceptions is represented
  by `()`, distinct from an unanalysable manager.
- Test evidence: fifteen new fixtures were first observed as 13 failures and
  two existing correct baselines. After the change, all fifteen pass. Together
  with the corrected module-scope tombstone fixture, all 102 isolated
  generator/auditor tests pass; Ruff, Ruff format, and `git diff --check` pass.
- Fixed-source result: vLLM
  `88402a41c4ab272ebbbd33f4a77fbbac0431cbb9`, vllm-ascend
  `81d3450128528be2c343232fcc28220814a15fd6`, and PyTorch
  `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0` generated in 164.134 seconds:
  971 relations and 44 findings (7 risk, 9 expected, 22 excluded, 6 verified),
  with zero unresolved or generator issues. Against the accepted v0.17
  semantics, all 971 relations and all 44 findings are exact matches. Against
  rejected v0.18, only `torch.nn.modules.module.Module.to` changes, restoring
  the real `(self, *args, **kwargs)` implementation. The class-callable alias
  binding evidence is unchanged.
- Coverage audit classified all 1,018 known sites in 36.270 seconds with zero
  missing, conflicting, or generator-issue records. It retains the same two
  known auditor-only orphans. Audited pre-version-bump hashes: relations
  `227fd7c38f60325abe1e56ae823d62c5552dfb42c8fe9ce0dd6293a9a077bc4e`,
  findings
  `f83164f52e5319ebb89f4ce5367ce4e511f586ce710cd200056f0f83dba6d6e7`,
  v0.17 comparison
  `95170c05e3fcb822ab379a40d48cc76a5755a42f26dbf96c5c9d0cd3af9dbfbc`,
  and coverage
  `130dfd282df2199e6574aec501ddb2872342b630ade68fca68399f021355a908`.
- This remains an explicit rollback checkpoint, not final acceptance.
  Independent generic review reproduced the next real generator boundaries:
  callable/non-callable conditional presence is currently filtered to the
  callable path; `try` handlers do not yet consume exact exception outcomes;
  conditional same-name classes are still chosen by traversal order;
  properties, unknown decorators, and assignment-form method descriptors need
  explicit binding kinds; exact `TYPE_CHECKING` paths, annotated callable
  aliases, loop `break`/`continue` plus `else`, exhaustive `match`, and dynamic
  namespace writes need modelling or review. The patch scanner also needs
  three-way handler matching, call-time global/closure state, explicit
  builtin-identity handling after deletion/rebinding, partial suppress state,
  correct bare-except evidence, definition-time/comprehension named
  expressions, and match-case traversal. None of these are hidden as upstream
  breaks; they remain work for the next Git iteration.

## v0.18.0 - conditional-callable and exception-flow checkpoint

- Starting commit: `d819fef0ee9cdf8f260ba32ef6700bcd2956901e`.
- Problem: nine independently reproduced boundaries could invent a handler
  patch, select one arbitrary conditional signature, miss a fallback override
  owner, index code after a terminal statement, lose state through multiple
  context managers, conflate a local exception with a same-named builtin,
  attach the wrong tuple-handler evidence, or reuse stale upstream provenance
  after an exact `None` refinement.
- Change: exclude the exact unshadowed `len(literal)` form from implicit-raise
  paths; retain callable variants; continue MRO lookup after a conditional
  owner; compute MUST bindings from normally completing paths; stop indexing
  after terminal statements; model multiple exact `nullcontext`/`suppress`
  managers; canonicalize builtin and local exception identities; preserve the
  real tuple handler as evidence; and clear active upstream provenance when an
  exact `None` path is selected while retaining enough history to emit an
  explicit dynamic-owner review.
- Safety boundary: multiple reachable signatures for one patch endpoint are
  reported as `review/conditional_callable_variants`; no signature is guessed.
  A dynamic owner that previously came from upstream remains an explicit
  `review/dynamic_patch_owner`, not a reconstructed stale target.
- Regression evidence: nine new fixtures fail on v0.17 and pass here. All 87
  isolated generator/auditor tests pass; Ruff, Ruff format, and
  `git diff --check` pass.
- Fixed-source checkpoint: vLLM
  `88402a41c4ab272ebbbd33f4a77fbbac0431cbb9`, vllm-ascend
  `81d3450128528be2c343232fcc28220814a15fd6`, and PyTorch
  `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0` generated in 164.940 seconds:
  971 relations and 44 findings (7 risk, 9 expected, 22 excluded, 6 verified),
  with zero unresolved or generator issues. Coverage audit classified all
  1,018 known sites in 36.334 seconds and retained the same two known
  auditor-only orphans.
- Rejected fixed-source result: this checkpoint is not accurate enough to
  accept. It changed `torch.nn.modules.module.Module.to` from the real final
  implementation `(self, *args, **kwargs)` to the first `@overload`
  declaration. The other 970 relations and all 44 findings are unchanged.
  This is a generator regression caused by collecting all same-name
  definitions and then choosing `candidates[0]`; it is not an upstream break.
- Independent review found three adjacent generator gaps: sequential and
  conditional definitions must be reduced to the final binding on every
  normally completing path; override and module-level callable paths must use
  the same variant check as monkey patches; and the real boundary UT must
  validate all active-main endpoint variants rather than searching only the
  first class-body node. These remain visible work for the next Git iteration.
- Audited rejected-output hashes: relations
  `7d01fdc746ceda6b0a6dec9a913f379aa220d1d1370f888eb567ae8549e0d7c9`,
  findings
  `f83164f52e5319ebb89f4ce5367ce4e511f586ce710cd200056f0f83dba6d6e7`,
  comparison report
  `878f3de93bbc66d2116d4511015c7835104daa6a7885860f08fc455722f3e1f6`,
  and coverage audit
  `03201511f89065f417dc202052e1e0cb7d65ab2ddfd437e7f3e360367be9fa74`.

## v0.17.0 - path-state and conditional-presence checkpoint

- Starting commit: `b9e3cb64c2c80d9a7359fd8b2f6574f602aaf175`.
- Problem: source dependencies could still disappear, use a stale owner, or
  take an impossible path when guards crossed helper scopes; an optional
  import was tested in a compound boolean; an upstream symbol was conditional;
  a `raise` entered a handler; a `with` body terminated or rebound a name; a
  dynamic value replaced an imported owner; or `setattr` used that dynamic
  owner. Negative `hasattr` on one member could also hide a stale patch to a
  different member.
- Guard and condition change: replace plain guard strings internally with
  scoped, activation-specific facts while retaining the same display strings
  in JSON. Split `and`/`or` conditions in short-circuit order, narrow exact or
  `None` alternatives on each feasible path, and attach canonical
  owner/member identity to `hasattr` facts. Only a guard for the exact patched
  target can classify an expected injection or inactive patch.
- Presence change: index symbols as possible separately from those present on
  every normally completing path. A callable below an unknown condition no
  longer proves `hasattr` true; a child module existing on disk no longer
  proves that its parent package exports it. Unknown `if` arms are intersected
  for MUST bindings, `finally` bindings apply to every path, and conditional
  class methods remain indexable without being labelled unconditional.
- Flow change: helper and patch scans propagate live, return, raise, break, and
  continue exits through exact `if`, `try`, `finally`, and `with` paths.
  Explicit built-in exception inheritance is respected, ordered handlers do
  not execute after an earlier covering handler, and a possible implicit
  exception uses the state at the potentially raising statement instead of the
  pre-`try` state. A known single `contextlib.suppress` restores only an exact
  matching explicit raise path.
- Binding change: represent exact, runtime-`None`, and unknown bindings
  separately. Sequential exact assignment replaces stale provenance; branch
  merge unions provenance. `nullcontext(value) as owner` binds the exact value,
  another dynamic `with ... as owner` creates a tombstone, and both attribute
  assignment and `setattr` emit
  `review/dynamic_patch_owner` with `generator_issue=false` instead of
  reverting to an old import or disappearing.
- Reason: an exact missing upstream owner/member remains a real downstream
  risk. A runtime-dynamic owner is an explicit manual review. Neither case is
  rewritten as a generator issue merely to make the result look clean.
- Regression evidence: nineteen new fixtures were first observed failing and
  then passed, covering scoped guards, compound optional imports, conditional
  `hasattr`, package exports, target-specific guards, explicit and implicit
  exception paths, built-in exception inheritance, `with` termination and
  bindings, dynamic `setattr`, all-path definitions, `finally`, `suppress`, and
  latest-owner provenance. All 78 isolated generator/auditor tests pass; Ruff,
  Ruff format, and `git diff --check` pass.
- Fixed-source effect: two complete runs against vLLM
  `88402a41c4ab272ebbbd33f4a77fbbac0431cbb9`, vllm-ascend
  `81d3450128528be2c343232fcc28220814a15fd6`, and PyTorch
  `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0` took 149.170 and
  148.948 seconds. Both produced 971 relations and 44 findings: 7 risk,
  9 expected, 22 excluded, 6 verified, and zero generator issues. All relation
  and finding semantics are identical to v0.16; the only relation-file change
  is generator metadata.
- Deterministic hashes for both runs: relations
  `9c558e4d1761803f9f619ba5f24c4d7bb2f3af0e9b8549995ca7ca94de95cee8`,
  findings
  `f83164f52e5319ebb89f4ce5367ce4e511f586ce710cd200056f0f83dba6d6e7`,
  and report
  `e736a61fd26afc790c70eaa4fa26eafcce7721f38346ffdc33fc0ecc2df871e5`.
- This remains an explicit rollback checkpoint, not final acceptance. A final
  independent review reproduced the next boundaries: multiple conditional
  definitions of the same method can collapse to one arbitrary signature or
  owner; a statically safe call can create a false implicit-exception handler
  edge; multi-manager `suppress`, namespaced same-name exceptions, terminal
  paths in MUST analysis, and provenance after a `None` branch require another
  iteration. The fixed-source pair does not prove those generic cases absent.

## v0.16.0 - preserve exact path state through helpers and termination

- Starting commit: `c3643d25f2d9384ec13ef2bb54f92d8435f1449d`.
- Problem: six generic control-flow forms either lost a real dependency,
  emitted a dead dependency, or hid a real upstream removal: a helper argument
  whose exact `vllm.*` owner no longer exists; an exact main/tag branch that
  returns; a patch in `finally` after a return; semantically opposite guards
  rendered with different text; a statically true `hasattr`; and an exact
  import merged with `None` then narrowed by a non-`None` guard.
- Change: preserve syntactically exact vLLM references until normal target
  validation classifies them; carry exact-reference and runtime-`None`
  alternatives separately; normalize guard predicates from their AST; prove
  only positive `hasattr` results from an exact indexed owner/member; propagate
  live, return, and raise exits through the helper and patch scanners; execute
  `finally` once per incoming path while keeping its path-specific bindings;
  and stop scanning after an exact selected branch terminates.
- Reason: a removed upstream module, class, or member is a downstream
  compatibility risk, not a generator failure. Conversely, an unreachable
  statement or a contradictory path must not create a dependency edge.
- Safety boundary: absence is never used to prove `hasattr(...) == False`;
  incomplete MRO remains unknown. A branch merge retains an exact owner only
  when the alternatives are explicitly known. The optional-owner finding no
  longer reconstructs a target from a stale module import. Duplicate findings
  reached through multiple `try` paths are collapsed at the source site.
- Regression evidence: six new fixtures first failed for the exact cases
  above and then passed. The existing fixture now proves that the statically
  selected `PatchTarget.hook` relation contains all three patch occurrences
  instead of leaving one dynamic-name finding. All 59 isolated generator and
  auditor tests pass; Ruff, Ruff format, and `git diff --check` pass.
- Fixed-source effect: vLLM `88402a41c4ab272ebbbd33f4a77fbbac0431cbb9`,
  vllm-ascend `81d3450128528be2c343232fcc28220814a15fd6`, and PyTorch
  `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0` produce 971 relations and
  44 findings: 7 risk, 9 expected, 22 excluded, and 6 verified. The relation
  and finding sets are semantically identical to v0.14. Relative to v0.15,
  all 969 prior relations remain and the MiniMax `forward_qk` and Qwen3.5 MTP
  `forward` edges are restored; both generator reviews disappear.
- Audited candidate hashes before the metadata version update: relations
  `a28a88771972ecdd6b6da6d00a32f6b3f28199ef24908c9f04f50e200436294a`,
  findings
  `f83164f52e5319ebb89f4ce5367ce4e511f586ce710cd200056f0f83dba6d6e7`,
  and report
  `a00bb2831ecc6121537d2871ce5c239132673bf9d5c2ab6f89311de3ab13e41b`.
- The final v0.16 JSONL hash is
  `bfe4903c722d4da30630ba150d739779304f6fb733a54afeecfb049ff755463f`.
  The independent site audit still classifies all 1,018 candidates and reports
  the same two known auditor-only orphans: the verified HunYuan helper-owner
  site and cached literal `sys.modules` site. This checkpoint does not use that
  site-only audit as proof that dynamic owner endpoints are complete.
- Independent review found the next path-state boundaries before final
  acceptance: guard identity across lexical scopes, compound non-`None`
  narrowing, conditional-definition `hasattr`, explicit-raise state entering
  a handler, target-specific `hasattr` classification, `with` termination,
  and loop `break`/`continue`. They remain visible work for the next Git
  iteration rather than being hidden by the restored fixed-source count.

## v0.15.0 - definition-site helper propagation checkpoint

- Starting commit: `b2ffb1e40`.
- Problem: a conservative branch join, same-name helper redefinition,
  conditional owner expression, multi-level direct `sys.modules` target, and
  helper-to-helper forwarding exposed both false edges and silent omissions.
- Change: distinguish module-level private helpers by module and definition
  location; retain one guarded invocation per exact call; propagate exact
  owner bindings through a finite worklist; preserve mutually compatible
  `IfExp`/`BoolOp` alternatives; treat an unknown branch value as a tombstone;
  respect simple terminal branches; and peel an arbitrary static attribute
  suffix from literal `sys.modules` targets.
- Safety boundary: an exact helper invocation state is processed once, so
  direct and mutual recursion terminate. Dynamic/star arguments are not
  guessed. An unresolved optional-import owner under an explicit non-`None`
  guard is emitted as `review/unresolved_patch_owner` instead of disappearing.
- Regression evidence: six fixtures first reproduced a bad branch join, a
  redefinition cross product, two missing conditional-owner edges, a missing
  deep `sys.modules` edge, missing forwarding, and constant short-circuit false
  positives. All 53 isolated generator/auditor tests pass; Ruff and
  `git diff --check` pass.
- Fixed-source checkpoint: 969 relations and 46 findings. All 969 shared
  relations and all prior 44 findings are byte-identical to v0.14. Exactly two
  prior verified patch edges are now honest generator reviews:
  MiniMax `forward_qk` needs static `hasattr` evaluation, and Qwen3.5 MTP
  `forward` needs `{exact import | None}` path narrowing. Seven upstream risks
  remain unchanged. A full run completed in 129.3 seconds.
- Independent coverage result: 1,018/1,018 known sites are classified. The two
  orphans are auditor gaps for the already verified HunYuan helper-parameter
  edge and cached literal `sys.modules.get` edge, not generator regressions.
- This is an intentionally incomplete rollback checkpoint. Independent review
  reproduced additional generator work: exact-main selected termination,
  `try/finally` state, semantic guard negation, removed-upstream owners passed
  through helpers, global late binding, nested/conditional helper identity,
  parameter-dependent control flow, and loop `break`/`continue`. These must be
  fixed or emitted as explicit review before final acceptance.

## v0.14.0 - preserve lexical scope and per-call patch ownership

- Starting commit: `d9a0c7c7e`.
- Problem: four generic cases could either invent or silently lose a patch
  edge: a function parameter reused an outer module binding; a release-only
  helper call in a short-circuited condition was still counted; two exact
  owners passed to one helper were collapsed into no result; and a direct
  literal `sys.modules["vllm..."]` target was ignored.
- Change: clear every positional, keyword-only, variadic, and keyword-variadic
  parameter when entering a function scope. Evaluate helper-call traversal on
  the active main path with boolean short-circuit semantics. Store one exact
  helper invocation context per active direct call and scan the helper once per
  context. Resolve both direct literal `sys.modules[...]` and
  `sys.modules.get(...)` patch targets.
- Safety boundary: a reassigned parameter, non-unique argument, dynamic module
  key, callback, reflection, or indirect helper transfer is not guessed. Exact
  multi-owner calls are independent real dependencies, not an ambiguity.
- Regression evidence: four new failing fixtures first reproduced two false
  positives and three missing edges; all pass after the change. An older test
  that treated two exact owners as ambiguous was corrected to require both
  edges. All 47 isolated generator/auditor tests pass; Ruff and
  `git diff --check` pass.
- Fixed-source effect: the exact vLLM, vllm-ascend, and PyTorch inputs still
  produce 971 relations and the same 44 findings. Relations and findings are
  byte-identical to v0.13 before the metadata version update, so all seven real
  upstream risks remain visible and generator issues remain zero. One full run
  completed in 129.7 seconds.
- Independent follow-up review found five additional generic cases that do not
  occur in the fixed source pair: conservative branch joins, helper
  redefinition identity, conditional owner values, multi-level direct
  `sys.modules` targets, and helper-to-helper forwarding. They are intentionally
  left for the next Git iteration instead of being hidden by this result.

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
