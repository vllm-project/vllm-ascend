## 1. Freeze the Baseline and Reproduce Native FULL

- [x] 1.1 Record commit `1a8feb60`, vLLM/CANN/driver/torch-npu versions, model hashes, dataset hash, NPU inventory, and process-free card selection in a Task 1 evidence report.
- [x] 1.2 Run the existing focused Eager, Piecewise, FULL_DECODE_ONLY, graph-input-contract, proposer, and 310P model-runner unit suites unchanged and preserve exact output.
- [x] 1.3 Run matched 4B TP1 Eager, Piecewise, and FULL_DECODE_ONLY C1 controls on server 1 and record health, success count, throughput, acceptance length, and graph evidence where applicable.
- [x] 1.4 Run the unchanged 4B TP1 DFlash `FULL` startup and one C1 request in strict evidence mode, stop at the first failure, and classify requested/resolved/runtime modes, component, descriptor, execution signature, rank, and first failing operation.
- [x] 1.5 Compare the fresh trace with historical hypotheses without copying old code, and mark each hypothesis confirmed, disproved, or not yet reached.

## 2. Add Exact Activation and Dispatch Policy

- [ ] 2.1 Add RED unit tests for the exact four-condition activation predicate and every one-condition-negative Eager, Piecewise, FDO, non-DFlash, and non-310P control.
- [ ] 2.2 Implement the immutable predicate and instance-owned controller in `vllm_ascend/_310p/dflash_full.py` until the activation tests pass.
- [ ] 2.3 Add RED tests that classify prefill, chunked prefill, mixed, decode, speculative decode, and mixed-with-spec without mutating or reconstructing the parent descriptor.
- [ ] 2.4 Implement classification and validate the final parent dispatcher result, including legitimate out-of-coverage fallback and fail-closed unexpected in-range fallback.
- [ ] 2.5 Add RED tests for the GDN capability hook and implement an exact-predicate override in `gdn_attn_builder_310.py` while preserving the baseline return for all negative controls.
- [ ] 2.6 Run the focused FDO/Piecewise/Eager runner regressions and commit the activation/dispatch slice with a signed-off commit.

## 3. Add the Private Qualified Graph Store

- [ ] 3.1 Add RED tests proving that component, TP rank, parent descriptor, and execution signature each change graph identity independently.
- [ ] 3.2 Implement the private key, entry, and store in `vllm_ascend/_310p/dflash_full_graph.py` without changing the generic ACL graph cache.
- [ ] 3.3 Add RED tests for duplicate capture, missing target or draft entry, missing TP rank, wrong signature lookup, and replay-counter ordering.
- [ ] 3.4 Implement per-rank startup manifests and increment replay counters only after successful graph launch completion is known.
- [ ] 3.5 Run graph-store tests plus generic ACL graph and FDO manifest regressions, then commit the graph-store slice with a signed-off commit.

## 4. Add Persistent FULL Input Contracts

- [ ] 4.1 Add RED tests for recursive tensor count/order, shape, dtype, device, relevant stride, persistent address, component, signature, and contract-version mismatches.
- [ ] 4.2 Implement per-entry immutable contracts, stable references to existing owned buffers, deterministic padding, value updates, and pre-launch validation in `vllm_ascend/_310p/dflash_full_inputs.py`; allocate a private buffer only for a freshly proven missing stable input.
- [ ] 4.3 Add RED tests proving two engine instances and two graph entries cannot alias mutable state or retained input objects.
- [ ] 4.4 Add tests proving capture/replay preparation performs no device-to-host read or data-dependent host branch.
- [ ] 4.5 Run input-contract tests plus existing 310P graph-input/FDO/Piecewise regressions, then commit the input slice with a signed-off commit.

## 5. Integrate Target and DFlash Draft Narrowly

- [ ] 5.1 Add RED runner tests that the exact controller routes target capture/replay while every negative mode follows the baseline call path byte-for-byte at the hook boundary.
- [ ] 5.2 Add RED proposer tests that draft routing exists only while DFlash proposal executes and that non-DFlash and inactive FULL never construct or query the private store.
- [ ] 5.3 Implement the narrow hooks in `model_runner_310p.py` and `_310p/spec_decode/llm_base_proposer_310.py` without modifying the generic runner, generic proposer, generic ACL wrapper, or FDO modules.
- [ ] 5.4 Add strict structured evidence for requested/resolved/runtime modes, qualified key, capture result, replay result, addresses, and closed fallback reason behind the existing logging configuration.
- [ ] 5.5 Run all focused 310P DFlash runner/proposer tests, frozen mode controls, lint, format check, and strict OpenSpec validation; commit the integration slice.

## 6. Clear Only Freshly Proven Capture Blockers

- [ ] 6.1 Rerun 4B TP1 FULL startup; preserve the first new RED boundary and do not investigate downstream failures until that boundary is green.
- [ ] 6.2 If reproduced, add a focused test for capture-time attention-mask host transfer and implement a FULL-only tensor-resident mask path; otherwise record the hypothesis as disproved.
- [ ] 6.3 If reproduced, add a focused test for boolean GDN initial-state selection lowering through `NonzeroV2` and implement a FULL-only capture-safe selection; otherwise record it as disproved.
- [ ] 6.4 If reproduced, add a focused test for GDN chunk metadata host-list conversion and implement a FULL-only tensor contract; otherwise record it as disproved.
- [ ] 6.5 If reproduced, add a focused test for dense-prefill attention not being recorded and implement the smallest FULL-only routing that produces identical Eager output; otherwise record it as disproved.
- [ ] 6.6 After each blocker, rerun the exact RED, 4B TP1 startup, numerical comparison, and frozen Eager/Piecewise/FDO controls before making a separate signed-off commit.
- [ ] 6.7 Stop and return to design review if a third independent operator family, a generic graph change, an upstream change, an eager island, or a baseline contract change is required.

## 7. Gate Any Dedicated FULL Operator Separately

- [ ] 7.1 For each unresolved blocker, record either `operator not needed` with passing Python evidence or an admission package containing the exact RED, failed safe alternative, tensor-only ABI, alias rules, and isolation proof.
- [ ] 7.2 If an admission package exists, pause for explicit user approval before adding operator source, registration, build files, or call sites.
- [ ] 7.3 If approved, add the separately named private operator and isolated Eager-equivalence, repeated ACL capture/replay, boundary-shape, and address-stability tests before connecting it to the controller.
- [ ] 7.4 Add negative call-count tests proving zero calls from Eager, Piecewise, FDO, non-DFlash, non-310P, and inactive FULL, and verify existing operator symbols and callers are unchanged.
- [ ] 7.5 Commit an admitted operator and its controller integration as separate signed-off commits; if no operator is admitted, close this section with the recorded `operator not needed` evidence and no code changes.

## 8. Pass Fast Hardware Gates on Server 1

- [ ] 8.1 Pass 4B TP1 C1 with complete target/draft manifests, real signature-specific replay, Eager-equivalent deterministic output, and no hidden fallback.
- [ ] 8.2 Pass 4B TP1 C10, then repeat C1/C10 on 4B TP2; stop at the first failed topology rather than continuing the matrix.
- [ ] 8.3 Pass 35B W8A8 TP2 C1 then C10, followed by TP4 C1 then C10, preserving first-error evidence for any `507xxx`, AICore, HCCL, contract, or numerical failure.
- [ ] 8.4 After each topology, rerun matched Eager, Piecewise, and FDO controls and block progression on unexplained output, acceptance-length, or throughput drift greater than five percent.

## 9. Complete Formal Acceptance

- [ ] 9.1 Run GSM8K output-256 at C1/4 requests and C10/20 requests on server 1 for 4B TP1/TP2 and 35B W8A8 TP2/TP4 with fixed order, temperature zero, ignore EOS, and recorded warmup.
- [ ] 9.2 Run random input/output-2048 at the same topologies and C1/C10 request counts using a fixed seed and saved prompts.
- [ ] 9.3 Run AISBench smoke for each model/topology after the vLLM bench groups pass.
- [ ] 9.4 Produce a report containing commit and environment identity, exact commands, result JSON paths, success counts, output throughput, acceptance length, Eager-relative percentages, memory, manifests, and real replay evidence per rank and signature.
- [ ] 9.5 Run the complete focused unit suite, strict OpenSpec validation, Ruff check, format check, diff check, and a final read-only scope review before declaring the change complete.

## 10. Independent Pull Validation and Handoff

- [ ] 10.1 After server-1 acceptance and push, use server 2 only to pull the committed branch into its independent repository and record commit identity and clean tracked status.
- [ ] 10.2 Rebuild or reinstall only what the committed diff requires, then rerun 2B TP1/TP2 minimal gates and a representative 35B FULL matrix without copying files from server 1.
- [ ] 10.3 Update the server-2 summary and PR change log with independently reproducible commands, results, limitations, and rollback instructions.
