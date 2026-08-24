# Ascend 310P DFlash FULL Minimal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement genuine `FULL` ACL graph capture and replay for DFlash on Ascend 310P while leaving Eager, Piecewise, FULL_DECODE_ONLY, non-DFlash, generic graph infrastructure, and upstream vLLM unchanged.

**Architecture:** An exact-scope 310P controller classifies execution without mutating upstream descriptors. A FULL-only router owns one ordinary `ACLGraphWrapper` per execution signature, so the generic descriptor cache stays untouched while the effective key becomes component + rank + signature + upstream descriptor. Entry-local contracts reference existing runner/proposer persistent buffers and reject incompatible replay before launch.

**Tech Stack:** Python 3.12, PyTorch/torch-npu, vLLM 0.24.0, vLLM Ascend plugin, pytest, Ruff, OpenSpec, Ascend 310P.

**Spec:** `openspec/changes/fix-310p-dflash-full-graph-minimal/design.md` and `openspec/changes/fix-310p-dflash-full-graph-minimal/specs/310p-dflash-full-graph/spec.md`

## Global Constraints

- Base implementation commit is exactly `1a8feb60d1d642c87feccdb9d1aee5d273f7197a`; the OpenSpec documentation commit is its only parented change before implementation.
- All diagnosis and implementation run in server 1 container `whn_310b`, worktree `/home/whn/vllm-ascend-full-minimal`.
- Server 2 remains independent pull validation only; no source or diagnostic files are copied from server 1.
- Activation requires Ascend 310P + DFlash + configured `CUDAGraphMode.FULL`; runtime routing additionally requires the parent-selected runtime mode `FULL`.
- Do not modify upstream vLLM, `vllm_ascend/compilation/acl_graph.py`, generic model runner/proposer, existing FDO modules, or existing operator schemas/callers.
- No production change is written before its focused test has failed for the intended missing behavior.
- Hardware investigation prints values at component boundaries and stops at the first failure; one hypothesis and one variable are tested at a time.
- A dedicated operator requires a separately recorded RED and an explicit user checkpoint before source, registration, build, or call-site changes.
- Stop if a third independent operator family, shared graph cache, upstream change, accepted-mode contract change, or hidden eager island is required.

---

### Task 1: Freeze Baselines and Locate the First Current FULL Boundary

**Files:**
- Create: `openspec/changes/fix-310p-dflash-full-graph-minimal/reports/task1-root-cause.md`
- Modify: `openspec/changes/fix-310p-dflash-full-graph-minimal/tasks.md`
- Evidence: `/home/whn/vllm_repair/full-minimal/task1/`

**Interfaces:**
- Consumes: unchanged commit `1a8feb60` production code and existing unit/hardware commands.
- Produces: one reproducible first failing boundary with requested/resolved/runtime modes, attention state, descriptor, component, rank, input structure, and exact first exception.

- [ ] **Step 1: Record immutable environment evidence**

Run inside `whn_310b` and save stdout under the evidence directory:

```bash
git rev-parse HEAD HEAD^
git status --short
python -m pip show vllm vllm-ascend torch torch-npu
npu-smi info
sha256sum /home/qzh/datasets/gsm8k/test_vllm.jsonl
find /home/models/Qwen3.5-4B /home/models/Qwen3.5-4B-DFlash -maxdepth 1 -type f -name 'config.json' -exec sha256sum {} +
```

- [ ] **Step 2: Run the unchanged focused unit baseline**

Run:

```bash
pytest -q \
  tests/ut/_310p/test_dflash_full_decode_only.py \
  tests/ut/_310p/test_dflash_full_decode_acl_graph.py \
  tests/ut/_310p/test_dflash_full_decode_contract.py \
  tests/ut/_310p/test_dflash_full_decode_manifest.py \
  tests/ut/_310p/test_graph_input_contract_310p.py \
  tests/ut/_310p/test_model_runner_310p.py \
  tests/ut/_310p/spec_decode/test_dflash_full_decode_proposer.py \
  tests/ut/_310p/spec_decode/test_llm_base_proposer_310.py \
  tests/ut/_310p/attention/test_attention_mask_310.py
```

Expected: exit 0. Any failure is a baseline blocker and is investigated before FULL work.

- [ ] **Step 3: Run matched 4B TP1 controls sequentially on a freshly verified free card**

For Eager, Piecewise, and FULL_DECODE_ONLY, use identical model/DFlash/K=15/scheduler settings and GSM8K C1/4/output256. Save launch command, server log, benchmark command, JSON, and NPU snapshot. Piecewise and FDO must include real capture/replay evidence.

- [ ] **Step 4: Reproduce unchanged FULL and preserve only the first boundary**

Launch with `VLLM_LOGGING_LEVEL=DEBUG`, DFlash K=15, `cudagraph_mode=FULL`, and capture sizes `[160,16]`. Run one fixed C1 request only if health succeeds. Extract:

```bash
grep -E 'cudagraph_mode|requested|resolved|runtime|attn_state|descriptor|component|rank|Capturing|Replaying|ERROR|Traceback|507[0-9]+|AICore|HCCL' server.log
```

Expected RED: current FULL does not satisfy the exact FULL contract. Stop at the earliest mode/dispatch/capture failure; do not inspect later historical blockers in the same cycle.

- [ ] **Step 5: Write the evidence report and mark OpenSpec 1.1-1.5**

The report contains command, environment, first-error stack, values immediately entering/exiting each observed boundary, and a table marking every historical hypothesis `confirmed`, `disproved`, or `not reached`.

- [ ] **Step 6: Commit baseline evidence**

```bash
git add openspec/changes/fix-310p-dflash-full-graph-minimal/reports/task1-root-cause.md \
  openspec/changes/fix-310p-dflash-full-graph-minimal/tasks.md
git commit -s -m "test(310p): record DFlash FULL baseline boundary"
```

### Task 2: Add the Exact Activation and Execution Classifier

**Files:**
- Create: `vllm_ascend/_310p/dflash_full.py`
- Create: `tests/ut/_310p/test_dflash_full.py`
- Modify: `openspec/changes/fix-310p-dflash-full-graph-minimal/tasks.md`

**Interfaces:**
- Produces: `DFlashFullExecutionSignature`, `DFlashFullDecision`, `DFlashFullController`, `is_310p_dflash_full(vllm_config)`, and `classify_dflash_full_execution(...)`.
- The classifier consumes parent state and returns a decision; it never constructs or mutates `BatchDescriptor`.

- [ ] **Step 1: Write activation RED tests**

```python
@pytest.mark.parametrize("platform,dflash,configured_full,expected", [
    (True, True, True, True),
    (False, True, True, False),
    (True, False, True, False),
    (True, True, False, False),
])
def test_exact_full_activation(monkeypatch, config, platform, dflash, configured_full, expected):
    monkeypatch.setattr("vllm_ascend._310p.dflash_full.is_310p", lambda: platform)
    config.speculative_config.method = "dflash" if dflash else "mtp"
    config.compilation_config.cudagraph_mode = (
        CUDAGraphMode.FULL if configured_full else CUDAGraphMode.PIECEWISE
    )
    assert is_310p_dflash_full(config) is expected
```

- [ ] **Step 2: Verify activation RED**

Run `pytest -q tests/ut/_310p/test_dflash_full.py -k activation`. Expected: import failure because `dflash_full.py` does not exist.

- [ ] **Step 3: Implement the minimal immutable activation API**

```python
def is_310p_dflash_full(vllm_config: VllmConfig) -> bool:
    speculative_config = vllm_config.speculative_config
    return (
        is_310p()
        and speculative_config is not None
        and speculative_config.method == "dflash"
        and vllm_config.compilation_config.cudagraph_mode is CUDAGraphMode.FULL
    )
```

- [ ] **Step 4: Write classifier RED tests for all six signatures**

```python
@pytest.mark.parametrize("state,all_decode,component,expected", [
    ("PrefillNoCache", False, "target", DFlashFullExecutionSignature.PREFILL),
    ("ChunkedPrefill", False, "target", DFlashFullExecutionSignature.CHUNKED_PREFILL),
    ("PrefillCacheHit", False, "target", DFlashFullExecutionSignature.MIXED),
    ("DecodeOnly", True, "target", DFlashFullExecutionSignature.DECODE),
    ("SpecDecoding", True, "draft", DFlashFullExecutionSignature.SPEC_DECODE),
    ("SpecDecoding", False, "target", DFlashFullExecutionSignature.MIXED_WITH_SPEC),
])
def test_classifies_without_mutating_descriptor(state, all_decode, component, expected):
    descriptor = BatchDescriptor(num_tokens=16, num_reqs=1, uniform=True)
    decision = classify_dflash_full_execution(
        attn_state=SimpleNamespace(name=state), all_decode=all_decode,
        component=component, parent_mode=CUDAGraphMode.FULL,
        descriptor=descriptor,
    )
    assert decision.signature is expected
    assert decision.descriptor is descriptor
```

- [ ] **Step 5: Verify classifier RED, implement the enum/dataclass/controller, and verify GREEN**

The controller keeps counters and the latest decision on the engine/proposer instance only. Run `pytest -q tests/ut/_310p/test_dflash_full.py`; expected all pass.

- [ ] **Step 6: Run negative frozen-mode tests and commit**

```bash
pytest -q tests/ut/_310p/test_dflash_full.py tests/ut/_310p/test_dflash_full_decode_only.py
git add vllm_ascend/_310p/dflash_full.py tests/ut/_310p/test_dflash_full.py \
  openspec/changes/fix-310p-dflash-full-graph-minimal/tasks.md
git commit -s -m "feat(310p): add isolated DFlash FULL policy"
```

### Task 3: Enable Native GDN FULL Without Broadening Other Modes

**Files:**
- Modify: `vllm_ascend/_310p/ops/gdn_attn_builder_310.py`
- Create: `tests/ut/_310p/ops/test_gdn_full_capability_310.py`
- Modify: `openspec/changes/fix-310p-dflash-full-graph-minimal/tasks.md`

**Interfaces:**
- Consumes: `is_310p_dflash_full(vllm_config)`.
- Produces: exact-scope `AttentionCGSupport.ALWAYS`; all negative scopes return `super().get_cudagraph_support(...)`.

- [ ] **Step 1: Write the capability RED**

```python
def test_gdn_reports_always_only_for_exact_dflash_full(monkeypatch, full_config, kv_spec):
    monkeypatch.setattr("vllm_ascend._310p.dflash_full.is_310p", lambda: True)
    assert GDNAttentionMetadataBuilder310.get_cudagraph_support(
        full_config, kv_spec
    ) is AttentionCGSupport.ALWAYS

def test_gdn_preserves_uniform_batch_for_piecewise(piecewise_config, kv_spec):
    assert GDNAttentionMetadataBuilder310.get_cudagraph_support(
        piecewise_config, kv_spec
    ) is AttentionCGSupport.UNIFORM_BATCH
```

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/ut/_310p/ops/test_gdn_full_capability_310.py`. Expected: FULL returns `UNIFORM_BATCH`, proving the downgrade.

- [ ] **Step 3: Add the exact classmethod override**

```python
@classmethod
def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
    if is_310p_dflash_full(vllm_config):
        return AttentionCGSupport.ALWAYS
    return super().get_cudagraph_support(vllm_config, kv_cache_spec)
```

- [ ] **Step 4: Verify GREEN and frozen controls**

Run the new test plus `tests/ut/_310p/ops/test_gdn_310.py`, FDO, Piecewise, and non-DFlash builder tests.

- [ ] **Step 5: Commit**

```bash
git add vllm_ascend/_310p/ops/gdn_attn_builder_310.py \
  tests/ut/_310p/ops/test_gdn_full_capability_310.py \
  openspec/changes/fix-310p-dflash-full-graph-minimal/tasks.md
git commit -s -m "feat(310p): enable GDN FULL only for DFlash FULL"
```

### Task 4: Add the Qualified FULL Graph Router and Manifest

**Files:**
- Create: `vllm_ascend/_310p/dflash_full_graph.py`
- Create: `tests/ut/_310p/test_dflash_full_graph.py`
- Modify: `openspec/changes/fix-310p-dflash-full-graph-minimal/tasks.md`

**Interfaces:**
- Produces: `DFlashFullGraphKey`, `DFlashFullGraphRecord`, and `DFlashFullGraphRouter`.
- Router owns `dict[DFlashFullExecutionSignature, ACLGraphWrapper]`; each inner wrapper retains the unchanged upstream `dict[BatchDescriptor, ACLGraphEntry]`.

- [ ] **Step 1: Write key-separation and duplicate-manifest RED tests**

```python
def test_key_separates_signature_component_and_rank(descriptor):
    keys = {
        DFlashFullGraphKey("target", 0, descriptor, DFlashFullExecutionSignature.PREFILL),
        DFlashFullGraphKey("target", 0, descriptor, DFlashFullExecutionSignature.DECODE),
        DFlashFullGraphKey("draft", 0, descriptor, DFlashFullExecutionSignature.DECODE),
        DFlashFullGraphKey("target", 1, descriptor, DFlashFullExecutionSignature.DECODE),
    }
    assert len(keys) == 4
```

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/ut/_310p/test_dflash_full_graph.py`. Expected: missing module/symbol failure.

- [ ] **Step 3: Implement router composition without copying ACL capture code**

```python
class DFlashFullGraphRouter:
    def __call__(self, *args, **kwargs):
        decision = self._decision_provider()
        wrapper = self._wrappers.get(decision.signature)
        if wrapper is None:
            wrapper = ACLGraphWrapper(
                self._runnable, self._vllm_config,
                runtime_mode=CUDAGraphMode.FULL,
                component=self._component,
            )
            self._wrappers[decision.signature] = wrapper
        return self._invoke_and_record(wrapper, decision, args, kwargs)
```

- [ ] **Step 4: Add RED tests for capture/replay accounting**

Tests assert: capture record appears once, duplicate capture fails, replay count changes only after successful return, wrong signature cannot find another signature's entry, and rank/component are included in every structured error.

- [ ] **Step 5: Implement manifest records and verify GREEN**

Run `pytest -q tests/ut/_310p/test_dflash_full_graph.py tests/ut/_310p/test_dflash_full_decode_manifest.py tests/ut/_310p/test_dflash_full_decode_acl_graph.py`.

- [ ] **Step 6: Commit**

```bash
git add vllm_ascend/_310p/dflash_full_graph.py \
  tests/ut/_310p/test_dflash_full_graph.py \
  openspec/changes/fix-310p-dflash-full-graph-minimal/tasks.md
git commit -s -m "feat(310p): isolate DFlash FULL graph entries"
```

### Task 5: Bind Entry-Local Contracts to Existing Persistent Inputs

**Files:**
- Create: `vllm_ascend/_310p/dflash_full_inputs.py`
- Create: `tests/ut/_310p/test_dflash_full_inputs.py`
- Modify: `vllm_ascend/_310p/dflash_full_graph.py`
- Modify: `openspec/changes/fix-310p-dflash-full-graph-minimal/tasks.md`

**Interfaces:**
- Produces: `DFlashFullInputBinding.capture(key, sources)` and `binding.validate(sources)`.
- Reuses `GraphInputSource`, `capture_graph_input_sources`, and `validate_graph_input_contracts`; no new device allocation is permitted in this task.

- [ ] **Step 1: Write contract RED tests**

```python
def test_binding_rejects_changed_address(key):
    first = torch.empty(16)
    binding = DFlashFullInputBinding.capture(key, (source("tokens", first),))
    with pytest.raises(GraphInputContractError, match="data_ptr changed"):
        binding.validate((source("tokens", torch.empty(16)),))

def test_bindings_do_not_share_mutable_state(key_a, key_b, tensor):
    a = DFlashFullInputBinding.capture(key_a, (source("tokens", tensor),))
    b = DFlashFullInputBinding.capture(key_b, (source("tokens", tensor),))
    assert a is not b
    assert a.contracts is not b.contracts
```

- [ ] **Step 2: Verify RED, implement immutable binding, verify GREEN**

Implementation stores an immutable tuple of contracts plus alias ownership; it does not clone tensors. Run `pytest -q tests/ut/_310p/test_dflash_full_inputs.py`.

- [ ] **Step 3: Add no-D2H/no-host-branch source test**

The test parses `dflash_full_inputs.py` and rejects `.cpu(`, `.item(`, `.tolist(`, and truth-value conversion of tensors in capture/replay preparation.

- [ ] **Step 4: Integrate binding into router and run graph regressions**

Run input tests, router tests, `test_graph_input_contract_310p.py`, FDO graph tests, and Piecewise graph tests.

- [ ] **Step 5: Commit**

```bash
git add vllm_ascend/_310p/dflash_full_inputs.py \
  vllm_ascend/_310p/dflash_full_graph.py \
  tests/ut/_310p/test_dflash_full_inputs.py \
  openspec/changes/fix-310p-dflash-full-graph-minimal/tasks.md
git commit -s -m "feat(310p): validate DFlash FULL persistent inputs"
```

### Task 6: Replace Only the 310P Target and Draft Wrappers

**Files:**
- Modify: `vllm_ascend/_310p/model_runner_310p.py`
- Modify: `vllm_ascend/_310p/spec_decode/llm_base_proposer_310.py`
- Modify: `tests/ut/_310p/test_model_runner_310p.py`
- Create: `tests/ut/_310p/spec_decode/test_dflash_full_proposer_310.py`
- Modify: `openspec/changes/fix-310p-dflash-full-graph-minimal/tasks.md`

**Interfaces:**
- Target `load_model()` calls `super()`, unwraps the just-created generic FULL wrapper only in exact scope, and replaces it with `DFlashFullGraphRouter`.
- Draft `load_model(model)` performs the same exact-scope replacement for `_runnable`.
- All negative scopes retain object identity and call sequence from baseline.

- [ ] **Step 1: Write target and draft replacement RED tests**

```python
def test_target_replaces_wrapper_only_for_exact_full(monkeypatch, runner):
    generic = fake_acl_wrapper()
    monkeypatch.setattr(NPUModelRunner, "load_model", lambda self: setattr(self, "model", generic))
    NPUModelRunner310.load_model(runner)
    assert isinstance(runner.model, DFlashFullGraphRouter)
    assert runner.model.unwrap() is generic.unwrap()

def test_piecewise_keeps_parent_wrapper_identity(monkeypatch, runner):
    generic = fake_acl_wrapper()
    monkeypatch.setattr(NPUModelRunner, "load_model", lambda self: setattr(self, "model", generic))
    NPUModelRunner310.load_model(runner)
    assert runner.model is generic
```

- [ ] **Step 2: Verify RED**

Run the two focused target/draft test files. Expected: 310P classes do not yet replace wrappers.

- [ ] **Step 3: Implement exact-scope `load_model` overrides**

```python
def load_model(self) -> None:
    super().load_model()
    if not is_310p_dflash_full(self.vllm_config):
        return
    parent_wrapper = self.model
    self.model = DFlashFullGraphRouter(
        parent_wrapper.unwrap(), self.vllm_config,
        component="target",
        decision_provider=self._dflash_full_controller.target_decision,
        retained_input_provider=self._full_decode_target_retained_inputs,
    )
```

The proposer override mirrors this with component `draft` and proposer-owned decision/input providers.

- [ ] **Step 4: Add structured boundary logging**

At DEBUG level log one line before and after dispatcher/router/input boundaries with: requested mode, resolved mode, runtime mode, attention state, execution signature, descriptor, component, rank, tensor role/count/address/shape/dtype, capture count, replay count, and first exception. Logs read host metadata only; they never read tensor contents.

- [ ] **Step 5: Verify GREEN and frozen regressions**

Run the focused FULL suite plus all existing FDO/Piecewise/model-runner/proposer tests. Run Ruff check and format check on only touched Python files.

- [ ] **Step 6: Commit**

```bash
git add vllm_ascend/_310p/model_runner_310p.py \
  vllm_ascend/_310p/spec_decode/llm_base_proposer_310.py \
  tests/ut/_310p/test_model_runner_310p.py \
  tests/ut/_310p/spec_decode/test_dflash_full_proposer_310.py \
  openspec/changes/fix-310p-dflash-full-graph-minimal/tasks.md
git commit -s -m "feat(310p): route DFlash FULL target and draft graphs"
```

### Task 7: Hardware Trace Loop and Optional Operator Gate

**Files:**
- Create/update: `openspec/changes/fix-310p-dflash-full-graph-minimal/reports/task7-hardware-trace.md`
- Conditional modify: only the first 310P file named by a fresh stack trace.
- Conditional operator files: none until explicit user approval after an admission report.

**Interfaces:**
- Consumes: structured logs from Tasks 2-6.
- Produces: one confirmed root cause and one RED/GREEN fix per iteration, or an operator admission report and a hard pause.

- [ ] **Step 1: Rerun 4B TP1 FULL startup with DEBUG evidence**

Use the exact Task 1 command and capture sizes. Extract the first boundary where input values or modes differ from the captured contract.

- [ ] **Step 2: Add only the missing boundary print**

If the first trace is insufficient, add one DEBUG statement immediately before and after the named operation. Print host-visible metadata only: shape, dtype, device, stride, data pointer, storage offset, extent, state name, descriptor, component, and rank. Rerun once.

- [ ] **Step 3: State and test one hypothesis**

Write in the report: `Root cause hypothesis: X because observed A changes to B between boundary Y and Z.` Add a focused failing test for X. Do not combine attention mask, GDN state, chunk metadata, or dense prefill in one patch.

- [ ] **Step 4: Implement one minimal FULL-only fix and verify**

Run the focused RED, 4B TP1 FULL startup/C1, numerical comparison, and frozen Eager/Piecewise/FDO controls. Commit only if all are green.

- [ ] **Step 5: Enforce the operator checkpoint**

If Python/plugin code cannot remove host synchronization or dynamic host metadata safely, write an admission report containing the exact RED, failed safe alternative, tensor-only ABI, shapes/dtypes/alias rules, private symbol name, and negative call-count tests. Stop and request explicit approval before modifying operator source or build files.

- [ ] **Step 6: Stop on architecture expansion**

After two failed implementation hypotheses or when a third independent operator family appears, stop, preserve evidence, and update OpenSpec instead of trying another fix.

### Task 8: Fast Gates, Formal Matrix, and Independent Pull Validation

**Files:**
- Create: `openspec/changes/fix-310p-dflash-full-graph-minimal/reports/final-acceptance.md`
- Modify: `openspec/changes/fix-310p-dflash-full-graph-minimal/tasks.md`
- Modify on server 2 only after push: `/home/whn/whn_dflash/results/SUMMARY.md`
- Modify on server 2 only after push: `/home/whn/whn_dflash/results/PR_CHANGELOG_73c6b830_1a8feb60.md`

**Interfaces:**
- Consumes: passing feature commits and fixed datasets.
- Produces: reproducible per-rank real-FULL evidence, metrics, regressions, and independent pull results.

- [ ] **Step 1: Pass server-1 fast gates in order**

Run 4B TP1 C1 then C10, 4B TP2 C1/C10, 35B TP2 C1/C10, then 35B TP4 C1/C10. Stop at the first failed group. Require target/draft capture manifests and signature-specific replay on every rank.

- [ ] **Step 2: Run the formal workloads**

For every topology run GSM8K output256 and fixed-seed random input/output2048: C1 uses 4 requests; C10 uses 20. Record output throughput, acceptance length, success count, memory, requested/resolved/runtime modes, manifest, replay counts, and Eager-relative percentages.

- [ ] **Step 3: Run frozen controls and AISBench**

Run matched Eager, Piecewise, and FDO controls. Any unexplained throughput or acceptance-length drift greater than five percent blocks completion. Run AISBench smoke after vLLM bench passes.

- [ ] **Step 4: Run final static/unit verification**

```bash
pytest -q tests/ut/_310p/test_dflash_full.py \
  tests/ut/_310p/test_dflash_full_graph.py \
  tests/ut/_310p/test_dflash_full_inputs.py \
  tests/ut/_310p/test_model_runner_310p.py \
  tests/ut/_310p/spec_decode/test_dflash_full_proposer_310.py \
  tests/ut/_310p/test_dflash_full_decode_only.py \
  tests/ut/_310p/test_dflash_full_decode_acl_graph.py \
  tests/ut/_310p/test_graph_input_contract_310p.py
openspec validate fix-310p-dflash-full-graph-minimal --strict
ruff check vllm_ascend/_310p/dflash_full.py \
  vllm_ascend/_310p/dflash_full_graph.py \
  vllm_ascend/_310p/dflash_full_inputs.py \
  vllm_ascend/_310p/model_runner_310p.py \
  vllm_ascend/_310p/ops/gdn_attn_builder_310.py \
  vllm_ascend/_310p/spec_decode/llm_base_proposer_310.py \
  tests/ut/_310p/test_dflash_full.py \
  tests/ut/_310p/test_dflash_full_graph.py \
  tests/ut/_310p/test_dflash_full_inputs.py \
  tests/ut/_310p/ops/test_gdn_full_capability_310.py \
  tests/ut/_310p/spec_decode/test_dflash_full_proposer_310.py
ruff format --check vllm_ascend/_310p/dflash_full.py \
  vllm_ascend/_310p/dflash_full_graph.py \
  vllm_ascend/_310p/dflash_full_inputs.py \
  vllm_ascend/_310p/model_runner_310p.py \
  vllm_ascend/_310p/ops/gdn_attn_builder_310.py \
  vllm_ascend/_310p/spec_decode/llm_base_proposer_310.py \
  tests/ut/_310p/test_dflash_full.py \
  tests/ut/_310p/test_dflash_full_graph.py \
  tests/ut/_310p/test_dflash_full_inputs.py \
  tests/ut/_310p/ops/test_gdn_full_capability_310.py \
  tests/ut/_310p/spec_decode/test_dflash_full_proposer_310.py
git diff --check
```

- [ ] **Step 5: Perform final read-only scope review and push**

Review `git diff 1a8feb60...HEAD` to prove no upstream vLLM, generic ACL wrapper, generic runner/proposer, FDO, Piecewise, or shared operator schema change. Push only after verification.

- [ ] **Step 6: Validate an independent pull on server 2**

Pull the commit into `/home/whn/whn_dflash/vllm-ascend`, verify clean tracked status, rebuild/reinstall only if the committed diff requires it, and run 2B TP1/TP2 plus representative 35B FULL gates without copying server-1 files.

- [ ] **Step 7: Complete reports and OpenSpec tasks**

Update the server-1 acceptance report and server-2 summary/PR log with exact commands, environment, commit, results, failures, limitations, and rollback. Mark tasks complete only when their saved evidence exists.
