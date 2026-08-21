## Purpose

Define a replay-safe and testable way to locate and correct deterministic numerical divergence between Ascend 310P DFlash Eager and genuine FULL_DECODE_ONLY execution without weakening graph authenticity or unrelated behavior.

## ADDED Requirements

### Requirement: Exact diagnostic scope
The numerical-equivalence capability SHALL activate only for an explicitly enabled diagnostic run using Ascend 310P, DFlash, and either Eager or `FULL_DECODE_ONLY`. Normal INFO/DEBUG serving, Piecewise, `FULL`, `FULL_AND_PIECEWISE`, non-DFlash, non-speculative, and non-310P execution MUST remain unchanged.

#### Scenario: Probe is not requested
- **WHEN** a service starts without the numerical-probe configuration
- **THEN** it allocates no probe buffers, installs no layer probes, emits no tensor artifacts, and follows the existing execution path

#### Scenario: Out-of-scope mode requests a probe
- **WHEN** numerical probing is requested outside the declared 310P DFlash Eager/FDO scope
- **THEN** startup fails with a scope error rather than silently instrumenting another path

### Requirement: Frozen paired reproduction
Every comparison SHALL use identical software, model weights, tokenizer, prompt tokens, generation settings, DFlash K, request order, TP size, and serving flags. The initial localization case MUST reproduce the stable Qwen3.6-35B-A3B-w8a8 TP2 concurrency-1 divergence on GSM8K record 12, whose archived first differing generated-token index is 48, before layer probes are used.

#### Scenario: Reproduction is stable
- **WHEN** the frozen eager and FDO diagnostic pair runs GSM8K record 12 with at least 64 output tokens
- **THEN** Eager repeats identically and FDO reproduces the archived branch at token index 48, or the investigation records that the baseline is no longer reproducible and stops before proposing a fix

### Requirement: Boundary-first localization
The diagnostic flow SHALL compare the earliest available scheduler/model inputs, target outputs, draft outputs, selected hidden states, logits, proposed tokens, and rejection-sampler outputs at matched speculative iterations before inspecting transformer layers.

#### Scenario: Boundary mismatch is found
- **WHEN** matched traces first differ at an outer target, draft, or sampler boundary
- **THEN** the report identifies the component, iteration, active request, logical token row, descriptor, tensor role, and preceding boundary that still matched

#### Scenario: Inputs differ before model execution
- **WHEN** input IDs, positions, active lengths, block/slot metadata, or selected rows differ for a matched iteration
- **THEN** layer-level probing stops and the input-state producer becomes the root-cause investigation boundary

### Requirement: Replay-safe selected-layer probing
FDO layer observation SHALL use persistent graph-owned device storage updated by the captured graph and exported only after replay completion. An explicit bounded set of target or draft layers SHALL be selected per diagnostic service, and observation MUST NOT create an eager model island.

#### Scenario: Selected FDO layers replay
- **WHEN** a selected layer executes in a captured FDO descriptor
- **THEN** its active-lane output is copied inside the real graph, the copy updates on each replay, and the post-replay artifact is associated with that replay identity

#### Scenario: Unselected layer executes
- **WHEN** a transformer layer is not selected for the current diagnostic service
- **THEN** no output-copy operation or probe storage is added for that layer

### Requirement: Active lanes and padding are distinguished
Every artifact SHALL record descriptor capacity, actual active request/token counts, logical row mapping, tensor shape, dtype, rank, and component. Comparisons MUST ignore initialized inactive descriptor tails while still verifying that active rows are complete and ordered identically.

#### Scenario: Descriptor contains padding
- **WHEN** an FDO descriptor has inactive request or token lanes
- **THEN** numerical comparison includes every active logical lane and excludes only the explicitly recorded inactive tail

#### Scenario: Active row mapping is incomplete
- **WHEN** either side cannot map an observed row to the same request, speculative iteration, and logical token position
- **THEN** the comparison fails as an alignment error rather than reporting a numerical mismatch

### Requirement: Numerical evidence is sufficient and bounded
For each matched tensor, the trace SHALL retain a deterministic identity plus shape/dtype metadata, finite-value status, nonzero-difference count, maximum and mean absolute difference, cosine similarity where meaningful, and selected top-k token IDs/logits with argmax margin for logits. Full active-row tensor payloads MAY be retained only for the bounded diagnostic prompt and iteration range.

#### Scenario: Logits choose different tokens
- **WHEN** Eager and FDO logits have different selected tokens
- **THEN** evidence contains both selected token IDs, both logits evaluated at those IDs, each side's top-k set, and each argmax margin

#### Scenario: Diagnostic bound is reached
- **WHEN** the configured maximum iteration or artifact-size bound is reached
- **THEN** probing stops cleanly and the manifest reports truncation without changing inference output

### Requirement: First differing layer is located monotonically
After identical inputs and an internal hidden-state divergence are established, the investigation SHALL locate the earliest differing transformer layer separately for target and draft. It MAY use a bounded ordered checkpoint sweep when the configured record/byte limits cover the selected layers; otherwise it SHALL use binary search followed by adjacent-layer confirmation.

#### Scenario: An ordered checkpoint matches
- **WHEN** all selected checkpoints through layer N match under the declared numerical comparison
- **THEN** the next comparison moves strictly later than N

#### Scenario: Earliest layer is claimed
- **WHEN** layer N differs and layer N-1 matches for the same paired iteration
- **THEN** the report identifies layer N as the first differing layer and records both artifacts

### Requirement: Hypotheses are isolated
The investigation SHALL test one implementation difference at a time against the reproduced prompt. The first candidates SHALL be FDO-only W8A8 linear chunking and FDO-only MoE event/ordering behavior, but neither MAY be changed as a repair until boundary/layer evidence connects it to the first divergence.

#### Scenario: Candidate does not move the first divergence
- **WHEN** a single-variable experiment leaves the first differing boundary/layer and numerical signature unchanged
- **THEN** the candidate is rejected, its experiment is removed, and no additional change is stacked on it

#### Scenario: Candidate removes the first divergence
- **WHEN** a single-variable experiment makes the previously differing layer and downstream logits match without graph fallback
- **THEN** it becomes the root-cause candidate and proceeds to a failing regression test and minimal production repair

### Requirement: Repair remains plugin-only and graph-authentic
The final repair SHALL be confined to vLLM Ascend Python code, MUST preserve genuine target and draft FULL replay, and MUST NOT modify upstream vLLM, C++, AscendC, custom operators, model weights, or create an eager island.

#### Scenario: Safe plugin correction exists
- **WHEN** first-divergence evidence supports a plugin-side correction
- **THEN** the smallest correction is implemented behind the exact FDO scope and covered by a test that failed before the correction

#### Scenario: Safe plugin correction does not exist
- **WHEN** equivalence requires an upstream/operator modification or removing required work from the graph
- **THEN** the change stops with the first-divergence evidence and requests a separate scope decision

### Requirement: Divergence triage follows observed frequency
An isolated wording-only branch MAY be recorded without immediate layer localization when it remains the sole stable branch after paired Eager and FDO reruns. Two or more branches that reproduce at the same requests and token positions SHALL be treated as a numerical correctness cluster and MUST be localized.

#### Scenario: One isolated wording branch occurs
- **WHEN** exactly one stable request branch remains after Eager and FDO repeat classification and its completed answer remains semantically equivalent
- **THEN** it is recorded as non-blocking diagnostic evidence without triggering a full layer search

#### Scenario: Multiple stable branches occur
- **WHEN** at least two requests reproduce the same first differing token positions across paired reruns
- **THEN** validation remains open until the first numerical divergence is localized and either repaired or explicitly accepted as a scoped limitation

### Requirement: Acceptance protects all established FDO behavior
After repair, the exact divergent 35B prompt SHALL pass first, followed by 35B TP2 C1/C4 and the established 9B TP1 C1/C4/C10 plus 9B TP2 C1/C4 gates. All groups MUST retain request success, graph proof, acceptance-length/rate, throughput, memory, and safety thresholds from the parent FDO change.

#### Scenario: Fast 35B prompt passes
- **WHEN** the repaired Eager/FDO pair runs GSM8K record 12
- **THEN** the first 64 generated token IDs match and target/draft FULL replay remains genuine on both TP ranks

#### Scenario: Existing graph or quality metric regresses
- **WHEN** any established group loses graph replay, request success, acceptance quality, throughput threshold, memory viability, or concurrency-10 safety
- **THEN** the repair is rejected even if the original 35B prompt matches
