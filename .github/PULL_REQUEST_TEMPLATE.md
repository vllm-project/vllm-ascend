<!--  Thanks for sending a pull request!

BEFORE SUBMITTING, PLEASE READ https://docs.vllm.ai/en/latest/contributing/overview.html

-->
### What this PR does / why we need it?
<!--
- Please clarify what changes you are proposing. The purpose of this section is to outline the changes and how this PR fixes the issue.
If possible, please consider writing useful notes for better and faster reviews in your PR.

- Please clarify why the changes are needed. For instance, the use case and bug description.

- Fixes #
-->

### Does this PR introduce _any_ user-facing change?
<!--
Note that it means *any* user-facing change including all aspects such as API, interface or other behavior changes.
Documentation-only updates are not considered user-facing changes.

If yes, add a release note fragment `release_notes/fragments/<PR number>.yaml` in this PR.
If no, no fragment is needed. When unsure, add one; the release manager can drop extras later.
See release_notes/README.md for the principles and format.
-->

### How was this patch tested?
<!--
Give the commands and observed results for this revision. If you did not add tests, explain why.
For inference accuracy or performance changes, identify the baseline and PR revisions and include
the applicable evidence below. Link longer logs and note any relevant checks still pending.

- For accuracy, name the model, dataset (split and sample count), and metric. Report baseline and PR scores
  and their difference under the same evaluation settings.
- For performance, name the model, NPU type, and card count. State whether
  prefill and decode are colocated or disaggregated. For disaggregation, give instances per side and cards
  per instance. Report input and output lengths in tokens (or their distributions) and request concurrency.
  Compare the same workload and report both values, units, and the change for the claimed metric
  (e.g. output tokens/s or P95 time per output token in ms).
- For an optimization, say whether it uses a shared path across NPUs (e.g. A3 and A5) or a chip-specific path.
  Give results separately for each NPU and deployment mode tested, and identify any untested configurations
  within the claimed scope.
-->

---

### Checklist

<!--
Please go through the checklist below before requesting review.
Tick the boxes that apply ([x]). If an item does not apply, leave it unchecked;
optionally explain why in the PR description.
-->

#### Testing

- [ ] **Boundary / edge-case coverage**: I reviewed whether this change involves
  boundary conditions (e.g. empty input, single/zero element, min/max values,
  off-by-one, sequence-length limits, integer overflow). Where applicable, I
  added unit tests covering these edge cases in `tests/ut/` or `tests/e2e/`.
- [ ] **Regression test**: If this PR fixes a bug, I added a regression test
  that would fail without the fix.
- [ ] **Error / failure path coverage**: I covered abnormal paths (exceptions,
  timeouts, out-of-range, division by zero) in addition to the happy path.
- [ ] **Meaningful & reliable tests**: New tests assert real behavior and are
  not written merely to satisfy CI.
- [ ] **Performance impact**: If this PR touches a performance-sensitive path,
  I attached benchmark results and added performance regression tests.
