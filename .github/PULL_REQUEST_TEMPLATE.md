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
-->

### How was this patch tested?
<!--
CI passed with new added/existing test.
If it was tested in a way different from regular unit tests, please clarify how you tested step by step, ideally copy and paste-able, so that other reviewers can test and check, and descendants can verify in the future.
If tests were not added, please describe why they were not added and/or why it was difficult to add.
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
