# Main2Main interface evidence

The scheduled Main2Main workflow runs this source-only detector on a CPU runner
before starting the existing adaptation job. No vLLM installation, model call,
adaptation or NPU is needed to generate the report.

1. `resolve-source` retains its fresh/accumulated-baseline selection and freezes
   the vLLM target SHA (manual target or the checked-out main HEAD).
2. `interface-report` checks out that exact vLLM target and the selected Ascend
   source, rebasing the latter onto the selected upstream base when required.
   The old vLLM SHA comes from that source's verified-commit marker.
3. The analyzer is the vendored `tools/vllm_interface_contracts` package from the
   **workflow revision**, in a separate checkout. It is not loaded from the
   older adaptation baseline or a personal repository. This is the optimized
   v2.34.0 engine, with full attribute and inherited-state analysis enabled.
4. `prepare` pins and checks all inputs. `scan` produces candidate contract
   findings and renders one evidence-bearing `qa-review.md`. Findings do not
   fail the scan (`--fail-on never`); invalid inputs, execution errors and
   invalid reports fail the CPU job before the NPU job can start.
5. An exact-key cache stores only generated text reports, with input identity
   and SHA-256 checks. Corrupt/mismatched results are rescanned. No executable
   engine cache is deserialized. Equal old/new SHAs produce a no-change report.
   The standalone engine also retains an optional private local AST pickle cache;
   it is explicitly allowlisted in the import check, disabled by this wrapper,
   and must never be restored from shared or untrusted artifacts.
6. Only the Markdown is uploaded for QA. The adaptation job downloads it outside
   source/build/workspace cleanup paths, checks that its baseline tree matches
   the scanned tree, and passes `MAIN2MAIN_INTERFACE_REPORT` to the existing
   `kickoff` call. Both jobs use the same frozen target. Existing adaptation,
   tests, QA model selection and push behavior otherwise remain in place.

The companion `main2main_flow` change providing
`Main2MainFlow.INTERFACE_REPORT_VERSION == 1` must merge first. The workflow
checks this capability before kickoff; older flow versions must not silently
ignore the report. The environment variable belongs to the flow's orchestration
API, not to vLLM Ascend's runtime environment settings.

QA treats the report as untrusted candidate evidence from the pre-adaptation
baseline. It still checks the actual code and diff, limits findings to the
current step, and owns the verdict. Static findings and model usage records
are not proof of exhaustive detection or QA accuracy.

Cache identity includes exact old/new/Ascend/engine SHAs, engine tree, wrapper
hash, Python minor version and analysis profile. A different rebased commit or
workflow revision intentionally invalidates reuse even if some content matches.
QA Markdown is regenerated from the verified report on each run.

## CPU-only regression tests

Run from a clean committed checkout (the same invariant as the workflow):

```sh
python -B tools/main2main_interface/test_run.py --engine-root . -v
```

Fixtures cover a required-argument break and fixed consumer, report cache hit,
corruption and forced rescan, dirty inputs, exact fingerprints, equal/reversed
ranges, marker mismatch, and typing-guard removal evidence. The tests never
install vLLM or call a model. Historical CPU smoke evidence is linked in the PR;
the production adaptation/NPU pipeline is not dispatched as a smoke test.
