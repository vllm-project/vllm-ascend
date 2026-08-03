# vLLM interface boundary tests

`singlecard/test_interface_boundaries.py` provides a CPU-only boundary check for vLLM callables coupled to
vllm-ascend. The existing upstream `vllm-interface` job collects it because that job runs the complete
`tests/e2e/vllm_interface` directory.

The compact `interface_boundaries.jsonl` file stores one upstream callable per line. Each record contains the upstream
signature boundary and all related vllm-ascend patch, override, direct-call, or inheritance endpoints.

The test checks:

- upstream files, classes, callables, and parameter boundaries;
- downstream patch and override endpoint boundaries;
- direct calls for missing/extra positional parameters and unsupported/missing keywords;
- direct inheritance edges.

For monkey-patched callables, direct calls are checked against the replacement signature. The test parses Python source
with `ast`; it does not import `torch_npu`, initialize an NPU, download a model, or execute inference.

## Source-based mapping generator (POC)

For a Chinese explanation of why the generator grew from its early version to more than 10,000 lines, including the
problem solved by every version from v0.3 to v0.36, see `接口映射生成器代码演进说明.md`.
For the architecture audit, completed refactor, and byte-for-byte accuracy evidence, see
`接口映射生成器架构评估与优化说明.md`.

`generate_interface_boundaries.py` rebuilds the low-noise subset of the mapping directly from a checked-out vLLM and
vllm-ascend source pair. It currently discovers:

- explicit monkey patches made with assignment or `setattr`;
- patch targets imported at module or function scope;
- simple target aliases such as `PATCH_TARGET = ImportedVllmClass`;
- `setattr` names resolved from string constants, string collections, or one live candidate;
- lambda, direct `property(...)`, class-body callable aliases, and statically provable wrapper factories;
- direct inheritance from a statically resolved vLLM class;
- verified overrides whose effective parent implementation is resolved through the combined MRO;
- generated dataclass constructors, typed lazy exports, patch save/restore lifecycle, and field-mutation findings;
- exact, source-pinned Triton `kernel[grid](...)` launch signatures, including literal heuristic-generated parameters;
- exact local helpers that select one literal-named class from a complete MRO, return its module, and patch that module
  through `sys.modules[name]` or `sys.modules.get(name)`;
- direct `contextlib.contextmanager` wrappers, keeping the source/reporting contract separate from the wrapper's broad
  runtime entry;
- optional exact external source indexes for methods inherited by a vLLM class, without treating external-only overrides as vLLM edges.

The POC targets vLLM main. Branches guarded by an exact `vllm_version_is("<tag>")` check are treated as release-only;
the opposite branch is indexed for main. Top-level imports under the selected branch and `try` blocks are included.
An incomplete or ambiguous vLLM/vllm-ascend MRO is reported as unresolved instead of choosing a likely parent.
The MRO-selected-module rule is fail-closed as well: the receiver class, complete MRO, literal selected class name, and
single vLLM owner must all be proven. Ordinary callables assigned to a proven instance are checked as instance
attributes; class descriptor rules are applied only when the target is a class namespace.

The generator is consumer-first. A downstream patch or inheritance declaration whose upstream target cannot be resolved
is kept in the main output as a finding instead of being silently dropped. Findings distinguish an upstream risk, an
expected injection, an excluded inactive branch, and a static-analysis review. The optional unresolved output mirrors
these findings for convenient review. It is AST-only and requires neither an NPU nor package imports.

Schema version 6 stores verified relations under `u`/`c`, candidate findings under `f`, the definition source package
under `p`, the replacement definition file
in each consumer, and patch evidence separately under `e`. Each finding includes `status`, `reason_code`, and whether it
represents a generator limitation. Evidence includes the assignment file and line, lexical scope, guards, patch kind,
and every statically discovered assignment occurrence. Signature contracts separately record source definitions,
runtime entries, reported signatures, receiver-bound calls, and access protocols. Python parse failures stop generation
instead of silently reducing coverage.

An external source root must be reproducible. The generator accepts either a Git checkout whose HEAD equals the expected
SHA or a `.interface-source.json` snapshot manifest that records the exact upstream commit and SHA-256 of every included
Python file. An unknown external parent keeps the MRO in review; the generator never chooses a later vLLM method through
an incomplete chain.

Example:

```bash
python tests/e2e/vllm_interface/generate_interface_boundaries.py \
  --vllm-root /path/to/vllm \
  --ascend-root /path/to/vllm-ascend \
  --expect-vllm-sha <vllm-sha> \
  --expect-ascend-sha <vllm-ascend-sha> \
  --external-root torch=/path/to/pytorch-source \
  --expect-external-sha torch=<pytorch-sha> \
  --output generated_boundaries.jsonl \
  --unresolved-output unresolved_relations.jsonl \
  --compare-with tests/e2e/vllm_interface/interface_boundaries.jsonl \
  --report comparison_report.json
```

The expected SHA options are recommended for reproducible local generation so that a comparison cannot accidentally use
a different source pair.
The comparison report separates exact edge matches from downstream endpoint coverage; this prevents a re-export path
change from being counted as a missing downstream dependency.

`audit_interface_boundary_coverage.py` independently enumerates source candidate sites and checks that each has exactly
one disposition in the generated mapping. It can follow direct downstream helpers with call-site module arguments, but
does not enter vLLM or external helper bodies and reinterpret their normal field assignments as downstream patches.
