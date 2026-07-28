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
