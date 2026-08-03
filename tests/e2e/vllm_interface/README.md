# vLLM interface contract tests

`singlecard/test_interface_contracts.py` watches the callable-level vLLM interfaces used by vllm-ascend. The first
baseline covers monkey patches, callable overrides, direct imported callable calls, and direct inheritance. The existing
upstream `vllm-interface` job collects this test because it runs the complete `tests/e2e/vllm_interface` directory.

The test only parses Python source code. It does not import `torch_npu`, initialize an NPU, download a model, or run
inference. Each manifest entry checks that:

- the upstream file, class, and callable still exist;
- positional, keyword-only, and variadic parameters keep the same names and kinds;
- parameters do not change between required and optional.
- direct calls do not use keyword arguments removed from the upstream callable.

Annotations, default values, return values, fields, and runtime behavior are intentionally outside the first version of
the contract. Direct calls to monkey-patched callables are excluded from keyword validation because their effective
signature comes from the patch rather than the original upstream callable. This keeps the signal focused on callable
interface breaks.

The vLLM baseline commit is recorded in `interface_contracts.json`. When a contract fails, review every downstream
consumer shown in the assertion message. Update the baseline only after confirming that the vllm-ascend implementation
remains compatible or has been adapted.
