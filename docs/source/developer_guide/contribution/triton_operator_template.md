# Triton operator documentation template

Use the callable name as the page title. A complete example is [fused_qkvzba_split_reshape_cat](../operator_reference/triton/fused_qkvzba_split_reshape_cat.md).

## Description

Describe the operation from the caller's perspective. Link to the qualified wrapper definition and a representative production call or registration. Use site-relative links between documentation pages. For implementation, caller and test references, link to the file on the repository’s main branch and name the symbol; line numbers move as the code changes. Summarize the algorithm or formula when it explains observable behavior.

## Parameters and results

| Name | Role | Shape, dtype and layout | Meaning |
| --- | --- | --- | --- |
| Input, output or mutable state | Required or optional | Include relevant strides | Describe ownership and per-mode effects |

Describe the return value, buffers written in place, aliasing and required call order. Explain modes and convenience aliases together with their owning wrapper.

## Constraints

Specify supported shapes, layouts, dtypes, execution modes and fallback conditions relevant to this operation. Distinguish implementation checks from configurations exercised by tests. Cite evidence for device and graph support.

If runtime shapes, dtypes or optional arguments select different JIT variants, explain the relevant specialization conditions. For an operator that registers warmup variants, link its registration site, identify the component that opens the registry window, and describe how the serving wrapper reuses the variants. See [Triton JIT warmup on Ascend](../Design_Documents/triton_jit_warmup.md).

## Origin and differences

Identify the original implementation or explain its origin in this repository. Record Ascend adaptations that affect callers.

## Test cases

Link to an actual test file and case. Give the test command, reference implementation and comparison criterion. State which inputs, boundaries and side effects the case checks. If only integration coverage exists, link that path and identify outstanding checks.

## Example

Include a minimal call when setup, state ownership or optional modes would otherwise be unclear.
