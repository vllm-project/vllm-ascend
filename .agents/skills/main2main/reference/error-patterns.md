# Common Error Patterns Reference

These are the most frequently seen failure patterns when upstream vLLM evolves. Use this reference when diagnosing CI failures or applying fixes.

---

## Method Signature Change

**Error:** `TypeError: forward_oot() got an unexpected keyword argument 'X'` or `missing 1 required positional argument: 'X'`

**Cause:** vLLM changed a method signature — parameter added, removed, renamed, or full API replacement (e.g., `disable_full` to `valid_modes`/`invalid_modes`).

**Fix:** Compare signatures at good vs bad commit, then adapt:

```python
from vllm_ascend.utils import vllm_version_is

# Option 1: Add parameter conditionally to call site
kwargs = {"existing_param": value}
if not vllm_version_is("0.16.0"):
    kwargs["new_param"] = new_value
function(**kwargs)

# Option 2: Add default parameter to OOT method signature
def forward_oot(self, query, key, value, cu_seqlens=None, max_seqlen=None, new_param=None):
    ...
```

For full API replacements, adapt the call site to match the new API — do NOT blindly add the old parameter.

**Important:** When creating version-guarded branches, all branches must define the function with identical signatures (convert lambdas to `def` if needed). Mismatched signatures across branches cause mypy `[call-arg]` errors.

---

## Config/Attribute Change

**Error:** `AttributeError: 'CompilationConfig' object has no attribute 'X'`, `KeyError: 'field_name'`, or `Config object has no attribute 'Y'`

**Cause:** Upstream moved an attribute/config field between classes, restructured a config class, or added a new required field (e.g., `bs_to_padded_graph_size` moved to `CudagraphDispatcher`, `uses_mrope` moved from target to draft model config, `enable_eplb` added to `FusedMoEParallelConfig`).

**Fix:** Use `vllm_version_is()` to access from the correct location:

```python
if vllm_version_is('0.16.0'):
    value = self.vllm_config.old_location.attribute
else:
    value = self.new_class.new_location.attribute
```

For config access that changes frequently, consider helper methods like `_get_positions()` / `_set_positions()` to abstract the logic. For new required fields, add them to the config wrapper.

---

## Custom Op Not Registered

**Error:** `AttributeError: '_OpNamespace' '_C' object has no attribute 'op_name'`

**Cause:** vLLM code references `torch.ops._C.op_name` — a CUDA custom op not available on Ascend.

**Fix:** Register an equivalent Ascend op, or override config to use a different code path (e.g., re-force `+rms_norm` in `custom_ops` for SP).

---

## Method Return Type Change

**Error:** `TypeError: '>' not supported between instances of 'NoneType' and 'NoneType'` or similar comparison errors on None.

**Cause:** Upstream changed a method from returning `None` to returning a value (e.g., `float`), and the caller now uses it.

**Fix:** Update the OOT override to return the expected value.

---

## Module Reorganization

**Error:** `ImportError: cannot import name 'X' from 'vllm.old.path'`, or `error: Cannot find implementation or library stub for module named "vllm.X" [import-not-found]`

**Cause:** vLLM moved/renamed a module, or removed it entirely (e.g., `vllm._bc_linter`).

**Fix:** For moved/renamed modules, use `vllm_version_is()` to branch imports:

```python
if vllm_version_is("0.16.0"):
    from vllm.old.path import X
else:
    from vllm.new.path import X
```

For removed modules, delete the import **and** all usages (decorators, function calls) — clean removal over `# type: ignore`.

---

## Platform Interface Addition

**Error:** `TypeError: Can't instantiate abstract class AscendPlatform with abstract method X`

**Cause:** New abstract method added to vLLM's `Platform` base class.

**Fix:** Implement the method in `vllm_ascend/platform.py`. Check the base class signature and return type, then provide an Ascend-appropriate implementation.

---

## Environment Flakes (NO FIX NEEDED)

These are transient infrastructure issues — note them in the report but require no code changes:

- `OSError: [Errno 116] Stale file handle` — multi-process NFS race
- `ConnectionResetError` — transient network failure
- `filelock` errors — model download contention
- `ConnectionRefusedError` — service not ready
- `TimeoutError` — transient timeout
- `torch.cuda.OutOfMemoryError` — resource exhaustion
- `OSError: No space left on device` — disk full
