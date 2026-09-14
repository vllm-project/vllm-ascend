# Mixed-window DFlash cache repair

## Scope and layout

This restores the cache repair for Ascend V2 with mixed Full/SWA DFlash and
Mamba/GDN target state. It leaves target-only, all-full DFlash, and all-sliding
DFlash paths unchanged. It does not change sampling, GDN rollback, attention
masks, model weights, or the official four SWA layers / one full layer / 2048
window configuration. Unsupported layouts fail at initialization.

Ascend stores contiguous conv/SSM planes at the start of each shared raw pool
and contiguous K/V planes at its tail. Equal `page_size_padded` alone is not
enough when the Full and SWA physical block sizes differ. In the reported TP2
geometry, only 12 physical blocks are needed for old SWA block 1 K/V writes to
overlap full-attention V blocks 10/11 despite their distinct block IDs.

The repair derives the attention storage block size from the SSM plane bytes
divided by the K/V row bytes. For this TP2 geometry, SWA storage changes from
128 to 1536 tokens; the kernel still uses 128-token blocks and the attention
window remains 2048. With conv page bytes `C=102400`, SSM page bytes
`S=1572864`, and common page bytes `P=C+2*S`, the actual layout becomes
`[all conv, all K/SSM, all V]`. Distinct physical IDs then own disjoint ranges.
Startup validates actual pointers, dtype, contiguity, and shared-arena offsets.

Separately, the supplied NPU probe passed scatter writes but failed FIA reads
at kernel block IDs 65536 and 65543. For that geometry, block 65536 starts at
element `2**32`. The repair conservatively bounds both the kernel-block count
and plane-element count, without claiming which CANN internal index caused
the failure. It also limits explicit overrides to the real profiled memory
budget and reruns the planner before allocation. The address-derived limit
is 5461 physical blocks for the reported geometry; memory or a smaller override
may reduce it further. This is an operator workaround, not a CANN kernel fix.

Both context prewrites and ordinary cache writes mask the entire reserved
physical null block and out-of-range slots. A 1536-token null block includes
kernel blocks 0 through 11. No per-step debug logging or CPU synchronization
is added.

## Validation

Run in the normal vLLM-Ascend development environment:

```bash
pytest -q tests/ut/worker/test_dflash_cache_layout.py tests/ut/worker/test_dflash_cache.py tests/ut/worker/test_dflash_cache_views.py
```

The first two suites cover the byte-level alias and planner contracts; the
metadata suite substitutes dependency fixtures and is not runtime integration.
The view suite uses real Torch tensors, real vLLM specs and the production
materializer. None replaces NPU end-to-end validation.

Deploy all changed Python modules together and restart all workers. Keep the
official draft config unchanged. First retain `--num-gpu-blocks-override 4096`
and `--enforce-eager`, replaying the previously failing GPQA request order and
warmup at concurrency 1, then 32. Exercise long outputs and block reuse with
different requests; a single short curl is insufficient. After this passes,
remove the override and repeat, then test graph mode.

Expected startup markers for the reported TP2 geometry:

```text
DFlash mixed cache layout aligned: ... block_size=128 -> 1536, sliding_window=2048 ...
DFlash mixed cache guard: physical_blocks=... -> ...
DFlash mixed cache plan ready: physical_blocks=... (memory/address safe)
DFlash mixed cache views verified: ... conv_bytes=102400, plane_bytes=1572864
```

The guard warning only appears when capacity is reduced. A startup layout
exception means validation rejected the configuration; absence of an exception
does not prove generation accuracy. Record raw responses, `finish_reason`,
completion-token counts and server logs. High acceptance alone is not a pass.

If repetition remains, compare the first differing output token against the
target-only baseline at an identical integer-token prefix, including top
logprobs. Replaying a prefix that has already looped for thousands of tokens
cannot establish that the earlier transition into the loop was correct.
This patch restores the cache repair only; it does not add repetition penalties
or claim to resolve all causes of repetitive generation.
