# DSv4 partial prefix cache — design, rationale, and benchmark method

This documents the three commits that make sub-block ("partial") prefix-cache
reuse a net win for DeepSeek-V4 compressed MLA, **why** each step was needed,
and **how** the numbers were measured so the result can be reproduced.

## Why a partial path exists at all

At `block-size=128` the DSv4 compressed cache block spans
`block_size * compress_ratio = 128 * 128 = 16384` tokens. vLLM's normal
prefix-cache hit is **whole-block**: a request only hits a block once a full
16384-token block matches. So **any prompt shorter than one compressed block
(2K–8K) gets 0% prefix-cache hit through the full-block path** — every request
pays full prefill recompute even with `--enable-prefix-caching`.

The partial path (`core/single_type_kv_cache_manager.py`) adds a private
short-key fallback: it caches sub-block boundaries and, on a lookup, copies the
matching cached KV into a fresh private block so the request can reuse it. This
turns the 0% into 76–83% hit for 2K–8K prompts.

## Why the naive partial path regressed short prompts

A whole-block hit is **free**: the block is shared by reference (refcount), no
data movement. A partial hit is **not** — the matched range is a sub-block that
cannot be shared, so its KV is **copied** into a private block (~60 per-layer
indexed-copy kernels on the worker). That copy has a large fixed cost.

Cost/benefit per request:
- **benefit** = prefill compute saved ∝ hit length
- **cost** = KV copy + boundary hashing (largely fixed)

For short prompts the prefill is cheap, so the saved prefill is small in
absolute terms and the copy cost dominates → **the hit made TTFT worse**. This
is the regression: high hit rate, slower TTFT.

## The fixes

1. **Batched copy** (`worker/model_runner_v1.py`). Materialize the hit with a
   single Triton `batch_memcpy` launch covering every (layer, block-pair) D2D
   copy, instead of ~60 per-layer indexed copies (2 kernels each). The fixed
   launch overhead of those ~60 copies — not the data volume — is what dominates
   the copy cost for short prompts. Collapsing it to one launch is what flips a
   short partial hit from a loss to a win (2K: +13% regression -> -42% vs
   no-cache on A2). Falls back to the per-layer copy only on 310P, where Triton
   batch_memcpy is unavailable.

2. **Copy/compute overlap** (`worker/model_runner_v1.py`). Issue the KV copy on
   a dedicated NPU stream (ordered after pending default-stream KV writes via
   `wait_stream`) and record an event; `_model_forward` waits on that event
   before the model reads the cache. The copy then overlaps with the
   default-stream input preparation (token gather, positions, attention
   metadata, H2D copies) that runs between `_update_states` and the forward and
   does **not** read KV. Correctness is unchanged — the forward still waits for a
   fully materialized cache.

3. **Hit-length gate** (`envs.py: VLLM_ASCEND_DSV4_PARTIAL_MIN_HIT_TOKENS`,
   **default 0 = never skip**). Optional policy knob: skip a partial hit whose
   matched length is below the threshold. With the batched + overlapped copy a
   short partial hit is already a net win, so the default uses every hit; raise
   it only on a deployment where short partial hits are measured unprofitable.

All three only act on prompts that get **zero** whole-block hits (i.e. <16K at
bs=128, where `computed_blocks[0]` is empty). Long-context whole-block prefix
hits use the standard zero-copy reference path; `_copy_prefix_cache_blocks`
returns early and the forward event wait is skipped, so long context is
completely unaffected.

## How the numbers were measured

Hardware/model: DeepSeek-V4-Flash w8a8 + MTP, 8×910B4, TP8, vLLM 0.21.0rc,
`block-size 128`, `max-num-seqs 32`.

Serve (see `start_matrix_np.sh` style invocation):

```bash
# PC ON, partial fully enabled (short prompts also hit):
VLLM_ASCEND_DSV4_PARTIAL_MIN_HIT_TOKENS=0 \
  vllm serve <model> --enable-prefix-caching --block-size 128 \
    --tensor-parallel-size 8 --enable-expert-parallel --max-num-seqs 32 \
    --quantization ascend --async-scheduling ...

# PC OFF baseline (no caching at all):
  vllm serve <model> --no-enable-prefix-caching --block-size 128 ...
```

Load probe — ais-bench, prefix-cache dataset with a shared (90% repeated)
prefix so partial hits are possible:

```bash
python aisbench_test.py \
  --input_len {2048|4096|8192} --output_len 1024 \
  --data_num 64 --concurrency 32 --request_rate 0 \
  --dataset_type prefix_cache --repeat_rate 90% --prefix_num 1 --prefix_test \
  --npu_num 8
```

Method per length: 1 warmup + 3 measured rounds; metrics read from the
ais-bench result CSV (TTFT avg, TPOT avg, output throughput, prefill
throughput). Each A/B leg restarts the serve so PC-on and PC-off are clean,
independent runs. Hit rate and copy count are cross-checked from the serve log
(`Prefix cache hit rate`, `DSV4_COPY_N`).

Correctness of the overlapped copy is verified separately: the same long prompt
is sent 4× with greedy decoding (`temperature=0`); the first is a cache miss and
the rest are partial hits consuming the overlapped copy. All four completions
must be byte-identical (a half-copied/stale KV would diverge).

## Results (concurrency 32, output 1k, median TTFT, A2 / 910B4)

Per-layer copy + overlap vs the final batched copy + overlap, both vs a
same-session PC-off baseline, gate open (`min_hit=0`) so short prompts hit:

| input | PC-off | per-layer copy | batched copy   | hit  |
|-------|--------|----------------|----------------|------|
| 2K    | 1741ms | 1974ms (+13%)  | 1007ms (-42%)  | ~78% |
| 4K    | 3205ms | 1594ms (-50%)  | 1558ms (-51%)  | ~80% |
| 8K    | 5512ms | 2233ms (-60%)  | 2726ms (-51%)  | ~83% |

Reading:
- **Per-layer copy**: hits but regresses 2K (the ~60 per-layer launch overhead
  exceeds the prefill saved on a short prompt); 4K/8K already win because their
  copy amortizes.
- **Batched copy**: collapses the launches into one, which flips 2K from a
  regression to a -42% gain — the breakthrough for short prompts. 4K is
  unchanged (-51%); 8K is slightly worse than per-layer (the large-copy Triton
  path is less efficient than the native indexed copy there) but still -51% vs
  no-cache. The short-prompt win is the goal; 8K is handled by a separate
  optimization.

Correctness is verified separately: the same long prompt is sent 4× with greedy
decoding (`temperature=0`); the first is a cache miss and the rest are partial
hits consuming the batched copy. All four completions are byte-identical (a
half-copied / stale KV would diverge).

## Caveats

- The gate default is `VLLM_ASCEND_DSV4_PARTIAL_MIN_HIT_TOKENS=0` (every partial
  hit is used). The batched + overlapped copy makes short hits profitable, so
  there is no reason to skip them by default; raise the knob only if a given
  deployment measures short hits to be unprofitable.
- TTFT is the prefill metric prefix caching targets. TPOT/throughput deltas
  between separate serve runs are mostly run-to-run variance — caching does not
  change decode — and should not be read as a caching gain.
- TTFT at concurrency 32 carries queueing noise; the warmup + 3-round protocol
  reduces but does not eliminate it.
