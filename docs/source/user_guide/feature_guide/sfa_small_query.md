# Small-query NoPE SparseFlashAttention

The A2/A3 native SparseFlashAttention path handles TND queries and a paged
PA_BSND cache. For a single KV head, NoPE and sparse block size one, the
existing kernel assigns one query group per task. When the query tensor has
fewer rows than available AIC cores, the host limits the launch to that row
count and sizes its scratch regions for the resulting launch.

The bound uses the query tensor shape, including graph padding. It does not
read actual query lengths from the device on the host. Device metadata can
therefore change between replays without changing the captured launch. Query
ownership, per-core scratch layouts and the system API workspace are retained.
RoPE, other layouts, larger query tensors and the A5 path use their existing
launch and workspace calculations.

## Reproduce component measurements

Build and install each revision using the repository's normal custom-op build
procedure. Use the same device, CANN/framework versions, compiler flags and
benchmark script for both revisions. Run only one benchmark or inference task
on the device at a time. Check the actual process after an observation timeout
before starting another run.

Capture inputs once using a full custom-op installation, which includes both
the native and quantized LightningIndexer operators:

```bash
python benchmarks/benchmark_sparse_flash_attention_small_query.py \
    --phase capture --inputs /tmp/sfa-small-query-inputs \
    --output /tmp/sfa-small-query-capture.json
```

The script generates synthetic features, runs the actual Indexer operators,
keeps their returned index order and freezes the SFA tensors. Both Indexer
variants feed the unquantized NoPE SFA operator; this does not benchmark
quantized SFA or a full model. Each replay checks against an independent FP64
attention calculation. Eager and graph outputs are checked.

Run `measure` under the baseline installation, then under the candidate,
candidate again, and baseline again, preserving every output file:

```bash
python benchmarks/benchmark_sparse_flash_attention_small_query.py \
    --phase measure --inputs /tmp/sfa-small-query-inputs \
    --output /tmp/sfa-small-query-baseline-1.json
```

Change the output name for each run. Compare records with identical case names
and capture hashes. The default reports five raw device-time samples per case,
each covering 100 replays of eight SFA calls. It includes query counts above
the small-query bound as controls. Keep every case, failed record and slowdown;
do not select only improved shapes or sort indices to manufacture locality.
`--return-lse` enables an additional diagnostic path; the production NoPE
helper normally disables LSE.

Measure memory in separate processes from latency:

```bash
python benchmarks/benchmark_sparse_flash_attention_small_query.py \
    --phase memory --inputs /tmp/sfa-small-query-inputs \
    --output /tmp/sfa-small-query-baseline-memory.json
```

Repeat under the candidate with a new output name. Results include live and
peak Torch allocator observations for one eager call and an eight-call graph.
Cached allocations, outputs and graph lifetimes affect these measurements.
They are not device-wide HBM usage, model memory savings or evidence of model
throughput improvement.

Verify the installed host library and actual profiler block counts as well as
source/build identity. The compatibility `liboptiling.so` entry and its actual
`lib/linux/aarch64/libcust_opmaster_rt2.0.so` target must resolve to the intended
build. Merely observing a library in the process does not establish which
registered tiling implementation executed.
