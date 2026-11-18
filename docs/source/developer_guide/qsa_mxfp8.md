# Qwen3.8 QSA cache on Ascend 950

Set `--kv-cache-dtype mxfp8` on both prefill and decode workers to use
ordinary GBSA with MXFP8 K/V. This path supports head dimensions 192 and
256 and one local KV head per TP rank. It requires CANN GBSA mode 6.

Q and K use dynamic group-of-32 quantization. Main attention layers require
static per-channel V scales from the checkpoint. MTP initializes its V scale
to the E8M0 encoding of one. The indexer uses per-token E4M3 keys and FP32
scales on layers with the `QK_FP8_DYNAMIC` recipe.

K and V occupy separate contiguous page pools. The planner shares these
regions with recurrent state when the manager-page geometry permits it.
A state larger than its corresponding page needs a separate region, which
can reduce available block capacity. Mooncake registers the K, V and scale
views separately. Use matching cache formats and checkpoints on P and D.
