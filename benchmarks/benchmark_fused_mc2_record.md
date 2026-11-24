# Fused MC2 + EPLB Record: separate operator benchmarks

Run on idle Ascend A3 NPUs with this branch's framework extension and custom CANN operators installed. Use the same checkout, image, CANN/torch_npu versions, devices, and clock/load conditions for both modes. The synthetic W8A8 workload uses the [DeepSeek-V4.1-Flash-1017 model geometry](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/config.json): K=5120, expert intermediate size=2304, first weight N=4608, TopK=6, 384 routed experts, and SwiGLU limit=10. `--world-size` sets the EP group size (default 2), so experts/rank is `384 / world-size`: 192 at EP2 or 48 at EP8, without EPLB redundancy. `--m` sets input token rows/rank (default 128); it is a workload choice, not a model constant. Each rank generates identical input and weights across both invocations. The script checks expert counts, accumulated load, and nonempty MoE output after timing. It does not load the model checkpoint, reproduce its weight values, or measure whole-model latency.

The two-card case holds about 6.3 GiB of int8 expert weights **per NPU** before operator workspaces; the eight-card case holds about 1.6 GiB per NPU. Weights are generated one expert at a time to limit host-memory use. Ensure enough free device memory. The A3 deployment guide lists W8A8 and W4A8 checkpoints, but this benchmark exercises the synthetic int8/W8A8 operator path only.

From the repository root, run the **unprofiled latency/throughput comparison** in two fresh processes:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
python3 benchmarks/benchmark_fused_mc2_record.py --mode original --world-size 2 --m 128 --warmup 10 --iterations 100 --repeats 5 | tee original_bench.log
python3 benchmarks/benchmark_fused_mc2_record.py --mode fused --world-size 2 --m 128 --warmup 10 --iterations 100 --repeats 5 | tee fused_bench.log
```

`original` times `dispatch_ffn_combine` **plus** standalone `record_expert_tokens`; `fused` times `dispatch_ffn_combine` with the integrated Record parameters. Both update an initially zero expert-load array. Confirm that both modes print identical `INPUT` and `OUTPUT` SHA-256 values for each rank. Each repeat synchronizes only after a batch of 100 calls, with a cross-rank barrier before the batch. The `RESULT` line reports the median of the slower rank's per-call batch time over five repeats. `wall` includes Python/CANN dispatch and waiting for device completion; `event` measures the NPU event span around the same batch and may include host enqueue gaps. Neither is a pure kernel duration. Compare the same field between the two **unprofiled** runs, including repeat-to-repeat spread and all ranks' lines. For a performance claim, repeat the two fresh-process runs in reverse order to expose drift; if a background workload appears, rerun both.

For an EP8 measurement, expose eight idle devices and pass `--world-size 8` to **both** commands, keeping `--m` and all other options equal. EP2 and EP8 have different communication behavior; compare original against fused only within the same EP size.

Collect **independent msprof traces** in two more fresh processes, using separate output directories:

```bash
msprof --output="$(pwd)/profile_original" --task-time=on --hccl=on --ai-core=on --application="python3 benchmarks/benchmark_fused_mc2_record.py --mode original --world-size 2 --m 128 --warmup 10 --iterations 40 --repeats 1" | tee original_msprof.log
msprof --output="$(pwd)/profile_fused" --task-time=on --hccl=on --ai-core=on --application="python3 benchmarks/benchmark_fused_mc2_record.py --mode fused --world-size 2 --m 128 --warmup 10 --iterations 40 --repeats 1" | tee fused_msprof.log
```

In `op_summary_*.csv` and the timeline, locate the 40 measured `DispatchFFNCombine` calls per rank after the 10 warmups. The original trace should have one `_record_expert_tokens_kernel` after each measured MC2 call; the fused trace should have none. Check for extra copy or synchronization tasks. Profiled `Task Duration` is diagnostic: do not subtract the two MC2 medians or use profiler-instrumented `RESULT` values to claim speedup. The unprofiled batch wall/event comparisons are the operator-chain measurement; whole-model throughput remains a separate validation step.

Record the Git commit, CANN/torch_npu versions, NPU type, device IDs, and `npu-smi info` alongside the four logs and two profile directories. These commands do not rebuild the operator package; verify that the package and extension loaded by Python correspond to the same commit before measuring.
