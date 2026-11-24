# Fused MC2 + EPLB Record: two separate operator benchmarks

Run on two **idle Ascend A3 NPUs** with this branch's framework extension and custom CANN operators installed. Use the same checkout, image, CANN/torch_npu versions, device pair, and clock/load conditions for both commands. The fixed workload matches the W8A8 operator correctness test: 2 ranks, M=128, K=N=1024, TopK=7, 40 experts/rank, list weights, seed 1234. Each rank generates the same inputs in both invocations. The script checks expert counts, accumulated load, and nonempty MoE output after timing.

From the repository root, run the **unprofiled latency/throughput comparison** in two fresh processes:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
python3 benchmarks/benchmark_fused_mc2_record.py --mode original --warmup 10 --iterations 100 --repeats 5 | tee original_bench.log
python3 benchmarks/benchmark_fused_mc2_record.py --mode fused --warmup 10 --iterations 100 --repeats 5 | tee fused_bench.log
```

`original` times `dispatch_ffn_combine` **plus** standalone `record_expert_tokens`; `fused` times `dispatch_ffn_combine` with the integrated Record parameters. Both update an initially zero expert-load array. Confirm that both modes print identical `INPUT` SHA-256 values for each rank. Each repeat synchronizes only after a batch of 100 calls, with a two-rank barrier before the batch. The `RESULT` line reports the median of the slower rank's per-call batch time over five repeats. `wall` includes Python/CANN dispatch and waiting for device completion; `event` measures the NPU event span around the same batch and may include host enqueue gaps. Neither is a pure kernel duration. Compare the same field between the two **unprofiled** runs, including repeat-to-repeat spread and both ranks' lines. For a performance claim, repeat the two fresh-process runs in reverse order to expose drift; if a background workload appears, rerun both.

Collect **independent msprof traces** in two more fresh processes, using separate output directories:

```bash
msprof --output="$(pwd)/profile_original" --task-time=on --hccl=on --ai-core=on --application="python3 benchmarks/benchmark_fused_mc2_record.py --mode original --warmup 10 --iterations 40 --repeats 1" | tee original_msprof.log
msprof --output="$(pwd)/profile_fused" --task-time=on --hccl=on --ai-core=on --application="python3 benchmarks/benchmark_fused_mc2_record.py --mode fused --warmup 10 --iterations 40 --repeats 1" | tee fused_msprof.log
```

In `op_summary_*.csv` and the timeline, locate the 40 measured `DispatchFFNCombine` calls per rank after the 10 warmups. The original trace should have one `_record_expert_tokens_kernel` after each measured MC2 call; the fused trace should have none. Check for extra copy or synchronization tasks. Profiled `Task Duration` is diagnostic: do not subtract the two MC2 medians or use profiler-instrumented `RESULT` values to claim speedup. The unprofiled batch wall/event comparisons are the operator-chain measurement; whole-model throughput remains a separate validation step.

Record the Git commit, CANN/torch_npu versions, NPU type, device IDs, and `npu-smi info` alongside the four logs and two profile directories. These commands do not rebuild the operator package; verify that the package and extension loaded by Python correspond to the same commit before measuring.
