# 统一算子回归入口

Build and install vLLM-Ascend with custom kernels enabled on an Atlas A2 host,
then run the equivalence test from the repository root. The test expects the
bundled `vllm_ascend/lib/libvllm_ascend_splitquant_dense.so` operator library.

运行run_equivalence.py --profile generic（14B/27B）或30b --out <新结果目录>；可添加--reference <原对照outputs.pt>进行逐元素零容差比较。原对照数据保留在实验目录operator/{generic,30b}/original，旧比较库已删除。

性能使用--profiler-per-case --perf-cases <本目录/splitquant_dense_fused_perf_cases.jsonl>，固定warmup5/active5、ProfilerLevel1。完整模型已验证结果见../../../../../SPLITQUANT_STATUS.md及实验REPORT.md，测试脚本均针对当前统一库。
