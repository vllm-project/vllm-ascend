import gc
import os

import pytest
import torch

# 强制 vLLM 使用 spawn 方式启动子进程，避免 pytest 环境下的 fork 死锁
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import LLM, SamplingParams


@pytest.mark.flaky(reruns=1, reruns_delay=10)
@pytest.mark.timeout(600)  # 给 (模型加载 + 推理) 留足时间，设为 10 分钟
def test_minimal_repro_310p_flaky():
    """
    310P 最小复现用例：验证基础推理链路，并测试显存清理边界。
    目的：防止长序列测试后 OOM，以及底层通信挂死导致的 CI 超时。
    """
    model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    llm = None

    try:
        # 使用相对保守的显存设置
        llm = LLM(
            model=model_path,
            max_model_len=1024,
            gpu_memory_utilization=0.8,  # 改回 0.8，0.6 可能导致无法分配 KV Cache
            enforce_eager=True,
            dtype="float16",
        )

        prompts = ["测试偶发失败重试。"]
        sampling_params = SamplingParams(temperature=0.0, max_tokens=10)

        outputs = llm.generate(prompts, sampling_params)
        assert len(outputs) == 1
        print("310P 基础推理链路验证成功。")

    except RuntimeError as e:
        if "NPU out of memory" in str(e):
            pytest.fail("触发 OOM：310P 显存清理边界失效，需检查前置测试是否泄漏显存。")
        else:
            raise e
    finally:
        # 强制清理：防止当前用例失败后，污染后续 CI 用例
        if llm is not None:
            from vllm.distributed.parallel_state import (
                destroy_model_parallel,
                destroy_distributed_environment,
            )
            destroy_model_parallel()
            destroy_distributed_environment()
            del llm
        gc.collect()
        torch.npu.empty_cache()
