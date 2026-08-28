# SPDX-License-Identifier: Apache-2.0

import ctypes
import gc
import os
import time

import torch
import torch.nn as nn
from vllm.logger import logger


def dump_state_dict(model: nn.Module, path: str) -> None:
    if os.path.exists(path):
        logger.info("model save path %s exists, skip dump model", path)
        return

    logger.info("[dump model] start dump model to %s (type=%s)", path, type(model))
    start = time.time()
    import psutil  # type: ignore[import-untyped]

    process = psutil.Process(os.getpid())
    logger.info("start dump_model() cpu memory use: %.2f MB", process.memory_info().rss / 1024**2)
    torch.save(model.state_dict(), path)
    gc.collect()
    logger.info("after gc.collect() cpu memory use: %.2f MB", process.memory_info().rss / 1024**2)
    torch.npu.empty_cache()
    logger.info("after torch.npu.empty_cache() cpu memory use: %.2f MB", process.memory_info().rss / 1024**2)
    try:
        libc = ctypes.CDLL("libc.so.6")
        result = libc.malloc_trim(0)
        if result == 1:
            print("exec malloc_trim(0) success")
        else:
            print("exec malloc_trim(0) fail")
    except Exception as e:
        print(f"exec malloc_trim(0) with error: {e}")

    logger.info("after dump_model() cpu memory use: %.2f MB", process.memory_info().rss / 1024**2)
    logger.info("[dump model] save model ckpt to %s, elapse %.4f s", path, time.time() - start)


def restore_state_dict(model: nn.Module, path: str, label: str) -> None:
    if not os.path.exists(path):
        logger.warning("[restore model] [%s] ckpt %s not found, skip", label, path)
        return

    start = time.time()
    state_dict = torch.load(path, map_location="cpu", mmap=True)
    logger.info(
        "[restore model] [%s] load model to cpu from %s, elapse %ss, the num of items is %s",
        label,
        path,
        time.time() - start,
        len(state_dict),
    )
    restored = 0
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    for name, cpu_tensor in state_dict.items():
        if name in parameters:
            parameters[name].data.copy_(cpu_tensor)
            restored += 1
        if name in buffers:
            buffers[name].data.copy_(cpu_tensor)
            restored += 1
    logger.info("[restore model] [%s] replace success %s / %s", label, restored, len(state_dict))
    logger.info(
        "[restore model] [%s] restore model ckpt from %s, elapse %.4f s",
        label,
        path,
        time.time() - start,
    )
