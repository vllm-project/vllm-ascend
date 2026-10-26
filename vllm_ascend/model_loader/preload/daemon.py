# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import gc
import socket

import torch
from torch.utils.hooks import RemovableHandle
from vllm.config import set_current_vllm_config
from vllm.model_executor.model_loader.weight_cache import daemon as upstream
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

from .config import validate_preload_config


class AscendWeightCacheDaemon(upstream.WeightCacheDaemon):
    def load_model(self) -> None:
        # Unlike engine workers, a preload process does not construct NPUWorker.
        from vllm_ascend import ops
        from vllm_ascend.ascend_forward_context import sync_v2_extra_kwargs
        from vllm_ascend.utils import adapt_patch, register_ascend_customop, register_ascend_fla_customop

        validate_preload_config(self.vllm_config)
        device_index = current_platform.logical_device_id_to_visible_device_id(self.local_rank)
        torch.npu.set_device(device_index)
        torch.npu.config.allow_internal_format = True
        sync_v2_extra_kwargs(self.vllm_config)
        adapt_patch()
        with set_current_vllm_config(self.vllm_config):
            ops.register_dummy_fusion_op()
            register_ascend_customop(self.vllm_config)
            register_ascend_fla_customop()
            set_random_seed(self.model_config.seed)
        super().load_model()

    def warmup(self) -> None:
        # The upstream daemon warmup is specific to FlashInfer. Ascend engine
        # workers own their kernel warmup and ACL graph capture.
        return

    def _handle_release(self, conn: socket.socket) -> None:
        from .loader import _clear_rope_caches

        # Parameters can own bound weight_loader methods, forming cycles.
        # Compiled models also register strong bytecode callbacks. Drop these
        # references and collect NPU IPC limbo before upstream acknowledges a
        # copy handoff; empty_cache alone cannot release live allocations.
        if self.model is not None:
            for module in self.model.modules():
                handle = getattr(module, "_bytecode_hook_handle", None)
                if isinstance(handle, RemovableHandle):
                    handle.remove()
            del module
        self.model = None
        self.vllm_config.compilation_config.static_forward_context.clear()
        _clear_rope_caches()
        gc.collect()
        torch.npu.ipc_collect()
        super()._handle_release(conn)


def run_ascend_daemon(*args, **kwargs) -> None:
    """Importable spawn target; install device adapters in every new process."""
    from vllm_ascend import _ensure_global_patch
    from vllm_ascend.patch.platform.patch_preload import apply_preload_patch

    _ensure_global_patch()
    apply_preload_patch()
    upstream._run_daemon_original(*args, **kwargs)
