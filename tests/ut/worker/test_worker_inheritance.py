# SPDX-License-Identifier: Apache-2.0
"""Runtime resolution guards for the RFC Worker migration."""


def test_inherited_worker_method_resolution():
    from vllm.v1.worker.gpu_worker import Worker as GPUWorker

    from vllm_ascend._310p.worker_310p import NPUWorker310
    from vllm_ascend.worker.worker import NPUWorker
    from vllm_ascend.worker.worker_interfaces import INHERITED_WORKER_METHODS
    from vllm_ascend.xlite.xlite_worker import XliteWorker

    for cls in (NPUWorker, NPUWorker310, XliteWorker):
        for name in INHERITED_WORKER_METHODS:
            assert getattr(cls, name) is getattr(GPUWorker, name), (cls.__name__, name)
