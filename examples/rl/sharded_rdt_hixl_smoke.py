# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-NPU HIXL READ smoke test, without model weights or a vLLM server.

Run against an idle Ray cluster exposing at least two NPU resources:
    python examples/rl/sharded_rdt_hixl_smoke.py
This tests CANN transfer and persistent buffers, not layerwise model reload.
"""

import ray


@ray.remote(resources={"NPU": 1})
class Endpoint:
    def __init__(self):
        import torch
        import torch_npu  # noqa: F401

        from vllm_ascend.distributed.weight_transfer.hixl_session import (
            ALIGNMENT,
            HixlSession,
        )

        self.torch = torch
        torch.npu.set_device(0)
        self.session = HixlSession(torch.device("npu", 0))
        self.buffer = self.session.allocate(ALIGNMENT)

    def publish(self, value):
        self.buffer.fill_(value)
        self.torch.npu.synchronize()
        return self.session.endpoint, self.buffer.data_ptr(), self.buffer.numel()

    def receive(self, descriptor, value):
        endpoint, address, size = descriptor
        self.buffer.zero_()
        self.torch.npu.synchronize()
        request = self.session.read(endpoint, self.buffer, address, size)
        self.session.wait(request)
        expected = self.torch.full_like(self.buffer, value)
        assert self.torch.equal(self.buffer, expected), "HIXL payload mismatch"
        return size

    def close(self):
        self.session.close()


if __name__ == "__main__":
    ray.init(address="auto", namespace="wx_sharded_rdt_hixl_smoke")
    if ray.available_resources().get("NPU", 0) < 2:
        raise RuntimeError("The smoke test requires two idle Ray NPU resources")
    source, consumer = Endpoint.remote(), Endpoint.remote()
    for value in (17, 93):
        descriptor = ray.get(source.publish.remote(value), timeout=300)
        nbytes = ray.get(consumer.receive.remote(descriptor, value), timeout=300)
        print(f"HIXL READ verified: value={value}, bytes={nbytes}")
    # Disconnect all readers before deregistering source memory.
    ray.get(consumer.close.remote(), timeout=300)
    ray.get(source.close.remote(), timeout=300)
    ray.shutdown()
