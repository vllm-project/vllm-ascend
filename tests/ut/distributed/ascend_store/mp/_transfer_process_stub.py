"""CPU subprocess fixture for the real ZMQ/lifetime loop; never imports NPU code."""

import importlib.util
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class FakeRuntime:
    def __init__(self, config):
        if config.get("fail_init"):
            raise RuntimeError("backend initialization failed")
        self.executor = ThreadPoolExecutor(max_workers=2)

    def execute(self, operation, payload):
        if operation == "fail":
            raise ValueError("backend operation failed")
        return payload

    def submit(self, operation, payload):
        def transfer():
            time.sleep(payload["delay"])
            return payload["value"]

        return self.executor.submit(transfer)

    def close(self):
        self.executor.shutdown(wait=True)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[5]
    path = root / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/mp/worker.py"
    spec = importlib.util.spec_from_file_location("transfer_worker_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    os._exit(module.run_worker(sys.argv[1], int(sys.argv[2]), FakeRuntime))
