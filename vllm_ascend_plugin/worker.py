from typing import List

from vllm_ascend_plugin.model_runner import DummyModelRunner


class DummyCacheEngine:
    pass


class NPUWorker:

    def __init__(self):
        self.cache_engine = List[DummyCacheEngine]
        self.model_runner = DummyModelRunner()
