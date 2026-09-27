from contextlib import contextmanager
from unittest.mock import PropertyMock, patch

from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

from vllm_ascend.worker.v2.spec_decode.dspark import speculator as module


def test_dspark_capture_uses_draft_config_and_restores_context():
    spec = object.__new__(module.AscendDSparkSpeculator)
    draft_config = object()
    events = []

    @contextmanager
    def set_config(value):
        assert value is draft_config
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def capture(self):
        assert self is spec
        assert events == ["enter"]
        events.append("capture")

    with (
        patch.object(
            module.AscendDSparkSpeculator, "attn_vllm_config", new_callable=PropertyMock, return_value=draft_config
        ),
        patch.object(module, "set_current_vllm_config", set_config),
        patch.object(DFlashSpeculator, "capture", capture),
    ):
        spec.capture()
    assert events == ["enter", "capture", "exit"]
