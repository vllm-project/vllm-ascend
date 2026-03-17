from vllm_ascend.model_loader.rfork.transfer_engine import RForkTransferEngineBackendWorker


class _StatusLike:
    def __init__(self, err: bool):
        self._err = err

    def is_error(self):
        return self._err



def test_status_ok_for_common_types() -> None:
    f = RForkTransferEngineBackendWorker._status_ok
    assert f(None) is True
    assert f(True) is True
    assert f(False) is False
    assert f(0) is True
    assert f(1) is False
    assert f(_StatusLike(False)) is True
    assert f(_StatusLike(True)) is False
