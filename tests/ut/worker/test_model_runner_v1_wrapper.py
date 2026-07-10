from types import SimpleNamespace

import vllm_ascend.worker.model_runner_v1 as model_runner_v1


def test_torch_cuda_wrapper_preserves_runtime_npu_event(monkeypatch):
    class FakeNPUEvent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_npu = SimpleNamespace(
        Event=FakeNPUEvent,
        Stream=object(),
        default_stream=object(),
        current_stream=object(),
        stream=object(),
        synchronize=object(),
        mem_get_info=object(),
    )
    fake_cuda = SimpleNamespace(
        Event=object(),
        Stream=object(),
        default_stream=object(),
        current_stream=object(),
        stream=object(),
        synchronize=object(),
        mem_get_info=object(),
    )
    fake_torch = SimpleNamespace(Event=object(), npu=fake_npu, cuda=fake_cuda)
    monkeypatch.setattr(model_runner_v1, "torch", fake_torch)

    with model_runner_v1._torch_cuda_wrapper():
        assert fake_torch.cuda.Event is FakeNPUEvent

    assert fake_torch.cuda.Event is FakeNPUEvent
    event = fake_torch.cuda.Event(blocking=True)
    assert event.kwargs == {"blocking": True}
