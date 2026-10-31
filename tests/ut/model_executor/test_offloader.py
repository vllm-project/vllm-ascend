from types import SimpleNamespace

from vllm.model_executor.offloader.base import NoopOffloader

from vllm_ascend.model_executor.offloader.base import create_offloader
from vllm_ascend.model_executor.offloader.prefetch import _is_using_nz_weight


def test_create_offloader_without_config_returns_noop():
    offloader = create_offloader(None)

    assert isinstance(offloader, NoopOffloader)


def test_create_offloader_for_non_prefetch_backend_returns_noop():
    offload_config = SimpleNamespace(
        offload_backend=None,
        prefetch=None,
    )

    offloader = create_offloader(offload_config)

    assert isinstance(offloader, NoopOffloader)


def test_is_using_nz_weight_handles_invalid_npu_format(monkeypatch):
    param = SimpleNamespace(
        data=SimpleNamespace(device=SimpleNamespace(type="npu")),
    )

    monkeypatch.setattr(
        "vllm_ascend.model_executor.offloader.prefetch.torch_npu.get_npu_format",
        lambda _: object(),
        raising=False,
    )

    assert not _is_using_nz_weight(param)
