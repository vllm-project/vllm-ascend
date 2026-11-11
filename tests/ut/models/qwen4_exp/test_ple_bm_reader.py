# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.qwen4_exp import ple_bm_reader


class Client:
    def __init__(self):
        self.calls = 0
        self.fail = False

    def close(self):
        self.calls += 1
        if self.fail:
            raise RuntimeError("close failed")


def test_reader_retains_client_and_pointers_until_successful_close(monkeypatch):
    monkeypatch.setattr(ple_bm_reader, "attach_table", lambda *args: torch.tensor([123], dtype=torch.int64))
    client = Client()
    checkpoint = SimpleNamespace(rows=10, width=160, weight_names=frozenset({"one.weight"}))
    reader = ple_bm_reader.PLEBMReader(client, checkpoint, "weight", 2, 8)
    assert reader.ple_external_checkpoint_names == frozenset({"one.weight"})
    client.fail = True
    with pytest.raises(RuntimeError, match="close failed"):
        reader.close_host_offload()
    assert reader._shared_pool_client is client
    assert reader.pointers.tolist() == [123]
    with pytest.raises(RuntimeError, match="closed"):
        reader(torch.tensor([2]))
    client.fail = False
    reader.close_host_offload()
    reader.close_host_offload()
    assert client.calls == 2
    assert reader.pointers is None


def test_failed_attachment_closes_client_and_does_not_expose_names(monkeypatch):
    def fail_attach(*args):
        raise ValueError("attach failed")

    monkeypatch.setattr(ple_bm_reader, "attach_table", fail_attach)
    client = Client()
    checkpoint = SimpleNamespace(rows=10, width=160, weight_names=frozenset({"one.weight"}))
    with pytest.raises(ValueError, match="attach failed"):
        ple_bm_reader.PLEBMReader(client, checkpoint, "weight", 0, 10)
    assert client.calls == 1


def test_cleanup_failure_preserves_original_error_without_add_note(monkeypatch):
    class AttachError(RuntimeError):
        add_note = None

    def fail_attach(*args):
        raise AttachError("attach failed")

    monkeypatch.setattr(ple_bm_reader, "attach_table", fail_attach)
    client = Client()
    client.fail = True
    checkpoint = SimpleNamespace(rows=10, width=160, weight_names=frozenset({"one.weight"}))
    with pytest.raises(AttachError, match="attach failed"):
        ple_bm_reader.PLEBMReader(client, checkpoint, "weight", 0, 10)
    assert client.calls == 1
