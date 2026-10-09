# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the reviewer-fixed VPP async-send paths.

Covers two Gemini code-review findings on PR #18018:

* ``handles[2:]`` was hard-coded to split gloo-metadata from hccl-tensor
  handles in ``_release_completed_send_buff``; if ``_isend_object`` ever
  returns fewer than 2 handles the slice mis-identifies the tensors and the
  batch is dropped too early. ``_async_send_buff`` now stores an explicit
  ``num_metadata`` count.
* ``send_tensor_dict_cpu`` sent ``tensor.detach().cpu()`` to Gloo ``isend``
  without ``.contiguous()``; a strided activation silently corrupted (or
  crashed) the transfer. The CPU copy is now made contiguous.

The class under test is constructed via ``object.__new__`` (bypassing the
real ``GroupCoordinator.__init__``) and only the attributes touched by each
method are populated, so no distributed runtime is required.
"""

from collections import deque

import torch

from vllm_ascend.patch.worker.patch_distributed import GroupCoordinatorPatch


class _FakeWork:
    """Minimal stand-in for ``torch.distributed.Work``."""

    def __init__(self, completed: bool = True):
        self._completed = completed
        self.waited = False

    def is_completed(self) -> bool:
        return self._completed

    def wait(self):
        self.waited = True


def _make_empty_patch() -> GroupCoordinatorPatch:
    obj = object.__new__(GroupCoordinatorPatch)
    obj._async_send_buff = deque()
    return obj


class TestReleaseCompletedSendBuffNumMetadata:
    """The async-send release loop must slice on num_metadata, not a hard Rate.

    Regression for the ``handles[2:]`` hard-coded-index finding. With a single
    metadata handle plus two hccl tensor handles (num_metadata == 1), the old
    code treated ``handles[2:]`` as the tensor gate and would have popped the
    entry while the first tensor send was still incomplete.
    """

    def test_keeps_buff_when_first_tensor_handle_incomplete(self):
        patch = _make_empty_patch()
        # handles = [metadata, tensor0(incomplete), tensor1(complete)]
        # num_metadata == 1 -> tensor gate is handles[1:], tensor0 blocks.
        meta = _FakeWork(completed=True)
        t0 = _FakeWork(completed=False)
        t1 = _FakeWork(completed=True)
        patch._async_send_buff.append(([meta, t0, t1], 1, [object()]))

        patch._release_completed_send_buff()

        # must NOT have been popped (first tensor send still in flight)
        assert len(patch._async_send_buff) == 1
        assert not meta.waited

    def test_pops_and_waits_metadata_when_all_tensors_complete(self):
        patch = _make_empty_patch()
        meta = _FakeWork(completed=True)
        t_hccl = _FakeWork(completed=True)
        patch._async_send_buff.append(([meta, t_hccl], 1, [object(), object()]))

        patch._release_completed_send_buff()

        assert len(patch._async_send_buff) == 0
        assert meta.waited  # gloo metadata waited as backstop before dropping

    def test_metadata_only_entry_is_dropped_without_wait(self):
        patch = _make_empty_patch()
        meta = _FakeWork(completed=True)
        # num_metadata == len(handles): no hccl tensor handles at all.
        patch._async_send_buff.append(([meta], 1, [object()]))

        patch._release_completed_send_buff()

        assert len(patch._async_send_buff) == 0
        # gloo metadata not waited (would risk fold-point deadlock)
        assert not meta.waited

    def test_wait_send_buff_unpacks_three_tuple(self):
        patch = _make_empty_patch()
        h = _FakeWork(completed=True)
        patch._async_send_buff.append(([h, h], 1, [object(), object()]))

        patch._wait_send_buff()

        assert len(patch._async_send_buff) == 0


class TestSendTensorDictCpuContiguous:
    """CPU-Gloo fold sends must carry a contiguous CPU copy."""

    def _send_cpu(self, tensor_dict, monkeypatch):
        patch = _make_empty_patch()
        patch.world_size = 2
        patch.rank_in_group = 0
        patch.ranks = [0, 1]
        patch.cpu_group = object()

        meta_handles = [_FakeWork(), _FakeWork()]
        monkeypatch.setattr(
            patch, "_isend_object",
            lambda obj, dst: (list(meta_handles), [_FakeWork(), object()]),
        )

        sent: list[torch.Tensor] = []

        def fake_isend(tensor, dst=None, group=None):
            sent.append(tensor)
            return _FakeWork()

        monkeypatch.setattr("torch.distributed.isend", fake_isend)
        patch.send_tensor_dict_cpu(tensor_dict, dst=1)
        return patch, sent, len(meta_handles)

    def test_cpu_copy_is_contiguous_for_strided_activation(self, monkeypatch):
        # A strided slice: shape (4,3) sliced to every-other row is non-contig.
        src = torch.arange(24, dtype=torch.float32).reshape(4, 6)[:, ::2]
        assert not src.is_contiguous()

        patch, sent, num_meta = self._send_cpu({"x": src}, monkeypatch)

        assert len(sent) == 1
        t_cpu = sent[0]
        assert t_cpu.is_contiguous()
        assert t_cpu.device.type == "cpu"
        assert t_cpu.data_ptr() != src.data_ptr()  # detached copy, new buffer
        # Values preserved by contiguous copy.
        assert torch.equal(t_cpu, src)
        # _async_send_buff now stores the explicit num_metadata count.
        handles, stored_num_meta, retained = patch._async_send_buff[0]
        assert stored_num_meta == num_meta
        assert len(retained) == num_meta + len(sent)

    def test_num_metadata_recorded_matches_metadata_handles(self, monkeypatch):
        # Even when a metadata send yields a different handle count, the
        # stored num_metadata must reflect it (not a hard-coded 2).
        patch = _make_empty_patch()
        patch.world_size = 2
        patch.rank_in_group = 0
        patch.ranks = [0, 1]
        patch.cpu_group = object()
        monkeypatch.setattr(
            patch, "_isend_object",
            lambda obj, dst: ([_FakeWork()], [object()]),  # exactly 1 metadata handle
        )
        monkeypatch.setattr(
            "torch.distributed.isend",
            lambda tensor, dst=None, group=None: _FakeWork(),
        )

        patch.send_tensor_dict_cpu({"x": torch.ones(4)}, dst=1)

        handles, num_metadata, _ = patch._async_send_buff[0]
        assert num_metadata == 1
        # hccl tensor gate = handles[num_metadata:], i.e. the tensor handle only.
        assert len(handles[num_metadata:]) == 1