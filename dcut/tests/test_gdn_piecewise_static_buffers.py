# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from dcut.gdn_buffers import (  # noqa: E402
    _dcut_get_gdn_piecewise_spec_bufs,
    _dcut_prepare_gdn_eager_state,
    _dcut_prepare_gdn_piecewise_replay,
)
from dcut.globals import _dcut_gdn_static  # noqa: E402


class _GDNMetadata:
    pass


def _make_metadata(
    query_start_loc: list[int],
    state_indices: list[list[int]],
    accepted_tokens: list[int],
) -> _GDNMetadata:
    meta = _GDNMetadata()
    meta.num_spec_decodes = len(state_indices)
    meta.num_prefills = 0
    meta.num_decodes = 0
    meta.spec_sequence_masks = torch.ones(
        len(state_indices), dtype=torch.bool
    )
    meta.spec_state_indices_tensor = torch.tensor(
        state_indices, dtype=torch.int32
    )
    conv_meta = SimpleNamespace(
        query_start_loc=torch.tensor(
            query_start_loc, dtype=torch.int32
        ),
        num_accepted_tokens=torch.tensor(
            accepted_tokens, dtype=torch.int64
        ),
        cache_indices=torch.arange(
            len(state_indices), dtype=torch.int32
        ),
    )
    meta.spec_decode_metadata = SimpleNamespace(spec_causal_conv1d=conv_meta)
    return meta


def test_piecewise_spec_buffers_keep_addresses_and_refresh_values() -> None:
    _dcut_gdn_static.clear()
    meta = _make_metadata(
        [0, 2, 3],
        [[10, 11, 12], [20, 21, 22]],
        [2, 3],
    )
    context = SimpleNamespace(
        model_instance=object(),
        attn_metadata={"layers.0.mixer": meta},
    )

    assert _dcut_prepare_gdn_piecewise_replay(
        context, 8, _GDNMetadata, 4
    )
    bufs = _dcut_get_gdn_piecewise_spec_bufs(
        context, "layers.0.mixer", 8
    )
    pointers = {
        name: tensor.data_ptr() for name, tensor in bufs.items()
    }

    assert set(bufs) == {"qsl", "nat", "ssi"}
    assert bufs["qsl"].tolist() == [0, 2, 3, 3, 3]
    # NAT selects state from the previous verifier step. The second request
    # accepted three tokens previously even though this step has one token, so
    # the accepted count must not be clamped to the current segment length.
    assert bufs["nat"].tolist() == [2, 3, 0, 0]
    assert bufs["ssi"].tolist() == [
        [10, 11, 12],
        [20, 21, 22],
        [-1, -1, -1],
        [-1, -1, -1],
    ]

    # FULL replay clears unused rows explicitly; PIECEWISE keeps the default
    # and preserves its original update path.
    meta.num_spec_decodes = 1
    meta.spec_decode_metadata.spec_causal_conv1d.query_start_loc = (
        torch.tensor([0, 1], dtype=torch.int32)
    )
    meta.spec_decode_metadata.spec_causal_conv1d.num_accepted_tokens = (
        torch.tensor([3], dtype=torch.int64)
    )
    meta.spec_state_indices_tensor = torch.tensor(
        [[30, 31, 32]], dtype=torch.int32
    )

    # Simulate arbitrary values left by a prior FULL capture/replay. The next
    # replay must reconstruct every fixed-capacity input, not merely overwrite
    # values that happen to be active in both batches.
    bufs["qsl"].fill_(99)
    bufs["nat"].fill_(99)
    bufs["ssi"].fill_(99)

    assert _dcut_prepare_gdn_piecewise_replay(
        context,
        8,
        _GDNMetadata,
        4,
        clear_unused_rows=True,
    )
    assert all(
        bufs[name].data_ptr() == pointer
        for name, pointer in pointers.items()
    )
    assert bufs["qsl"].tolist() == [0, 1, 1, 1, 1]
    assert bufs["nat"].tolist() == [3, 1, 1, 1]
    assert bufs["ssi"].tolist() == [
        [30, 31, 32],
        [-1, -1, -1],
        [-1, -1, -1],
        [-1, -1, -1],
    ]


def test_full_fixed_capacity_survives_live_batch_size_changes() -> None:
    _dcut_gdn_static.clear()
    capacity = 96
    meta = _make_metadata(
        list(range(capacity + 1)),
        [[row, row + 1, row + 2] for row in range(capacity)],
        [1] * capacity,
    )
    context = SimpleNamespace(
        model_instance=object(),
        attn_metadata={"layers.0.mixer": meta},
    )

    assert _dcut_prepare_gdn_piecewise_replay(
        context,
        512,
        _GDNMetadata,
        capacity,
        clear_unused_rows=True,
    )
    bufs = _dcut_get_gdn_piecewise_spec_bufs(
        context, "layers.0.mixer", 512
    )
    pointers = {
        name: tensor.data_ptr() for name, tensor in bufs.items()
    }

    for live_batch_size in (32, 64, 1):
        meta.num_spec_decodes = live_batch_size
        meta.spec_decode_metadata.spec_causal_conv1d.query_start_loc = (
            torch.arange(live_batch_size + 1, dtype=torch.int32)
        )
        meta.spec_decode_metadata.spec_causal_conv1d.num_accepted_tokens = (
            torch.ones(live_batch_size, dtype=torch.int64)
        )
        meta.spec_state_indices_tensor = torch.tensor(
            [
                [1000 + row, 2000 + row, 3000 + row]
                for row in range(live_batch_size)
            ],
            dtype=torch.int32,
        )

        assert _dcut_prepare_gdn_piecewise_replay(
            context,
            512,
            _GDNMetadata,
            capacity,
            clear_unused_rows=True,
        )
        assert all(
            bufs[name].data_ptr() == pointer
            for name, pointer in pointers.items()
        )
        assert bufs["qsl"][: live_batch_size + 1].tolist() == list(
            range(live_batch_size + 1)
        )
        assert bufs["qsl"][live_batch_size + 1 :].tolist() == [
            live_batch_size
        ] * (capacity - live_batch_size)
        assert bufs["nat"].tolist() == [1] * capacity
        assert torch.all(bufs["ssi"][live_batch_size:] == -1)


def test_piecewise_batch_buffers_are_shared_across_gdn_layers() -> None:
    _dcut_gdn_static.clear()
    first = _make_metadata(
        [0, 2, 3],
        [[10, 11, 12], [20, 21, 22]],
        [2, 3],
    )
    second = _make_metadata(
        [0, 2, 3],
        [[30, 31, 32], [40, 41, 42]],
        [2, 3],
    )
    context = SimpleNamespace(
        model_instance=object(),
        attn_metadata={
            "layers.0.mixer": first,
            "layers.1.mixer": second,
        },
    )

    assert _dcut_prepare_gdn_piecewise_replay(
        context, 8, _GDNMetadata, 4
    )
    first_bufs = _dcut_get_gdn_piecewise_spec_bufs(
        context, "layers.0.mixer", 8
    )
    second_bufs = _dcut_get_gdn_piecewise_spec_bufs(
        context, "layers.1.mixer", 8
    )

    for name in ("qsl", "nat"):
        assert first_bufs[name].data_ptr() == second_bufs[name].data_ptr()
    assert first_bufs["ssi"].data_ptr() != second_bufs["ssi"].data_ptr()
    assert first_bufs["ssi"][:2].tolist() == [
        [10, 11, 12],
        [20, 21, 22],
    ]
    assert second_bufs["ssi"][:2].tolist() == [
        [30, 31, 32],
        [40, 41, 42],
    ]


def test_piecewise_metadata_group_shares_state_indices_once() -> None:
    _dcut_gdn_static.clear()
    shared = _make_metadata(
        [0, 2, 3],
        [[10, 11, 12], [20, 21, 22]],
        [2, 3],
    )
    context = SimpleNamespace(
        model_instance=object(),
        attn_metadata={
            "layers.0.mixer": shared,
            "layers.1.mixer": shared,
            "layers.2.mixer": shared,
        },
    )

    assert _dcut_prepare_gdn_piecewise_replay(
        context, 8, _GDNMetadata, 4, clear_unused_rows=True
    )
    grouped_bufs = [
        _dcut_get_gdn_piecewise_spec_bufs(context, prefix, 8)
        for prefix in context.attn_metadata
    ]

    for name in ("qsl", "nat", "ssi"):
        assert len({bufs[name].data_ptr() for bufs in grouped_bufs}) == 1
    assert grouped_bufs[0]["ssi"].tolist() == [
        [10, 11, 12],
        [20, 21, 22],
        [-1, -1, -1],
        [-1, -1, -1],
    ]

    # A shrinking replay clears inactive rows and preserves the fixed tensor
    # addresses captured by FULL graphs.
    shared.num_spec_decodes = 1
    shared.spec_decode_metadata.spec_causal_conv1d.query_start_loc = (
        torch.tensor([0, 1], dtype=torch.int32)
    )
    shared.spec_decode_metadata.spec_causal_conv1d.num_accepted_tokens = (
        torch.tensor([4], dtype=torch.int64)
    )
    shared.spec_state_indices_tensor = torch.tensor(
        [[30, 31, 32]], dtype=torch.int32
    )
    pointers = {
        name: grouped_bufs[0][name].data_ptr()
        for name in ("qsl", "nat", "ssi")
    }

    assert _dcut_prepare_gdn_piecewise_replay(
        context, 8, _GDNMetadata, 4, clear_unused_rows=True
    )
    assert all(
        grouped_bufs[0][name].data_ptr() == pointer
        for name, pointer in pointers.items()
    )
    assert grouped_bufs[0]["nat"].tolist() == [4, 1, 1, 1]
    assert grouped_bufs[0]["ssi"].tolist() == [
        [30, 31, 32],
        [-1, -1, -1],
        [-1, -1, -1],
        [-1, -1, -1],
    ]


def test_piecewise_replay_rejects_metadata_group_topology_changes() -> None:
    _dcut_gdn_static.clear()
    shared = _make_metadata([0, 1], [[10, 11, 12]], [1])
    context = SimpleNamespace(
        model_instance=object(),
        attn_metadata={
            "layers.0.mixer": shared,
            "layers.1.mixer": shared,
        },
    )

    assert _dcut_prepare_gdn_piecewise_replay(
        context, 8, _GDNMetadata, 4, clear_unused_rows=True
    )

    # These prefixes were captured with one shared SSI address. Splitting them
    # into two runtime metadata groups must fall back instead of letting either
    # group overwrite the other group's captured input.
    context.attn_metadata["layers.1.mixer"] = _make_metadata(
        [0, 1], [[20, 21, 22]], [1]
    )
    assert not _dcut_prepare_gdn_piecewise_replay(
        context, 8, _GDNMetadata, 4, clear_unused_rows=True
    )


def test_graph_replay_normalizes_only_initial_handoff_state_rows() -> None:
    _dcut_gdn_static.clear()
    meta = _make_metadata(
        [0, 1, 2],
        [[10, 11, 12], [20, 21, 22]],
        [7, 5],
    )
    context = SimpleNamespace(
        model_instance=object(),
        attn_metadata={"layers.0.mixer": meta},
    )

    assert _dcut_prepare_gdn_piecewise_replay(
        context,
        8,
        _GDNMetadata,
        4,
        clear_unused_rows=True,
        initial_spec_rows=(1,),
    )
    bufs = _dcut_get_gdn_piecewise_spec_bufs(
        context, "layers.0.mixer", 8
    )
    assert bufs["nat"].tolist() == [7, 1, 1, 1]

    # The override belongs only to the first consumer step. A subsequent
    # verifier replay must use the accepted-token offsets it actually receives.
    meta.spec_decode_metadata.spec_causal_conv1d.num_accepted_tokens = (
        torch.tensor([4, 3], dtype=torch.int64)
    )
    assert _dcut_prepare_gdn_piecewise_replay(
        context,
        8,
        _GDNMetadata,
        4,
        clear_unused_rows=True,
    )
    assert bufs["nat"].tolist() == [4, 3, 1, 1]


def test_eager_spec_state_is_prepared_once_per_forward() -> None:
    first = _make_metadata(
        [0, 2, 3],
        [[10, 11, 12], [20, 21, 22]],
        [2, 0],
    )
    second = _make_metadata(
        [0, 2, 3],
        [[30, 31, 32], [40, 41, 42]],
        [2, 0],
    )
    context = SimpleNamespace(
        attn_metadata={
            "layers.0.mixer": first,
            "layers.1.mixer": second,
        }
    )

    assert _dcut_prepare_gdn_eager_state(context, _GDNMetadata)
    state = context._dcut_gdn_eager_spec_state
    assert state["num_accepted_tokens"].dtype == torch.int32
    assert state["num_accepted_tokens"].tolist() == [2, 0]
    assert set(state) == {"query_start_loc", "num_accepted_tokens"}
    assert (
        state["query_start_loc"]
        is first.spec_decode_metadata.spec_causal_conv1d.query_start_loc
    )

    assert _dcut_prepare_gdn_eager_state(
        context,
        _GDNMetadata,
        initial_spec_rows=(1,),
    )
    assert context._dcut_gdn_eager_spec_state[
        "num_accepted_tokens"
    ].tolist() == [2, 1]
    original_accepted = (
        first.spec_decode_metadata.spec_causal_conv1d.num_accepted_tokens
    )
    assert original_accepted.tolist() == [2, 0]
    first.spec_sequence_masks = None
    second.spec_sequence_masks = None
    assert not _dcut_prepare_gdn_eager_state(
        context, _GDNMetadata
    )
    assert context._dcut_gdn_eager_spec_state is None


def test_piecewise_replay_rejects_non_spec_and_mixed_batches() -> None:
    _dcut_gdn_static.clear()
    meta = _make_metadata([0, 1], [[10, 11, 12]], [1])
    context = SimpleNamespace(
        model_instance=object(),
        attn_metadata={"layers.0.mixer": meta},
    )

    meta.spec_sequence_masks = None
    assert not _dcut_prepare_gdn_piecewise_replay(
        context, 8, _GDNMetadata, 4
    )

    meta.spec_sequence_masks = torch.ones(1, dtype=torch.bool)
    meta.num_prefills = 1
    assert not _dcut_prepare_gdn_piecewise_replay(
        context, 8, _GDNMetadata, 4
    )

    meta.num_prefills = 0
    meta.num_decodes = 1
    assert not _dcut_prepare_gdn_piecewise_replay(
        context, 8, _GDNMetadata, 4
    )


def test_piecewise_buffers_do_not_alias_model_instances() -> None:
    _dcut_gdn_static.clear()
    meta = _make_metadata([0, 1], [[10, 11]], [1])
    first = SimpleNamespace(
        model_instance=object(),
        attn_metadata={"layers.0.mixer": meta},
    )
    second = SimpleNamespace(
        model_instance=object(),
        attn_metadata={"layers.0.mixer": meta},
    )

    assert _dcut_prepare_gdn_piecewise_replay(
        first, 4, _GDNMetadata, 2
    )
    assert _dcut_prepare_gdn_piecewise_replay(
        second, 4, _GDNMetadata, 2
    )
    first_bufs = _dcut_get_gdn_piecewise_spec_bufs(
        first, "layers.0.mixer", 4
    )
    second_bufs = _dcut_get_gdn_piecewise_spec_bufs(
        second, "layers.0.mixer", 4
    )

    assert first_bufs["qsl"].data_ptr() != second_bufs["qsl"].data_ptr()
