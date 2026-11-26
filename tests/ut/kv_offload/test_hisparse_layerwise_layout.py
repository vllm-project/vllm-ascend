# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU byte references for public HiSparse page-copy descriptors."""

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.hisparse_layout import (
    PageBuffer,
    check_page_range,
    indexer_spans,
    main_components,
    main_spans,
    page_offsets,
)


def _apply_spans(spans, sources, destinations):
    """Resolve fake addresses without dereferencing memory or using a device."""

    def resolve(address, size, buffers):
        for base, buffer in buffers:
            offset = address - base
            if offset >= 0 and offset + size <= len(buffer):
                return buffer, offset
        pytest.fail(f"Copy outside reference buffers: address={address}, size={size}")

    for destination, source, size in spans:
        assert size > 0
        source_buffer, source_offset = resolve(source, size, sources)
        destination_buffer, destination_offset = resolve(destination, size, destinations)
        destination_buffer[destination_offset : destination_offset + size] = source_buffer[
            source_offset : source_offset + size
        ]


@pytest.mark.parametrize("padding", [0, 3])
@pytest.mark.parametrize("source_page", [0, 1])
def test_packed_component_major_pages(padding, source_page):
    page_tokens = 3
    latent_bytes, rope_bytes = 6, 3
    payload = latent_bytes + rope_bytes
    source_base, destination_base = 1000, 2000
    slot_size = payload + padding
    source = bytearray(b"abcdefXYZ" + b"!" * padding + b"ghijklUVW" + b"!" * padding)
    destination = bytearray(b"?" * 36)
    host = PageBuffer(destination_base, payload=payload, stride=12, pages=3)

    bases, sizes, strides = main_components([source_base], [slot_size], [(slot_size,)], latent_bytes, rope_bytes)
    assert bases == [source_base, source_base + latent_bytes]
    assert sizes == [latent_bytes, rope_bytes]
    assert strides == [slot_size, slot_size]
    spans = list(
        main_spans(
            host,
            bases,
            sizes,
            page_tokens,
            (2, source_page),
            page_tokens,
            source_strides=strides,
        )
    )
    assert len(spans) == 2 * page_tokens
    _apply_spans(spans, [(source_base, source)], [(destination_base, destination)])

    expected = bytearray(b"?" * len(destination))
    expected[24:33] = b"abXcdYefZ" if source_page == 0 else b"ghUijVklW"
    assert destination == expected


def test_split_components_one_token_final_page():
    page_tokens = 3
    latent_base, rope_base, destination_base = 1000, 2000, 3000
    latent = bytearray(b"abcdefghijkl")
    rope = bytearray(b"XYZUVW")
    destination = bytearray(b"?" * 24)
    host = PageBuffer(destination_base, payload=9, stride=12, pages=2)
    bases, sizes, strides = main_components([latent_base, rope_base], [6, 3], [(2, 3, 2), (2, 3, 1)], 6, 3)
    assert bases == [latent_base, rope_base]
    assert sizes == [6, 3]
    assert strides == [6, 3]

    for page, valid_tokens in [(0, page_tokens), (1, 1)]:
        spans = list(
            main_spans(
                host,
                bases,
                sizes,
                page_tokens,
                (page, page),
                valid_tokens,
                source_strides=strides,
            )
        )
        assert len(spans) == 2 * valid_tokens
        _apply_spans(
            spans,
            [(latent_base, latent), (rope_base, rope)],
            [(destination_base, destination)],
        )

    assert destination == bytearray(b"abXcdYefZ???ghU?????????")


@pytest.mark.parametrize("valid_tokens, expected_rows", [(3, b"ghUijVklW"), (1, b"ghU")])
def test_row_major_pages(valid_tokens, expected_rows):
    source_base, destination_base = 1000, 2000
    source = bytearray(b"abXcdYefZghUijVklW")
    destination = bytearray(b"?" * 24)
    host = PageBuffer(destination_base, payload=9, stride=12, pages=2)
    bases, sizes, strides = main_components([source_base], [9], [(2, 3, 3)], 6, 3)
    assert (bases, sizes, strides) == ([source_base], [9], [9])

    spans = list(main_spans(host, bases, sizes, 3, (0, 1), valid_tokens))
    assert spans == [(destination_base, source_base + 9, len(expected_rows))]
    _apply_spans(spans, [(source_base, source)], [(destination_base, destination)])
    expected = bytearray(b"?" * len(destination))
    expected[: len(expected_rows)] = expected_rows
    assert destination == expected


@pytest.mark.parametrize("valid_tokens", [1, 3])
def test_indexer_and_scale_use_independent_destination_strides(valid_tokens):
    indexer_base, scale_base = 1000, 2000
    destination_indexer_base, destination_scale_base = 3000, 4000
    indexer = bytearray(b"abcdefghijkl")
    scale = bytearray(b"XYZUVW")
    destination_indexer = bytearray(b"?" * 27)
    destination_scale = bytearray(b"?" * 15)
    destinations = [
        PageBuffer(destination_indexer_base, payload=6, stride=9, pages=3),
        PageBuffer(destination_scale_base, payload=3, stride=5, pages=3),
    ]

    spans = list(indexer_spans(destinations, [indexer_base, scale_base], [6, 3], 3, (2, 1), valid_tokens))
    assert spans == [
        (destination_indexer_base + 18, indexer_base + 6, 2 * valid_tokens),
        (destination_scale_base + 10, scale_base + 3, valid_tokens),
    ]
    _apply_spans(
        spans,
        [(indexer_base, indexer), (scale_base, scale)],
        [(destination_indexer_base, destination_indexer), (destination_scale_base, destination_scale)],
    )
    expected_indexer = bytearray(b"?" * len(destination_indexer))
    expected_scale = bytearray(b"?" * len(destination_scale))
    expected_indexer[18 : 18 + 2 * valid_tokens] = b"ghijkl"[: 2 * valid_tokens]
    expected_scale[10 : 10 + valid_tokens] = b"UVW"[:valid_tokens]
    assert destination_indexer == expected_indexer
    assert destination_scale == expected_scale


def test_page_buffer_addresses_include_stride_not_payload():
    buffer = PageBuffer(base=1000, payload=6, stride=9, pages=3)
    assert [buffer.address(page) for page in range(3)] == [1000, 1009, 1018]


@pytest.mark.parametrize("page", [-1, 3])
def test_page_buffer_rejects_unregistered_page_ids(page):
    buffer = PageBuffer(base=1000, payload=6, stride=9, pages=3)
    with pytest.raises(ValueError, match="outside the registered"):
        buffer.address(page)


@pytest.mark.parametrize("page_ids", [(-1, 0), (2, 0), (0, -1)])
def test_main_rejects_invalid_page_ids(page_ids):
    host = PageBuffer(base=2000, payload=9, stride=12, pages=2)
    with pytest.raises(ValueError, match="page index|page ID"):
        list(main_spans(host, [1000, 1100], [6, 3], 3, page_ids, 1))


def test_main_rejects_destination_payload_mismatch():
    host = PageBuffer(base=2000, payload=8, stride=12, pages=2)
    with pytest.raises(ValueError, match="Main KV sizes"):
        list(main_spans(host, [1000, 1100], [6, 3], 3, (0, 0), 1))


@pytest.mark.parametrize("page_ids", [(-1, 0), (2, 0), (0, -1)])
def test_indexer_rejects_invalid_page_ids(page_ids):
    destinations = [PageBuffer(3000, 6, 9, 2), PageBuffer(4000, 3, 5, 2)]
    with pytest.raises(ValueError, match="page index|source Indexer page"):
        list(indexer_spans(destinations, [1000, 2000], [6, 3], 3, page_ids, 1))


@pytest.mark.parametrize("payloads", [(5, 3), (6, 2)])
def test_indexer_rejects_either_destination_payload_mismatch(payloads):
    destinations = [PageBuffer(3000, payloads[0], 9, 2), PageBuffer(4000, payloads[1], 5, 2)]
    with pytest.raises(ValueError, match="Indexer or scale page bytes disagree"):
        list(indexer_spans(destinations, [1000, 2000], [6, 3], 3, (0, 0), 1))


@pytest.mark.parametrize("latent_bytes, rope_bytes", [(0, 3), (6, 0), (-1, 3), (6, -1), (6, 4)])
def test_packed_components_reject_invalid_payload(latent_bytes, rope_bytes):
    with pytest.raises(ValueError, match="Packed Prefill MLA slot"):
        main_components([1000], [9], [(9,)], latent_bytes, rope_bytes)


def test_native_c8_rows_copy_rope_and_scales_on_a_partial_page():
    tokens, row = 128, 656
    source = bytearray((b"x" * 512 + b"r" * 128 + b"s" * 16) * tokens)
    dest = bytearray(b"?" * len(source))
    host = PageBuffer(1000000, len(dest), len(dest), 1)
    bases, sizes, strides = main_components([4096], [len(source)], [(tokens, 1, row)], tokens * 512, tokens * 64)
    _apply_spans(
        main_spans(host, bases, sizes, tokens, (0, 0), 3, source_strides=strides), [(4096, source)], [(host.base, dest)]
    )
    assert dest[: 3 * row] == source[: 3 * row]
    assert dest[3 * row :] == b"?" * ((tokens - 3) * row)


@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_chunked_shared_main_has_one_writer_per_page(width):
    for rank in range(width):
        whole = list(page_offsets(3, 14, rank, width))
        split = list(page_offsets(3, 4, rank, width)) + list(page_offsets(7, 10, rank, width))
        assert whole == split == [page for page in range(3, 17) if page % width == rank]


@pytest.mark.parametrize(
    "source_ids,start,dest_count,source_count,computed",
    [
        ([1], -1, 5, 2, 32),
        ([1], 1, 1, 2, 32),
        ([2], 0, 2, 2, 32),
        ([-1], 0, 2, 2, 32),
        ([False], 0, 2, 2, 32),
        ([1], 1, 2, 2, 16),
    ],
)
def test_invalid_source_allocation_and_watermark(source_ids, start, dest_count, source_count, computed):
    with pytest.raises(ValueError):
        check_page_range(source_ids, start, dest_count, source_count, 16, computed)


def test_last_valid_row_and_empty_chunk():
    check_page_range([0], 1, 2, 1, 16, 17)
    check_page_range([], 2, 2, 1, 16, 17)
