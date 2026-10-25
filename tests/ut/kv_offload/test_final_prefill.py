# SPDX-License-Identifier: Apache-2.0
import contextlib
import ctypes
import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.quantization.methods.kv_cache.turboquant.latent import pack_bf16_kv, unpack_c8_kv

from vllm_ascend.distributed.kv_transfer.kv_p2p.final_prefill import (
    C8_TO_TQ4,
    TQ_ABI,
    FinalPrefillExporter,
    describe_layout,
    final_prefill_mode,
    pack_c8_kv,
    validate_remote_layout,
)


def metadata(tensors, *, mode=C8_TO_TQ4, logical_block=4, group=0, indexer=False, tq=False):
    meta = SimpleNamespace(
        tensor_group_idx=[group] * len(tensors),
        kv_caches_base_addr=[t.data_ptr() for t in tensors],
        block_len=[math.prod(t.shape[1:]) * t.element_size() for t in tensors],
        block_size_scale=[logical_block // t.shape[1] for t in tensors],
    )
    cls = type("AscendSFAIndexerCacheSpec" if indexer else "AscendMLAAttentionSpec", (), {})
    spec = cls()
    spec.block_size, spec.cache_sparse_sfa_turboquant = logical_block, tq
    spec.cache_sparse_sfa_c8 = not indexer and not tq and len(tensors) == 1
    describe_layout(meta, tensors, spec, mode, not tq)
    return meta


class CopyEngine:

    def __init__(self):
        self.calls = []

    def batch_transfer_sync_write(self, session, src, dst, lengths):
        self.calls.append((session, src, dst, lengths))
        for s, d, size in zip(src, dst, lengths):
            ctypes.memmove(d, s, size)
        return 0


def padded(shape, dtype, pad=32):
    elements = math.prod(shape[1:])
    storage = torch.full((shape[0], elements + pad), 7, dtype=dtype)
    view = storage[:, :elements].view(shape)
    return view, storage


def c8_cache(nope, rope):
    return torch.cat((nope.to(torch.int8), rope.contiguous().view(torch.int8),
                      torch.ones((*nope.shape[:-1], 4), dtype=torch.float32).view(torch.int8)), -1)


def fake_pack(packed, quantizer):
    nope, rope = unpack_c8_kv(packed)
    result = torch.zeros((nope.shape[0], 386), dtype=torch.uint8)
    result[:, :256] = nope[:, :256].to(torch.uint8)
    result[:, 256:384] = rope.contiguous().view(torch.uint8)
    result[:, 384:] = torch.ones((nope.shape[0], 1), dtype=torch.float16).view(torch.uint8)
    return result


@pytest.mark.parametrize("incremental", [False, True])
def test_export_scattered_pages_partial_tail_and_shared_prefix_unchanged(incremental):
    mode = C8_TO_TQ4
    nope, ns = padded((8, 2, 1, 512), torch.bfloat16)
    rope, rs = padded((8, 2, 1, 64), torch.bfloat16)
    nope.copy_(torch.arange(16).reshape(8, 2, 1, 1))
    rope.copy_(torch.arange(16).reshape(8, 2, 1, 1) / 10)
    indexer, ins = padded((8, 2, 1, 8), torch.int8)
    indexer.copy_(torch.arange(16).reshape(8, 2, 1, 1))
    before = [t.clone() for t in (ns, rs, ins)]
    packed, ps = padded((4, 4, 1, 386), torch.uint8)
    dest_main, dest_storage = (packed, ), [ps]
    active, active_storage = padded((8, 2, 1, 656), torch.int8)
    active.copy_(c8_cache(nope, rope))
    active_before = active_storage.clone()
    di, dis = padded(indexer.shape, indexer.dtype)
    caches = {"main": (active,), "indexer": (indexer, )}
    local = {
        "main": metadata(caches["main"], mode=mode),
        "indexer": metadata(caches["indexer"], mode=mode, group=1, indexer=True),
    }
    remote = {
        "main": metadata(dest_main, mode=mode, tq=mode == C8_TO_TQ4),
        "indexer": metadata((di, ), mode=mode, group=1, indexer=True),
    }
    request = SimpleNamespace(
        chunk_finish=not incremental,
        remote_host="localhost",
        remote_te_rpc_port=100,
        # Transfer two non-contiguous pages, starting beyond a D cache hit.
        local_block_ids=[[2, 0], [2, 0]],
        remote_block_ids=[[1, 3], [1, 3]],
        remote_layer_metadata=remote,
    )
    exporter = FinalPrefillExporter.__new__(FinalPrefillExporter)
    exporter.caches, exporter.metadata, exporter.mode = caches, local, mode
    exporter.device, exporter.stream = torch.device("cpu"), MagicMock()
    exporter.ready = MagicMock()
    exporter.quantizer, exporter.chunk_tokens = None, 4
    exporter.staging = torch.empty((4, 386), dtype=torch.uint8)
    exporter.engine = CopyEngine()
    exporter.incremental = incremental
    exporter.staging_buffers = [exporter.staging, torch.empty_like(exporter.staging)]
    exporter.quantizer = MagicMock()

    task = SimpleNamespace(wait_event=MagicMock(), send_request={"r": request}, layer_names=list(caches))
    with patch("torch.npu.Event"), patch("torch.npu.stream", side_effect=lambda _: contextlib.nullcontext()), patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.final_prefill.pack_c8_kv", side_effect=fake_pack) as pack:
        exporter.transfer(task)
    assert pack.call_count == 2
    torch.testing.assert_close(active_storage, active_before)
    for source, target in ((2, 1), (0, 3)):
        expected = fake_pack(active[source * 2:source * 2 + 2].reshape(4, 656), None)
        torch.testing.assert_close(dest_main[0][target].reshape(4, 386), expected)
    for source, target in ((2, 1), (0, 3)):
        torch.testing.assert_close(di[target * 2:target * 2 + 2], indexer[source * 2:source * 2 + 2])
    for original, current in zip(before, (ns, rs, ins)):
        torch.testing.assert_close(current, original)
    for storage in (*dest_storage, dis):
        assert torch.all(storage[:, -32:] == 7)  # Never copy source padding.
    assert torch.all(dest_main[0][0] == 7)  # D's cached prefix remains intact.
    if mode == C8_TO_TQ4:
        assert max(sum(call[3]) for call in exporter.engine.calls[:2]) <= 4 * 386


def test_packer_keeps_scale_and_predivides_rope_without_normalizing_again():
    nope = torch.randn(3, 512, dtype=torch.bfloat16)
    rope = torch.ones(3, 64, dtype=torch.bfloat16) * 4
    codes = torch.arange(256, dtype=torch.uint8).expand(3, -1)
    scales = torch.tensor([[2.], [4.], [8.]], dtype=torch.float16)
    quantizer = MagicMock()
    quantizer.compress.return_value = torch.cat((codes, scales.view(torch.uint8)), -1).unsqueeze(1)
    packed = pack_bf16_kv(nope, rope, quantizer)
    torch.testing.assert_close(quantizer.compress.call_args.args[0], nope)
    torch.testing.assert_close(packed[:, :256], codes)
    torch.testing.assert_close(packed[:, 384:].contiguous().view(torch.float16), scales)
    torch.testing.assert_close(packed[:, 256:384].contiguous().view(torch.bfloat16), (rope / scales).bfloat16())


@pytest.mark.parametrize("field,value", [
    ("export_mode", None),
    ("cache_format", "c8"),
    ("logical_block_size", 8),
    ("component_shapes", [[4, 1, 644]]),
    ("tensor_group_idx", [1]),
])
def test_reject_incompatible_remote(field, value):
    local = metadata((torch.empty(4, 2, 1, 656, dtype=torch.int8),))
    remote = metadata((torch.empty(2, 4, 1, 386, dtype=torch.uint8), ), tq=True)
    assert remote.cache_format == TQ_ABI
    validate_remote_layout(local, remote)
    setattr(remote, field, value)
    with pytest.raises(ValueError):
        validate_remote_layout(local, remote)


def test_export_write_failure_is_not_silently_successful():
    exporter = FinalPrefillExporter.__new__(FinalPrefillExporter)
    exporter.engine = MagicMock()
    exporter.engine.batch_transfer_sync_write.return_value = -1
    with pytest.raises(RuntimeError, match="transfer failed"):
        exporter._write("remote", [1], [2], [3])


@pytest.mark.parametrize("producer,mode,dtype", [(True, C8_TO_TQ4, "int8"),
                                                 (False, C8_TO_TQ4, "turboquant_4bit_nc")])
def test_role_config_and_reject_context_parallel(producer, mode, dtype):
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={"final_prefill_export": mode},
                                           kv_connector="MooncakeLayerwiseConnector",
                                           is_kv_producer=producer,
                                           is_kv_consumer=not producer),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1,
                                        decode_context_parallel_size=1,
                                        pipeline_parallel_size=1),
        cache_config=SimpleNamespace(cache_dtype=dtype),
        model_config=SimpleNamespace(dtype=torch.bfloat16, enforce_eager=True),
        speculative_config=None,
    )
    assert final_prefill_mode(config) == mode
    from vllm.config.compilation import CUDAGraphMode

    config.model_config.enforce_eager = False
    config.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)
    if producer:
        with pytest.raises(ValueError, match="requires eager P"):
            final_prefill_mode(config)
    else:
        assert final_prefill_mode(config) == mode
        config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
        with pytest.raises(ValueError, match="requires eager P"):
            final_prefill_mode(config)
    config.model_config.enforce_eager = True
    config.parallel_config.decode_context_parallel_size = 2
    with pytest.raises(ValueError, match="PCP=DCP"):
        final_prefill_mode(config)


def test_staging_is_aligned_and_bounded_and_registration_failure_raises():
    caches = {
        'main': (torch.empty(4, 128, 1, 656, dtype=torch.int8),)
    }
    meta = {'main': metadata(caches['main'], logical_block=256)}
    engine = MagicMock()
    engine.register_memory.return_value = 0
    with patch('torch.npu.Stream'), patch('torch.npu.Event'), patch(
            'vllm_ascend.distributed.kv_transfer.kv_p2p.final_prefill.pack_c8_kv', side_effect=fake_pack) as pack:
        exporter = FinalPrefillExporter(caches, meta, C8_TO_TQ4, 1024, engine)
        assert exporter.staging.data_ptr() % (2 * 1024**2) == 0
        assert exporter.staging.numel() == 1024 * 386
        assert exporter._staging_storage.numel() == 1024 * 386 + 2 * 1024**2
        engine.register_memory.assert_called_once_with(exporter.staging.data_ptr(), 1024 * 386)
        pack.assert_called_once()
        assert pack.call_args.args[0].shape == (256, 656)
        with pytest.raises(ValueError, match='at least one'):
            FinalPrefillExporter(caches, meta, C8_TO_TQ4, 128, engine)
        engine.register_memory.return_value = -1
        with pytest.raises(RuntimeError, match='registration failed'):
            FinalPrefillExporter(caches, meta, C8_TO_TQ4, 1024, engine)
        pack.assert_called_once()


def test_incremental_export_drains_queued_conversion_on_failure():
    exporter = FinalPrefillExporter.__new__(FinalPrefillExporter)
    exporter.incremental = True
    exporter.stream = MagicMock()
    with patch.object(exporter, "_pipeline_convert_and_write", side_effect=RuntimeError("failed")):
        with pytest.raises(RuntimeError, match="failed"):
            exporter._convert_and_write(None, None, None, [], [], "remote")
    exporter.stream.synchronize.assert_called_once()


def test_incremental_double_buffer_reuse_waits_for_previous_transfer():
    exporter = FinalPrefillExporter.__new__(FinalPrefillExporter)
    exporter.incremental = True
    exporter.device = torch.device("cpu")
    exporter.stream = MagicMock()
    exporter.chunk_tokens = 4
    exporter.staging_buffers = [torch.empty(4, 386, dtype=torch.uint8) for _ in range(2)]
    nope = torch.arange(16, dtype=torch.bfloat16).view(8, 2, 1, 1).expand(-1, -1, -1, 512).contiguous()
    rope = torch.zeros(8, 2, 1, 64, dtype=torch.bfloat16)
    destination = torch.zeros(4, 4, 1, 386, dtype=torch.uint8)
    active = c8_cache(nope, rope)
    exporter.caches = {"main": (active,)}
    local = metadata((active,))
    remote = metadata((destination,), tq=True)
    order = []
    exporter.quantizer = MagicMock()

    def store(packed, quantizer):
        order.append("convert")
        return fake_pack(packed, quantizer)
    engine = CopyEngine()

    def write(*args):
        order.append("send")
        return engine.batch_transfer_sync_write(*args)

    exporter.engine = SimpleNamespace(batch_transfer_sync_write=write)
    with patch("torch.npu.Event"), patch("torch.npu.stream", side_effect=lambda _: contextlib.nullcontext()), patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.final_prefill.pack_c8_kv", side_effect=store):
        exporter._convert_and_write("main", local, remote, [3, 1, 2, 0], [0, 1, 2, 3], "remote")
    assert order == ["convert", "convert", "send", "convert", "send", "convert", "send", "send"]
    for target, source in enumerate([3, 1, 2, 0]):
        expected = fake_pack(active[source * 2:source * 2 + 2].reshape(4, 656), None)
        torch.testing.assert_close(destination[target].reshape(4, 386), expected)


@pytest.mark.parametrize("pp", [1, 2, 3, 4])
@pytest.mark.parametrize("mtp", [False, True])
def test_bulk_export_accepts_configured_pipeline_and_draft(pp, mtp):
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="MooncakeConnectorV1",
            kv_connector_extra_config={"final_prefill_export": C8_TO_TQ4},
            is_kv_producer=True, is_kv_consumer=False,
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, decode_context_parallel_size=1,
            pipeline_parallel_size=pp,
        ),
        cache_config=SimpleNamespace(cache_dtype="int8"),
        model_config=SimpleNamespace(dtype=torch.bfloat16, enforce_eager=True),
        speculative_config=SimpleNamespace(num_speculative_tokens=3) if mtp else None,
    )
    assert final_prefill_mode(config) == C8_TO_TQ4
    if pp != 1 or mtp:
        config.kv_transfer_config.kv_connector = "MooncakeLayerwiseConnector"
        with pytest.raises(ValueError, match="requires MooncakeConnectorV1"):
            final_prefill_mode(config)


def test_bulk_export_preserves_unquantized_layers_with_different_group_ids():
    source = (torch.randn(8, 2, 1, 512, dtype=torch.bfloat16),
              torch.randn(8, 2, 1, 64, dtype=torch.bfloat16))
    destination = tuple(torch.full_like(t, 7) for t in source)
    local = metadata(source, group=0)
    remote = metadata(destination, group=2)
    exporter = FinalPrefillExporter.__new__(FinalPrefillExporter)
    exporter.metadata = {"mtp.attn": local}
    exporter.caches = {"mtp.attn": source}
    exporter.mode = C8_TO_TQ4
    exporter.ready = MagicMock()
    exporter.stream = MagicMock()
    exporter.engine = CopyEngine()
    original = tuple(t.clone() for t in source)
    with patch.object(exporter, "_convert_and_write") as convert:
        exporter.export_layers({"mtp.attn": (remote, [2, 0], [1, 3])}, "destination")
    convert.assert_not_called()
    for src, dst, before in zip(source, destination, original):
        torch.testing.assert_close(src, before)
        torch.testing.assert_close(dst[2:4], src[4:6])
        torch.testing.assert_close(dst[6:8], src[0:2])
        assert torch.all(dst[:2] == 7)
    exporter.stream.synchronize.assert_called_once()


def make_bulk_exporter():
    exporter = FinalPrefillExporter.__new__(FinalPrefillExporter)
    exporter.incremental = True
    exporter.device = torch.device("cpu")
    exporter.stream, exporter.ready = MagicMock(), MagicMock()
    exporter.chunk_tokens = 8
    exporter.staging_buffers = [torch.empty(8, 386, dtype=torch.uint8) for _ in range(2)]
    exporter.quantizer = MagicMock()
    exporter.engine = CopyEngine()
    exporter.caches, exporter.metadata = {}, {}
    layers, destinations = {}, {}
    # Different physical page scales/group IDs, with equal and unequal maps.
    for name, page_tokens, group, ids in [("a", 2, 0, [3, 1, 2]), ("b", 2, 0, [3, 1, 2]),
                                         ("draft", 4, 2, [2, 0, 3]), ("empty", 2, 3, [])]:
        nope = torch.arange(16, dtype=torch.bfloat16).view(-1, 1).expand(-1, 512)
        rope = torch.ones(16, 64, dtype=torch.bfloat16)
        active = c8_cache(nope, rope).view(16 // page_tokens, page_tokens, 1, 656)
        dest = torch.full((4, 4, 1, 386), 7, dtype=torch.uint8)
        exporter.caches[name] = (active,)
        exporter.metadata[name] = metadata((active,), group=group)
        remote = metadata((dest,), group=group + 4, tq=True)
        layers[name] = (remote, ids, [0, 2, 3][:len(ids)])
        destinations[name] = dest
    return exporter, layers, destinations


@pytest.mark.parametrize("incremental", [False, True])
def test_bulk_indices_reused_only_within_request_and_cross_layer_bytes_match(incremental):
    exporter, layers, destinations = make_bulk_exporter()
    exporter.incremental = incremental
    exporter.staging = exporter.staging_buffers[0]
    originals = {name: tensors[0].clone() for name, tensors in exporter.caches.items()}
    order = []

    def pack(packed, quantizer):
        order.append("convert")
        return fake_pack(packed, quantizer)

    original_write = exporter.engine.batch_transfer_sync_write

    def write(*args):
        order.append("send")
        return original_write(*args)

    exporter.engine.batch_transfer_sync_write = write
    with patch("torch.npu.Event"), patch("torch.npu.stream", side_effect=lambda _: contextlib.nullcontext()), patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.final_prefill.pack_c8_kv", side_effect=pack), patch(
            "torch.tensor", wraps=torch.tensor) as upload:
        exporter.export_layers(layers, "first")
        first_uploads = upload.call_count
        # a/b share a page map; draft has a different map/scale; empty uploads nothing.
        assert first_uploads == 2
        exporter.export_layers(layers, "second")
        assert upload.call_count == 2 * first_uploads
    if incremental:
        # The next layer's first conversion precedes the previous layer's last
        # transfer, and the last partial chunk is sent before returning.
        assert order[:12] == ["convert", "convert", "send", "convert", "send", "convert",
                              "send", "convert", "send", "convert", "send", "send"]
    for name, (remote, source, target) in layers.items():
        active = exporter.caches[name][0]
        scale = exporter.metadata[name].block_size_scale[0]
        for src, dst in zip(source, target):
            expected = fake_pack(active[src * scale:(src + 1) * scale].reshape(4, 656), None)
            torch.testing.assert_close(destinations[name][dst].reshape(4, 386), expected, rtol=0, atol=0)
        assert torch.all(destinations[name][1] == 7)
        torch.testing.assert_close(active, originals[name], rtol=0, atol=0)


def test_bulk_pending_transfer_failure_drains_next_layer_conversion():
    exporter, layers, _ = make_bulk_exporter()
    # One chunk per layer makes the first failed write occur after the next
    # layer has already queued its conversion in the other buffer.
    for name, (remote, src, dst) in layers.items():
        layers[name] = (remote, src[:1], dst[:1])
    exporter.engine.batch_transfer_sync_write = MagicMock(return_value=-1)
    with patch("torch.npu.Event"), patch("torch.npu.stream", side_effect=lambda _: contextlib.nullcontext()), patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.final_prefill.pack_c8_kv", side_effect=fake_pack) as pack:
        with pytest.raises(RuntimeError, match="transfer failed"):
            exporter.export_layers(layers, "failed")
        assert pack.call_count == 2
    exporter.stream.synchronize.assert_called_once()


def test_bulk_validates_all_layers_before_first_write():
    exporter, layers, _ = make_bulk_exporter()
    remote, _, dst = layers["draft"]
    layers["draft"] = (remote, [999, 0, 3], dst)
    with pytest.raises(ValueError, match="outside registered cache"):
        exporter.export_layers(layers, "bad-pages")
    assert not exporter.engine.calls


def test_bulk_overlap_keeps_passthrough_pages_and_final_empty_layer_safe():
    exporter, layers, destinations = make_bulk_exporter()
    indexer = torch.arange(64, dtype=torch.int8).view(8, 2, 1, 4)
    target = torch.full_like(indexer, 7)
    exporter.caches["indexer"] = (indexer,)
    exporter.metadata["indexer"] = metadata((indexer,), group=8, indexer=True)
    remote = metadata((target,), group=9, indexer=True)
    mixed = {"a": layers["a"], "indexer": (remote, [3, 0], [0, 3]),
             "b": layers["b"], "empty": layers["empty"]}
    with patch("torch.npu.Event"), patch("torch.npu.stream", side_effect=lambda _: contextlib.nullcontext()), patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.final_prefill.pack_c8_kv", side_effect=fake_pack):
        exporter.export_layers(mixed, "mixed")
    torch.testing.assert_close(target[:2], indexer[6:8], rtol=0, atol=0)
    torch.testing.assert_close(target[6:8], indexer[:2], rtol=0, atol=0)
    assert torch.all(target[2:6] == 7)
    # Last pending TQ chunk is flushed even when the final layer has no pages.
    expected = fake_pack(exporter.caches["b"][0][4:6].reshape(4, 656), None)
    torch.testing.assert_close(destinations["b"][3].reshape(4, 386), expected, rtol=0, atol=0)
    exporter.stream.synchronize.assert_called_once()
