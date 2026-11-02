# SPDX-License-Identifier: Apache-2.0
"""CPU tests of address ownership and pre-dlopen URMA configuration."""

import ctypes
import gc
import hashlib
import importlib
import os
import sys
import weakref
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import torch


@pytest.fixture
def configuration(modules):
    return importlib.import_module("pd_under_test.configuration")


@pytest.mark.parametrize("value", [1, 16, 64, "1", "16", "64"])
def test_canonical_bounded_mib(configuration, value):
    assert configuration.parse_urma_swap_mib(value) == int(value)


@pytest.mark.parametrize(
    "value", [0, -1, True, False, 16.0, None, "", "0", "-1", "+16", " 16", "16 ", "016", "16MiB", 65, 2**64]
)
def test_invalid_mib(configuration, value):
    with pytest.raises(ValueError):
        configuration.parse_urma_swap_mib(value)


@pytest.mark.parametrize("protocol", [None, "sdma", "device_rdma", "DEVICE_URMA", True])
def test_pd_protocol_must_be_explicit(configuration, protocol):
    with pytest.raises(ValueError, match="explicit"):
        configuration.validate_transport_options({"memfabric_transfer_protocol": protocol})


def test_default_is_sixteen_mib_only_for_pd(configuration):
    assert configuration.validate_transport_options({"memfabric_transfer_protocol": "device_urma"}) == (
        "device_urma",
        16,
    )


@pytest.fixture
def unloaded(configuration, monkeypatch, tmp_path):
    """Model proc metadata only; never import MemFabric or create an engine."""
    monkeypatch.delenv(configuration.URMA_SWAP_ENV, raising=False)
    monkeypatch.delenv(configuration.DEPRECATED_URMA_SWAP_ENV, raising=False)
    for name in tuple(sys.modules):
        if name == "memfabric_hybrid" or name.startswith("memfabric_hybrid.") or name.startswith("_pymf_"):
            monkeypatch.delitem(sys.modules, name)
    core = tmp_path / "libmf_hybm_core.so"
    core.write_bytes(b"CPU fixture; not an executable library")
    state = NS(maps="1000-2000 r-xp /usr/lib/libc.so\n", version="1.3.0", metadata_calls=[], path_calls=[])
    original_read_text = Path.read_text

    def read_text(path, *args, **kwargs):
        if path == Path("/proc/self/maps"):
            if isinstance(state.maps, BaseException):
                raise state.maps
            return state.maps
        return original_read_text(path, *args, **kwargs)

    def locate_file(path):
        state.path_calls.append(path)
        assert path == "memfabric_hybrid/lib/libmf_hybm_core.so"
        return core

    def distribution(name):
        state.metadata_calls.append(name)
        assert name == "memfabric-hybrid"
        return NS(version=state.version, locate_file=locate_file)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(configuration.importlib.metadata, "distribution", distribution)
    # Resolving a submodule can import its parent. It must never be used here.
    monkeypatch.setattr(importlib.util, "find_spec", lambda *a, **kw: pytest.fail("find_spec can load the package"))
    state.core = core
    state.configuration = configuration
    return state


@pytest.mark.parametrize("inherited", [None, "16"])
def test_configure_before_loading_records_version_hash_and_units(unloaded, monkeypatch, inherited):
    config = unloaded.configuration
    if inherited is not None:
        monkeypatch.setenv(config.URMA_SWAP_ENV, inherited)
    proof = config.configure_urma_staging(16)
    assert proof == {
        "package_version": "1.3.0",
        "library_path": str(unloaded.core),
        "library_sha256": hashlib.sha256(unloaded.core.read_bytes()).hexdigest(),
        "swap_mib": 16,
        "swap_bytes": 16 * 1024 * 1024,
        "configured_before_library_load": True,
    }
    assert os.environ[config.URMA_SWAP_ENV] == "16"
    assert unloaded.metadata_calls == ["memfabric-hybrid"]
    assert not any(name.startswith("memfabric_hybrid") for name in sys.modules)


@pytest.mark.parametrize("inherited", ["", "0", "32", "016", " 16", "16777216"])
def test_conflict_does_not_change_environment(unloaded, monkeypatch, inherited):
    config = unloaded.configuration
    monkeypatch.setenv(config.URMA_SWAP_ENV, inherited)
    with pytest.raises(ValueError, match="conflicts"):
        config.configure_urma_staging(16)
    assert os.environ[config.URMA_SWAP_ENV] == inherited
    assert not unloaded.metadata_calls


def test_deprecated_even_equal_is_rejected(unloaded, monkeypatch):
    config = unloaded.configuration
    monkeypatch.setenv(config.DEPRECATED_URMA_SWAP_ENV, "16")
    with pytest.raises(ValueError, match="deprecated"):
        config.configure_urma_staging(16)
    assert config.URMA_SWAP_ENV not in os.environ


@pytest.mark.parametrize("kind", ["package", "submodule", "extension", "maps_without_modules"])
@pytest.mark.parametrize("inherited", [None, "16"])
def test_already_loaded_cannot_be_repaired_by_environment(unloaded, monkeypatch, kind, inherited):
    config = unloaded.configuration
    if inherited is not None:
        monkeypatch.setenv(config.URMA_SWAP_ENV, inherited)
    if kind == "maps_without_modules":
        unloaded.maps += "2000-3000 r-xp /other/libmf_hybm_core.so (deleted)\n"
    else:
        name = {
            "package": "memfabric_hybrid",
            "submodule": "memfabric_hybrid._pymf_transfer",
            "extension": "_pymf_transfer",
        }[kind]
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    with pytest.raises(RuntimeError, match="before any MemFabric import"):
        config.configure_urma_staging(16)
    assert os.environ.get(config.URMA_SWAP_ENV) == inherited
    assert not unloaded.metadata_calls


@pytest.mark.parametrize("failure", ["maps", "version", "library"])
def test_unverifiable_library_cannot_set_environment(unloaded, failure):
    config = unloaded.configuration
    if failure == "maps":
        unloaded.maps = OSError("proc unavailable")
    elif failure == "version":
        unloaded.version = "1.2.0"
    else:
        unloaded.core.unlink()
    with pytest.raises((RuntimeError, ValueError, FileNotFoundError)):
        config.configure_urma_staging(16)
    assert config.URMA_SWAP_ENV not in os.environ


@pytest.mark.parametrize("producer", [True, False])
def test_real_configuration_precedes_engine_registration(modules, unloaded, monkeypatch, producer):
    config = unloaded.configuration
    calls = []

    def configure(**kwargs):
        assert os.environ[config.URMA_SWAP_ENV] == "16"
        assert kwargs["transfer_protocol"] == "device_urma"
        calls.append("configure")

    engine = NS(
        configure=configure,
        unique_id="cpu-session",
        register_memory=lambda ptr, size: calls.append((ptr, size)) or 0,
        destroy=lambda: calls.append("destroy") or 0,
        unInitialize=lambda: calls.append("unInitialize") or 0,
    )
    engine._engine = engine
    engine.get_transfer_engine = lambda host: engine
    name = "vllm_ascend.distributed.kv_transfer.utils.memfabric_transfer_engine"
    module = ModuleType(name)
    module.global_memfabric_te = engine
    monkeypatch.setitem(sys.modules, name, module)
    allocation = torch.empty(8)
    reader = modules.transport.MemfabricReader(
        producer=producer,
        device=1,
        host="127.0.0.1",
        protocol="device_urma",
        regions={(2**44, 1024): "hisparse_source", (2**45, 2048): "hisparse_indexer"},
        owners=[allocation],
    )
    reader.initialize()
    assert calls == ["configure", *(([(2**44, 1024)]) if producer else []), (2**45, 2048)]
    assert reader.staging_config["swap_bytes"] == 16777216
    reader.close()
    assert calls[-2:] == ["destroy", "unInitialize"]


class Mapping:
    def __init__(self, tensor):
        self.tensor = tensor
        self.host_address = tensor.data_ptr()
        self.device_ptr = self.host_address + 2**44
        self.nbytes = tensor.numel() * tensor.element_size()
        self.closed = False

    def address_of(self, tensor):
        if self.closed:
            raise RuntimeError("mapping closed")
        offset = tensor.data_ptr() - self.host_address
        if offset < 0 or offset + tensor.numel() * tensor.element_size() > self.nbytes:
            raise ValueError("mapping bounds")
        return self.device_ptr + offset


def cache_fixture(*, factor=7, dtype=torch.bfloat16, blocks=8, guard=19):
    block_size, width = 4, 6
    layer_elements = blocks * block_size * width
    host = torch.full((2 * layer_elements + 2 * guard,), -19, dtype=dtype)
    mapping = Mapping(host)
    raw = torch.full((blocks * factor * block_size * 2 + 2 * guard,), -19, dtype=dtype)
    caches, context, groups, tensors = {}, {}, [], []
    for layer in range(2):
        name = f"layer{layer}"
        caches[name] = host[guard + layer * layer_elements : guard + (layer + 1) * layer_elements].view(
            blocks, block_size, width
        )
        context[name] = NS(hisparse_cache=NS(runtime=NS(cache_ops=NS(host_mapping=mapping))))
        indexer = f"indexer{layer}"
        caches[indexer] = (
            torch.as_strided(raw, ((blocks - 1) * factor + 1, block_size, 1, 2), (8, 2, 2, 1), guard + layer * 8),
        )
        context[indexer] = NS(hisparse_block_stride=factor)
        tensors.append(NS(layers=[indexer], block_stride=factor * 8 * 2))
        groups.append(
            NS(
                role="hisparse_indexer",
                enable_kv_transfer=True,
                is_eagle_group=False,
                layer_names=[indexer],
                kv_cache_spec=NS(block_size=block_size),
            )
        )
    groups.insert(
        0,
        NS(
            role="hisparse_source",
            enable_kv_transfer=True,
            is_eagle_group=False,
            layer_names=["layer0", "layer1"],
            kv_cache_spec=NS(block_size=block_size),
        ),
    )
    return NS(
        caches=caches,
        context=context,
        mapping=mapping,
        host=host,
        raw=raw,
        config=NS(kv_cache_groups=groups, kv_cache_tensors=tensors, num_blocks=blocks, hisparse_host_num_blocks=blocks),
    )


@pytest.mark.parametrize("producer", [True, False])
def test_role_addresses_offsets_multilayer_stride_and_actual_owners(modules, producer):
    fixture = cache_fixture()
    result, regions, owners = modules.layout.build_layouts(
        fixture.caches, fixture.config, fixture.context, producer=producer, allow_cpu_indexer=True
    )
    expected_host_base = fixture.mapping.device_ptr if producer else fixture.mapping.host_address
    assert regions[(expected_host_base, fixture.mapping.nbytes)] == "hisparse_source"
    assert len(regions) == 2  # One mapped host owner and one HMA device backing.
    for layer in result:
        cache = fixture.caches[layer.name]
        if layer.role == "hisparse_source":
            assert layer.base == expected_host_base + cache.data_ptr() - fixture.host.data_ptr()
            assert layer.address(3, 2) == layer.base + 3 * 4 * 12 + 2 * 12
            assert any(owner is cache for owner in owners)
        else:
            cache = cache[0]
            assert layer.base == cache.data_ptr()
            assert layer.address(3, 2) == cache.data_ptr() + 3 * 7 * 16 + 2 * 4
    mapping_ref = weakref.ref(fixture.mapping)
    host_ref = weakref.ref(fixture.host)
    indexer_ref = weakref.ref(fixture.caches["indexer1"][0])
    del fixture, cache
    gc.collect()
    assert all(ref() is not None for ref in (mapping_ref, host_ref, indexer_ref))
    owners.clear()
    gc.collect()
    assert all(ref() is None for ref in (mapping_ref, host_ref, indexer_ref))


@pytest.mark.parametrize("producer", [True, False])
def test_source_mapping_must_still_be_live_for_both_addresses(modules, producer):
    fixture = cache_fixture()
    fixture.mapping.closed = True
    with pytest.raises(RuntimeError, match="mapping closed"):
        modules.layout.build_layouts(
            fixture.caches, fixture.config, fixture.context, producer=producer, allow_cpu_indexer=True
        )


def test_cpu_indexer_remains_test_only_and_bf16_required(modules):
    fixture = cache_fixture()
    with pytest.raises(ValueError, match="Ascend device"):
        modules.layout.build_layouts(fixture.caches, fixture.config, fixture.context, producer=False)
    fixture = cache_fixture(dtype=torch.float16)
    with pytest.raises(ValueError, match="BF16"):
        modules.layout.build_layouts(
            fixture.caches, fixture.config, fixture.context, producer=False, allow_cpu_indexer=True
        )


@pytest.mark.parametrize("producer", [None, 0, 1, "producer"])
def test_role_cannot_be_inferred_or_truthy(modules, producer):
    fixture = cache_fixture()
    with pytest.raises(TypeError, match="explicit producer boolean"):
        modules.layout.build_layouts(fixture.caches, fixture.config, fixture.context, producer=producer)


@pytest.mark.parametrize("start", [0, 37])
def test_role_layout_partial_transfer_preserves_entire_backing(modules, start):
    """CPU oracle translates fake SOURCE alias only; not an URMA capability test."""
    source = cache_fixture(factor=7, blocks=80, guard=19)
    destination = cache_fixture(factor=11, blocks=80, guard=23)
    source_layouts, _, source_owners = modules.layout.build_layouts(
        source.caches, source.config, source.context, producer=True, allow_cpu_indexer=True
    )
    destination_layouts, _, destination_owners = modules.layout.build_layouts(
        destination.caches, destination.config, destination.context, producer=False, allow_cpu_indexer=True
    )
    assert source_owners and destination_owners
    source_blocks = {0: list(range(65)), 1: list(range(65)), 2: list(range(65))}
    destination_blocks = {group: list(reversed(range(65))) for group in source_blocks}
    expected_host, expected_indexer = destination.host.clone(), destination.raw.clone()
    for index, layer in enumerate(source_layouts):
        source_cache = source.caches[layer.name]
        destination_cache = destination.caches[layer.name]
        if layer.role == "hisparse_source":
            source_view = source_cache
            destination_view = destination_cache
            expected_storage = expected_host
        else:
            source_cache, destination_cache = source_cache[0], destination_cache[0]
            source_view = torch.as_strided(source_cache, (80, 4, 2), (7 * 8, 2, 1), source_cache.storage_offset())
            destination_view = torch.as_strided(
                destination_cache, (80, 4, 2), (11 * 8, 2, 1), destination_cache.storage_offset()
            )
            expected_storage = expected_indexer
        source_view.copy_(torch.arange(source_view.numel()).reshape(source_view.shape) + index * 1000)
        expected_view = torch.as_strided(
            expected_storage, destination_view.shape, destination_view.stride(), destination_view.storage_offset()
        )
        for token in range(start, 257):
            block, offset = divmod(token, 4)
            expected_view[destination_blocks[layer.group_id][block], offset].copy_(
                source_view[source_blocks[layer.group_id][block], offset]
            )
    for target, origin, size in modules.protocol.plan_reads(
        source_layouts, destination_layouts, source_blocks, destination_blocks, start, 257
    ):
        if source.mapping.device_ptr <= origin < source.mapping.device_ptr + source.mapping.nbytes:
            origin -= source.mapping.device_ptr - source.mapping.host_address
        ctypes.memmove(target, origin, size)
    assert torch.equal(destination.host, expected_host)
    assert torch.equal(destination.raw, expected_indexer)
