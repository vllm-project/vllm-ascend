# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Ascend setup for the cross-encoder Mooncake Store client."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.data import (
    TensorSpec,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.store_client import (
    _MOONCAKE_TENSOR_HEADER,
    _SAFE_PUT_REJECTIONS,
    EmbeddingStoreError,
    EmbeddingStoreOperationError,
    MooncakeEmbeddingStoreClient,
    _batch_get_completed_safely,
    _encode_mooncake_tensor_metadata,
    _validate_mooncake_tensor_metadata,
)
from vllm.distributed.mooncake_store import DEFAULT_TENANT_ID, MooncakeStoreConfig
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.bounce import (
    ASCEND_DIRECT_MEMORY_ALIGNMENT,
    AscendBounceArena,
    _RegistrationRangePlan,
    _plan_source,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.transfer import (
    AscendMooncakeTransfer,
    _RegistrationCleanupError,
)

logger = init_logger(__name__)

# KV Pool keeps independently owned Ascend Store engines and their registered
# KV backing alive for the worker-process lifetime.  The encoder Store follows
# the same rule for a producer slab registered in both the EC and Store
# engines: neither owner may tear down the shared NPU allocation first.
_PROCESS_LIFETIME_STORE_OWNERS: list[tuple[Any, Any]] = []


def use_store_independent_te() -> bool:
    """Select the KV-Pool-compatible independent Store topology."""
    if os.getenv("ASCEND_ENABLE_USE_FABRIC_MEM", "0") == "1":
        return False
    raw = os.getenv("ASCEND_GLOBAL_RESOURCE_CONFIG")
    if not raw:
        return False
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(
            "ASCEND_GLOBAL_RESOURCE_CONFIG must be a JSON object"
        ) from error
    if not isinstance(config, dict):
        raise ValueError(
            "ASCEND_GLOBAL_RESOURCE_CONFIG must be a JSON object"
        )
    return True


def ensure_mooncake_store_imported() -> None:
    """Load Store bindings before the TransferEngine bindings.

    Mooncake's Python extensions share TransferEngine type registration. The
    Store extension must be loaded before ``mooncake.engine`` creates the
    process-wide engine that is later passed to ``MooncakeDistributedStore``.
    """
    try:
        import mooncake.store  # noqa: F401  # type: ignore
    except ImportError as error:
        raise ImportError("Install mooncake with Store support to enable cross_encoder_cache.") from error


class AscendMooncakeEmbeddingStoreClient(MooncakeEmbeddingStoreClient):
    """Use the Ascend producer arena with role-appropriate engines."""

    def __init__(
        self,
        store: Any,
        bounce_arena: AscendBounceArena,
        registration_transfer: AscendMooncakeTransfer,
        replicate_config: Any | None = None,
        *,
        read_buffer_bytes: int = 128 * 1024**2,
        use_fabric_mem: bool = False,
        use_store_independent_te: bool = False,
    ) -> None:
        super().__init__(
            store,
            replicate_config=replicate_config,
            read_buffer_bytes=read_buffer_bytes,
        )
        self._bounce_arena = bounce_arena
        self._registration_transfer = registration_transfer
        self._use_fabric_mem = use_fabric_mem
        self._use_store_independent_te = use_store_independent_te
        self._put_header_tensor: torch.Tensor | None = None
        self._independent_staging_owner: torch.Tensor | None = None
        self._retained_for_process_lifetime = False

    @property
    def uses_store_independent_te(self) -> bool:
        return self._use_store_independent_te

    def register_staging_buffer(self, tensor: torch.Tensor) -> None:
        """Register one long-lived EC staging slab in the Store-owned engine."""
        if not self._use_store_independent_te:
            raise RuntimeError("Store-independent staging registration is not enabled")
        if self._independent_staging_owner is not None:
            if self._independent_staging_owner is tensor:
                return
            raise RuntimeError("A different Store-independent staging slab is already registered")
        ret = self.store.register_buffer(tensor.data_ptr(), tensor.nbytes)
        if ret != 0:
            raise EmbeddingStoreOperationError(
                f"Failed to register producer staging slab with Store: {ret}"
            )
        self._independent_staging_owner = tensor

    def _ensure_bounce_available(self) -> torch.Tensor:
        bounce = self._bounce_arena.tensor
        if bounce is None:
            raise EmbeddingStoreOperationError("Ascend Mooncake bounce arena is unavailable")

        address = bounce.data_ptr()
        if address % ASCEND_DIRECT_MEMORY_ALIGNMENT != 0:
            raise RuntimeError("Ascend Mooncake bounce arena is not 2 MiB aligned")

        return bounce

    @contextmanager
    def _registered_io(
        self,
        buffers: list[tuple[Any, int, int]],
    ) -> Iterator[Callable[[int], None]]:
        """Register direct ranges with the engine used by Store I/O."""
        self._check_healthy()
        addresses: list[int] = []
        submitted = False
        confirmed = False

        def confirm(result: int) -> None:
            nonlocal confirmed
            if type(result) is not int or (result < 0 and result not in _SAFE_PUT_REJECTIONS):
                raise EmbeddingStoreError("Mooncake I/O completion is unconfirmed; retaining buffers")
            confirmed = True

        owners = [owner for owner, _, _ in buffers]
        try:
            if self._use_store_independent_te and buffers:
                try:
                    for owner, address, size in buffers:
                        status = self.store.register_buffer(address, size)
                        if status != 0:
                            raise EmbeddingStoreOperationError(
                                f"Failed to register embedding buffer: {status}"
                            )
                        addresses.append(address)
                except BaseException as error:
                    cleanup_failed = False
                    for address in reversed(addresses):
                        cleanup_failed |= self.store.unregister_buffer(address) != 0
                    if cleanup_failed:
                        self._poison(owners)
                        raise EmbeddingStoreError(
                            "independent Store registration rollback is unconfirmed"
                        ) from error
                    raise
            elif not self._use_fabric_mem and buffers:
                try:
                    addresses = self._registration_transfer.acquire_registration_ranges(
                        tuple(
                            _RegistrationRangePlan(
                                address=address,
                                nbytes=size,
                                owners=(owner,),
                            )
                            for owner, address, size in buffers
                        )
                    )
                except _RegistrationCleanupError as error:
                    self._poison(owners)
                    raise EmbeddingStoreError(
                        "shared Store registration rollback is unconfirmed"
                    ) from error
                except RuntimeError as error:
                    raise EmbeddingStoreOperationError(
                        f"Failed to register embedding buffer: {error}"
                    ) from error
            submitted = True
            yield confirm
        except BaseException as error:
            if submitted and not confirmed:
                self._poison(owners)
                raise EmbeddingStoreError("Mooncake I/O completion is unconfirmed; retaining buffers") from error
            raise
        finally:
            if addresses and (not submitted or confirmed):
                try:
                    if self._use_store_independent_te:
                        cleanup_failed = False
                        for address in reversed(addresses):
                            status = self.store.unregister_buffer(address)
                            if status != 0:
                                cleanup_failed = True
                        if cleanup_failed:
                            raise EmbeddingStoreError(
                                "independent Store registration release is unconfirmed"
                            )
                    elif not self._registration_transfer.release_registration_ranges(
                        addresses
                    ):
                        raise EmbeddingStoreError(
                            "shared Store registration release is unconfirmed"
                        )
                except BaseException:
                    self._poison(owners)
                    raise

    def put_tensor(self, pool_key: str, tensor: torch.Tensor) -> None:
        self._check_healthy()
        if tensor.device.type != "npu":
            raise EmbeddingStoreOperationError("embedding tensor must be on an NPU")
        if not tensor.is_contiguous():
            raise EmbeddingStoreOperationError("embedding tensor must be contiguous")

        # Store publication runs on a background CPU thread. Wait for NPU
        # producer work before Mooncake reads a direct range or a prefix copied
        # through the private bounce-copy stream.
        torch.npu.set_device(tensor.device)
        torch.npu.synchronize(tensor.device)
        lease = None
        retain_lease = False

        try:
            metadata = _encode_mooncake_tensor_metadata(tensor)
            metadata_tensor = torch.frombuffer(bytearray(metadata), dtype=torch.uint8)
            source = _plan_source(tensor)
            if self._use_fabric_mem:
                # Preserve the proven FabricMem layout: the Store-owned engine
                # handles the small NPU header allocation implicitly.
                if self._put_header_tensor is None:
                    self._put_header_tensor = torch.empty(
                        len(metadata),
                        dtype=torch.uint8,
                        device=tensor.device,
                    )
                header = self._put_header_tensor
                if header.numel() != len(metadata):
                    raise EmbeddingStoreOperationError("Embedding header size changed")
                header.copy_(metadata_tensor)
                torch.npu.synchronize(tensor.device)
                addresses = [header.data_ptr()]
                sizes = [len(metadata)]

                if source.prefix_nbytes > 0:
                    self._ensure_bounce_available()
                    lease = self._bounce_arena.acquire(source.prefix_nbytes)
                    assert lease is not None
                    bounce_address = self._bounce_arena.copy(
                        lease,
                        [(tensor, 0, source.prefix_nbytes)],
                    )
                    addresses.append(bounce_address)
                    sizes.append(source.prefix_nbytes)
            else:
                # HIXL cannot select one transfer type for a batch containing
                # both Host and device sources. Pack the 24-byte header and the
                # unaligned payload prefix into the selected engine's
                # pre-registered NPU bounce arena so every slice is device memory.
                self._ensure_bounce_available()
                bounce_nbytes = len(metadata) + source.prefix_nbytes
                lease = self._bounce_arena.acquire(bounce_nbytes)
                assert lease is not None
                bounce_address = self._bounce_arena.copy(
                    lease,
                    [
                        (metadata_tensor, 0, len(metadata)),
                        *(
                            [(tensor, len(metadata), source.prefix_nbytes)]
                            if source.prefix_nbytes > 0
                            else []
                        ),
                    ],
                )
                addresses = [bounce_address]
                sizes = [bounce_nbytes]

            direct_buffers = []
            if source.registration_address is not None:
                direct_buffers.append(
                    (
                        tensor,
                        source.registration_address,
                        source.registration_nbytes,
                    )
                )
            if source.direct_address is not None:
                addresses.append(source.direct_address)
                sizes.append(source.direct_nbytes)

            with self._registered_io(direct_buffers) as confirm:
                [result] = self.store.batch_put_from_multi_buffers(
                    [pool_key],
                    [addresses],
                    [sizes],
                    self.replicate_config,
                )
                confirm(result)
                if result < 0:
                    raise EmbeddingStoreOperationError(f"failed to put embedding tensor for {pool_key}")
            logger.info("Stored encoder output in Mooncake Store")
        except BaseException:
            if lease is not None and self._poisoned:
                self._poison([self._bounce_arena, tensor, lease])
                retain_lease = True
            raise
        finally:
            if lease is not None and not retain_lease:
                self._bounce_arena.release(lease)

    def prepare_put_header(self, tensor: torch.Tensor) -> torch.Tensor:
        """Build the small CPU source copied into a fallback wave's bounce."""
        metadata = _encode_mooncake_tensor_metadata(tensor)
        return torch.frombuffer(bytearray(metadata), dtype=torch.uint8)

    def put_prepared_tensor(
        self,
        pool_key: str,
        addresses: list[int],
        sizes: list[int],
        owners: list[Any],
    ) -> None:
        """PUT buffers already registered by the joint EC/Store wave."""
        self._check_healthy()
        try:
            [result] = self.store.batch_put_from_multi_buffers(
                [pool_key],
                [addresses],
                [sizes],
                self.replicate_config,
            )
            if type(result) is not int or (
                result < 0 and result not in _SAFE_PUT_REJECTIONS
            ):
                raise EmbeddingStoreError(
                    "Mooncake I/O completion is unconfirmed; retaining buffers"
                )
        except EmbeddingStoreOperationError:
            raise
        except BaseException as error:
            self._poison(owners)
            raise EmbeddingStoreError(
                "Mooncake I/O completion is unconfirmed; retaining buffers"
            ) from error
        if result < 0:
            raise EmbeddingStoreOperationError(
                f"failed to put embedding tensor for {pool_key}"
            )
        logger.info("Stored encoder output in Mooncake Store")

    def retain_fallback_owners(self, owners: list[Any]) -> None:
        """Poison owners whose submitted fallback completion is uncertain."""
        self._poison(owners)

    def _load_chunk(
        self,
        expected: dict[str, TensorSpec],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        if device.type != "npu":
            raise EmbeddingStoreOperationError("Ascend embedding target must be on an NPU")

        if not self._use_fabric_mem:
            return self._load_chunk_shared_roce(expected, device)

        self._check_healthy()
        torch.npu.set_device(device)
        if self._read_buffer is None:
            self._read_buffer = torch.empty(
                self._read_buffer_bytes,
                dtype=torch.uint8,
                device=device,
            )
        buffer = self._read_buffer
        base = buffer.data_ptr()
        sizes = [_MOONCAKE_TENSOR_HEADER.size + spec.nbytes for spec in expected.values()]
        offsets: list[int] = []
        end = 0
        for size in sizes:
            offsets.append(end)
            end += size
        try:
            results = self.store.batch_get_into(
                list(expected),
                [base + offset for offset in offsets],
                sizes,
            )
            if not _batch_get_completed_safely(results, sizes):
                raise EmbeddingStoreError("Unsafe batch GET results")
        except BaseException as error:
            self._poison([buffer])
            raise EmbeddingStoreError("Mooncake I/O completion is unconfirmed; retaining read buffer") from error

        valid: list[tuple[str, TensorSpec, int]] = []
        for (key, spec), offset, size, result in zip(expected.items(), offsets, sizes, results, strict=True):
            if result != size:
                logger.warning(
                    "Store GET skipped for %s: result=%d, expected bytes=%d",
                    key,
                    result,
                    size,
                )
                continue
            metadata = bytes(buffer[offset : offset + _MOONCAKE_TENSOR_HEADER.size].cpu().tolist())
            try:
                _validate_mooncake_tensor_metadata(key, metadata, spec)
            except EmbeddingStoreOperationError as error:
                logger.warning("Store GET skipped for %s: %s", key, error)
                continue
            valid.append((key, spec, offset + _MOONCAKE_TENSOR_HEADER.size))
        if not valid:
            return {}

        targets = {
            key: torch.empty(
                spec.shape,
                dtype=getattr(torch, spec.dtype.removeprefix("torch.")),
                device=device,
            )
            for key, spec, _ in valid
        }
        try:
            for key, spec, offset in valid:
                targets[key].view(torch.uint8).view(-1).copy_(buffer[offset : offset + spec.nbytes])
            torch.npu.synchronize(device)
        except BaseException as error:
            self._poison([buffer, *targets.values()])
            raise EmbeddingStoreError("Embedding copy completion is unconfirmed; retaining buffers") from error
        return targets

    def _load_chunk_shared_roce(
        self,
        expected: dict[str, TensorSpec],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Scatter a Store object directly into its final NPU tensor.

        Only each 24-byte header and an unalignable payload prefix use the
        selected topology's NPU bounce arena. Aligned payload suffixes are
        received into their final tensors, so no complete CPU or second NPU
        read buffer is needed.
        """
        self._check_healthy()
        torch.npu.set_device(device)
        targets = {
            key: torch.empty(
                spec.shape,
                dtype=getattr(torch, spec.dtype.removeprefix("torch.")),
                device=device,
            )
            for key, spec in expected.items()
        }
        plans = {key: _plan_source(target) for key, target in targets.items()}
        header_size = _MOONCAKE_TENSOR_HEADER.size
        bounce_nbytes = sum(header_size + plan.prefix_nbytes for plan in plans.values())
        self._ensure_bounce_available()
        lease = self._bounce_arena.acquire(bounce_nbytes)
        assert lease is not None
        bounce = self._bounce_arena.tensor
        assert bounce is not None

        all_addresses: list[list[int]] = []
        all_sizes: list[list[int]] = []
        header_offsets: dict[str, int] = {}
        prefix_offsets: dict[str, int] = {}
        direct_buffers: list[tuple[Any, int, int]] = []
        offset = 0
        for key, target in targets.items():
            plan = plans[key]
            header_offsets[key] = offset
            addresses = [bounce.data_ptr() + lease.offset + offset]
            sizes = [header_size]
            offset += header_size
            if plan.prefix_nbytes > 0:
                prefix_offsets[key] = offset
                addresses.append(bounce.data_ptr() + lease.offset + offset)
                sizes.append(plan.prefix_nbytes)
                offset += plan.prefix_nbytes
            if plan.direct_address is not None:
                addresses.append(plan.direct_address)
                sizes.append(plan.direct_nbytes)
                direct_buffers.append(
                    (target, plan.registration_address, plan.registration_nbytes)
                )
            all_addresses.append(addresses)
            all_sizes.append(sizes)

        reuse_safe = False
        retain_lease = False
        try:
            with self._registered_io(direct_buffers) as confirm:
                results = self.store.batch_get_into_multi_buffers(
                    list(expected),
                    all_addresses,
                    all_sizes,
                )
                capacities = [header_size + spec.nbytes for spec in expected.values()]
                if not _batch_get_completed_safely(results, capacities):
                    raise EmbeddingStoreError("Unsafe batch GET results")
                confirm(0)

            loaded: dict[str, torch.Tensor] = {}
            for (key, spec), result in zip(expected.items(), results, strict=True):
                expected_size = header_size + spec.nbytes
                if result != expected_size:
                    logger.warning(
                        "Store GET skipped for %s: result=%d, expected bytes=%d",
                        key,
                        result,
                        expected_size,
                    )
                    continue
                header_offset = header_offsets[key]
                metadata = bytes(
                    bounce.narrow(0, lease.offset + header_offset, header_size)
                    .cpu()
                    .tolist()
                )
                try:
                    _validate_mooncake_tensor_metadata(key, metadata, spec)
                except EmbeddingStoreOperationError as error:
                    logger.warning("Store GET skipped for %s: %s", key, error)
                    continue
                plan = plans[key]
                if plan.prefix_nbytes > 0:
                    target_prefix = targets[key].view(torch.uint8).view(-1).narrow(
                        0, 0, plan.prefix_nbytes
                    )
                    target_prefix.copy_(
                        bounce.narrow(
                            0,
                            lease.offset + prefix_offsets[key],
                            plan.prefix_nbytes,
                        )
                    )
                loaded[key] = targets[key]
            torch.npu.synchronize(device)
            reuse_safe = True
            return loaded
        except BaseException:
            if not reuse_safe:
                self._poison([self._bounce_arena, lease, *targets.values()])
                retain_lease = True
            raise
        finally:
            if not retain_lease:
                self._bounce_arena.release(lease)

    def close(self) -> None:
        if self._retained_for_process_lifetime:
            return
        self._check_healthy()

        self._put_header_tensor = None

        if self._use_fabric_mem:
            # FabricMem never explicitly registered these NPU allocations.
            self._read_buffer = None
        if self._use_store_independent_te and self._independent_staging_owner is not None:
            # Do not explicitly unregister or close either native owner of the
            # dual-registered slab.  The installed AscendDirect binding has
            # not demonstrated independent unregister safety; this mirrors KV
            # Pool's process-lifetime registration contract.
            _PROCESS_LIFETIME_STORE_OWNERS.append(
                (self.store, self._independent_staging_owner)
            )
            self._independent_staging_owner = None
            self._retained_for_process_lifetime = True
            return

        # The base header remains unused, so the base close tears down only
        # the Store and any base-class read buffer.
        super().close()

    def load_tensors(
        self,
        expected: Mapping[str, TensorSpec],
        device: torch.device | str,
    ) -> dict[str, torch.Tensor]:
        loaded = super().load_tensors(expected, device)
        if loaded:
            logger.info("Loaded %d encoder output(s) from Mooncake Store", len(loaded))
        return loaded


def _setup_store_owned_engine(
    store: Any,
    config: MooncakeStoreConfig,
    local_hostname: str,
    topology: str,
) -> None:
    """Let Store create the TransferEngine required by this topology."""
    setup_kwargs: dict[str, Any] = {}
    if config.tenant_id != DEFAULT_TENANT_ID:
        setup_kwargs["tenant_id"] = config.tenant_id
    ret = store.setup(
        local_hostname,
        config.metadata_server,
        config.global_segment_size,
        0,
        config.protocol,
        config.device_name,
        config.master_server_address,
        **setup_kwargs,
    )
    if ret != 0:
        raise RuntimeError(
            f"Initialize {topology} MooncakeDistributedStore failed: {ret}"
        )


def _setup_store_with_shared_engine(
    store: Any,
    config: MooncakeStoreConfig,
    local_hostname: str,
    transfer_engine: Any,
) -> None:
    """Attach Store to the already initialized process-wide RoCE engine."""
    setup_kwargs: dict[str, Any] = {"engine": transfer_engine.get_engine()}
    if config.tenant_id != DEFAULT_TENANT_ID:
        setup_kwargs["tenant_id"] = config.tenant_id
    endpoint = f"{local_hostname}:{transfer_engine.get_rpc_port()}"
    ret = store.setup(
        endpoint,
        config.metadata_server,
        config.global_segment_size,
        config.local_buffer_size,
        config.protocol,
        config.device_name,
        config.master_server_address,
        **setup_kwargs,
    )
    if ret != 0:
        raise RuntimeError("Initialize MooncakeDistributedStore failed.")


def configure_non_fabric_roce() -> None:
    """Select the probe-verified RoCE link before either engine is created."""
    # Mooncake Store's native setup checks only getenv() presence, not whether
    # the value equals "1".  Leaving ASCEND_ENABLE_USE_FABRIC_MEM=0 in the
    # environment therefore enables FabricMem on a Store-owned engine.
    os.environ.pop("ASCEND_ENABLE_USE_FABRIC_MEM", None)
    os.environ["HCCL_INTRA_PCIE_ENABLE"] = "0"
    os.environ["HCCL_INTRA_ROCE_ENABLE"] = "1"


def _configure_shared_roce() -> None:
    """Validate configuration specific to the shared-engine RoCE path."""
    if os.getenv("ASCEND_GLOBAL_RESOURCE_CONFIG"):
        raise ValueError(
            "shared EC/Store RoCE requires ASCEND_GLOBAL_RESOURCE_CONFIG to be unset"
        )
    if os.getenv("ASCEND_BUFFER_POOL"):
        raise ValueError(
            "shared EC/Store RoCE requires ASCEND_BUFFER_POOL to be unset"
        )
    configure_non_fabric_roce()


def create_ascend_mooncake_embedding_store_client(
    bounce_arena: AscendBounceArena,
    registration_transfer: AscendMooncakeTransfer,
    read_buffer_bytes: int = 128 * 1024**2,  # 128 MiB
) -> AscendMooncakeEmbeddingStoreClient:
    """Create an embedding Store using the selected Ascend engine topology."""
    config = MooncakeStoreConfig.load_from_config()
    if config.protocol != "ascend":
        raise ValueError("Ascend cross_encoder_cache requires Mooncake Store protocol='ascend'")
    if config.enable_offload:
        raise ValueError("cross_encoder_cache supports a RAM Store; disable enable_offload")

    try:
        from mooncake.store import (  # type: ignore
            MooncakeDistributedStore,
            ObjectDataType,
            ReplicateConfig,
        )
    except ImportError as error:
        raise ImportError("Install mooncake with Store support to enable cross_encoder_cache.") from error

    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import rdma_utils

    store = MooncakeDistributedStore()
    local_hostname = rdma_utils.get_requester_local_hostname(get_ip())
    use_fabric_mem = os.getenv("ASCEND_ENABLE_USE_FABRIC_MEM", "0") == "1"
    use_store_independent = use_store_independent_te()
    if use_store_independent:
        configure_non_fabric_roce()
    if not use_fabric_mem and os.getenv("ASCEND_BUFFER_POOL"):
        raise ValueError(
            "Ascend cross_encoder_cache does not support ASCEND_BUFFER_POOL"
        )
    if use_fabric_mem:
        # Mooncake only tags a TransferEngine as Store-owned and reads the
        # FabricMem setting while Store initializes that engine. Passing the
        # already initialized EC P2P engine would keep store=false and
        # use_fabric_mem=false. As in Ascend's KV Store backend, the effective
        # local buffer is zero in this mode; Store owns only its FabricMem
        # global segment. The EC bounce arena remains registered solely with
        # the worker-owned P2P engine and is not registered again here.
        _setup_store_owned_engine(store, config, local_hostname, "FabricMem")
        engine_topology = "Store-owned FabricMem TransferEngine"
        effective_local_buffer_size = 0
    elif use_store_independent:
        _setup_store_owned_engine(store, config, local_hostname, "independent")
        engine_topology = "Store-owned RoCE TransferEngine"
        effective_local_buffer_size = 0
    else:
        _configure_shared_roce()
        transfer_engine = registration_transfer._ensure_engine()
        _setup_store_with_shared_engine(
            store,
            config,
            local_hostname,
            transfer_engine,
        )
        engine_topology = "shared EC/Store RoCE TransferEngine"
        effective_local_buffer_size = config.local_buffer_size

    replicate_config = ReplicateConfig()
    replicate_config.data_type = ObjectDataType.TENSOR
    logger.info(
        "Initialized embedding Mooncake store with %s mode=%s global_segment_size=%d local_buffer_size=%d",
        engine_topology,
        config.mode,
        config.global_segment_size,
        effective_local_buffer_size,
    )
    return AscendMooncakeEmbeddingStoreClient(
        store,
        bounce_arena,
        registration_transfer,
        replicate_config=replicate_config,
        read_buffer_bytes=read_buffer_bytes,
        use_fabric_mem=use_fabric_mem,
        use_store_independent_te=use_store_independent,
    )
