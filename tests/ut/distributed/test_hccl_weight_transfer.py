# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for HCCL weight transfer engine backend.

Unit tests for engine classes (parsing, validation, registry).
Integration tests for HCCL weight transfer between processes using Ray.
"""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer import WeightTransferEngineFactory

from vllm_ascend.distributed.weight_transfer import register_engine
from vllm_ascend.distributed.weight_transfer.hccl_engine import (
    HCCLTrainerSendWeightsArgs,
    HCCLWeightTransferEngine,
    HCCLWeightTransferInitInfo,
    HCCLWeightTransferUpdateInfo,
)
from vllm_ascend.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    DEFAULT_PACKED_NUM_BUFFERS,
)


@pytest.fixture(autouse=True, scope="module")
def _register_hccl_engine():
    """Ensure HCCL engine is registered before tests (idempotent)."""
    try:
        register_engine()
    except ValueError:
        # Already registered from a previous test module
        pass


def create_mock_parallel_config(
    rank: int = 0,
    world_size: int = 1,
    dp_rank: int = 0,
) -> ParallelConfig:
    """Create a mock ParallelConfig for testing."""
    config = MagicMock(spec=ParallelConfig)
    config.rank = rank
    config.world_size = world_size
    config.data_parallel_rank = dp_rank
    config.data_parallel_index = dp_rank
    return config


def create_mock_weight_transfer_config(backend: str = "hccl") -> MagicMock:
    """Create a mock WeightTransferConfig for testing.

    Uses MagicMock instead of real WeightTransferConfig because "hccl"
    is not in the upstream Pydantic Literal type. In production, vllm-ascend
    patches the config to accept "hccl".
    """
    config = MagicMock(spec=WeightTransferConfig)
    config.backend = backend
    return config


# ---------------------------------------------------------------------------
# Unit Tests: HCCLWeightTransferUpdateInfo Validation
# ---------------------------------------------------------------------------


class TestHCCLWeightTransferUpdateInfoValidation:
    """Test HCCLWeightTransferUpdateInfo dataclass validation."""

    def test_valid_update_info(self):
        """Test creating valid HCCLWeightTransferUpdateInfo."""
        info = HCCLWeightTransferUpdateInfo(
            names=["layer.weight", "layer.bias"],
            dtype_names=["float32", "float32"],
            shapes=[[10, 10], [10]],
        )
        assert info.names == ["layer.weight", "layer.bias"]
        assert info.dtype_names == ["float32", "float32"]
        assert info.shapes == [[10, 10], [10]]
        assert info.packed is False

    def test_mismatched_dtype_names_raises(self):
        """Test that mismatched dtype_names length raises ValueError."""
        with pytest.raises(ValueError, match="dtype_names"):
            HCCLWeightTransferUpdateInfo(
                names=["layer.weight", "layer.bias"],
                dtype_names=["float32"],  # Only one dtype
                shapes=[[10, 10], [10]],
            )

    def test_mismatched_shapes_raises(self):
        """Test that mismatched shapes length raises ValueError."""
        with pytest.raises(ValueError, match="shapes"):
            HCCLWeightTransferUpdateInfo(
                names=["layer.weight", "layer.bias"],
                dtype_names=["float32", "float32"],
                shapes=[[10, 10]],  # Only one shape
            )

    def test_empty_lists_valid(self):
        """Test that empty lists are valid."""
        info = HCCLWeightTransferUpdateInfo(
            names=[],
            dtype_names=[],
            shapes=[],
        )
        assert len(info.names) == 0

    def test_packed_default_false(self):
        """Test that packed defaults to False."""
        info = HCCLWeightTransferUpdateInfo(
            names=["layer.weight"],
            dtype_names=["float32"],
            shapes=[[10, 10]],
        )
        assert info.packed is False

    def test_packed_can_be_set_true(self):
        """Test that packed can be set to True."""
        info = HCCLWeightTransferUpdateInfo(
            names=["layer.weight"],
            dtype_names=["float32"],
            shapes=[[10, 10]],
            packed=True,
        )
        assert info.packed is True

    def test_packed_buffer_size_default(self):
        """Test default packed_buffer_size_bytes."""
        info = HCCLWeightTransferUpdateInfo(
            names=["w"],
            dtype_names=["float32"],
            shapes=[[10]],
        )
        assert info.packed_buffer_size_bytes == DEFAULT_PACKED_BUFFER_SIZE_BYTES

    def test_packed_num_buffers_default(self):
        """Test default packed_num_buffers."""
        info = HCCLWeightTransferUpdateInfo(
            names=["w"],
            dtype_names=["float32"],
            shapes=[[10]],
        )
        assert info.packed_num_buffers == DEFAULT_PACKED_NUM_BUFFERS


# ---------------------------------------------------------------------------
# Unit Tests: HCCLWeightTransferInitInfo
# ---------------------------------------------------------------------------


class TestHCCLWeightTransferInitInfo:
    """Test HCCLWeightTransferInitInfo dataclass."""

    def test_valid_init_info(self):
        """Test creating valid HCCLWeightTransferInitInfo."""
        info = HCCLWeightTransferInitInfo(
            master_address="127.0.0.1",
            master_port=12345,
            rank_offset=1,
            world_size=3,
        )
        assert info.master_address == "127.0.0.1"
        assert info.master_port == 12345
        assert info.rank_offset == 1
        assert info.world_size == 3


# ---------------------------------------------------------------------------
# Unit Tests: HCCLTrainerSendWeightsArgs
# ---------------------------------------------------------------------------


class TestHCCLTrainerSendWeightsArgs:
    """Test HCCLTrainerSendWeightsArgs dataclass defaults."""

    def test_defaults(self):
        """Test default values for trainer send weights args."""
        mock_group = MagicMock()
        args = HCCLTrainerSendWeightsArgs(group=mock_group)
        assert args.group is mock_group
        assert args.src == 0
        assert args.post_iter_func is None
        assert args.packed is False
        assert args.stream is None
        assert args.packed_buffer_size_bytes == DEFAULT_PACKED_BUFFER_SIZE_BYTES
        assert args.packed_num_buffers == DEFAULT_PACKED_NUM_BUFFERS

    def test_packed_can_be_set_true(self):
        """Test packed can be enabled."""
        mock_group = MagicMock()
        args = HCCLTrainerSendWeightsArgs(group=mock_group, packed=True)
        assert args.packed is True

    def test_custom_post_iter_func(self):
        """Test custom post_iter_func can be set."""
        mock_group = MagicMock()
        custom_func = lambda x: x[1]
        args = HCCLTrainerSendWeightsArgs(
            group=mock_group, post_iter_func=custom_func
        )
        assert args.post_iter_func is custom_func


# ---------------------------------------------------------------------------
# Unit Tests: HCCL Engine Parsing
# ---------------------------------------------------------------------------


class TestHCCLEngineParsing:
    """Test HCCLWeightTransferEngine parsing methods."""

    def test_parse_init_info_valid(self):
        """Test parsing valid init info dict."""
        config = create_mock_weight_transfer_config()
        parallel_config = create_mock_parallel_config()
        engine = HCCLWeightTransferEngine(config, parallel_config)

        init_info = engine.parse_init_info(
            {
                "master_address": "127.0.0.1",
                "master_port": 12345,
                "rank_offset": 1,
                "world_size": 3,
            }
        )

        assert isinstance(init_info, HCCLWeightTransferInitInfo)
        assert init_info.master_address == "127.0.0.1"
        assert init_info.master_port == 12345
        assert init_info.rank_offset == 1
        assert init_info.world_size == 3

    def test_parse_init_info_missing_field_raises(self):
        """Test parsing init info with missing required field."""
        config = create_mock_weight_transfer_config()
        parallel_config = create_mock_parallel_config()
        engine = HCCLWeightTransferEngine(config, parallel_config)

        with pytest.raises(ValueError, match="Invalid init_info"):
            engine.parse_init_info(
                {
                    "master_address": "127.0.0.1",
                    # Missing master_port, rank_offset, world_size
                }
            )

    def test_parse_update_info_valid(self):
        """Test parsing valid update info dict."""
        config = create_mock_weight_transfer_config()
        parallel_config = create_mock_parallel_config()
        engine = HCCLWeightTransferEngine(config, parallel_config)

        update_info = engine.parse_update_info(
            {
                "names": ["w1", "w2"],
                "dtype_names": ["float32", "bfloat16"],
                "shapes": [[100, 100], [50]],
            }
        )

        assert isinstance(update_info, HCCLWeightTransferUpdateInfo)
        assert update_info.names == ["w1", "w2"]
        assert update_info.dtype_names == ["float32", "bfloat16"]
        assert update_info.shapes == [[100, 100], [50]]
        assert update_info.packed is False

    def test_parse_update_info_with_packed(self):
        """Test parsing update info dict with packed enabled."""
        config = create_mock_weight_transfer_config()
        parallel_config = create_mock_parallel_config()
        engine = HCCLWeightTransferEngine(config, parallel_config)

        update_info = engine.parse_update_info(
            {
                "names": ["w1"],
                "dtype_names": ["float32"],
                "shapes": [[100, 100]],
                "packed": True,
                "packed_buffer_size_bytes": 1024,
                "packed_num_buffers": 3,
            }
        )

        assert isinstance(update_info, HCCLWeightTransferUpdateInfo)
        assert update_info.packed is True
        assert update_info.packed_buffer_size_bytes == 1024
        assert update_info.packed_num_buffers == 3


# ---------------------------------------------------------------------------
# Unit Tests: Engine Registry
# ---------------------------------------------------------------------------


class TestEngineRegistry:
    """Test weight transfer engine registry for HCCL backend."""

    def test_create_engine_hccl(self):
        """Test factory creates HCCL engine."""
        config = create_mock_weight_transfer_config()
        parallel_config = create_mock_parallel_config()
        engine = WeightTransferEngineFactory.create_engine(config, parallel_config)
        assert isinstance(engine, HCCLWeightTransferEngine)

    def test_create_engine_nccl_still_works(self):
        """Test factory still creates NCCL engine for 'nccl' backend."""
        config = create_mock_weight_transfer_config(backend="nccl")
        parallel_config = create_mock_parallel_config()
        engine = WeightTransferEngineFactory.create_engine(config, parallel_config)
        # NCCL engine should be importable
        from vllm.distributed.weight_transfer.nccl_engine import (
            NCCLWeightTransferEngine,
        )
        assert isinstance(engine, NCCLWeightTransferEngine)

    def test_register_duplicate_raises(self):
        """Test registering duplicate engine name raises."""
        with pytest.raises(ValueError, match="already registered"):
            WeightTransferEngineFactory.register_engine(
                "hccl", HCCLWeightTransferEngine
            )

    def test_create_engine_invalid_backend(self):
        """Test factory raises for invalid backend."""
        from pydantic import ValidationError

        # Pydantic prevents invalid backend at construction
        with pytest.raises(ValidationError):
            WeightTransferConfig(backend="invalid_backend_name")

        # Test factory error by creating config with valid backend then
        # bypassing Pydantic validation
        config = WeightTransferConfig(backend="nccl")
        object.__setattr__(config, "backend", "invalid_backend_name")
        parallel_config = create_mock_parallel_config()
        with pytest.raises(ValueError, match="Invalid weight transfer backend"):
            WeightTransferEngineFactory.create_engine(config, parallel_config)


# ---------------------------------------------------------------------------
# Error Case: receive_weights without init
# ---------------------------------------------------------------------------


def test_hccl_receive_weights_without_init_raises():
    """Test that receive_weights raises if init_transfer_engine wasn't called."""
    if torch.accelerator.device_count() < 1:
        pytest.skip("Need at least 1 NPU for this test")

    config = create_mock_weight_transfer_config()
    parallel_config = create_mock_parallel_config()
    engine = HCCLWeightTransferEngine(config, parallel_config)

    update_info = HCCLWeightTransferUpdateInfo(
        names=["w"],
        dtype_names=["float32"],
        shapes=[[10]],
    )

    with pytest.raises(RuntimeError, match="not initialized"):
        engine.receive_weights(update_info, lambda x: None)


# ---------------------------------------------------------------------------
# Unit Tests: Non-packed weight transfer (with mocked communicator)
# ---------------------------------------------------------------------------


def test_hccl_receive_weights_non_packed():
    """Test receive_weights in non-packed mode with mocked communicator."""
    if torch.accelerator.device_count() < 1:
        pytest.skip("Need at least 1 NPU for this test")

    config = create_mock_weight_transfer_config()
    parallel_config = create_mock_parallel_config()
    engine = HCCLWeightTransferEngine(config, parallel_config)

    # Mock the model_update_group
    mock_group = MagicMock()
    engine.model_update_group = mock_group

    # Set up update info for two tensors
    update_info = HCCLWeightTransferUpdateInfo(
        names=["w1", "w2"],
        dtype_names=["float32", "float16"],
        shapes=[[100, 100], [50]],
        packed=False,
    )

    received = []

    def load_weights(weights):
        for name, tensor in weights:
            received.append((name, tensor.shape, tensor.dtype))

    engine.receive_weights(update_info, load_weights)

    # Verify broadcast was called for each tensor
    assert mock_group.broadcast.call_count == 2

    # Verify load_weights received both tensors
    assert len(received) == 2
    assert received[0][0] == "w1"
    assert received[0][1] == (100, 100)
    assert received[0][2] == torch.float32
    assert received[1][0] == "w2"
    assert received[1][1] == (50,)
    assert received[1][2] == torch.float16


def test_hccl_trainer_send_weights_non_packed():
    """Test trainer_send_weights in non-packed mode with mocked communicator."""
    if torch.accelerator.device_count() < 1:
        pytest.skip("Need at least 1 NPU for this test")

    # Create mock group
    mock_group = MagicMock()

    # Create params as real tensors on NPU
    params = [
        ("w1", torch.ones(10, 10, dtype=torch.float32, device="npu")),
        ("w2", torch.ones(5, dtype=torch.float16, device="npu")),
    ]

    args = HCCLTrainerSendWeightsArgs(group=mock_group, src=0)
    HCCLWeightTransferEngine.trainer_send_weights(iter(params), args)

    # Verify broadcast was called for each tensor
    assert mock_group.broadcast.call_count == 2


def test_hccl_trainer_send_weights_with_dict_args():
    """Test trainer_send_weights accepts dict arguments."""
    if torch.accelerator.device_count() < 1:
        pytest.skip("Need at least 1 NPU for this test")

    mock_group = MagicMock()

    params = [
        ("w1", torch.ones(10, 10, dtype=torch.float32, device="npu")),
    ]

    dict_args = {"group": mock_group, "src": 0}
    HCCLWeightTransferEngine.trainer_send_weights(iter(params), dict_args)

    assert mock_group.broadcast.call_count == 1


def test_hccl_shutdown_clears_group():
    """Test shutdown clears the model_update_group."""
    config = create_mock_weight_transfer_config()
    parallel_config = create_mock_parallel_config()
    engine = HCCLWeightTransferEngine(config, parallel_config)

    engine.model_update_group = MagicMock()
    engine.shutdown()
    assert engine.model_update_group is None


# ---------------------------------------------------------------------------
# Integration Test: HCCL Weight Transfer Between Ray Tasks
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 NPUs to run HCCL weight transfer test.",
)
def test_hccl_weight_transfer_between_processes():
    """Test HCCL weight transfer from trainer to inference process using Ray.

    This test verifies that the HCCLWeightTransferEngine can receive
    tensors broadcast by a trainer process via HCCL.
    """
    import ray

    from vllm.utils.network_utils import get_open_port

    ray.init(ignore_reinit_error=True)

    @ray.remote(num_gpus=1)
    def trainer_broadcast_tensor(
        master_address: str,
        master_port: int,
        world_size: int,
        tensor_shape: list[int],
        tensor_dtype: str,
    ) -> bool:
        """Trainer task that broadcasts a tensor via HCCL."""
        import torch

        from vllm.distributed.utils import StatelessProcessGroup
        from vllm_ascend.distributed.device_communicators.pyhccl import (
            PyHcclCommunicator,
        )

        pg = StatelessProcessGroup.create(
            host=master_address,
            port=master_port,
            rank=0,
            world_size=world_size,
        )
        device = torch.accelerator.current_device_index()
        comm = PyHcclCommunicator(pg, device=device)

        dtype = getattr(torch, tensor_dtype)
        tensor_to_send = torch.ones(tensor_shape, dtype=dtype, device="npu")
        comm.broadcast(
            tensor_to_send, src=0, stream=torch.npu.current_stream()
        )
        torch.accelerator.synchronize()

        return True

    @ray.remote(num_gpus=1)
    def inference_receive_tensor(
        master_address: str,
        master_port: int,
        world_size: int,
        tensor_shape: list[int],
        tensor_dtype: str,
    ) -> dict:
        """Inference task that receives tensor via HCCLWeightTransferEngine."""
        from unittest.mock import MagicMock

        import torch

        from vllm.config.parallel import ParallelConfig
        from vllm_ascend.distributed.weight_transfer.hccl_engine import (
            HCCLWeightTransferEngine,
            HCCLWeightTransferInitInfo,
            HCCLWeightTransferUpdateInfo,
        )

        config = MagicMock()
        config.backend = "hccl"

        parallel_config = MagicMock(spec=ParallelConfig)
        parallel_config.rank = 0
        parallel_config.world_size = 1
        parallel_config.data_parallel_rank = 0
        parallel_config.data_parallel_index = 0

        engine = HCCLWeightTransferEngine(config, parallel_config)

        init_info = HCCLWeightTransferInitInfo(
            master_address=master_address,
            master_port=master_port,
            rank_offset=1,
            world_size=world_size,
        )
        engine.init_transfer_engine(init_info)

        received_tensors = []

        def noop_load_weights(weights: list[tuple[str, torch.Tensor]]):
            for name, tensor in weights:
                received_tensors.append((name, tensor.clone()))

        update_info = HCCLWeightTransferUpdateInfo(
            names=["test.weight"],
            dtype_names=[tensor_dtype],
            shapes=[tensor_shape],
        )
        engine.receive_weights(update_info, noop_load_weights)
        torch.accelerator.synchronize()

        success = False
        received_shape = None
        received_sum = None

        if len(received_tensors) == 1:
            name, tensor = received_tensors[0]
            received_shape = list(tensor.shape)
            received_sum = tensor.sum().item()
            if received_shape == tensor_shape:
                expected_sum = 1.0 * torch.tensor(tensor_shape).prod().item()
                if abs(received_sum - expected_sum) < 0.01:
                    success = True

        engine.shutdown()

        return {
            "success": success,
            "received_shape": received_shape,
            "received_sum": received_sum,
        }

    master_address = "127.0.0.1"
    master_port = get_open_port()
    world_size = 2

    tensor_shape = [100, 100]
    tensor_dtype = "float32"

    inference_future = inference_receive_tensor.remote(
        master_address, master_port, world_size, tensor_shape, tensor_dtype
    )
    trainer_future = trainer_broadcast_tensor.remote(
        master_address, master_port, world_size, tensor_shape, tensor_dtype
    )

    trainer_result, result = ray.get([trainer_future, inference_future])

    assert trainer_result, "Trainer should complete successfully"
    assert result["success"], (
        f"Weight transfer failed. "
        f"Received shape: {result['received_shape']}, "
        f"Received sum: {result['received_sum']}"
    )


# ---------------------------------------------------------------------------
# Packed Tensor Unit Tests (NPU)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    torch.accelerator.device_count() < 1,
    reason="Need at least 1 NPU for packed tensor tests.",
)
class TestPackedBroadcastProducer:
    """Test packed_broadcast_producer function with NPU."""

    @staticmethod
    def _create_params(
        num_layers: int = 3,
        dtype: torch.dtype = torch.float32,
    ) -> list[tuple[str, torch.Tensor]]:
        """Create mock model parameters on NPU."""
        params = []
        for i in range(num_layers):
            params.append((f"layer{i}.weight", torch.randn(10, 20, dtype=dtype,
                             device="npu")))
            params.append((f"layer{i}.bias", torch.randn(10, dtype=dtype,
                             device="npu")))
        return params

    def test_producer_broadcasts_tensors(self):
        """Test that producer broadcasts all tensors."""
        from vllm_ascend.distributed.weight_transfer.packed_tensor import (
            packed_broadcast_producer,
        )

        params = self._create_params()
        mock_group = MagicMock()
        mock_group.device = torch.device("npu:0")

        packed_broadcast_producer(
            iterator=iter(params),
            group=mock_group,
            src=0,
            post_iter_func=lambda x: x[1],
            buffer_size_bytes=500,
        )

        assert mock_group.broadcast.call_count > 0

    def test_producer_single_large_tensor(self):
        """Test with a single tensor larger than target size."""
        from vllm_ascend.distributed.weight_transfer.packed_tensor import (
            packed_broadcast_producer,
        )

        large_tensor = torch.randn(1000, 1000, dtype=torch.float32, device="npu")
        params = [("large_weight", large_tensor)]

        mock_group = MagicMock()
        mock_group.device = torch.device("npu:0")

        packed_broadcast_producer(
            iterator=iter(params),
            group=mock_group,
            src=0,
            post_iter_func=lambda x: x[1],
            buffer_size_bytes=100,
        )

        assert mock_group.broadcast.call_count >= 1

        # Verify the total broadcasted size matches the tensor
        broadcasted_tensors = [
            call.args[0]
            for call in mock_group.broadcast.call_args_list
        ]
        actual_size = sum(t.numel() for t in broadcasted_tensors)
        expected_size = large_tensor.numel() * large_tensor.element_size()
        assert actual_size == expected_size

    def test_producer_multiple_batches(self):
        """Test that tensors are properly batched when exceeding target size."""
        from vllm_ascend.distributed.weight_transfer.packed_tensor import (
            packed_broadcast_producer,
        )

        params = [
            (f"weight_{i}", torch.randn(10, 10, dtype=torch.float32, device="npu"))
            for i in range(20)
        ]

        mock_group = MagicMock()
        mock_group.device = torch.device("npu:0")

        packed_broadcast_producer(
            iterator=iter(params),
            group=mock_group,
            src=0,
            post_iter_func=lambda x: x[1],
            buffer_size_bytes=2000,
        )

        assert mock_group.broadcast.call_count > 1

        broadcasted_tensors = [
            call.args[0]
            for call in mock_group.broadcast.call_args_list
        ]
        expected_total = sum(
            t.numel() * t.element_size() for _, t in params
        )
        actual_total = sum(t.numel() for t in broadcasted_tensors)
        assert actual_total == expected_total

    def test_producer_empty_iterator(self):
        """Test producer handles empty iterator gracefully."""
        from vllm_ascend.distributed.weight_transfer.packed_tensor import (
            packed_broadcast_producer,
        )

        mock_group = MagicMock()
        mock_group.device = torch.device("npu:0")

        packed_broadcast_producer(
            iterator=iter([]),
            group=mock_group,
            src=0,
            post_iter_func=lambda x: x[1],
            buffer_size_bytes=1000,
        )

        assert mock_group.broadcast.call_count == 0


@pytest.mark.skipif(
    torch.accelerator.device_count() < 1,
    reason="Need at least 1 NPU for packed tensor tests.",
)
class TestPackedBroadcastConsumer:
    """Test packed_broadcast_consumer function with NPU."""

    @staticmethod
    def _create_params(
        num_layers: int = 3,
        dtype: torch.dtype = torch.float32,
    ) -> list[tuple[str, torch.Tensor]]:
        params = []
        for i in range(num_layers):
            params.append((f"layer{i}.weight", torch.randn(10, 20, dtype=dtype,
                             device="npu")))
            params.append((f"layer{i}.bias", torch.randn(10, dtype=dtype,
                             device="npu")))
        return params

    @staticmethod
    def _state_dict_info(params):
        return {
            name: (tuple(tensor.shape), tensor.dtype)
            for name, tensor in params
        }

    def test_consumer_receives_tensors(self):
        """Test that consumer receives and unpacks tensors."""
        from vllm_ascend.distributed.weight_transfer.packed_tensor import (
            packed_broadcast_consumer,
            packed_broadcast_producer,
        )

        params = self._create_params()
        buffer_size = 2000

        # First run producer to capture broadcasted tensors
        producer_group = MagicMock()
        producer_group.device = torch.device("npu:0")
        producer_broadcasted = []

        def producer_broadcast(tensor, src):
            producer_broadcasted.append(tensor.clone())

        producer_group.broadcast = producer_broadcast

        packed_broadcast_producer(
            iterator=iter(params),
            group=producer_group,
            src=0,
            post_iter_func=lambda x: x[1],
            buffer_size_bytes=buffer_size,
        )

        # Now run consumer with captured tensors
        consumer_group = MagicMock()
        consumer_group.device = torch.device("npu:0")
        call_idx = [0]

        def consumer_broadcast(tensor, src):
            if call_idx[0] < len(producer_broadcasted):
                tensor.copy_(producer_broadcasted[call_idx[0]])
                call_idx[0] += 1

        consumer_group.broadcast = consumer_broadcast

        state_dict_info = self._state_dict_info(params)
        unpacked = {}

        def post_unpack_func(tensor_list):
            for name, tensor in tensor_list:
                unpacked[name] = tensor.clone()

        packed_broadcast_consumer(
            iterator=iter(state_dict_info.items()),
            group=consumer_group,
            src=0,
            post_unpack_func=post_unpack_func,
            buffer_size_bytes=buffer_size,
        )

        assert len(unpacked) == len(params)
        for name, original_tensor in params:
            assert name in unpacked
            assert unpacked[name].shape == original_tensor.shape
            assert unpacked[name].dtype == original_tensor.dtype
            assert torch.allclose(
                unpacked[name], original_tensor, rtol=1e-5, atol=1e-7
            )


@pytest.mark.skipif(
    torch.accelerator.device_count() < 1,
    reason="Need at least 1 NPU for packed tensor roundtrip tests.",
)
class TestPackedBroadcastRoundtrip:
    """Test producer-consumer roundtrip with different configurations."""

    @staticmethod
    def _create_params(
        num_layers: int = 2,
        dtype: torch.dtype = torch.float32,
    ) -> list[tuple[str, torch.Tensor]]:
        params = []
        for i in range(num_layers):
            params.append((f"layer{i}.weight", torch.randn(10, 20, dtype=dtype,
                             device="npu")))
            params.append((f"layer{i}.bias", torch.randn(10, dtype=dtype,
                             device="npu")))
        return params

    @staticmethod
    def _state_dict_info(params):
        return {
            name: (tuple(tensor.shape), tensor.dtype)
            for name, tensor in params
        }

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_roundtrip_different_dtypes(self, dtype):
        """Test roundtrip with different data types."""
        from vllm_ascend.distributed.weight_transfer.packed_tensor import (
            packed_broadcast_consumer,
            packed_broadcast_producer,
        )

        params = self._create_params(dtype=dtype)
        buffer_size = 1000

        producer_group = MagicMock()
        producer_group.device = torch.device("npu:0")
        producer_broadcasted = []

        def producer_broadcast(tensor, src):
            producer_broadcasted.append(tensor.clone())

        producer_group.broadcast = producer_broadcast

        packed_broadcast_producer(
            iterator=iter(params),
            group=producer_group,
            src=0,
            post_iter_func=lambda x: x[1],
            buffer_size_bytes=buffer_size,
        )

        consumer_group = MagicMock()
        consumer_group.device = torch.device("npu:0")
        call_idx = [0]

        def consumer_broadcast(tensor, src):
            if call_idx[0] < len(producer_broadcasted):
                tensor.copy_(producer_broadcasted[call_idx[0]])
                call_idx[0] += 1

        consumer_group.broadcast = consumer_broadcast

        state_dict_info = self._state_dict_info(params)
        unpacked = {}

        def post_unpack_func(tensor_list):
            for name, tensor in tensor_list:
                unpacked[name] = tensor.clone()

        packed_broadcast_consumer(
            iterator=iter(state_dict_info.items()),
            group=consumer_group,
            src=0,
            post_unpack_func=post_unpack_func,
            buffer_size_bytes=buffer_size,
        )

        for name, original_tensor in params:
            assert name in unpacked
            assert unpacked[name].dtype == dtype
            assert torch.allclose(
                unpacked[name], original_tensor, rtol=1e-4, atol=1e-6
            )

    def test_roundtrip_mixed_dtypes(self):
        """Test roundtrip with mixed data types."""
        from vllm_ascend.distributed.weight_transfer.packed_tensor import (
            packed_broadcast_consumer,
            packed_broadcast_producer,
        )

        params = [
            ("layer1.weight", torch.randn(10, 20, dtype=torch.float32, device="npu")),
            ("layer1.bias", torch.randn(10, dtype=torch.float16, device="npu")),
            ("layer2.weight", torch.randn(20, 30, dtype=torch.bfloat16, device="npu")),
        ]

        buffer_size = 500

        producer_group = MagicMock()
        producer_group.device = torch.device("npu:0")
        producer_broadcasted = []

        def producer_broadcast(tensor, src):
            producer_broadcasted.append(tensor.clone())

        producer_group.broadcast = producer_broadcast

        packed_broadcast_producer(
            iterator=iter(params),
            group=producer_group,
            src=0,
            post_iter_func=lambda x: x[1],
            buffer_size_bytes=buffer_size,
        )

        consumer_group = MagicMock()
        consumer_group.device = torch.device("npu:0")
        call_idx = [0]

        def consumer_broadcast(tensor, src):
            if call_idx[0] < len(producer_broadcasted):
                tensor.copy_(producer_broadcasted[call_idx[0]])
                call_idx[0] += 1

        consumer_group.broadcast = consumer_broadcast

        state_dict_info = self._state_dict_info(params)
        unpacked = {}

        def post_unpack_func(tensor_list):
            for name, tensor in tensor_list:
                unpacked[name] = tensor.clone()

        packed_broadcast_consumer(
            iterator=iter(state_dict_info.items()),
            group=consumer_group,
            src=0,
            post_unpack_func=post_unpack_func,
            buffer_size_bytes=buffer_size,
        )

        for name, original_tensor in params:
            assert name in unpacked
            assert unpacked[name].shape == original_tensor.shape
            assert unpacked[name].dtype == original_tensor.dtype
            assert torch.allclose(
                unpacked[name], original_tensor, rtol=1e-4, atol=1e-6
            )

    @pytest.mark.parametrize("target_size", [100, 1000, 10000, 100000])
    def test_roundtrip_different_batch_sizes(self, target_size):
        """Test roundtrip with different target batch sizes."""
        from vllm_ascend.distributed.weight_transfer.packed_tensor import (
            packed_broadcast_consumer,
            packed_broadcast_producer,
        )

        params = self._create_params(num_layers=5)

        producer_group = MagicMock()
        producer_group.device = torch.device("npu:0")
        producer_broadcasted = []

        def producer_broadcast(tensor, src):
            producer_broadcasted.append(tensor.clone())

        producer_group.broadcast = producer_broadcast

        packed_broadcast_producer(
            iterator=iter(params),
            group=producer_group,
            src=0,
            post_iter_func=lambda x: x[1],
            buffer_size_bytes=target_size,
        )

        consumer_group = MagicMock()
        consumer_group.device = torch.device("npu:0")
        call_idx = [0]

        def consumer_broadcast(tensor, src):
            if call_idx[0] < len(producer_broadcasted):
                tensor.copy_(producer_broadcasted[call_idx[0]])
                call_idx[0] += 1

        consumer_group.broadcast = consumer_broadcast

        state_dict_info = self._state_dict_info(params)
        unpacked = {}

        def post_unpack_func(tensor_list):
            for name, tensor in tensor_list:
                unpacked[name] = tensor.clone()

        packed_broadcast_consumer(
            iterator=iter(state_dict_info.items()),
            group=consumer_group,
            src=0,
            post_unpack_func=post_unpack_func,
            buffer_size_bytes=target_size,
        )

        assert len(unpacked) == len(params)
        for name, original_tensor in params:
            assert name in unpacked
            assert torch.allclose(
                unpacked[name], original_tensor, rtol=1e-5, atol=1e-7
            )

    def test_roundtrip_non_contiguous_tensors(self):
        """Test roundtrip with non-contiguous tensors from the trainer."""
        from vllm_ascend.distributed.weight_transfer.packed_tensor import (
            packed_broadcast_consumer,
            packed_broadcast_producer,
        )

        weight1 = torch.randn(20, 10, dtype=torch.float32, device="npu").T
        weight2 = torch.randn(40, 30, dtype=torch.float16, device="npu")[::2, ::2]
        weight3 = (
            torch.randn(5, 10, 15, dtype=torch.bfloat16, device="npu")
            .permute(2, 0, 1)
        )

        params = [
            ("layer1.weight", weight1),
            ("layer2.weight", weight2),
            ("layer3.weight", weight3),
        ]

        for name, tensor in params:
            assert not tensor.is_contiguous(), f"{name} should be non-contiguous"

        buffer_size = 500

        producer_group = MagicMock()
        producer_group.device = torch.device("npu:0")
        producer_broadcasted = []

        def producer_broadcast(tensor, src):
            producer_broadcasted.append(tensor.clone())

        producer_group.broadcast = producer_broadcast

        packed_broadcast_producer(
            iterator=iter(params),
            group=producer_group,
            src=0,
            post_iter_func=lambda x: x[1],
            buffer_size_bytes=buffer_size,
        )

        consumer_group = MagicMock()
        consumer_group.device = torch.device("npu:0")
        call_idx = [0]

        def consumer_broadcast(tensor, src):
            if call_idx[0] < len(producer_broadcasted):
                tensor.copy_(producer_broadcasted[call_idx[0]])
                call_idx[0] += 1

        consumer_group.broadcast = consumer_broadcast

        state_dict_info = self._state_dict_info(params)
        unpacked = {}

        def post_unpack_func(tensor_list):
            for name, tensor in tensor_list:
                unpacked[name] = tensor.clone()

        packed_broadcast_consumer(
            iterator=iter(state_dict_info.items()),
            group=consumer_group,
            src=0,
            post_unpack_func=post_unpack_func,
            buffer_size_bytes=buffer_size,
        )

        for name, original_tensor in params:
            assert name in unpacked
            assert unpacked[name].shape == original_tensor.shape
            assert unpacked[name].dtype == original_tensor.dtype
            assert torch.allclose(
                unpacked[name], original_tensor, rtol=1e-4, atol=1e-6
            )
