import pytest
import torch

from vllm_ascend.ops.moe.moe_mlp import cumsum_group_list


# Test configuration: Cover all supported type conversion combinations
@pytest.mark.parametrize(
    "src_type, dst_type, input_tensor, kwargs, expected_output",
    [
        # 1. Same source and destination type (0→0)
        (
            0,
            0,
            torch.tensor([1, 3, 5, 7]),
            {},
            torch.tensor([1, 3, 5, 7]),
        ),
        # 2. Same source and destination type (1→1)
        (
            1,
            1,
            torch.tensor([2, 4, 6]),
            {},
            torch.tensor([2, 4, 6]),
        ),
        # 3. Same source and destination type (2→2)
        (
            2,
            2,
            torch.tensor([[0, 2], [2, 3], [5, 1]]),
            {},
            torch.tensor([[0, 2], [2, 3], [5, 1]]),
        ),
        # 4. 1→0 (cumsum conversion)
        (
            1,
            0,
            torch.tensor([2, 1, 3, 4]),
            {},
            torch.tensor([2, 3, 6, 10]),
        ),
        # 5. 0→1 (difference conversion)
        (
            0,
            1,
            torch.tensor([2, 3, 6, 10]),
            {},
            torch.tensor([2, 1, 3, 4]),
        ),
        # 6. 2→0 (expert-token mapping conversion) - Basic scenario
        (
            2,
            0,
            torch.tensor([[0, 2], [2, 3], [5, 1]]),
            {
                "active_num": 0,
                "expert_num": 6
            },
            torch.tensor([2, 0, 3, 0, 0, 1]),
        ),
        # 7. 2→0 - Edge scenario (no expert interval)
        (
            2,
            0,
            torch.tensor([[1, 5], [3, 2], [4, 4]]),
            {
                "active_num": -1,
                "expert_num": 5
            },
            torch.tensor([-1, 5, -1, 2, 4]),
        ),
        # 8. 2→0 - Single expert
        (
            2,
            0,
            torch.tensor([[0, 10]]),
            {
                "active_num": 5,
                "expert_num": 1
            },
            torch.tensor([10]),
        ),
    ],
)
def test_cumsum_group_list_valid_cases(src_type, dst_type, input_tensor,
                                       kwargs, expected_output):
    """Test scenarios with valid type conversions"""
    result = cumsum_group_list(input_tensor, src_type, dst_type, **kwargs)
    # Verify result shape and values
    assert result.shape == expected_output.shape
    assert torch.allclose(result, expected_output)


def test_cumsum_group_list_invalid_src_type():
    """Test invalid source type"""
    input_tensor = torch.tensor([1, 2, 3])
    with pytest.raises(ValueError) as excinfo:
        cumsum_group_list(input_tensor, src_list_type=3, dst_list_type=0)
    assert "group_list_type should be in [0, 1, 2], but received 3" in str(
        excinfo.value)


def test_cumsum_group_list_unimplemented_conversion():
    """Test unimplemented type conversions"""
    input_tensor = torch.tensor([1, 2, 3])
    # Test 0→2 (unimplemented)
    with pytest.raises(NotImplementedError) as excinfo:
        cumsum_group_list(input_tensor, src_list_type=0, dst_list_type=2)
    assert "Conversion from src_list_type=0 to dst_list_type=2 is not implemented yet" in str(
        excinfo.value)

    # Test 1→2 (unimplemented)
    with pytest.raises(NotImplementedError):
        cumsum_group_list(input_tensor, src_list_type=1, dst_list_type=2)

    # Test 2→1 (unimplemented)
    input_2d = torch.tensor([[0, 1], [2, 3]])
    with pytest.raises(NotImplementedError):
        cumsum_group_list(input_2d, src_list_type=2, dst_list_type=1)


def test_cumsum_group_list_edge_cases():
    """Test edge cases"""
    # Empty tensor (1→0)
    empty_tensor = torch.tensor([], dtype=torch.int64)
    result = cumsum_group_list(empty_tensor, src_list_type=1, dst_list_type=0)
    assert torch.equal(result, empty_tensor)

    # Single-element tensor (0→1)
    single_tensor = torch.tensor([5])
    result = cumsum_group_list(single_tensor, src_list_type=0, dst_list_type=1)
    assert torch.equal(result, torch.tensor([5]))

    # 2→0 - Empty input
    empty_2d = torch.tensor([], dtype=torch.int64).reshape(0, 2)
    result = cumsum_group_list(empty_2d,
                               src_list_type=2,
                               dst_list_type=0,
                               active_num=0,
                               expert_num=3)
    assert torch.equal(result, torch.tensor([0, 0, 0]))


def test_cumsum_group_list_dtype_device_consistency():
    """Test consistency of output dtype and device with input"""
    # Test GPU (if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.tensor([[1, 2], [3, 4]],
                                dtype=torch.float32,
                                device=device)
    result = cumsum_group_list(
        input_tensor,
        src_list_type=2,
        dst_list_type=0,
        active_num=0.0,
        expert_num=4,
    )
    assert result.dtype == torch.float32
    assert result.device == device

    # Test int64 dtype
    input_int = torch.tensor([2, 4, 6], dtype=torch.int64)
    result_int = cumsum_group_list(input_int, src_list_type=0, dst_list_type=1)
    assert result_int.dtype == torch.int64
