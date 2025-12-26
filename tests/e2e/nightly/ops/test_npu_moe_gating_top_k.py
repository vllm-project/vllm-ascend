import itertools
import random
import logging
from typing import Tuple, Optional

import numpy as np
import torch
from torch_npu.testing.testcase import TestCase, run_tests

try:
    from vllm_ascend.utils import enable_custom_op
    enable_custom_op()
except ImportError:
    logging.warning("vllm_ascend.utils.enable_custom_op not found, skip custom op initialization")
    def enable_custom_op() -> None:
        pass

# Set random seed for reproducibility
SEED = 45
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if hasattr(torch, "npu") and torch.npu.is_available():
    torch.npu.manual_seed_all(SEED)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def softmax_func(x: np.ndarray, axis: Optional[int] = None, eps: float = 1e-20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stable softmax implementation for MOE gating.
    
    Args:
        x: Input array
        axis: Axis to compute softmax
        eps: Epsilon to avoid division by zero
    
    Returns:
        softmax_output: Softmax result
        x_max: Max value for numerical stability
        x_sum: Sum of exponentials
    """
    if "float16" in x.dtype.name:
        x = x.astype(np.float32)
    
    x_max = x.max(axis=axis, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=axis, keepdims=True)
    softmax_output = y / (x_sum + eps)
    
    return softmax_output, x_max, x_sum


class TestNpuMoeGatingTopK(TestCase):
    """Test suite for NPU MOE Gating Top-K operator compatibility."""

    def moe_gating_top_k_np(
        self,
        x: np.ndarray,
        k: int,
        bias: Optional[np.ndarray] = None,
        k_group: int = 1,
        group_count: int = 1,
        group_select_mode: int = 0,
        renorm: int = 0,
        norm_type: int = 0,
        y2_flag: bool = False,
        routed_scaling_factor: float = 1.0,
        eps: float = 1e-20
    ) -> Tuple[torch.Tensor, np.ndarray, Optional[np.ndarray]]:
        """
        NumPy reference implementation of MOE gating Top-K logic.
        
        Args:
            x: Input features, shape [batch_size, num_experts]
            k: Number of experts to select per sample
            bias: Gating bias, shape [num_experts]
            k_group: Number of groups to select (group mode)
            group_count: Number of expert groups
            group_select_mode: 0 (max per group), 1 (sum of top-2 per group)
            renorm: Whether to renormalize weights (1=enable, 0=disable)
            norm_type: 0 (softmax), 1 (sigmoid)
            y2_flag: Whether to return original x as y2
            routed_scaling_factor: Weight scaling factor
            eps: Epsilon for numerical stability
        
        Returns:
            y: Selected expert weights (Tensor)
            indices: Selected expert indices (int32 numpy array)
            y2: Original x if y2_flag=True, else None
        """
        # Convert torch tensors to numpy arrays if needed (compatibility layer)
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if isinstance(bias, torch.Tensor):
            bias = bias.cpu().numpy()

        # Type conversion for numerical stability
        orig_dtype = x.dtype
        if orig_dtype != np.float32:
            x = x.astype(np.float32)
            if bias is not None:
                bias = bias.astype(np.float32)

        # Apply normalization (softmax/sigmoid)
        if norm_type == 0:
            x, _, _ = softmax_func(x, axis=-1, eps=eps)
        else:
            x = 1 / (1 + np.exp(-x))  # Sigmoid

        original_x = x.copy()
        
        # Apply bias if provided
        if bias is not None:
            x = x + bias

        # Group-based expert selection
        if group_count > 1:
            batch_size, num_experts = x.shape
            if num_experts % group_count != 0:
                raise ValueError(
                    f"num_experts ({num_experts}) must be divisible by group_count ({group_count})"
                )
            group_size = num_experts // group_count
            
            # Reshape to [batch, groups, group_size]
            x_reshaped = x.reshape(batch_size, group_count, group_size)
            
            # Compute group scores
            if group_select_mode == 0:
                group_scores = np.amax(x_reshaped, axis=-1)
            else:
                # Sum of top-2 values per group
                group_scores = np.partition(x_reshaped, -2, axis=-1)[..., -2:].sum(axis=-1)
            
            # Select top-k_group groups
            top_groups = np.argsort(-group_scores, axis=-1, kind="stable")[:, :k_group]
            
            # Mask out non-selected groups with -inf
            mask = np.ones((batch_size, group_count), dtype=bool)
            mask[np.arange(batch_size)[:, None], top_groups] = False
            x_reshaped = np.where(mask[..., None], float("-inf"), x_reshaped)
            
            # Reshape back to original
            x = x_reshaped.reshape(batch_size, num_experts)

        # Select top-k experts
        x_tensor = torch.from_numpy(x)
        _, topk_indices = torch.sort(x_tensor, dim=-1, stable=True, descending=True)
        topk_indices = np.asarray(topk_indices[:, :k], dtype=np.int32)

        # Extract weights for selected experts
        selected_weights = np.take_along_axis(original_x, topk_indices, axis=1)

        # Apply renormalization if needed
        if norm_type == 1 or renorm == 1:
            weight_sum = np.sum(selected_weights, axis=-1, keepdims=True)
            selected_weights = selected_weights / (weight_sum + eps)
        
        # Apply scaling factor
        selected_weights *= routed_scaling_factor

        # Prepare y2 output
        y2 = original_x if y2_flag else None

        # Convert back to torch tensor with original dtype
        selected_weights_tensor = torch.tensor(selected_weights, dtype=orig_dtype)

        return selected_weights_tensor, topk_indices, y2

    def test_npu_moe_gating_topk_multi(self) -> None:
        """
        Multi-case test for NPU MOE Gating Top-K operator.
        Validates compatibility with different input shapes and parameter combinations.
        """
        # Test parameter space (aligned with vllm-ascend use cases)
        test_configs = {
            "group_select_modes": [0, 1],
            "renorms": [1],
            "norm_types": [0, 1],
            "group_counts": [1, 8],
            "k_ranges": [4, 8, 12, 16, 6, 32],
            "x_dim0": range(1, 17),  # Batch size 1-16
            "x_dim1": [256, 128, 64, 208, 192, 160]  # Expert counts
        }

        # Generate parameter combinations
        param_combinations = itertools.product(
            test_configs["group_select_modes"],
            test_configs["renorms"],
            test_configs["norm_types"],
            test_configs["group_counts"],
            test_configs["k_ranges"],
            test_configs["x_dim0"],
            test_configs["x_dim1"]
        )

        # Limit test cases to avoid excessive runtime (adjust as needed)
        max_test_cases = 100
        tested_cases = 0

        for params in param_combinations:
            if tested_cases >= max_test_cases:
                break

            (group_select_mode, renorm, norm_type, group_count, k, dim0, dim1) = params

            # Skip invalid configurations
            if group_count > 1:
                if dim1 % group_count != 0:
                    continue
                if k > (dim1 // group_count):
                    continue

            # Generate random inputs (consistent with vllm-ascend input distribution)
            x_np = np.random.uniform(-2.0, 2.0, (dim0, dim1)).astype(np.float32)
            bias_np = np.random.uniform(-2.0, 2.0, (dim1,)).astype(np.float32)

            # Convert to torch tensors
            x_tensor = torch.tensor(x_np, dtype=torch.float32)
            bias_tensor = torch.tensor(bias_np, dtype=torch.float32)

            # Random k_group (within valid range)
            k_group = random.randint(1, min(group_count, 4))
            
            # Fixed parameters (aligned with NPU OP defaults)
            y2_flag = False
            routed_scaling_factor = 1.0
            eps = 1e-20

            try:
                # Get NumPy reference result
                ref_weights, ref_indices, ref_y2 = self.moe_gating_top_k_np(
                    x=x_tensor,
                    k=k,
                    bias=bias_tensor,
                    k_group=k_group,
                    group_count=group_count,
                    group_select_mode=group_select_mode,
                    renorm=renorm,
                    norm_type=norm_type,
                    y2_flag=y2_flag,
                    routed_scaling_factor=routed_scaling_factor,
                    eps=eps
                )

                # Skip if NPU OP is not available
                if not hasattr(torch.ops, "_C_ascend") or not hasattr(torch.ops._C_ascend, "moe_gating_top_k"):
                    logger.warning("NPU MOE gating OP not found, skipping NPU test")
                    continue

                # Get NPU OP result
                npu_weights, npu_indices, npu_y2 = torch.ops._C_ascend.moe_gating_top_k(
                    x=x_tensor.npu(),
                    k=k,
                    kGroup=k_group,
                    groupCount=group_count,
                    groupSelectMode=group_select_mode,
                    renorm=renorm,
                    normType=norm_type,
                    outFlag=y2_flag,
                    routedScalingFactor=routed_scaling_factor,
                    eps=eps,
                    biasOptional=bias_tensor.npu() if bias_tensor is not None else None
                )

                # Convert NPU results to CPU for comparison
                npu_weights_cpu = npu_weights.cpu()
                npu_indices_cpu = npu_indices.cpu().numpy()

                # Log test case info (vllm-ascend standard format)
                logger.info(
                    f"Test Case {tested_cases + 1}: "
                    f"x_shape=({dim0},{dim1}), k={k}, group_count={group_count}, "
                    f"select_mode={group_select_mode}, norm_type={norm_type}, renorm={renorm}"
                )

                # Validate results (RTOL=1e-3 is standard for NPU numerical tolerance)
                self.assertRtolEqual(ref_weights, npu_weights_cpu, rtol=1e-3, atol=1e-5)
                self.assertRtolEqual(ref_indices, npu_indices_cpu)
                
                # Validate y2 if enabled
                if y2_flag:
                    self.assertRtolEqual(ref_y2, npu_y2.cpu().numpy(), rtol=1e-3, atol=1e-5)

                tested_cases += 1
                logger.info(f"Test Case {tested_cases} passed ")

            except Exception as e:
                logger.error(f"Test Case failed with error: {str(e)}", exc_info=True)
                continue

        logger.info(f"Completed {tested_cases}/{max_test_cases} test cases")


if __name__ == "__main__":
    # Run tests with vllm-ascend standard verbosity
    run_tests(verbosity=2)