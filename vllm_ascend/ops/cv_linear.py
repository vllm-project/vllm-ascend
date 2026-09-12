#
# CVLinearWrapper - Splits a Linear layer into quantize(Vector) + matmul(Cube)
#
import torch
import torch_npu

from vllm_ascend.quantization.methods import (
    AscendW8A8DynamicLinearMethod,
    AscendW8A8MXFP8DynamicLinearMethod,
)


class CVLinearWrapper:
    """
    Splits a Linear layer into quantize(Vector) + matmul(Cube).

    Automatically detects TP communication operations:
    - No communication (ReplicatedLinear): W8A8 is split into independent quantize + matmul
    - Has communication (ColumnParallelLinear with custom_op): automatically falls back to full forward

    Usage example:
        wrapper = CVLinearWrapper(linear)

        # Step 1: Quantize (Vector)
        q_quant, q_scale = wrapper.quantize(x)

        # Step 2: Matrix multiply (Cube)
        result = wrapper.matmul(q_quant, q_scale)
    """

    def __init__(self, linear):
        self.linear = linear

        # Detect whether TP communication operations exist
        self._has_communication = self._detect_communication(linear)

        # Detect quantization scheme
        # Handles two cases:
        # 1. linear.quant_method is directly AscendW8A8DynamicLinearMethod or AscendW8A8MXFP8DynamicLinearMethod
        # 2. linear.quant_method is a wrapper class, requiring .quant_method to get the actual quantization method
        self._quant_method = linear.quant_method
        self._is_w8a8_dynamic = self._detect_w8a8_dynamic(linear.quant_method)
        self._mxfp8_quant_method = self._detect_mxfp8_dynamic(linear.quant_method)
        self._is_mxfp8_dynamic = self._mxfp8_quant_method is not None

    @staticmethod
    def _detect_w8a8_dynamic(quant_method):
        """Detect whether the quantization method is W8A8 Dynamic"""
        # Case 1: quant_method is directly AscendW8A8DynamicLinearMethod
        if isinstance(quant_method, AscendW8A8DynamicLinearMethod):
            return True
        # Case 2: quant_method is a wrapper class, requiring .quant_method to get the actual method
        return hasattr(quant_method, "quant_method") and isinstance(
            quant_method.quant_method, AscendW8A8DynamicLinearMethod
        )

    @staticmethod
    def _detect_mxfp8_dynamic(quant_method):
        """Detect whether the quantization method is W8A8 MXFP8 Dynamic.

        Returns the inner scheme instance (needed for scale_alg/group_size)
        or None. isinstance also covers the DS block-quant subclass.
        """
        if isinstance(quant_method, AscendW8A8MXFP8DynamicLinearMethod):
            return quant_method
        if hasattr(quant_method, "quant_method") and isinstance(
            quant_method.quant_method, AscendW8A8MXFP8DynamicLinearMethod
        ):
            return quant_method.quant_method
        return None

    @staticmethod
    def _detect_communication(linear):
        """
        Detect whether the Linear layer has TP communication during forward.

        Criteria:
        - custom_op is None or CustomReplicatedOp: no TP communication
        - Other custom_op (e.g., MLPColumnParallelOp with all_gather): has TP communication
        - ColumnParallelLinear with gather_output=True: has all-gather communication
        Note: ColumnParallelLinear even with custom_op=None only communicates when gather_output=True.
              wq_b uses default gather_output=False, so no communication and can be split.
        """
        custom_op = getattr(linear, "custom_op", None)
        if custom_op is not None:
            from vllm_ascend.ops.linear_op import CustomReplicatedOp

            if not isinstance(custom_op, CustomReplicatedOp):
                return True

        return hasattr(linear, "gather_output") and linear.gather_output

    def quantize(self, x: torch.Tensor):
        """
        Execute only the quantization step (Vector operator).

        Args:
            x: Input tensor

        Returns:
            (quantized_x, pertoken_scale): Quantized tensor and scaling factor.
            For linear layers with communication or without quantization, returns (x, None).
        """
        if self._has_communication:
            return x, None

        if self._is_w8a8_dynamic:
            quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x)
            return quantized_x, pertoken_scale

        if self._is_mxfp8_dynamic:
            quantized_x, pertoken_scale = torch_npu.npu_dynamic_mx_quant(
                x,
                dst_type=torch.float8_e4m3fn,
                scale_alg=self._mxfp8_quant_method.dynamic_mx_quant_scale_alg,
            )
            return quantized_x, pertoken_scale

        return x, None

    def matmul(self, quantized_x: torch.Tensor, pertoken_scale=None, bias=None):
        """
        Execute only the matrix multiplication step (Cube operator).

        Args:
            quantized_x: Quantized input (original input when communication is present)
            pertoken_scale: Per-token scaling factor for W8A8_DYNAMIC
            bias: Bias

        Returns:
            Matrix multiplication result
        """
        if self._has_communication:
            return self.linear.forward(quantized_x)

        if self._is_w8a8_dynamic:
            need_unsqz = False
            if pertoken_scale is not None and pertoken_scale.dim() == 2:
                need_unsqz = True
                quantized_x = quantized_x.squeeze(dim=1)
                pertoken_scale = pertoken_scale.squeeze(dim=1)

            output = torch_npu.npu_quant_matmul(
                quantized_x,
                self.linear.weight,
                self.linear.weight_scale,
                pertoken_scale=pertoken_scale,
                bias=bias,
                output_dtype=self.linear.weight_scale.dtype,
            )

            if need_unsqz:
                output = output.unsqueeze(dim=1)
            return output

        if self._is_mxfp8_dynamic:
            # Mirrors the pre-quantized path of the MXFP8 scheme's apply()
            if bias is not None and bias.dtype != torch.float32:
                bias = bias.to(torch.float32)
            return torch_npu.npu_quant_matmul(
                quantized_x,
                self.linear.weight,
                self.linear.weight_scale,
                scale_dtype=torch_npu.float8_e8m0fnu,
                pertoken_scale=pertoken_scale,
                pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
                bias=bias,
                output_dtype=torch.bfloat16,
                group_sizes=[1, 1, self._mxfp8_quant_method.group_size],
            )

        return self.linear.quant_method.apply(self.linear, quantized_x, bias)

    def forward(self, x: torch.Tensor, bias=None):
        """Full forward (equivalent to the original Linear.forward)"""
        q_quant, q_scale = self.quantize(x)
        return self.matmul(q_quant, q_scale, bias)

    @property
    def weight(self):
        return self.linear.weight

    @weight.setter
    def weight(self, value):
        self.linear.weight = value

    def __getattr__(self, name):
        """Delegate undefined attributes to the inner linear object"""
        return getattr(self.linear, name)
