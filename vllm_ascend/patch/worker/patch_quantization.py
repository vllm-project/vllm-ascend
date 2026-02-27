import vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe as ct_moe_module
import vllm.model_executor.layers.quantization.kernels.mixed_precision as mixed_precision_module
import vllm.model_executor.layers.quantization.kernels.scaled_mm as scaled_mm_module
from vllm.platforms import PlatformEnum

from vllm_ascend.quantization.compressed_tensors.schemes.w4a8 import (
    CompressedTensorsAscendW4A8DynamicFusedMoEMethod,
)
from vllm_ascend.quantization.compressed_tensors.schemes.w8a8 import (
    AscendCompressedTensorsW8A8Int8DynamicFusedMoEMethod,
)
from vllm_ascend.quantization.compressed_tensors.schemes.wNa16 import AscendW4A16FusedMoEMethod
from vllm_ascend.quantization.kernels.mixed_precision.npu import (
    AscendW4A8LinearKernel,
    AscendwNa16LinearKernel,
)
from vllm_ascend.quantization.kernels.scaled_mm.npu import (
    AscendDynamicInt8ScaledMMLinearKernel,
    AscendStaticInt8ScaledMMLinearKernel,
)

mixed_precision_module._POSSIBLE_KERNELS[PlatformEnum.OOT] = [
    AscendW4A8LinearKernel,
    AscendwNa16LinearKernel,
]
scaled_mm_module._POSSIBLE_INT8_KERNELS[PlatformEnum.OOT] = [
    AscendDynamicInt8ScaledMMLinearKernel,
    AscendStaticInt8ScaledMMLinearKernel,
]

ct_moe_module.CompressedTensorsWNA16MarlinMoEMethod.apply = AscendW4A16FusedMoEMethod.apply
ct_moe_module.CompressedTensorsWNA16MarlinMoEMethod.process_weights_after_loading = (
    AscendW4A16FusedMoEMethod.process_weights_after_loading
)

ct_moe_module.CompressedTensorsW8A8Int8MoEMethod.create_weights = (
    AscendCompressedTensorsW8A8Int8DynamicFusedMoEMethod.create_weights
)
ct_moe_module.CompressedTensorsW8A8Int8MoEMethod.apply = AscendCompressedTensorsW8A8Int8DynamicFusedMoEMethod.apply
ct_moe_module.CompressedTensorsW8A8Int8MoEMethod.process_weights_after_loading = (
    AscendCompressedTensorsW8A8Int8DynamicFusedMoEMethod.process_weights_after_loading
)

ct_moe_module.CompressedTensorsW4A8Int8MoEMethod.__init__ = CompressedTensorsAscendW4A8DynamicFusedMoEMethod.__init__
ct_moe_module.CompressedTensorsW4A8Int8MoEMethod.apply = CompressedTensorsAscendW4A8DynamicFusedMoEMethod.apply
ct_moe_module.CompressedTensorsW4A8Int8MoEMethod.process_weights_after_loading = (
    CompressedTensorsAscendW4A8DynamicFusedMoEMethod.process_weights_after_loading
)
