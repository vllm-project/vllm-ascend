from vllm_ascend.quantization.quantizer import VLLMAscendQuantizer
from vllm_ascend.torchair.quantization.torchair_w4a8_dynamic import (
    TorchairAscendW4A8DynamicFusedMoEMethod,
    TorchairAscendW4A8DynamicLinearMethod)
from vllm_ascend.torchair.quantization.torchair_w8a8_dynamic import (
    TorchairAscendW8A8DynamicFusedMoEMethod,
    TorchairAscendW8A8DynamicLinearMethod)


class TorchairW8A8DYNAMICQuantizer(VLLMAscendQuantizer):

    @staticmethod
    def build_linear_method():
        return TorchairAscendW8A8DynamicLinearMethod()

    @staticmethod
    def build_moe_method():
        return TorchairAscendW8A8DynamicFusedMoEMethod()


class TorchairW4A8DYNAMICQuantizer(VLLMAscendQuantizer):

    @staticmethod
    def build_linear_method():
        return TorchairAscendW4A8DynamicLinearMethod()

    @staticmethod
    def build_moe_method():
        return TorchairAscendW4A8DynamicFusedMoEMethod()
