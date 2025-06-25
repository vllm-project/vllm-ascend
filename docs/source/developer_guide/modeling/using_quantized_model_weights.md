# Using Quantized Model Weights

Computation modules (e.g., variants of Linear layer) in vllm and vllm-ascend keep an attribute named `quant_method` indicating weights for a layer. It means that vllm-ascend will automatically load quantized weights and apply forward process according to different quantization schemes. vllm-ascend now delivers both static and dynamic `W8A8` quantization solutions on Ascend platforms, achieving optimized inference speed with significant memory savings.​

:::{note}
Please refer to [msit user guidance for W8A8 quantization and accuracy calibration](https://gitee.com/ascend/msit/blob/master/msmodelslim/docs/w8a8%E7%B2%BE%E5%BA%A6%E8%B0%83%E4%BC%98%E7%AD%96%E7%95%A5.md) to quantize your own models. It is recommended to quantize models with scripts in [msit quantization examples for prevailing models](https://gitee.com/ascend/msit/tree/master/msmodelslim/example) for common models.
:::