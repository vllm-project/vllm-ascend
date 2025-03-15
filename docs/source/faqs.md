# FAQs

## Version Specific FAQs

- [[v0.7.1rc1] FAQ & Feedback](https://github.com/vllm-project/vllm-ascend/issues/19)
- [[v0.7.3rc1] FAQ & Feedback](https://github.com/vllm-project/vllm-ascend/issues/267)

## General FAQs

### 1. What devices are currently supported?

Currently, **ONLY Atlas A2 series**  (Ascend-cann-kernels-910b) are supported:

- Atlas A2 Training series (Atlas 800T A2, Atlas 900 A2 PoD, Atlas 200T A2 Box16, Atlas 300T A2)
- Atlas 800I A2 Inference series (Atlas 800I A2)

Below series are NOT supported yet:
- Atlas 300I Duo、Atlas 300I Pro (Ascend-cann-kernels-310p) might be supported on 2025.Q2
- Atlas 200I A2 (Ascend-cann-kernels-310b) unplanned yet
- Ascend 910, Ascend 910 Pro B (Ascend-cann-kernels-910) unplanned yet

From a technical view, vllm-ascend support would be possible if the torch-npu is supported. Otherwise, we have to implement it by using custom ops. We are also welcome to join us to improve together.
