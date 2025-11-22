# Principles of Modeling files in vllm-ascend

To keep a more concise software architecture and avoid maintaining a wide variety of models, vllm-ascend will not maintain any modeling files in the future, instead we just register and replace some `CustomOp` in vllm to support the model inference on Ascend devices.

Currently, we plan to remove all the VL modeling files, find more details at [#4084](https://github.com/vllm-project/vllm-ascend/issues/4084).
