# LLaMA-Factory

**About / Introduction**

[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) is an easy-to-use and efficient platform for training and fine-tuning large language models. With LLaMA-Factory, you can fine-tune hundreds of pre-trained models locally without writing any code.

LLaMA-Facotory users need to evaluate and inference the model after fine-tuning the model.

**The Business Challenge**

LLaMA-Factory used transformers to perform inference on Ascend NPU, but the speed was slow.

**Solving Challenges and Benefits with vLLM Ascend**

With the joint efforts of LLaMA-Factory and vLLM Ascend ([LLaMA-Factory#7739](https://github.com/hiyouga/LLaMA-Factory/pull/7739)), the performance of LLaMA-Factory in the model inference stage has been significantly improved. According to the test results, the inference speed of LLaMA-Factory has been increased to 2x compared to the transformers version.

**Learn more**

See more about LLaMA-Factory and how it uses vLLM Ascend for inference on the Ascend NPU in the following documentation: [LLaMA-Factory Ascend NPU Inference](https://llamafactory.readthedocs.io/en/latest/advanced/npu_inference.html).
