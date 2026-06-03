# LLaMA-Factory

**Introduction**

[LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) is an easy-to-use and efficient platform for training and fine-tuning large language models. With LLaMA-Factory, you can fine-tune hundreds of pre-trained models locally without writing any code.

LLaMA-Factory users need to evaluate the model and perform inference after fine-tuning.

**Business challenge**

LLaMA-Factory uses Transformers to perform inference on Ascend NPUs, but the speed is slow.

**Benefits with vLLM Ascend**

With the joint efforts of LLaMA-Factory and vLLM Ascend ([LLaMA-Factory#7739](https://github.com/hiyouga/LlamaFactory/pull/7739)), LLaMA-Factory has achieved significant performance gains during model inference. Benchmark results show that its inference speed is now up to 2× faster compared to the Transformers implementation.

**Learn more**

For more details about LLaMA-Factory, please refer to the [official documentation](https://llamafactory.readthedocs.io/en/latest/index.html).
