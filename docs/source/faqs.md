# FAQs

## Version Specific FAQs

- [[v0.7.1rc1] FAQ & Feedback](https://github.com/vllm-project/vllm-ascend/issues/19)
- [[v0.7.3rc1] FAQ & Feedback](https://github.com/vllm-project/vllm-ascend/issues/267)
- [[v0.7.3rc2] FAQ & Feedback](https://github.com/vllm-project/vllm-ascend/issues/418)

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

### 2. How to get our docker containers?

You can get our containers at `Quay.io`, e.g., [<u>vllm-ascend</u>](https://quay.io/repository/ascend/vllm-ascend?tab=tags) and [<u>cann</u>](https://quay.io/repository/ascend/cann?tab=tags).

Plus, if you are in China, you can config your docker proxy to accelerate downloading:

```bash
vim /etc/docker/daemon.json
# Add `https://quay.io` to `registry-mirrors` and `insecure-mirrors`

vim /etc/systemd/system/docker.service.d/https-proxy.conf
# Config proxy
[Service]
Environment="HTTP_PROXY=xxx"

sudo systemctl daemon-reload
sudo systemctl restart docker
```

### 3. What models does vllm-ascend supports?

Currently, we have already supported `Qwen` / `Deepseek` (V0 only) / `Llama` models, other models we have tested are shown [<u>here</u>](https://vllm-ascend.readthedocs.io/en/latest/user_guide/supported_models.html). Plus, accoding to users' feedback, `gemma3` and `glm4` are not supported yet. Besides, more models need test.

### 4. How to get in touch with our community?

There are many channels that you can communicate with our community developers / users:

- Submit a GitHub [<u>issue</u>](https://github.com/vllm-project/vllm-ascend/issues?page=1)
- Join our [<u>weekly meeting</u>](https://docs.google.com/document/d/1hCSzRTMZhIB8vRq1_qOOjx4c9uYUxvdQvDsMV2JcSrw/edit?tab=t.0#heading=h.911qu8j8h35z) and share your ideas
- Join our [<u>WeChat</u>](https://github.com/vllm-project/vllm-ascend/issues/227) group and ask your quenstions
- Join [<u>vLLM forums</u>](https://discuss.vllm.ai/top?period=monthly) and publish your topics

### 5. What features does vllm-ascend V1 supports?

Find more details [<u>here</u>](https://github.com/vllm-project/vllm-ascend/issues/414).
