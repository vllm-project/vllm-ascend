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

If you are in China, you can use `daocloud` to accelerate your downloading:

1) Open `daemon.json`:

```bash
vi /etc/docker/daemon.json
```

2) Add `https://docker.m.daocloud.io` to `registry-mirrors`:

```json
{
  "registry-mirrors": [
        "https://docker.m.daocloud.io"
    ]
}
```

3) Restart your docker service:

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

After configuration, you can download our container from `m.daocloud.io/quay.io/ascend/vllm-ascend:v0.7.3rc2`.

### 3. What models does vllm-ascend supports?

Currently, we have already fully tested and supported `Qwen` / `Deepseek` (V0 only) / `Llama` models, other models we have tested are shown [<u>here</u>](https://vllm-ascend.readthedocs.io/en/latest/user_guide/supported_models.html). Plus, according to users' feedback, `gemma3` and `glm4` are not supported yet. Besides, more models need test.

### 4. How to get in touch with our community?

There are many channels that you can communicate with our community developers / users:

- Submit a GitHub [<u>issue</u>](https://github.com/vllm-project/vllm-ascend/issues?page=1).
- Join our [<u>weekly meeting</u>](https://docs.google.com/document/d/1hCSzRTMZhIB8vRq1_qOOjx4c9uYUxvdQvDsMV2JcSrw/edit?tab=t.0#heading=h.911qu8j8h35z) and share your ideas.
- Join our [<u>WeChat</u>](https://github.com/vllm-project/vllm-ascend/issues/227) group and ask your quenstions.
- Join our ascend channel in [<u>vLLM forums</u>](https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support/6) and publish your topics.

### 5. What features does vllm-ascend V1 supports?

Find more details [<u>here</u>](https://github.com/vllm-project/vllm-ascend/issues/414).

### 6. How to solve the problem of "Failed to infer device type" or "libatb.so: cannot open shared object file"?

Basically, the reason is that the NPU environment is not configured correctly. You can:
1. try `source /usr/local/Ascend/nnal/atb/set_env.sh` to enable NNAL package.
2. try `source /usr/local/Ascend/ascend-toolkit/set_env.sh` to enable CANN package.
3. try `npu-smi info` to check whether the NPU is working.

If all above steps are not working, you can try the following code with python to check whether there is any error:

```
import torch
import torch_npu
import vllm
```

If all above steps are not working, feel free to submit a GitHub issue.

### 7. Does vllm-ascend support Atlas 300I Duo?

No, vllm-ascend now only supports Atlas A2 series. We are working on it.

### 8. How does vllm-ascend perform?

Currently, only some models are improved. Such as `Qwen2 VL`, `Deepseek  V3`. Others are not good enough. In the future, we will support graph mode and custom ops to improve the performance of vllm-ascend. And when the official release of vllm-ascend is released, you can install `mindie-turbo` with `vllm-ascend` to speed up the inference as well.

### 9. How vllm-ascend work with vllm?
vllm-ascend is a plugin for vllm. Basically, the version of vllm-ascend is the same as the version of vllm. For example, if you use vllm 0.7.3, you should use vllm-ascend 0.7.3 as well. For main branch, we will make sure `vllm-ascend` and `vllm` are compatible by each commit.

### 10. Does vllm-ascend support Prefill Disaggregation feature?

Currently, only 1P1D is supported by vllm. For vllm-ascend, it'll be done by [this PR](https://github.com/vllm-project/vllm-ascend/pull/432). For NPND, vllm is not stable and fully supported yet. We will make it stable and supported by vllm-ascend in the future.
