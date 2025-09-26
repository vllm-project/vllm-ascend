✨ 特性
==================================== **SYSTEM INFO** ===========================================
Kernel: 4.19.90-vhulk2211.3.0.h1543.eulerosv2r10.aarch64
CPU: Kunpeng-920
name: HUAWEI Kunpeng 920 5250
Memory: 1.5Ti
Distro: Huawei Cloud EulerOS 2.0 (aarch64)
Python Version: Python 3.10.6
System Architecture: aarch64
torch-npu: 2.5.1
vllm: 0.8.5.post2.dev200+g1ff751ca1.d20250517.empty
vllm-ascend: 0.8.5rc2.dev33+gdc1e55b
nnal:
    Ascend-cann-atb : 8.1.RC1
    Ascend-cann-atb Version : 8.1.RC1.B110
    Platform : aarch64
    branch : br_release_cann_8.1.RC1_20250925
    commit id : 429f2ba259d45cb8fd0e23e218f4685bed3855d7
==================================== **SYSTEM INFO** ===========================================

🛠️ 安装
# **一. Docker镜像**
1.拉取镜像：
    docker pull registry-cbu.huawei.com/vllm_npu/vllm_a2_0.8.5_omni_infer_v1:B001
2.运行镜像：
    docker run -it --name my_vllm_container registry-cbu.huawei.com/vllm_npu/vllm_a2_0.8.5_omni_infer_v1:B001 /bin/bash

# **二.测试所需第三方依赖**
1.在omni_infer\tests\requirements.txt下执行pip
    pip install requirements.txt

📖 demo用例
# **三.测试用例**
1. 参考用例,eg:test_demo.py
    # platform_ascend910b:昇腾910b，level1：全量用例
    from tests.mark_utils import arg_mark
    @arg_mark(['platform_ascend910b'], 'level1')
    def test_example():
        assert True

2. 执行用例
    pytest -vra --disable-warnings -m "platform_ascend910b and level1" test_demo.py
