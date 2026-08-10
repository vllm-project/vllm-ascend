# 环境信息

42 环境上：
| 容器 | vllm 版本 | vllm-ascend commitid | 精度结果 |
| ------ | -------- | -------------------- | ----- |
| zty_b060 | releases/v0.21.0 | 971d50b3f | 正确 |
| zty_b060_erfen | releases/v0.22.1 | 43c7c9a48 | 错误（存在截断） |

# 版本信息

| 版本号 | vllm 版本 | vllm-ascend commitid |
| ------ | -------- | -------------------- |
| B060 | releases/v0.21.0 | 971d50b3f |
| B070 | releases/v0.22.1 | 350d1f0ff |

# 权重下载链接

https://www.modelscope.cn/models/Eco-Tech/Qwen3-235B-A22B-w8a8-QuaRot

# 权重 sha256sum

```shell
9552c8c45dd023912205ed0b6fbb45acf11377590579eb9721152315491c2561  config_backup.json
e10186cd5391e13034c6e0d4068fc220bbdb61b829757cd58d4545ea9959a1ce  config_bakk.json
299bbed39b8e408b9cec48be37b681361e673d36de7c3261f3bcea5b8e5b17f4  config.json
625c81a3527ce4f7a739854b431f380a987dc4af5e07e099b568e50e2e3b248c  configuration.json
e10186cd5391e13034c6e0d4068fc220bbdb61b829757cd58d4545ea9959a1ce  config_yarn.json
2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2  generation_config.json
7bc04fa8717878ae8448d9285ea90bf763676f85d84a15b0e91bbc4bbeb8b811  quant_model_description.json
e8c834d8f99992b2162d493124f6fcb9386d16c14bf15f5abca902f8c32a54c9  quant_model_weight_w8a8_dynamic-00001-of-00056.safetensors
3a083d67b20f07c874462ceb37150cbd12d42924cbc4b7702b5589f282a656ea  quant_model_weight_w8a8_dynamic-00002-of-00056.safetensors
cadb45080803b4d36da711cef398c4e197715b4417346e2e6c9364af75378b8f  quant_model_weight_w8a8_dynamic-00003-of-00056.safetensors
48a8475325d301e36f6cba2ba6375858698e9a110b1b7f620821a35d5f22b098  quant_model_weight_w8a8_dynamic-00004-of-00056.safetensors
903d6200d227ac34e66a41c823f9c3efe61cd548227f5d0f6d70d980c56f2608  quant_model_weight_w8a8_dynamic-00005-of-00056.safetensors
fb1a7059a31962fad6b9f6065736278d3a92626d1b85b6dff87e464d20cdfbe8  quant_model_weight_w8a8_dynamic-00006-of-00056.safetensors
70ba13e80e57976369f3075ed8baed172bb73c5a1d88e36e13312ea387a7b139  quant_model_weight_w8a8_dynamic-00007-of-00056.safetensors
47e6780fcf6bc61da7ede59e48467ecca6dbf9272a420e607eaeb5d9b38506f8  quant_model_weight_w8a8_dynamic-00008-of-00056.safetensors
93342ee0bc2f1901c9e6dd8b2c90700d28024158505d9eff3154ac86223c197b  quant_model_weight_w8a8_dynamic-00009-of-00056.safetensors
e85ea673ed2d994d6e17c45fffe5ed684d794a59a204ce82feccc5e8c04bdbd0  quant_model_weight_w8a8_dynamic-00010-of-00056.safetensors
b537bc76328d6596ae9f7f0e094556f0d0516d1c489c5e380378e77bb8234e23  quant_model_weight_w8a8_dynamic-00011-of-00056.safetensors
4f8b1f6193a9b5fa2c14a950e459bf31f912139c0065f9a31152103678a3de57  quant_model_weight_w8a8_dynamic-00012-of-00056.safetensors
7a999b2beb8ce42df1d73d591b3fc7f66bdd2c06c6bd17590110ade510c999e3  quant_model_weight_w8a8_dynamic-00013-of-00056.safetensors
3daa8f96265106543238799ea057cd4d5f821b63cee51ff84de433e7167f433d  quant_model_weight_w8a8_dynamic-00014-of-00056.safetensors
4ddc33a5660b4e0d7238d3aac357eab926bed072f6fad392d2bc559282487bfb  quant_model_weight_w8a8_dynamic-00015-of-00056.safetensors
3e04741c319a78ae8f8fa491d1a187ba8ff7bf554539410c69ae285f01904ebe  quant_model_weight_w8a8_dynamic-00016-of-00056.safetensors
8355734f62d61e9f675b9861b17ba20ed9905a5690e8b4c8e8ebee09be5878f5  quant_model_weight_w8a8_dynamic-00017-of-00056.safetensors
be13968c9a4533bb4fd7c7fbb0ef35cb209a14d68a56f448ba114769dccc202a  quant_model_weight_w8a8_dynamic-00018-of-00056.safetensors
f9bb015708090ac1c2581fca10153e76fd8f113ba767f1417dc7c2138681089c  quant_model_weight_w8a8_dynamic-00019-of-00056.safetensors
2021107ec6ce14ab34a2f19dbd78192f95e6c114e9e357ceb202ddb46bebb415  quant_model_weight_w8a8_dynamic-00020-of-00056.safetensors
bfe0342725cb5f7e0596045b22c46c43aec6baa7c4fd90d1b28228b74ed45c05  quant_model_weight_w8a8_dynamic-00021-of-00056.safetensors
375021b945a716cf8cf962bfac6ffc287925535b91b8151cd009304a91df1e64  quant_model_weight_w8a8_dynamic-00022-of-00056.safetensors
10de40e02079c74eda3669d84f6d7fbe32b75c20243ad78cb5ba7b493341ce2c  quant_model_weight_w8a8_dynamic-00023-of-00056.safetensors
220131197a0790ba68cc4e057f426594768f244c80c20ce7a1c14e4407a6ee53  quant_model_weight_w8a8_dynamic-00024-of-00056.safetensors
9f6a35978880941e62e23ae8f556d576791ccb683835b85c6e0a9210a0f30dc2  quant_model_weight_w8a8_dynamic-00025-of-00056.safetensors
bd16040511c1fd052feb47cd98f9cfe612df0a6f5363b233d3b5737fe6c7ef60  quant_model_weight_w8a8_dynamic-00026-of-00056.safetensors
777ae865a87cb1bf0888d66918222f9c3f186a1669a95973bb3ba8933f86e717  quant_model_weight_w8a8_dynamic-00027-of-00056.safetensors
81b2244ecc771886082843e3cff1093339405bd36ebfddeb5f051992d2f0ee70  quant_model_weight_w8a8_dynamic-00028-of-00056.safetensors
bb3fb76411f94183bcb7736d9704ed2f9d9a11e0ce4eaa4b83e95baff39b9edb  quant_model_weight_w8a8_dynamic-00029-of-00056.safetensors
31791ae10ed6ce47012ed4e2862ca747f2667806b612a0acbcdf9ae41296adf3  quant_model_weight_w8a8_dynamic-00030-of-00056.safetensors
443bc6c0b64db7a239b75bd3fbfdf94312cac511d13e8c8a022be90dffaa951a  quant_model_weight_w8a8_dynamic-00031-of-00056.safetensors
7060427d763ec63f74860378e15d48df451fdcc5fe5aca932fb65e0a37c36667  quant_model_weight_w8a8_dynamic-00032-of-00056.safetensors
da519ea13ecf92c32bf4a36742b41be99f42cd90d500db6246082f3e6f1c6579  quant_model_weight_w8a8_dynamic-00033-of-00056.safetensors
46729754128342ee8abc0a14dd6957e74f8da7eb8e72bcae8c1e509c877e7c62  quant_model_weight_w8a8_dynamic-00034-of-00056.safetensors
8a7a77e8dc81b3f550cd5445039d8a96c1ae9481f48432a8a6244e8d563b570d  quant_model_weight_w8a8_dynamic-00035-of-00056.safetensors
fe256abba277912cb5fda2915431a88b1fe3328ee39c71e24c3dbe643fe120de  quant_model_weight_w8a8_dynamic-00036-of-00056.safetensors
42d7d41eb8433657b20b5cef32e29d158a56918a98afc23ae4888d60ac988273  quant_model_weight_w8a8_dynamic-00037-of-00056.safetensors
25a3eb9de8dc6617ba199eddf30c7d1a9fedaeccd709af996fcd5e3eb3171b74  quant_model_weight_w8a8_dynamic-00038-of-00056.safetensors
6d32c15b3c69bf90400488a62a090eea36d1a40cda98cd2202fa44c945c13bfd  quant_model_weight_w8a8_dynamic-00039-of-00056.safetensors
b45f53f92463ba86a0dbba6193e43ecee664146b7f0546f333ccbff44b589f1b  quant_model_weight_w8a8_dynamic-00040-of-00056.safetensors
191f4e5a59cc67a9158f210508943dd027ec6cee73499a19c53e5dd53bf1408f  quant_model_weight_w8a8_dynamic-00041-of-00056.safetensors
faad9aad367bea0c370f59adec81131cf1f8c8e61e034df31dc383ace377848f  quant_model_weight_w8a8_dynamic-00042-of-00056.safetensors
98b31245c04942e647371c56ac718f4dd4bb1da940188bf72658e3383be25789  quant_model_weight_w8a8_dynamic-00043-of-00056.safetensors
4db32339e8e94f13f57365e22c1e64e2439b0754eb6d6a34786fa331a578a433  quant_model_weight_w8a8_dynamic-00044-of-00056.safetensors
ad53516a99ed09bdaa441e4d4e728aa2f31b251e642ac83c8f25abb28b5c81c1  quant_model_weight_w8a8_dynamic-00045-of-00056.safetensors
93e7d04850e7fc272bce4ac003ad0bcde44b8d27baa6a3683316d70acfd79080  quant_model_weight_w8a8_dynamic-00046-of-00056.safetensors
31fba174404e61817b8a56e225b2ed717a4164bc360063fd49f6a9e689ab20f6  quant_model_weight_w8a8_dynamic-00047-of-00056.safetensors
87ae371f128ea19e6dc0bc1b8598b9aa99aeb442b5372351a11c50d021c2f816  quant_model_weight_w8a8_dynamic-00048-of-00056.safetensors
83e36c8aeb4326677e3a639f36f83f9e6b25ef4aeb9c7db8cf8804aba2548363  quant_model_weight_w8a8_dynamic-00049-of-00056.safetensors
853f3bce0acda231ebbf20e6abf2a6d7b7af63ce3cc57f430668424e8a27b9a1  quant_model_weight_w8a8_dynamic-00050-of-00056.safetensors
08cf792dc63ad0a86a0ffc7306101bad8ffa99cde47bfa0b990ea639d89a9438  quant_model_weight_w8a8_dynamic-00051-of-00056.safetensors
112b27465591dae9c01896cf03631a7985672ffdb0953bcf3fa21576c2bf2180  quant_model_weight_w8a8_dynamic-00052-of-00056.safetensors
4aff17df4fa751808b7c567826e8b796b8569f2ed0042710b454fdced21444fa  quant_model_weight_w8a8_dynamic-00053-of-00056.safetensors
47ace7558a2471f9d7ebceabf82cf56efb28681f1c10666c52410f4dbc137268  quant_model_weight_w8a8_dynamic-00054-of-00056.safetensors
cddb550976b4114b71de07f7743e08bdb063aa2292af985535de7d1c41915d2d  quant_model_weight_w8a8_dynamic-00055-of-00056.safetensors
b1af43c5fe331b17ba2ed442ba1d41996540441df22c07bc7ae7a2da15d2cafe  quant_model_weight_w8a8_dynamic-00056-of-00056.safetensors
ef65039b127fc673cecd5b3aa4fcc68d4f07d6e4857f87e9c6c30a6a3200a45f  quant_model_weight_w8a8_dynamic.safetensors.index.json
9af6158e0c017ec705587fd03708f6e44fc85b188a472fc07a211abe3d916f94  tokenizer_config.json
aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4  tokenizer.json
ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910  vocab.json
```

# 服务拉起脚本

```shell
#!/bin/bash
set -e

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export VLLM_USE_V1=1;
export HCCL_OP_EXPANSION_MODE="AIV";
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True;
export HCCL_BUFFSIZE=1024;
export OMP_PROC_BIND=false;
export OMP_NUM_THREADS=1;
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1;
export VLLM_ENGINE_READY_TIMEOUT_S=3600

vllm serve /mnt/a800_weight/Qwen3-235B-A22B-W8A8-rot \
    --served-model-name qwen3_235b \
    --host 0.0.0.0 \
    --port 8000 \
    --async-scheduling \
    --tensor-parallel-size 4 \
    --data-parallel-size 4 \
    --data-parallel-size-local 4 \
    --data-parallel-start-rank 0 \
    --data-parallel-address 90.90.97.37 \
    --data-parallel-rpc-port 2345 \
    --max-num-seqs 40 \
    --max-model-len 40960 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.9 \
    --enable-expert-parallel \
    --quantization "ascend" \
    --trust-remote-code \
    --compilation-config '{"cudagraph_mode": "PIECEWISE"}'
```

# curl 命令

curl.sh
```shell
curl http://90.90.97.37:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @data.json
```

data.json
```shell
{
  "model": "qwen3_235b",
  "messages": [
    {
      "role": "user",
      "content": "Let $z = 2 + \\sqrt{2} - (3 + 3 \\sqrt{2})i$, and let $c = 2 - 3i$.  Let $w$ be the result when $z$ is rotated around $c$ by $\\frac{\\pi}{4}$ counter-clockwise.\n\n[asy]\nunitsize(0.6 cm);\n\npair C, W, Z;\n\nZ = (2 + sqrt(2), -3 - 3*sqrt(2));\nC = (2,-3);\nW = rotate(45,C)*(Z);\n\ndraw(Z--C--W);\n\ndot(\"$c$\", C, N);\ndot(\"$w$\", W, SE);\ndot(\"$z$\", Z, S);\nlabel(\"$\\frac{\\pi}{4}$\", C + (0.6,-1));\n[/asy]\n\nFind $w.$\nPlease reason step by step, and put your final answer within \\boxed{}."
    }
  ],
  "temperature": 0.6,
  "top_k": 20,
  "top_p": 0.95,
  "max_tokens": 4096,
  "stream": false,
  "chat_template_kwargs": {
    "thinking": true
  }
}
```

# aisbench 脚本

```python
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-general-chat",
        path="/mnt/a800_weight/Qwen3-235B-A22B-W8A8-rot",
        model="qwen3_235b",
        stream=False,
        request_rate = 0,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip = "90.90.97.37",
        host_port = 8000,
        url="",
        max_out_len = 32768,
        batch_size = 64,
        trust_remote_code = False,
        generation_kwargs=dict(
            top_k = 20,
            top_p = 0.95,
            temperature = 0.6,
            ignore_eos = False,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
```

aisbench 命令
```shell
ais_bench --models vllm_api_general_chat --datasets math500_gen_0_shot_cot_chat_prompt --debug
```
