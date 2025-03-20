# Using OpenCompass 
This document will guide you have a accuracy testing using [OpenCompass](https://github.com/open-compass/opencompass).
## 1. Online Serving

Please install the vLLM and vLLM Ascend first according to [Installation](https://vllm-ascend.readthedocs.io/en/latest/installation.html) doc, Then you can refer to [Quickstart](https://vllm-ascend.readthedocs.io/en/latest/quick_start.html) or [Tutorials](https://vllm-ascend.readthedocs.io/en/latest/tutorials/index.html) to start vLLM on Ascend NPU.

Or you can run docker container to start the vLLM server on a single NPU:
```
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
--name vllm-ascend \
--device /dev/davinci0 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-p 8000:8000 \
-e VLLM_USE_MODELSCOPE=True \
-e PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256 \
-it $IMAGE \
vllm serve Qwen/Qwen2.5-7B-Instruct --max_model_len 26240
```
If your service start successfully, you can see the info shown below:
```
INFO:     Started server process [6873]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```
Once your server is started, you can query the model with input prompts:
```
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "prompt": "The future of AI is",
        "max_tokens": 7,
        "temperature": 0
    }'
```
## 2. Acceleration evaluation by deploying the inference acceleration service API
Install opencompass and configure the environment variables in the container.
```
pip install -U opencompass
pip install datasets==2.18.0
git clone https://github.com/open-compass/opencompass.git
export DATASET_SOURCE=ModelScope
```
Add `opencompass/configs/eval_ch_api_demo.py` with the following content:
```
from mmengine.config import read_base
from opencompass.models import OpenAISDK

with read_base():
    from opencompass.configs.datasets.ceval.ceval_gen import ceval_datasets
datasets = ceval_datasets# + math_datasets

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

models = [
    dict(
        abbr='Qwen2.5-7B-Instruct-vLLM-API',
        type=OpenAISDK,
        key='EMPTY', # API key
        openai_api_base='http://127.0.0.1:8000/v1', 
        path='Qwen/Qwen2.5-7B-Instruct', 
        tokenizer_path='Qwen/Qwen2.5-7B-Instruct', 
        rpm_verbose=True, 
        meta_template=api_meta_template,
        query_per_second=1, 
        max_out_len=1024, 
        max_seq_len=4096, 
        temperature=0.01, 
        batch_size=8,
        retry=3,
    )
]
```
Run the following command:
```
python3 run.py opencompass/configs/eval_ch_api_demo.py --debug
```
The output is as shown below:

| dataset | version | metric | mode | Qwen2.5-7B-Instruct-vLLM-API |
|----- | ----- | ----- | ----- | -----|
| ceval-computer_network | db9ce2 | accuracy | gen | 68.42 |
| ceval-operating_system | 1c2571 | accuracy | gen | 89.47 |
| ceval-computer_architecture | a74dad | accuracy | gen | 76.19 |
| ceval-college_programming | 4ca32a | accuracy | gen | 86.49 |
| ceval-college_physics | 963fa8 | accuracy | gen | 68.42 |
| ceval-college_chemistry | e78857 | accuracy | gen | 66.67 |
| ceval-advanced_mathematics | ce03e2 | accuracy | gen | 31.58 |
| ceval-probability_and_statistics | 65e812 | accuracy | gen | 27.78 |
| ceval-discrete_mathematics | e894ae | accuracy | gen | 18.75 |
| ceval-electrical_engineer | ae42b9 | accuracy | gen | 62.16 |
| ceval-metrology_engineer | ee34ea | accuracy | gen | 83.33 |
| ceval-high_school_mathematics | 1dc5bf | accuracy | gen | 38.89 |
| ceval-high_school_physics | adf25f | accuracy | gen | 73.68 |
| ceval-high_school_chemistry | 2ed27f | accuracy | gen | 73.68 |
| ceval-high_school_biology | 8e2b9a | accuracy | gen | 89.47 |
| ceval-middle_school_mathematics | bee8d5 | accuracy | gen | 73.68 |
| ceval-middle_school_biology | 86817c | accuracy | gen | 90.48 |
| ceval-middle_school_physics | 8accf6 | accuracy | gen | 94.74 |
| ceval-middle_school_chemistry | 167a15 | accuracy | gen | 95.00 |
| ceval-veterinary_medicine | b4e08d | accuracy | gen | 82.61 |
| ceval-college_economics | f3f4e6 | accuracy | gen | 69.09 |
| ceval-business_administration | c1614e | accuracy | gen | 84.85 |
| ceval-marxism | cf874c | accuracy | gen | 94.74 |
| ceval-mao_zedong_thought | 51c7a4 | accuracy | gen | 95.83 |
| ceval-education_science | 591fee | accuracy | gen | 86.21 |
| ceval-teacher_qualification | 4e4ced | accuracy | gen | 88.64 |
| ceval-high_school_politics | 5c0de2 | accuracy | gen | 89.47 |
| ceval-high_school_geography | 865461 | accuracy | gen | 89.47 |
| ceval-middle_school_politics | 5be3e7 | accuracy | gen | 100.00 |
| ceval-middle_school_geography | 8a63be | accuracy | gen | 83.33 |
| ceval-modern_chinese_history | fc01af | accuracy | gen | 91.30 |
| ceval-ideological_and_moral_cultivation | a2aa4a | accuracy | gen | 100.00 |
| ceval-logic | f5b022 | accuracy | gen | 68.18 |
| ceval-law | a110a1 | accuracy | gen | 66.67 |
| ceval-chinese_language_and_literature | 0f8b68 | accuracy | gen | 65.22 |
| ceval-art_studies | 2a1300 | accuracy | gen | 75.76 |
| ceval-professional_tour_guide | 4e673e | accuracy | gen | 89.66 |
| ceval-legal_professional | ce8787 | accuracy | gen | 73.91 |
| ceval-high_school_chinese | 315705 | accuracy | gen | 68.42 |
| ceval-high_school_history | 7eb30a | accuracy | gen | 95.00 |
| ceval-middle_school_history | 48ab4a | accuracy | gen | 90.91 |
| ceval-civil_servant | 87d061 | accuracy | gen | 78.72 |
| ceval-sports_science | 70f27b | accuracy | gen | 94.74 |
| ceval-plant_protection | 8941f9 | accuracy | gen | 86.36 |
| ceval-basic_medicine | c409d6 | accuracy | gen | 94.74 |
| ceval-clinical_medicine | 49e82d | accuracy | gen | 77.27 |
| ceval-urban_and_rural_planner | 95b885 | accuracy | gen | 78.26 |
| ceval-accountant | 002837 | accuracy | gen | 89.80 |
| ceval-fire_engineer | bc23f5 | accuracy | gen | 67.74 |
| ceval-environmental_impact_assessment_engineer | c64e2d | accuracy | gen | 74.19 |
| ceval-tax_accountant | 3a5e3c | accuracy | gen | 85.71 |
| ceval-physician | 6e277d | accuracy | gen | 81.63 |
