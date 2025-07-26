import yaml
from base import MultiProcessModelTester

def test_model_functions():
    assert True
    
def test_model_accuracy():
    "base_config" = {
        "model": "vllm",
        "model_args": "pretrained=Qwen/Qwen3-8B-Base,max_model_len=4096,dtype=auto,tensor_parallel_size=2,gpu_memory_utilization=0.6",
        "apply_chat_template": true,
        "fewshot_as_multiturn": true,
        "num_fewshot": 5
    }
    
    "datasets" = {
        "ceval-valid": {
            "batch_size": 1,
            "groundtruth": 0.82,
            "filter": "acc,none"
        },
        "gsm8k": {
            "batch_size": "auto",
            "groundtruth": 0.83,
            "filter": "exact_match,flexible-extract"
        }
    }
    
    datasets_config = {}

    for name, spec in datasets.items():
        eval_params = {
            "model": base_config["model"],
            "model_args": base_config["model_args"],
            "apply_chat_template": base_config["apply_chat_template"],
            "fewshot_as_multiturn": base_config["fewshot_as_multiturn"],
            "num_fewshot": base_config["num_fewshot"],
            "tasks": name,
            "batch_size": spec["batch_size"]
        }
        
        datasets_config[name] = {
            "eval_params": eval_params,
            "groundtruth": spec["groundtruth"],
            "filter": spec["filter"]
        }

    tester = MultiProcessModelTester(rtol=0.03)
    results, all_success = tester.run_multi_process_tests(datasets_config)
    assert all_success, "Model accuracy did not meet expectations for one or more datasets."
    
    print(results)
    
    # 该部分应该是写入报告的部分了
    # execute_command = "
    #     export MODEL_ARGS='pretrained=Qwen/Qwen3-8B-Base,max_model_len=4096,dtype=auto,tensor_parallel_size=2,gpu_memory_utilization=0.6'
    #     lm_eval --model vllm --model_args $MODEL_ARGS --tasks ceval-valid,gsm8k \ 
    #     --apply_chat_template --fewshot_as_multiturn --num_fewshot 5 --batch_size 1
    # "
    
    
    
    
    