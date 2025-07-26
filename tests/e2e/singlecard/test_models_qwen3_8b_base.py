import lm_eval
import numpy as np

def test_model_function():
    """Placeholder test for Qwen3 8B Base model."""
    assert True
    
    
def test_model_accuracy():
    
    datasets = ["ceval-valid", "gsm8k"]
    batch_size = {
        "ceval-valid": 1, 
        "gsm8k": "auto"
    }
    groundtruths = {
        "ceval-valid": 0.82,
        "gsm8k": 0.83
    }
    rtol = 0.03
    
    filter = {
    "gsm8k": "exact_match,flexible-extract",
    "ceval-valid": "acc,none",
    }
    
    flag = True

    for dataset in datasets:
        eval_params = {
            "model": "vllm",
            "model_args": "pretrained=Qwen/Qwen3-8B-Base,max_model_len=4096, \
                           dtype=auto,tensor_parallel_size=2,gpu_memory_utilization=0.6",
            "tasks": dataset,
            "apply_chat_template": True,
            "fewshot_as_multiturn": True,
            "batch_size": batch_size[dataset],
            "num_fewshot": 5,
        }
        results = lm_eval.simple_evaluate(**eval_params)
        print(f"Success: Qwen3-8B-BASE on {dataset} ")
        measured_value = results["results"]
        print(f"Measured value: {measured_value}")
        if not np.isclose(measured_value, groundtruths[dataset], rtol):
            flag = False
            
    assert flag, "Model accuracy did not meet expectations for one or more datasets."
    
    
    # 要写一个write函数
    # 该函数将精度结果写入文件
    
    

        
        

        
        
        
            
            
            
            
        
        
    
    
    # load yaml
    # run-lmeval
    # assert >= xxx
    # write()
    
