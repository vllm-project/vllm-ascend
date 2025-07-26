import multiprocessing
import numpy as np
import lm_eval

class MultiProcessModelTester:
    def __init__(self, rtol=0.03):
        self.rtol = rtol
        self.manager = multiprocessing.Manager()
        self.results_dict = self.manager.dict()
        self.error_flag = self.manager.Value('b', False)
    
    def _evaluate_single_dataset(self, dataset, eval_params, groundtruth, filter):
        try:
            results = lm_eval.simple_evaluate(**eval_params)
            print(f"Success: {eval_params['model']} on {dataset}")
            
            measured_value = results["results"][filter]
            print(f"Measured value: {measured_value}")
            
            if not np.isclose(measured_value, groundtruth, rtol=self.rtol):
                self.error_flag.value = True
                
            self.results_dict[dataset] = {
                'measured': measured_value,
                'expected': groundtruth,
                'success': True
            }
        except Exception as e:
            print(f"Error evaluating {dataset}: {str(e)}")
            self.results_dict[dataset] = {
                'success': False,
                'error': str(e)
            }
            self.error_flag.value = True
    
    def run_multi_process_tests(self, datasets_config):
        """
        datasets_config: dict {
            "dataset_name": {
                "eval_params": dict,  
                "groundtruth": float,
                "filter": str  
            }
        }
        """
        processes = []
        for dataset, config in datasets_config.items():
            p = multiprocessing.Process(
                target=self._evaluate_single_dataset,
                args=(dataset, config["eval_params"], config["groundtruth"], config["filter"])
            )
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()
        
        return dict(self.results_dict), not self.error_flag.value
    
    