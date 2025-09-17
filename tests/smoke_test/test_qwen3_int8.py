import subprocess
import time

import requests

Url = "http://localhost:20002/v1/completions"
Headers = {"Content-Type": "application/json"}
Data = {
    "model": "Qwen3",
    "prompt": "San Francisco is a",
    "max_tokens": 10,
    "temperature": 0
}


def test_vllm_aclgraph_qwen3_32b_server_A2():
    script_path = "tests/smoke_test/qwen3_32b/run_dp_server_qwen3_32B_aclgraph.sh"
    output_file = "qwen3_32b_int8_output.txt"
    server_proc = None
    output_fp = None
    try:
        output_fp = open(output_file, "w+")
        server_proc = subprocess.Popen(["bash", script_path],
                                       stdout=output_fp,
                                       stderr=subprocess.STDOUT)
        for i in range(30):
            time.sleep(10)
            #ret = os.popen(f'tail -n 50 {output_file}').read()
            #ret = server_proc.stdout.read().decode()
            ret = subprocess.run(["tail", "-n", "50", output_file],
                                 capture_output=True,
                                 text=True)
            print(ret.stdout)
            assert "ERROR" not in ret.stdout, "some errors happen."
            if "startup complete" in ret.stdout:
                break
        else:
            assert False, "max tries achieved, server may not start."
        response = requests.post(Url, headers=Headers, json=Data)
        assert response.status_code == 200, "failed to get response."
        print(response.json())
    finally:
        if server_proc is not None:
            if server_proc.poll() is None:
                server_proc.terminate()
                server_proc.wait()
        if output_fp is not None:
            output_fp.close()
