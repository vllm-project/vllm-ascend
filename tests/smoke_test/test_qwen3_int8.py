
import subprocess
import time

def test_vllm_aclgraph_qwen3_32b_server_A2(weight_path):
    script_path = "tests/smoke_test/qwen3_32b/run_dp_server_qianwen3_32B_aclgraph.sh"
    subprocess.run(["chmod", "+x", script_path], capture_output=True, text=True)
    subprocess.run(["bash", "-c", f"bash {script_path} {weight_path} > output.txt 2>&1 &"], capture_output=True, text=True)
    time.sleep(1)
    for i in range(30):
        time.sleep(10)
        result = subprocess.run(["cat", "output.txt"], capture_output=True, text=True)
        ret = result.stdout.strip()
        print(ret)
        assert "ERROR" not in ret, "some errors happen."
        if "startup complete" in ret:
            break
    else:
        assert False, "max tries achieved, server may not start."
    curl_request = '''
    curl -X POST -s http://localhost:20002/v1/completions -H "Content-Type: application/json" -d '{"model": "Qwen3","prompt": "San Francisco is a","max_tokens": 10,"temperature": 0}';echo
    '''
    result = subprocess.run(["bash", "-c", curl_request], capture_output=True, text=True)
    ret = result.stdout.strip()
    assert "text" in ret, "failed to get response."

