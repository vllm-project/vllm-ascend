import subprocess
import time
import os


def test_vllm_aclgraph_qwen3_32b_server_A2():
    script_path = "tests/smoke_test/qwen3_32b/run_dp_server_qwen3_32B_aclgraph.sh"
    output_file = "qwen3_32b_int8_output.txt"
    try:
        server_proc = subprocess.Popen(["bash", script_path],
                                       stdout=open(output_file, "w+"),
                                       stderr=subprocess.STDOUT)
        for i in range(30):
            time.sleep(10)
            ret = os.popen(f'tail -n 50 {output_file}').read()
            #ret = server_proc.stdout.read().decode()
            print(ret)
            assert "ERROR" not in ret, "some errors happen."
            if "startup complete" in ret:
                break
        else:
            assert False, "max tries achieved, server may not start."
        curl_request = '''
        curl -X POST -s http://localhost:20002/v1/completions -H "Content-Type: application/json" -d '{"model": "Qwen3","prompt": "San Francisco is a","max_tokens": 10,"temperature": 0}';echo
        '''
        result = subprocess.run(["bash", "-c", curl_request],
                                capture_output=True,
                                text=True)
        ret = result.stdout.strip()
        assert "text" in ret, "failed to get response."
    finally:
        if 'server_proc' in locals() and server_proc.poll() is None:
            server_proc.terminate()
            server_proc.wait()
            if server_proc.stdout is not None:
                server_proc.stdout.close()
