### 1 Performance Benchmark Script
This script automates the deployment and execution of performance benchmarks for SGLang-based inference services. It handles service deployment via Ansible, monitors service readiness, executes performance tests, and collects results.

### 2 Features
* Automated Deployment: Deploys inference services using Ansible playbooks
* Service Monitoring: Verifies service readiness through API health checks
* Performance Testing: Executes standardized benchmark tests
* Result Collection: Automatically retrieves benchmark results
* Robust Error Handling: Implements retry logic and comprehensive logging

### 3 Prerequisites
#### 1 Python 3.8+
#### 2 Required Python packages:
```
pip install paramiko requests scp
```
### 3 Remote Server Access:
* SSH access to deployment servers
* Password or SSH key authentication

#### 4 Ansible Setup:
* Ansible installed on target servers
* Valid inventory and playbook files
### 4 Usage
Basic Command
```
python benchmark.py \
  --benchmark_src_dir <remote_results_dir> \
  --benchmark_target_dir <local_results_dir> \
  --host <proxy_host> \
  --port <proxy_port> \
  --model <model_path> \
  --ansible-ssh-host <ansible_host> \
  --benchmark-ssh-host <benchmark_host> \
  [authentication options] \
  [ansible options]
```
### 5 Full Parameter List

Parameter                 | Required | Default                             | Description
---------------------------|----------|-------------------------------------|---------------------------------------------------------------------
 --benchmark-src-dir       | Yes      | -                                   | Remote directory for benchmark results (e.g., /path/to/0808)
 --benchmark-target-dir    | Yes      | -                                   | Local directory to save results
 --host                    | Yes      | -                                   | Global proxy host address
 --port                    | No       | 8011                                | Global proxy port
 --model                   | Yes      | -                                   | Path to model files
 --ansible-ssh-host        | Yes      | -                                   | Ansible server host address
 --ansible-ssh-port        | No       | 22                                  | Ansible server SSH port
 --ansible-ssh-username    | No       | root                                | Ansible server username
 --benchmark-ssh-host      | Yes      | -                                   | Benchmark server host address
 --benchmark-ssh-port      | No       | 22                                  | Benchmark server SSH port
 --benchmark-ssh-username  | No       | root                                | Benchmark server username
 --ansible-ssh-password    | Either   | -                                   | Ansible server password
 --ansible-ssh-key-file    | Either   | -                                   | Ansible server private key file
 --benchmark-ssh-password  | Either   | -                                   | Benchmark server password
 --benchmark-ssh-key-file  | Either   | -                                   | Benchmark server private key file
 --inventory-path          | No       | omni_infer_inventory_used_for_1P1D.yml | Ansible inventory file path
 --playbook-path           | No       | omni_infer_server_used_for_1P1D.yml | Ansible playbook file path
 --wait-timeout            | No       | 120                                 | Service startup timeout (seconds)

### 6 Authentication Options
* Use either password or key-based authentication for each server
* For key-based authentication, provide path to private key file
* For password authentication, provide the password

### 7 Workflow
#### 1 Establish SSH Connections:
* Connect to Ansible deployment server
* Connect to benchmark results server

#### 2 Service Deployment:
* Execute Ansible playbook with specified tag
* Retry deployment up to 3 times if service doesn't start
#### 3 Service Readiness Check:
* Verify service availability via API endpoint
* /v1/chat/completions endpoint is monitored
* Timeout configurable via --wait-timeout

#### 4 Performance Benchmark:
* Execute Ansible playbook with performance_benchmark tag
* Monitor remote directory for result files (*.jsonl)

#### 5 Result Collection:
* Copy benchmark results from remote server to local machine
* Results stored in specified benchmark_target_dir

### 8 Example
```
python benchmark.py \
  --benchmark-src-dir /data/benchmarks/0815 \
  --benchmark-target-dir ./results \
  --host 192.168.1.100 \
  --port 8011 \
  --model /data/models/DeepSeek-R1 \
  --ansible-ssh-host 192.168.1.101 \
  --ansible-ssh-key-file /xx/xx/ansible_server.pem \
  --benchmark-ssh-host 192.168.1.102 \
  --benchmark-ssh-key-file /xx/xx/benchmark_server.pem \
  --inventory-path omni_infer_inventory.yml \
  --playbook-path omni_infer_server.yml \
  --wait-timeout 300
```

### 9 Output Files
Benchmark results include:
* sglang_performance_metrics.jsonl: Performance metrics
* sglang_system_info.json: System configuration details
* sglang_benchmark.log: Execution logs

### 10 Logging
The script generates detailed logs with timestamps:
```
2025-08-15 14:30:45 - root - INFO - <<< Starting initial deployment...
2025-08-15 14:31:15 - root - INFO - <<< Executing remote command (Ansible deployment): ansible-playbook -i my_inventory.yml deploy_playbook.yml --tags "full_deploy"
2025-08-15 14:33:22 - root - INFO - <<< Remote command executed successfully (Ansible deployment)
2025-08-15 14:33:22 - root - INFO - <<< Waiting for inference service 192.168.1.100:8011 to be ready (timeout: 300s, retry interval: 30s)...
```
### 11 Error Handling
The script implements:
* Connection retries for SSH
* Deployment retries (up to 3 times)
* Service startup timeout detection
* Comprehensive error logging
* Resource cleanup (SSH connections always closed)

Common errors include:
* SSH connection failed: Verify credentials and network connectivity
* Ansible deployment failed: Check playbook and inventory files
* Service not ready after timeout: Investigate service logs
* Directory copy failed: Verify path permissions

### 12 Troubleshooting
#### 1 Service Not Starting:
* Check Ansible playbook execution logs
* Verify model path accessibility
* Ensure ports are not blocked by firewall

#### 2 Connection Issues:
```
ssh -v -i key_file.pem user@server.com
```
* Test SSH connectivity manually
* Verify key permissions (chmod 600 key_file.pem)

#### 3 Permission Denied Errors:
* Ensure remote user has:
  * Permission to execute Ansible
  * Access to model files
  * Write access to results directory

#### 4 API Health Check Failures:
```
curl -v http://<host>:<port>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "test", "messages": [{"role": "user", "content": "Hello"}]}'
```
* Manually test the API endpoint

### 13 License
Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.