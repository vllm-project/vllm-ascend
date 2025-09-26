### 1 Performance Accuracy Script
This script automates the deployment and execution of performance accuracy for SGLang-based inference services. It handles service deployment via Ansible, monitors service readiness, executes performance tests, and collects results.

### 2 Features
* Automated Deployment: Deploys inference services using Ansible playbooks
* Service Monitoring: Verifies service readiness through API health checks
* Performance Testing: Executes standardized accuracy tests
* Result Collection: Automatically retrieves accuracy results
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
python perfomance_accuracy.py \
  --accuracy_src_dir <remote_results_dir> \
  --accuracy_target_dir <local_results_dir> \
  --host <proxy_host> \
  --port <proxy_port> \
  --model <model_path> \
  --ansible-ssh-host <ansible_host> \
  --accuracy-ssh-host <accuracy_host> \
  [authentication options] \
  [ansible options]
```
### 5 Full Parameter List

Parameter                 | Required | Default                             | Description
---------------------------|----------|-------------------------------------|---------------------------------------------------------------------
 --accuracy-src-dir       | Yes      | -                                   | Remote directory for accuracy results (e.g., /path/to/0808)
 --accuracy-target-dir    | Yes      | -                                   | Local directory to save results
 --host                    | Yes      | -                                   | Global proxy host address
 --port                    | No       | 8011                                | Global proxy port
 --model                   | Yes      | -                                   | Path to model files
 --ansible-ssh-host        | Yes      | -                                   | Ansible server host address
 --ansible-ssh-port        | No       | 22                                  | Ansible server SSH port
 --ansible-ssh-username    | No       | root                                | Ansible server username
 --accuracy-ssh-host      | Yes      | -                                   | Accuracy server host address
 --accuracy-ssh-port      | No       | 22                                  | Accuracy server SSH port
 --accuracy-ssh-username  | No       | root                                | Accuracy server username
 --ansible-ssh-password    | Either   | -                                   | Ansible server password
 --ansible-ssh-key-file    | Either   | -                                   | Ansible server private key file
 --accuracy-ssh-password  | Either   | -                                   | Accuracy server password
 --accuracy-ssh-key-file  | Either   | -                                   | Accuracy server private key file
 --ansible-tag  | Either   | -                                   | Accuracy server private key file
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
* Connect to accuracy results server

#### 2 Service Deployment:
* Execute Ansible playbook with specified tag
* Retry deployment up to 3 times if service doesn't start
#### 3 Service Readiness Check:
* Verify service availability via API endpoint
* /v1/chat/completions endpoint is monitored
* Timeout configurable via --wait-timeout

#### 4 Performance Accuracy:
* Automatically execute accuracy test cases(The third child stole food) and record the result
* Monitor remote directory for result files

#### 5 Result Collection:
* Copy accuracy results from remote server to local machine
* Results stored in specified accuracy_target_dir

### 8 Example
```
python performance_accuracy.py \
  --accuracy-src-dir /data/accuracy/0815 \
  --accuracy-target-dir ./results \
  --host 192.168.1.100 \
  --port 8011 \
  --model /data/models/DeepSeek-R1 \
  --ansible-ssh-host 192.168.1.101 \
  --ansible-ssh-key-file /xx/xx/ansible_server.pem \
  --accuracy-ssh-host 192.168.1.102 \
  --accuracy-ssh-key-file /xx/xx/accuracy_server.pem \
  --inventory-path omni_infer_inventory.yml \
  --playbook-path omni_infer_server.yml \
  --wait-timeout 300
```

### 9 Output Files
Accuracy results include:
* accuracy_result_log.txt: record the answer and output the accuracy

### 10 Logging
The script generates detailed logs with timestamps:
```
2025-08-15 14:30:45 - root - INFO - <<< Starting initial deployment...
2025-08-15 14:31:15 - root - INFO - <<< Executing remote command (Ansible deployment): ansible-playbook -i my_inventory.yml deploy_playbook.yml
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