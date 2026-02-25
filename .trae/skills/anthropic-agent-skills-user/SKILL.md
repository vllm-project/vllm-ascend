---
name: "anthropic-agent-skills-user"
description: "Extracts inference examples from markdown documentation into bash test scripts. Invoke when the user wants to generate test cases from documentation."
---

# Anthropic Agent Skills User

This skill automates the extraction of inference examples that can be run on single Ascend NPU node from markdown documentation and creates robust bash test scripts.

## Functionality

1.  **Reads Documentation**: Parses markdown files (e.g., `docs/source/tutorials/*.md`) to find inference commands that can run on single Ascend NPU node.
2.  **Identifies Command Types**:
    *   **Server Startup**: Commands starting with `vllm serve`. These must be run in the background.
    *   **Client Requests**: Commands using `curl` or `vllm bench` that require a running server.
        *   **Parameter Preservation**: Ensure that if `--tokenizer` is present in the documentation (especially for `vllm-ascend/` models), it is preserved in the extracted command, even if the `--model` argument is modified to match the served model name.
3.  **Generates Test Script**: Creates a bash script (`test_<model>_inference.sh`) with:
    *   **Environment Setup**: Includes logic to detect `vllm` executable in common paths if not in PATH.
    *   **Server Management**:
        *   Starts `vllm serve` in the background (`&`).
        *   Captures the server PID.
        *   **Health Check**: Implements a `wait_for_server` function that:
            *   Polls `http://localhost:8000/health` (or configured port).
            *   **Monitors PID**: Checks if the server process is still running during the wait loop to fail fast on crash.
        *   **Cleanup**: Uses `trap` to ensure the server is killed when the script exits (success or failure).
    *   **Wrapper Function**: Includes a `run_test_case` helper for error handling and logging.
        *   **Log Checking**: Checks command output for failure indicators (e.g., "Failed requests", "Error", "404 Not Found") even if the exit code is 0.
    *   **Environment Isolation**: Wraps each test case in a subshell `( ... )` to prevent environment variable leakage.
    *   **Global vs Local Envs**: Correctly identifies and places global variables outside subshells and test-specific variables inside.
    *   **Non-Blocking Execution**: Ensures failure in one test case does not stop the entire script.

## Usage Instructions

When invoked, the skill should:
1.  Analyze the provided documentation file to identify the model name and relevant commands.
2.  Create a bash script named `test_<model_name>_inference.sh`.
3.  Include environment detection logic for `vllm`.
4.  Extract the `vllm serve` command (including its environment variables).
5.  Generate code to start the server in the background and wait for it (with PID monitoring).
6.  Extract `curl` and `vllm bench` commands as test cases.
    *   **CRITICAL**: Check for `--tokenizer` in the original command. If present, it MUST be included in the test script command, even if `--model` is changed to the short name (e.g., `qwen3`).
7.  Make the script executable.

## Example Output Structure

```bash
#!/bin/bash

# Setup Environment
setup_env() {
    if ! command -v vllm &> /dev/null; then
        echo "vllm not found in PATH. Searching..."
        VLLM_PATH=$(find /usr/local/python* -name vllm -type f -executable 2>/dev/null | head -n 1)
        if [ -n "$VLLM_PATH" ]; then
            echo "Found vllm at $VLLM_PATH"
        else
            echo "Error: vllm not found."
            exit 1
        fi
    fi
}
setup_env

# Function to run a test case
run_test_case() {
    echo "Running: $1"
    # Capture output and exit code
    OUTPUT=$(eval "$2" 2>&1)
    EXIT_CODE=$?
    echo "$OUTPUT"
    
    # Check for specific failure patterns in output even if exit code is 0
    if [[ "$OUTPUT" == *"Failed requests"* ]] || [[ "$OUTPUT" == *"Error"* ]] || [[ "$OUTPUT" == *"404 Not Found"* ]]; then
        # Check if Failed requests is not 0
        if echo "$OUTPUT" | grep -q "Failed requests:\s*[1-9]"; then
             EXIT_CODE=1
        fi
    fi

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Test Case '$1' PASSED"
    else
        echo "Test Case '$1' FAILED"
    fi
    echo "--------------------------------------------------"
}

# Function to wait for server
wait_for_server() {
    local pid=$1
    echo "Waiting for server to start..."
    local retries=0
    local max_retries=120 # 20 minutes (10s * 120) for large models
    while ! curl -s http://localhost:8000/health > /dev/null; do
        if ! kill -0 $pid 2>/dev/null; then
            echo "Server process $pid has terminated unexpectedly."
            return 1
        fi
        sleep 10
        retries=$((retries+1))
        if [ $retries -ge $max_retries ]; then
            echo "Server failed to start within timeout."
            return 1
        fi
        echo "Waiting... ($retries/$max_retries)"
    done
    echo "Server is ready!"
}

# Global Envs
export GLOBAL_VAR=value

# Start Server
echo "Starting vLLM server..."
(
    export SERVER_ENV=value
    # Use setsid or just & to run in background
    vllm serve ... &
    echo $! > vllm_server.pid
)
# Main scope
export SERVER_ENV=value
vllm serve ... &
SERVER_PID=$!
echo "Server started with PID $SERVER_PID"

# Ensure server is killed on exit
trap 'kill $SERVER_PID' EXIT

# Wait for readiness
wait_for_server $SERVER_PID

# Test Case 1
(
    export LOCAL_VAR=value
    CMD="curl ..."
    run_test_case "Description" "$CMD"
)
```
