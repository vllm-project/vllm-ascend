    !!! tip "Network environments in the Chinese mainland"

        If the current environment cannot access Hugging Face reliably, a failed model download often first appears as a connection timeout, DNS failure, or other network error. The error message might not clearly identify the download source as the cause.

        Before running the offline inference or online serving examples below, consider switching to ModelScope:

        ```bash
        export VLLM_USE_MODELSCOPE=True
        pip install "modelscope>=1.18.1,<1.38"
        ```

        If the model has already been downloaded locally, replace the model ID in the example with the local directory. You do not need to set this environment variable.
