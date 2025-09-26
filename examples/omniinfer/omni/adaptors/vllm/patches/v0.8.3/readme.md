## How to Apply and Install the Patch

1. **Apply the Patch**  
    Run the provided bash script to apply the patch and install vllm:
    ```bash
    bash infer_engines/bash_install_code_v0.8.3.sh
    ```

2. **Compatibility Notice**  
    - This patch is designed **only** for the standard [vllm tags/v0.8.3](https://github.com/vllm-project/vllm) from GitHub.
    - If you need to modify the code further, please do so **after** applying the patch.

3. **Re-installation Requirement**  
    - If you did **not** install vllm using `pip install -e .`, please re-install vllm after applying the patch to ensure all changes take effect.

---

**Note:**  
Always verify your vllm version and backup your code before applying patches.
