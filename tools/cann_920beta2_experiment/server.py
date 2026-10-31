"""带实验诊断的 vLLM 入口；spawn 子进程也安装相同的记录钩子。"""

import os
from pathlib import Path

from diagnostic_hooks import install_trace, trace_fusion_config

diagnostics = Path(os.environ["VLLM_CACHE_ROOT"]) / "diagnostics"
install_trace(diagnostics)
trace_fusion_config(diagnostics)

if __name__ == "__main__":
    from vllm.entrypoints.cli.main import main

    main()
