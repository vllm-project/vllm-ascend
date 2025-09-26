# Copyright (c) HuaWei Technologies Co., Ltd. 2025-2025. All rights reserved

import os


class EmsEnv:
    enable_vllm_ems: bool = os.environ.get("ENABLE_VLLM_EMS", "0") == "1"
    enable_ems_profiling = os.environ.get("ENABLE_EMS_PROFILING", "0") == "1"
    llm_engine = os.environ.get("LLM_ENGINE", "vllm")
    model_id = os.environ.get("MODEL_ID", "cc_kvstore@_@ds_default_ns_001")
    access_id = os.environ.get("ACCELERATE_ID", "access_id")
    access_key = os.environ.get("ACCELERATE_KEY", "")
    enable_write_rcache: bool = os.environ.get("ENABLE_WRITE_RCACHE", "1") == "1"
    enable_read_local_only: bool = os.environ.get("ENABLE_READ_LOCAL_ONLY", "0") == "1"
    ems_store_local: bool = os.environ.get("EMS_STORE_LOCAL", "0") == "1"