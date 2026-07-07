"""MkDocs hook: set per-page and per-section titles from a bilingual mapping.

Page and section titles for the English and Chinese sites are centralized
in the ``TITLES`` and ``SECTION_TITLES`` dicts below. The active language
is selected via the ``DOCS_LANG`` environment variable (default: ``en``).

Sections are keyed by their first child file path (the section index
page), which is stable as long as the nav structure does not change.

For pages and sections not present in the mapping, titles fall back to
the H1 in the markdown (the mkdocs default behaviour). The Chinese site
also relies on ``tools/generate_zh_docs.py`` to translate those H1s from
``.po`` files.
"""

import os

TITLES = {
    "community/contributors.md": {"en": "Contributors", "zh": "贡献者"},
    "community/governance.md": {"en": "Governance", "zh": "治理"},
    "community/index.md": {"en": "Overview", "zh": "概览"},
    "community/issue-workflow-guidelines.md": {"en": "Issue Workflow Guidelines", "zh": "Issue 工作流指南"},
    "community/slash-commands.md": {"en": "Slash Commands", "zh": "Slash 命令"},
    "community/user_stories/llamafactory.md": {"en": "LLaMA-Factory", "zh": "LLaMA-Factory"},
    "community/versioning_policy.md": {"en": "Versioning Policy", "zh": "版本策略"},
    "developer_guide/Design_Documents/ACL_Graph.md": {"en": "ACL Graph", "zh": "ACL 图"},
    "developer_guide/Design_Documents/KV_Cache_Pool_Guide.md": {
        "en": "KV Cache Pool Guide",
        "zh": "KV Cache 池指南",
    },
    "developer_guide/Design_Documents/ModelRunner_prepare_inputs.md": {
        "en": "ModelRunner Prepare Inputs",
        "zh": "ModelRunner 输入准备",
    },
    "developer_guide/Design_Documents/add_custom_aclnn_op.md": {
        "en": "Add Custom ACLNN Op",
        "zh": "添加自定义 ACLNN 算子",
    },
    "developer_guide/Design_Documents/context_parallel.md": {"en": "Context Parallel", "zh": "上下文并行"},
    "developer_guide/Design_Documents/cpu_binding.md": {"en": "CPU Binding", "zh": "CPU 绑定"},
    "developer_guide/Design_Documents/disaggregated_prefill.md": {
        "en": "Disaggregated Prefill",
        "zh": "分离式预填充",
    },
    "developer_guide/Design_Documents/dynamic_chunked_pipeline_parallel.md": {
        "en": "Dynamic Chunked Pipeline Parallel",
        "zh": "动态分块流水线并行",
    },
    "developer_guide/Design_Documents/eplb_swift_balancer.md": {
        "en": "EPLB Swift Balancer",
        "zh": "EPLB 快速均衡器",
    },
    "developer_guide/Design_Documents/npugraph_ex.md": {"en": "NPUGraph Ex", "zh": "NPUGraph 扩展"},
    "developer_guide/Design_Documents/patch.md": {"en": "Patch", "zh": "补丁"},
    "developer_guide/Design_Documents/quantization.md": {"en": "Quantization", "zh": "量化"},
    "developer_guide/contribution/doc_writing.md": {"en": "Doc Writing", "zh": "文档编写"},
    "developer_guide/contribution/e2e_ci_test.md": {"en": "E2E CI Test", "zh": "端到端 CI 测试"},
    "developer_guide/contribution/multi_node_test.md": {"en": "Multi-Node Test", "zh": "多节点测试"},
    "developer_guide/contribution/nightly_ci_test.md": {"en": "Nightly CI Test", "zh": "夜间 CI 测试"},
    "developer_guide/contribution/testing.md": {"en": "Testing", "zh": "测试"},
    "developer_guide/evaluation/using_ais_bench.md": {"en": "Using AISBench", "zh": "使用 AISBench"},
    "developer_guide/evaluation/using_evalscope.md": {"en": "Using EvalScope", "zh": "使用 EvalScope"},
    "developer_guide/evaluation/using_lm_eval.md": {"en": "Using lm_eval", "zh": "使用 lm_eval"},
    "developer_guide/evaluation/using_opencompass.md": {"en": "Using OpenCompass", "zh": "使用 OpenCompass"},
    "developer_guide/index.md": {"en": "Overview", "zh": "概览"},
    "developer_guide/performance_and_debug/msprobe_guide.md": {"en": "msprobe Guide", "zh": "msprobe 指南"},
    "developer_guide/performance_and_debug/optimization_and_tuning.md": {
        "en": "Optimization and Tuning",
        "zh": "优化与调优",
    },
    "developer_guide/performance_and_debug/performance_benchmark.md": {
        "en": "Performance Benchmark",
        "zh": "性能基准测试",
    },
    "developer_guide/performance_and_debug/service_profiling_guide.md": {
        "en": "Service Profiling Guide",
        "zh": "服务性能分析指南",
    },
    "faqs.md": {"en": "FAQs", "zh": "常见问题"},
    "getting_started.md": {"en": "Overview", "zh": "概览"},
    "index.md": {"en": "Home", "zh": "首页"},
    "installation.md": {"en": "Installation", "zh": "安装"},
    "quick_start.md": {"en": "Quick Start", "zh": "快速开始"},
    "tutorials/features/dynamic_chunked_pipeline_parallel.md": {
        "en": "Dynamic Chunked Pipeline Parallel",
        "zh": "动态分块流水线并行",
    },
    "tutorials/features/long_sequence_context_parallel_multi_node.md": {
        "en": "Long Sequence Context Parallel (Multi Node)",
        "zh": "长序列上下文并行（多节点）",
    },
    "tutorials/features/long_sequence_context_parallel_single_node.md": {
        "en": "Long Sequence Context Parallel (Single Node)",
        "zh": "长序列上下文并行（单节点）",
    },
    "tutorials/features/pd_colocated_mooncake_multi_instance.md": {
        "en": "PD Colocated (Mooncake, Multi-Instance)",
        "zh": "PD 共置（Mooncake，多实例）",
    },
    "tutorials/features/pd_disaggregation_mooncake_multi_node.md": {
        "en": "PD Disaggregation (Mooncake, Multi Node)",
        "zh": "PD 分离（Mooncake，多节点）",
    },
    "tutorials/features/pd_disaggregation_mooncake_single_node.md": {
        "en": "PD Disaggregation (Mooncake, Single Node)",
        "zh": "PD 分离（Mooncake，单节点）",
    },
    "tutorials/features/ray.md": {"en": "Ray", "zh": "Ray"},
    "tutorials/features/suffix_speculative_decoding.md": {
        "en": "Suffix Speculative Decoding",
        "zh": "后缀推测解码",
    },
    "tutorials/hardwares/310p.md": {"en": "310P", "zh": "310P"},
    "tutorials/models/DeepSeek-R1.md": {"en": "DeepSeek-R1", "zh": "DeepSeek-R1"},
    "tutorials/models/DeepSeek-V3.1.md": {"en": "DeepSeek-V3.1", "zh": "DeepSeek-V3.1"},
    "tutorials/models/DeepSeek-V3.2.md": {"en": "DeepSeek-V3.2", "zh": "DeepSeek-V3.2"},
    "tutorials/models/DeepSeek-V4-Flash.md": {"en": "DeepSeek-V4-Flash", "zh": "DeepSeek-V4-Flash"},
    "tutorials/models/DeepSeek-V4-Pro.md": {"en": "DeepSeek-V4-Pro", "zh": "DeepSeek-V4-Pro"},
    "tutorials/models/DeepSeekOCR2.md": {"en": "DeepSeekOCR2", "zh": "DeepSeekOCR2"},
    "tutorials/models/GLM4.x.md": {"en": "GLM-4.x", "zh": "GLM-4.x"},
    "tutorials/models/GLM5.2.md": {"en": "GLM-5.2", "zh": "GLM-5.2"},
    "tutorials/models/GLM5.md": {"en": "GLM-5", "zh": "GLM-5"},
    "tutorials/models/Hunyuan-A13B-Instruct.md": {"en": "Hunyuan-A13B-Instruct", "zh": "Hunyuan-A13B-Instruct"},
    "tutorials/models/Hy3-preview.md": {"en": "Hy3-preview", "zh": "Hy3-preview"},
    "tutorials/models/InternVL3.5.md": {"en": "InternVL3.5", "zh": "InternVL3.5"},
    "tutorials/models/Kimi-K2-Thinking.md": {"en": "Kimi-K2-Thinking", "zh": "Kimi-K2-Thinking"},
    "tutorials/models/Kimi-K2.5.md": {"en": "Kimi-K2.5", "zh": "Kimi-K2.5"},
    "tutorials/models/Kimi-K2.6.md": {"en": "Kimi-K2.6", "zh": "Kimi-K2.6"},
    "tutorials/models/LLaVA-OneVision-Qwen2-0.5B-OV.md": {
        "en": "LLaVA-OneVision-Qwen2-0.5B-OV",
        "zh": "LLaVA-OneVision-Qwen2-0.5B-OV",
    },
    "tutorials/models/MiniMax-M2.md": {"en": "MiniMax-M2", "zh": "MiniMax-M2"},
    "tutorials/models/Minitron-8B-Base.md": {"en": "Minitron-8B-Base", "zh": "Minitron-8B-Base"},
    "tutorials/models/Mixtral-8x7B-Instruct-v0.1.md": {
        "en": "Mixtral-8x7B-Instruct-v0.1",
        "zh": "Mixtral-8x7B-Instruct-v0.1",
    },
    "tutorials/models/PaddleOCR-VL.md": {"en": "PaddleOCR-VL", "zh": "PaddleOCR-VL"},
    "tutorials/models/Qwen-VL-Dense.md": {"en": "Qwen-VL-Dense", "zh": "Qwen-VL-Dense"},
    "tutorials/models/Qwen2.5-Math-RM-72B.md": {"en": "Qwen2.5-Math-RM-72B", "zh": "Qwen2.5-Math-RM-72B"},
    "tutorials/models/Qwen3-235B-A22B.md": {"en": "Qwen3-235B-A22B", "zh": "Qwen3-235B-A22B"},
    "tutorials/models/Qwen3-30B-A3B.md": {"en": "Qwen3-30B-A3B", "zh": "Qwen3-30B-A3B"},
    "tutorials/models/Qwen3-ASR-1.7B.md": {"en": "Qwen3-ASR-1.7B", "zh": "Qwen3-ASR-1.7B"},
    "tutorials/models/Qwen3-Coder-30B-A3B.md": {"en": "Qwen3-Coder-30B-A3B", "zh": "Qwen3-Coder-30B-A3B"},
    "tutorials/models/Qwen3-Dense.md": {"en": "Qwen3-Dense", "zh": "Qwen3-Dense"},
    "tutorials/models/Qwen3-Next.md": {"en": "Qwen3-Next", "zh": "Qwen3-Next"},
    "tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md": {
        "en": "Qwen3-Omni-30B-A3B-Thinking",
        "zh": "Qwen3-Omni-30B-A3B-Thinking",
    },
    "tutorials/models/Qwen3-VL-235B-A22B-Instruct.md": {
        "en": "Qwen3-VL-235B-A22B-Instruct",
        "zh": "Qwen3-VL-235B-A22B-Instruct",
    },
    "tutorials/models/Qwen3-VL-30B-A3B-Instruct.md": {
        "en": "Qwen3-VL-30B-A3B-Instruct",
        "zh": "Qwen3-VL-30B-A3B-Instruct",
    },
    "tutorials/models/Qwen3-VL-Embedding.md": {"en": "Qwen3-VL-Embedding", "zh": "Qwen3-VL-Embedding"},
    "tutorials/models/Qwen3-VL-Reranker.md": {"en": "Qwen3-VL-Reranker", "zh": "Qwen3-VL-Reranker"},
    "tutorials/models/Qwen3.5-27B-Qwen3.6-27B.md": {
        "en": "Qwen3.5-27B & Qwen3.6-27B",
        "zh": "Qwen3.5-27B & Qwen3.6-27B",
    },
    "tutorials/models/Qwen3.5-397B-A17B.md": {"en": "Qwen3.5-397B-A17B", "zh": "Qwen3.5-397B-A17B"},
    "tutorials/models/Qwen3.6-35B-A3B.md": {"en": "Qwen3.6-35B-A3B", "zh": "Qwen3.6-35B-A3B"},
    "tutorials/models/Qwen3_embedding.md": {"en": "Qwen3 Embedding", "zh": "Qwen3 Embedding"},
    "tutorials/models/Qwen3_reranker.md": {"en": "Qwen3 Reranker", "zh": "Qwen3 Reranker"},
    "tutorials/models/gpt-oss-120b.md": {"en": "gpt-oss-120b", "zh": "gpt-oss-120b"},
    "user_guide/configuration/additional_config.md": {"en": "Additional Configuration", "zh": "附加配置"},
    "user_guide/configuration/env_vars.md": {"en": "Environment Variables", "zh": "环境变量"},
    "user_guide/deployment_guide/using_mindie_motor.md": {"en": "Using MindIE Motor", "zh": "使用 MindIE Motor"},
    "user_guide/deployment_guide/using_volcano_kthena.md": {"en": "Using Volcano Kthena", "zh": "使用 Volcano Kthena"},
    "user_guide/feature_guide/Ai_QoS_introduction_en.md": {"en": "AI QoS", "zh": "AI QoS"},
    "user_guide/feature_guide/Fine_grained_TP.md": {"en": "Fine-grained TP", "zh": "细粒度 TP"},
    "user_guide/feature_guide/batch_invariance.md": {"en": "Batch Invariance", "zh": "批次不变性"},
    "user_guide/feature_guide/context_parallel.md": {"en": "Context Parallel", "zh": "上下文并行"},
    "user_guide/feature_guide/cpu_binding.md": {"en": "CPU Binding", "zh": "CPU 绑定"},
    "user_guide/feature_guide/dynamic_batch.md": {"en": "Dynamic Batch", "zh": "动态批次"},
    "user_guide/feature_guide/dynamic_chunk_pipeline_parallel.md": {
        "en": "Dynamic Chunk Pipeline Parallel",
        "zh": "动态分块流水线并行",
    },
    "user_guide/feature_guide/epd_disaggregation.md": {"en": "EPD Disaggregation", "zh": "EPD 分离"},
    "user_guide/feature_guide/expert_parallelism_load_balancer.md": {
        "en": "Expert Parallelism Load Balancer",
        "zh": "专家并行负载均衡",
    },
    "user_guide/feature_guide/external_dp.md": {"en": "External DP", "zh": "外部数据并行"},
    "user_guide/feature_guide/flash_attention.md": {"en": "Flash Attention", "zh": "Flash 注意力"},
    "user_guide/feature_guide/graph_mode.md": {"en": "Graph Mode", "zh": "图模式"},
    "user_guide/feature_guide/kv_cache_cpu_offload.md": {"en": "KV Cache CPU Offload", "zh": "KV Cache CPU 卸载"},
    "user_guide/feature_guide/kv_pool.md": {"en": "KV Pool", "zh": "KV 池"},
    "user_guide/feature_guide/large_scale_ep.md": {"en": "Large Scale EP", "zh": "大规模 EP"},
    "user_guide/feature_guide/layer_sharding.md": {"en": "Layer Sharding", "zh": "层分片"},
    "user_guide/feature_guide/lmcache_ascend_deployment.md": {
        "en": "LMCache Ascend Deployment",
        "zh": "LMCache Ascend 部署",
    },
    "user_guide/feature_guide/lora.md": {"en": "LoRA", "zh": "LoRA"},
    "user_guide/feature_guide/netloader.md": {"en": "NetLoader", "zh": "NetLoader"},
    "user_guide/feature_guide/quantization.md": {"en": "Quantization", "zh": "量化"},
    "user_guide/feature_guide/rfork.md": {"en": "RFork", "zh": "RFork"},
    "user_guide/feature_guide/sequence_parallelism.md": {"en": "Sequence Parallelism", "zh": "序列并行"},
    "user_guide/feature_guide/sleep_mode.md": {"en": "Sleep Mode", "zh": "休眠模式"},
    "user_guide/feature_guide/speculative_decoding.md": {"en": "Speculative Decoding", "zh": "推测解码"},
    "user_guide/feature_guide/structured_output.md": {"en": "Structured Output", "zh": "结构化输出"},
    "user_guide/feature_guide/ucm_deployment.md": {"en": "UCM Deployment", "zh": "UCM 部署"},
    "user_guide/feature_guide/weight_prefetch.md": {"en": "Weight Prefetch", "zh": "权重预取"},
    "user_guide/index.md": {"en": "Overview", "zh": "概览"},
    "user_guide/release_notes.md": {"en": "Release Notes", "zh": "发布说明"},
    "user_guide/support_matrix/feature_matrix.md": {"en": "Feature Matrix", "zh": "功能矩阵"},
    "user_guide/support_matrix/supported_features.md": {"en": "Supported Features", "zh": "支持的功能"},
    "user_guide/support_matrix/supported_models.md": {"en": "Supported Models", "zh": "支持的模型"},
}

# Section titles keyed by the original English title (the dict key in
# the nav config). This is more stable than matching by first child,
# which can shift when a translated page is missing from a build.
SECTION_TITLES = {
    "Getting Started": {"en": "Getting Started", "zh": "快速开始"},
    "Model Tutorials": {"en": "Model Tutorials", "zh": "模型教程"},
    "Feature Tutorials": {"en": "Feature Tutorials", "zh": "功能教程"},
    "Hardware Tutorials": {"en": "Hardware Tutorials", "zh": "硬件教程"},
    "User Guide": {"en": "User Guide", "zh": "用户指南"},
    "Configuration": {"en": "Configuration", "zh": "配置"},
    "Feature Guide": {"en": "Feature Guide", "zh": "功能指南"},
    "Deployment Guide": {"en": "Deployment Guide", "zh": "部署指南"},
    "Developer Guide": {"en": "Developer Guide", "zh": "开发者指南"},
    "Contribution": {"en": "Contribution", "zh": "贡献"},
    "Design Documents": {"en": "Design Documents", "zh": "设计文档"},
    "Evaluation": {"en": "Evaluation", "zh": "评估"},
    "Performance and Debug": {"en": "Performance and Debug", "zh": "性能与调试"},
    "Community": {"en": "Community", "zh": "社区"},
    "User Stories": {"en": "User Stories", "zh": "用户故事"},
}


def _apply_section_titles(items, lang):
    """Walk the nav tree and update section titles.

    Each section is matched by its current title (the original English
    dict key from the nav config); see ``SECTION_TITLES`` for the
    mapping. Section ``title`` is a plain attribute on ``Section``
    objects, so it can be assigned directly.
    """
    for item in items:
        if not item.is_section:
            continue
        titles = SECTION_TITLES.get(item.title)
        if titles and titles.get(lang):
            item.title = titles[lang]
        _apply_section_titles(item.children, lang)


def on_nav(nav, *, config, files, **kwargs):
    """Translate section titles.

    Page titles are set in ``on_page_markdown`` instead, because
    ``Page.read_source()`` (which runs after this event) rebuilds
    ``page.meta`` from front matter and would discard any
    ``meta['title']`` we set here.
    """
    lang = os.environ.get("DOCS_LANG", "en")
    _apply_section_titles(nav.items, lang)
    return nav


def on_page_markdown(markdown, page, **kwargs):
    """Override the page title for the active language (DOCS_LANG)."""
    lang = os.environ.get("DOCS_LANG", "en")
    titles = TITLES.get(page.file.src_path)
    if titles and titles.get(lang):
        # ``title`` is a ``weak_property`` without a setter; assigning to
        # ``page.title`` shadows the descriptor with an instance attribute.
        page.title = titles[lang]
    return markdown
