# 重构:把 enable_fused_mc2 + mega_moe_max_tokens 嵌套进 fused_moe_config

## 设计决策
- **硬切**:删除顶层 `enable_fused_mc2`/`mega_moe_max_tokens`,只接受 `{"fused_moe_config": {"enable_fused_mc2": 1, "mega_moe_max_tokens": 65536}}`。
- **旧键直接报错**:在 `init_ascend_config` 里 pydantic 构造**之前**拦截顶层旧键(line 1336-1340 的 `enable_flashcomm1` 守卫旁),`raise ValueError` 给带迁移指引的清晰报错(而非 pydantic 的 "extra fields not permitted")。
- **子配置范式**:镜像 `AscendCompilationConfig`(`@config` dataclass + `field(default_factory=...)`,ascend_config.py:55-87/423),pydantic 直接从嵌套 dict 构造。
- **全量同步**:81 个 e2e YAML + ~15 教程 .md + .po 镜像 + 4 个 e2e Python 脚本 + additional_config.md 文档表,全部改嵌套形式(旧键现在报错,必须全改)。

## Phase 1 — ascend_config.py
- **1a** 新增 `FusedMoeConfig`(`@config`,含 `enable_fused_mc2:int=0`、`mega_moe_max_tokens:int=65536` + after-validator 校验 `mega_moe_max_tokens<=0`),放 `AscendFusionConfig` 后。
- **1b** 删顶层 `enable_fused_mc2`(409-410)和 `mega_moe_max_tokens`(394-401);子配置块(422-427)加 `fused_moe_config: FusedMoeConfig = field(default_factory=FusedMoeConfig)`。
- **1c** `init_ascend_config`(line ~1340 后)加旧键报错守卫:检测 `enable_fused_mc2`/`mega_moe_max_tokens` 顶层键 → `raise ValueError(...)` 给新写法示例。
- **1d** 验证逻辑落点:`_validate_user_input_ranges`(452-470)保留 megamoe 回滚,`self.enable_fused_mc2`→`self.fused_moe_config.enable_fused_mc2`;`derive_and_validate`(552-569、700-705)保留 enum/MiniMax/multistream/megamoe-by-config/hierarchy,改属性路径;`mega_moe_max_tokens<=0` 检查**移入** FusedMoeConfig after-validator(删 646-648)。
- 更新 docstring JSON 示例:`"enable_fused_mc2":0`+`"mega_moe_max_tokens":65536` → `"fused_moe_config":{"enable_fused_mc2":0,"mega_moe_max_tokens":65536}`。

## Phase 2 — 生产读取点(18 处)
`get_ascend_config().enable_fused_mc2`→`.fused_moe_config.enable_fused_mc2`(16 处);`.mega_moe_max_tokens`→`.fused_moe_config.mega_moe_max_tokens`(2 处)。涉及 moe_comm_method.py、ascend_forward_context.py、lora/fused_moe.py、eplb/adaptor/vllm_adaptor.py、routed_experts.py、w4a8.py、w8a8_dynamic.py。moe_comm_method.py:329/334 日志**文本串**含 "mega_moe_max_tokens" 保持不变。

## Phase 3 — 测试(~15 文件 + 1 新 UT)
- additional_config dict(test_ascend_config.py)→ `{"fused_moe_config":{...}}`,断言改 `.fused_moe_config.<field>`。
- SimpleNamespace mock(硬报错→嵌套 namespace):test_ascend_forward_context.py、test_fused_moe.py、test_lora.py。
- MagicMock mock(静默误过→显式设):test_moe_comm_method.py、eplb/w4a8/eagle_proposer/speculators/test_platform/w8a8_dynamic。
- 新增 `test_init_ascend_config_raises_on_legacy_flat_keys`:传 `{"enable_fused_mc2":1}` 断言抛 `ValueError` 且消息含 "fused_moe_config"。

## Phase 4 — e2e Python 脚本(4 文件)
- `_FEATURE_CONFIGS` 查找改读 `additional_config["fused_moe_config"]["enable_fused_mc2"]`(test_single_node.py:352、test_multi_node.py:34)。
- utils.py:182 `.get("enable_fused_mc2")`→`.get("fused_moe_config",{}).get("enable_fused_mc2")`。
- test_kimi_k3.py:351 扁平→嵌套。

## Phase 5 — 批量改 YAML + 教程 .md + .po(81 yaml + ~15 md + .po)
Python 脚本扫 `tests/e2e/**/*.yaml`、`docs/source/tutorials/**/*.md`,正则把单行 JSON 内 `"enable_fused_mc2"\s*:\s*(\d)` 替换成 `"fused_moe_config":{"enable_fused_mc2":$1}`(无预存 fused_moe_config,安全;边角回退 JSON parse 合并)。`.po` 镜像同步。验证:`grep -rn '"enable_fused_mc2"' tests/e2e/ docs/` 仅剩嵌套内。

## Phase 6 — additional_config.md 文档表
- `:69`/`:74` 标注路径 `fused_moe_config.mega_moe_max_tokens`/`fused_moe_config.enable_fused_mc2`。
- `:23` 迁移表目标列→`fused_moe_config.enable_fused_mc2`。

## Phase 7 — 验证
1. `ruff format/check` + `bash format.sh ci`(含 markdownlint)
2. `pytest tests/ut/test_ascend_config.py tests/ut/ops/test_moe_comm_method.py tests/ut/test_ascend_forward_context.py tests/ut/ops/test_fused_moe.py tests/ut/lora/test_lora.py`
3. `grep -rn 'get_ascend_config()\.enable_fused_mc2\|get_ascend_config()\.mega_moe_max_tokens' vllm_ascend/ tests/` 无残留顶层访问
4. `grep -rn '"enable_fused_mc2"' tests/e2e/ docs/` 全嵌套化

## 备注
- 旧键报错是硬切 + 友好 UX:外部旧配置启动即报错并给确切新写法,不静默失效。需在 release notes 标注 breaking change。
- 验证逻辑保留父类仅改属性路径,最低风险;仅 `mega_moe_max_tokens` 范围检查下沉子配置。
- 配置治理机制(迁移注册表 + CI lint)另行开 PR,本次不涉及。