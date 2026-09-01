# 检视计划设计

file-review 与 pr-review 共用,在 code-summarize / api-prestudy / docs-detect 返回后派发。

## 派发

```
Agent({
  subagent_type: "general",
  description: "检视计划设计",
  prompt: "检视计划设计

【输入】
- 代码概要路径：{code_summary_path}
- API 预研路径：{api_prestudy_path}（仅 Kernel 侧,若存在）
- docs_input：{docs_input}（设计文档路径/目录或空）
- scope_hint：{scope_hint}（条例类别,空则全量）
- （PR 模式）diff 路径：{diff_file_path}
- 代码文件路径：{file_input 或 repo_path}
- 检视类型：{review_type}（file 或 pr）
- 检视标识：{review_id}（PR 检视时为 PR 号；文件检视时留空）
- mode：{minimal|compact|standard}（由 code-fetch/file-review 调 workflow.review_mode.py 判定,阈值见 scripts/workflow.review-thresholds.yaml）
- guidance：{对应 mode 的编排指导,同上脚本输出}

【执行】

Step 0 创建 yaml 输出目录。调用 `{skill_base}/scripts/workflow.create_review_dir.py` 创建本次检视的结构化 yaml 输出目录：
- 命令：`python3 {skill_base}/scripts/workflow.create_review_dir.py --type {review_type} --id {review_id}`
- 捕获 stdout 输出作为 `yaml_dir`（绝对路径，如 `/tmp/pr1234_a3b7x9`）
- 将 `yaml_dir` 作为返回值之一回传主 Agent。**主 Agent 保留 yaml_dir 用于启动 collector 服务和阶段2/3 脚本调用，不传给 clause-review / design-check 子 Agent**（子 Agent 通过 collector HTTP 端点提交 yaml，不接触目录路径）
- 目录创建失败（exit code 非 0）则终止，报错返回

Step 1 读上游产出。Read 概要提取侧别/API 调用索引/变量溯源/跨文件防御摘要/函数清单;Read API 预研（若有）。检视模式直接用上游传入的 mode（不再自行统计行数判档）。
提取 references 触发关键词:执行 sed -n '/<适用>/,/<\/适用>/p' references/*.md,对领域=true 文件取 触发: 字段关键词去重。
禁止 Read 源码仓;reference 只 sed 取快速索引段。

Step 2 风险研判。读概要变量溯源表「来源类型」列:外部输入→高风险（数值安全/输入验证/数组索引类必查）;TilingData/硬件配置/编译期常量→上游已防御（降级仅核对）;其余→正常检视。变量溯源缺失→全部正常检视,不产出仅核对清单。

Step 3 声明式匹配。基于 <适用> 区块逐文件判定:语言匹配（规则语言=代码语言或不限）;侧别匹配（规则侧别=All 或含代码侧别）;默认启用=false 跳过;领域规则（领域=true）:触发:必须触发→无条件通过,否则概要 API 索引/变量溯源命中其触发关键词→通过（概要不足时 diff/代码 grep 兜底）,未命中→跳过。文件 <适用> 含 排除场景: 字段则按其自定义逻辑排除。

Step 4 条例级侧别过滤。对匹配文件 sed 取快速索引段,逐条例按侧别过滤:Kernel 侧保留 [适用:All]+[适用:Kernel];Tiling 侧保留 [适用:All]+[适用:Tiling]/[适用:Host];混合侧保留全部并标注;无标记按文件全局侧别。

Step 4.5 范围过滤。scope_hint 非空时:类别名（如\"数值安全\"）→ 只保留该类条例;\"secure\" → 只保留 cpp-secure+ascendc-topk;空 → 全保留。

Step 5 内容筛查（所有模式）。
PR 模式:只认概要 API 调用索引/函数清单/常量清单中「变更?」列标记为新增/修改的模式;空白既有模式和已删除模式跳过。文件检视模式:看概要全表。
示例:除零条例→变量溯源含新增/修改除法取模才保留;DataCopy 对齐→API 索引含新增/修改 DataCopy 才保留;指针判空→变量溯源含指针解引用;外部输入校验→变量溯源含外部输入变量。
无对应模式→跳过,记入跳过清单写理由。Step 2 判为上游已防御→记入仅核对清单。领域=true 且 触发:必须触发 的文件不筛查,无条件保留。兜底:概要表缺失→diff grep 兜底（PR 限定 diff 变更行）;防御摘要缺失→不产出仅核对清单。

Step 6 合并 + 容量。
6.1 跨文档同类合并:按快速索引「类别」字段合并,不同文档同类别→同组,同文档内保持一起。
6.2 同根因合并:同一代码位置（行号/函数名）被 ≥2 条条例命中→合并;多条条例命中同一变量且同风险等级→合并。禁止合并:跨侧别;红线/必触发条例独立成组。合并后容量超限→按主条例拆同波多子组。
6.3 容量:Read core/review-load-balance.md 获取规则,扫描各文件 <检视负载> 头取容量,合并组取 min。

Step 7 分组 + 波次。按上游传入的 mode 走对应 guidance 执行（mode/guidance 由 code-fetch/file-review 调 workflow.review_mode.py 判定,阈值见 scripts/workflow.review-thresholds.yaml,plan-design 不自行判档）。
红线（ascendc-red-line）与 topK（ascendc-topk）永远是最高优先级:优先级序中排在 1 之前,独立成组（不参与 Step 6.2 合并,见 6.2 禁止合并项）,必进第一波;模式分档与每波 ≤6 组限制不削弱其优先级——分波/挤波时若第一波已满,红线/topK 仍优先占位,其余条例让位后延。
cpp-style 专项（所有模式追加）:19 条单独成 1 组,组名 style,标签 [全部],并入第一波,全文 Read references/cpp-style.md,结果以 [STYLE] 前缀输出,不进 PASS/FAIL 统计。

Step 8 design-check 发射。docs_input 非空→发射（与波次1 同消息并行,独立轨道不进通用分组）;为空→不发射。写入专项清单。

Step 9 输出检视计划。

【输出格式】
yaml_dir: /tmp/{目录名}
检视计划 [{极简/紧凑/标准}]
代码行数: {N} | 检视范围: {全量/scope_hint}
代码语言: {C++/Python/Build/混合} | 侧别: {Kernel/Tiling/混合}
风险分级表: {变量→来源类型→风险等级}
匹配规则文件: {文件列表}（共 N 条）
跳过文件: {文件}:{原因}
跳过清单: {条例ID+理由}
仅核对清单: {条例ID+防御证据位置}
文件分组: {G_file} 组（多文件时列文件清单,单文件省略）
专项清单: {design-check（若发射）}

波次1:
  {组1（来源,标签）}: 条例ID {标题},...
     代码范围: {仅Kernel/仅Tiling/全部}
     文件范围: {file_group_name}（{文件列表}）← 多文件填,单文件省略
  style（cpp-style,[全部]）: 全部 19 条 | 代码范围:全部 | 读取:全文 Read cpp-style.md | 标记:[STYLE]

波次2（如有）:
  ...

共 G 组,分 W 波。

工具调用预算 ≤10 次:概要1+预研0-1+docs0-1+sed1-2+兜底grep0-3。超预算砍兜底 grep。
禁止生成报告文件。"
})
```
