# GitCode Issue 评论通用工作流

本文只定义 GitCode Issue 评论的通用平台能力：目标解析、写前确认、幂等检查、评论 POST、
GET 回查、Token 安全和错误处理。评论正文的业务模板、指派策略、状态迁移及后续跟踪由调用方
负责，不属于 toolkit。

> API 端点和字段见 [gitcode-api.md](gitcode-api.md)，Token 获取与传递见
> [token-config.md](token-config.md)，外部写入的通用授权边界见
> [authorization-contract.md](authorization-contract.md)。

## 1. 解析目标

评论目标必须包含 `owner`、`repo` 和 `issue_number`。输入为 Issue URL 时使用共享解析器：

```bash
python scripts/parse_gitcode_url.py \
  "https://gitcode.com/cann/ops-math/issues/2170"
# {"owner":"cann","repo":"ops-math","type":"issue","number":2170}
```

调用方已经提供结构化目标时直接使用，不重复解析。解析结果不是写入授权。

## 2. 写前检查

调用方负责提供最终评论正文。toolkit 在提交前只执行通用检查：

1. 确认目标 Issue 和完整正文已经获得当前写入的明确授权；正文或目标实质变化时重新确认。
2. 检查正文不包含 Token、密码、私钥、环境变量值、本机绝对路径或其他凭据。
3. GET 近期评论并进行幂等判断；已有相同正文时复用已有评论，不重复 POST。
4. 确认 `GITCODE_TOKEN` 已配置且对目标仓库具有所需权限。

长正文无法在交互界面完整显示时，调用方应提供完整正文产物的可读路径和摘要，不能用截断
内容授权未知正文。用户确认只覆盖展示的目标和正文，不覆盖调用方未展示的其他写操作。

## 3. 提交评论

```text
POST /repos/{owner}/{repo}/issues/{number}/comments
Content-Type: application/json

{"access_token":"<token>","body":"<comment_body>"}
```

使用共享客户端时，由调用方完成业务授权后传入准确的路径和 JSON；toolkit 不根据调用方名称
或任务描述推断授权。

```bash
curl -X POST \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/issues/${issue_number}/comments" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d "{\"access_token\":\"${token}\",\"body\":\"${comment_body}\"}" \
  --connect-timeout 30 --max-time 120
```

## 4. GET 回查

HTTP 成功只表示请求已受理。POST 后必须 GET 评论列表，以返回的评论 ID、作者、创建时间和
正文确认写入结果；仅比较状态码或截断前缀不足以证明长正文完整写入。

```text
GET /repos/{owner}/{repo}/issues/{number}/comments?per_page=100&page=<n>
```

优先按 POST 返回的评论 ID 定位；没有可靠 ID 时，按当前认证用户、提交时间窗和完整正文
联合匹配。回查失败不得标记评论成功，也不要在结果未知时立即重复 POST，应先重新 GET，
避免制造重复评论。

## 5. 通用错误处理

| 状态/错误 | 处理 |
| --- | --- |
| 401 | Token 无效或过期；重新进入 Token 检查点，替换后从该请求恢复 |
| 403 | 检查目标仓库权限；替换凭据后仍拒绝则返回权限 blocker |
| 404 | 校验 owner/repo/Issue 编号；目标不存在时不重试 |
| 422 | 校验正文和参数，修正后最多重试一次 |
| 429 | 遵守 `Retry-After` 或服务端 reset 窗口，超过调用方预算时返回限流状态 |
| 5xx/网络错误 | 有界退避重试；最终状态未知时先 GET 回查再决定是否重试 POST |

日志只能记录目标、评论 ID、时间、状态码和脱敏摘要，不记录 Token 或完整敏感正文。

## 通用边界

- toolkit 不决定评论写什么、由谁处理或评论之后进入什么业务状态。
- toolkit 不维护调用方的处理队列、跟踪清单、运行目录或限速状态文件。
- 评论中的 `@user` 只产生平台通知；若调用方需要指派，应使用平台支持的独立指派流程并
  单独回查结果。
- Issue 评论 API 不支持 PR 行内评论的 `in_reply_to` 语义；回复既有讨论时新建评论，并由
  调用方决定需要引用的上下文。
