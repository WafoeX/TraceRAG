# 配置说明

[返回 README](../README.md) · [部署与维护](OPERATIONS.md)

## 配置入口

- [`.env.example`](../.env.example)：直接运行 Python 应用时的配置模板；真实 `.env` 不提交。
- [`deploy/application.env.example`](../deploy/application.env.example)：Compose 专用变量模板。每次显式传 `--env-file`，不复用开发/实验凭据。
- [`app/config.py`](../app/config.py)：设置字段、默认值及校验的最终依据。
- [`configs/providers/application-budget.example.json`](../configs/providers/application-budget.example.json)：**未核验**的应用预算价目示例，默认阻止真实调用。

Compose 只映射其 YAML 中明确列出的环境变量，不会把 env 文件内所有 Python 配置自动传入容器。更换模型路由或其他字段时，应审查专门的 Compose override，并同时应用于 API 和 worker。shell 变量优先于 Compose env 文件，启动前检查残留 `TRACERAG_*`；不要公开未经脱敏的 `docker compose config` 输出。

## 模型与检索

| 设置 | 默认值/含义 |
| --- | --- |
| `MODEL_PROVIDER` | `openai`；另支持显式 `siliconflow` |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | 共享 OpenAI 兼容客户端凭据/地址 |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` |
| `LLM_RERANK_MODEL` | 未设置时沿用 chat model |
| `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` | 官方 DeepSeek 的生成/LLM 重排覆盖；不替换共享 embedding |
| `EMBEDDING_DIM` | `1536`，必须与模型输出、数据库 schema 一致 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` tokens |
| `TOP_K` / `RETRIEVAL_CANDIDATES` | `4` / 每路 `10` |
| `USE_HYBRID` / `USE_RERANK` | 均为 `true` |
| `RERANK_PASSAGE_MODE` | `prefix_600`；`full_chunk` 为显式选项 |
| `RERANK_PROMPT_VERSION` | `baseline`；另有 `keyed_v1`、`relevance_v1` |
| `MIN_RELEVANCE` | `0.12`，不是正确率或语义支持概率 |

SiliconFlow 路由有专门 endpoint、模型和维度约束，见配置校验及模板。native rerank 仍为实验能力，不是默认检索链路。换 embedding 模型不只是换字符串：应使用新库或明确重建向量，不能把不兼容空间混用。

## 调用与估算预算

| 设置 | 默认值 |
| --- | --- |
| `RATE_LIMIT_PER_MINUTE` | 每账号 `10` 次受控 POST/分钟，跨知识库共享 |
| `ACCOUNT_CONCURRENCY` | 每账号 `2` |
| `APPLICATION_CONCURRENCY` | 跨 API/worker 的全局 provider 操作槽位 `4` |
| `PROVIDER_TIMEOUT_SECONDS` | `20`，网络阶段超时，不是整请求硬截止 |
| `PROVIDER_MAX_ATTEMPTS` | 总尝试数 `2`，应用内禁用 SDK 隐藏重试 |
| `PROVIDER_RETRY_BACKOFF_SECONDS` | `0.25` |
| `APPLICATION_GENERATION_MAX_TOKENS` | `2048` |
| `APPLICATION_RERANK_MAX_TOKENS` | `1024` |

`RATE_LIMIT_ENABLED=false` 仅适合隔离测试，**不能关闭应用预算**。新应用库没有默认额度；必须显式分配 global 和账号累计 CNY 额度，无自动日重置。

价目须精确匹配实际 base URL、model、role（embedding / llm_rerank / generation），包含可核验来源、检查时间、input/output 每百万 token 费率，以及覆盖**完整请求与全部计费项**的 input/output token 上界和 request byte 上限。仅填写 `verified:true` 不等于完成了核验，不可把测试费率当真实价格。

调用前按完整上界预占；合法 usage 才结算差额。未知费用为 `null` 而非 0；unknown/遗留 held 必须停相关调用并人工核账。历史 provider 费用 JSONL 不是应用余额来源，可信实验脚本也不自动受多用户应用预算托管。

## 会话、隐私与启动

- `SESSION_COOKIE_SECURE=true` 默认要求 HTTPS；仅在 localhost HTTP 演示时显式设为 false。
- 正式 Compose 强制 `RUN_DB_INIT=false`、`APPLICATION_STARTUP_CHECK=true`；初始化和升级走维护 CLI，不在启动时隐式迁移。
- `DATABASE_CONNECT_TIMEOUT_SECONDS=3`，取值 1–10 秒；readiness 不核验 provider 可用性或剩余额度。
- 问答追踪默认不存问题、回答、文献正文。`retain_trace_content:true` 是单次明确同意，不是长期授权；超过 1 MiB 的正文快照整份不存。
- trace 最多 7 天可见，到期即从 API 隐藏；**物理删除需显式清理作业**。反馈随 trace 到期/删除，不延长期限。
- 完整反馈导出须逐次人工编辑和确认；不会自动脱敏、重放真实模型或验证语义正确性。
- 不记录 HTTP body、Cookie、Authorization 或未经脱敏的配置。备份和供应商副本有独立保留责任，删 trace 不等于撤回所有副本。
