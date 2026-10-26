# 部署与维护

[返回 README](../README.md) · [配置说明](CONFIGURATION.md)

仅面向本机或受控私有环境。需要 Docker Compose、Python 3.12.13 和 uv 0.12.16。真实模型可能收费；本说明不是公网生产安全认证。

## 1. 准备专用应用环境

```bash
uv sync --frozen --group dev
uv run --frozen python -m scripts.prepare_test_cache
cp deploy/application.env.example deploy/application.env
chmod 600 deploy/application.env
```

手动填写独立数据库密码、模型密钥及已创建的私有备份绝对目录（建议权限 700、加密磁盘、位于仓库外）。不要复制开发 `.env` 或复用实验库。默认 Secure Cookie 需 HTTPS；本机普通 HTTP 体验才显式设置 `TRACERAG_COOKIE_SECURE=false`。

以下命令假定默认数据库名 `tracerag_application`、维度 `1536`；若更改，必须同步修改命令与所有服务配置。先清除 shell 中残留的 `TRACERAG_*`，避免覆盖 env 文件。

```bash
APP_ENV=deploy/application.env
APP_PROJECT=tracerag-application
dc() {
  docker compose --env-file "$APP_ENV" -p "$APP_PROJECT" \
    -f compose.application.yml "$@"
}
dc config --quiet
dc build
dc up -d --wait db
# 仅新空数据库；确认所有连接该库的 API/worker 均未运行。
dc run --rm tools init --apply \
  --database-url postgresql+psycopg://tracerag@db:5432/tracerag_application \
  --confirm-database tracerag_application --embedding-dim 1536 \
  --confirm-services-stopped
dc up -d --wait api worker
curl --fail http://127.0.0.1:18000/health
curl --fail http://127.0.0.1:18000/ready
```

API 默认仅绑定 `127.0.0.1:18000`，数据库不发布主机端口，API/worker 使用独立命名卷。不要挂载历史实验卷；`docker compose down` 保留命名卷，**不要随意加 `-v`**。容器以非 root 运行，tokenizer 构建前准备并校验，运行时不下载。

## 2. 开户、核价与额度

无默认真实用户、公开注册或 HTTP 调额接口。交互输入密码，不把密码放 argv：

```bash
dc run --rm --entrypoint /app/.venv/bin/python tools \
  -m scripts.manage_application_users \
  --database-url postgresql+psycopg://tracerag@db:5432/tracerag_application \
  --username researchadmin
```

默认挂载价目未经核验，模型调用会失败关闭。先按[配置说明](CONFIGURATION.md)核验价格、usage 契约和完整请求上界，再通过 `TRACERAG_PRICING_FILE` 挂载经审查的私有价目文件，并让 API/worker 一致使用。不要沿用演示密钥、虚构 usage 或测试费率。

管理员在明确决定可承担的累计 CNY 额度后分别设置 global 和账号：

```bash
# TOTAL_CNY / USER_TOTAL_CNY 由管理员自行决定，不是本说明授予额度。
dc run --rm --entrypoint /app/.venv/bin/python tools \
  -m scripts.manage_application_budget \
  --database-url postgresql+psycopg://tracerag@db:5432/tracerag_application \
  set-limit --global --cny "$TOTAL_CNY" --reason '显式预算分配'
dc run --rm --entrypoint /app/.venv/bin/python tools \
  -m scripts.manage_application_budget \
  --database-url postgresql+psycopg://tracerag@db:5432/tracerag_application \
  set-limit --username researchadmin --cny "$USER_TOTAL_CNY" --reason '显式账号分配'
```

`set-limit` 是绝对累计上限，不是充值，不清零 spent/held。unknown 或遗留预占不按超时退款；停止相关调用、核账后才使用 `manage_application_budget resolve --confirm-audited`，上界违约还需修正配置并分别显式 unblock。用 `--help` 查看参数，不把未知费用记为 0。

## 3. 状态与常用接口

| 入口 | 含义 |
| --- | --- |
| `/`、`/docs` | 浏览器界面、交互 API 文档 |
| `GET /health` | 进程存活，不访问数据库 |
| `GET /ready` | 数据库、schema、维度、迁移校验与恢复状态；失败 503 |
| `/auth/login`、`/auth/me`、`/auth/logout` | 会话登录、恢复与退出 |
| `/knowledge-bases` | 当前用户的知识库 |
| `/documents/jobs`、`/documents/jobs/upload` | 异步文本/文件入库及任务管理 |
| `/documents/catalog` | 活跃版本文献目录 |
| `POST /query/evidence` | 问答、完整上下文、生成/降级状态 |
| `GET /budget` | 本人累计估算额度及近期记录，只读 |
| `/traces`、`/traces/{id}` | 本人的未到期追踪 |
| `/traces/{id}/feedback`、`/traces/{id}/export` | 固定分类反馈、人工审查导出 |

私有接口须登录；集合/问答须明确知识库，变更请求还需 `X-CSRF-Token` 与同源校验。历史同步 `/documents`、`/query` 和 `/query/stream` 也受鉴权和预算约束。具体请求 schema 以运行中的 `/docs` 为准，旧匿名 curl 样例不能直接沿用。

worker healthcheck 只验证数据库/schema，不证明任务吞吐或不存在卡死；readiness 也不证明模型服务可用、已核价或有额度。

## 4. 显式迁移

当前原生新库为 schema v7；已有应用库通过 001–006 增量链升级。管理命令默认 plan，不读 `.env`、不连库：

```bash
uv run --frozen python -m scripts.manage_application_database
```

实际迁移前停 API/worker 和所有其他调用者、先备份，再确认真实源版本。以下仅为已有 v1 应用库的示例，**不是历史 SciFact/实验库自动接管入口**：

```bash
dc stop api worker
dc run --rm tools migrate --apply \
  --database-url postgresql+psycopg://tracerag@db:5432/tracerag_application \
  --confirm-database tracerag_application --from-version 1 --embedding-dim 1536 \
  --owner legacyowner --library-name 'Legacy papers' --confirm-services-stopped
```

v1/v2 升级需明确旧数据接收者，交互输入其密码；后续版本不重新分配旧数据。不得猜测 `--from-version`。每步事务提交并核对迁移源码 SHA，整条链不是一个大事务；中断后先核对已完成步骤，再显式续跑。原生 schema、不匹配的 ledger/校验和或非空目标会拒绝盲目接管。不要用 `create_all` 或删库重建代替升级。

## 5. 备份与隔离恢复

备份包含原文件、历史版本、文本、向量、密码 hash、trace、任务 payload 和预算等敏感数据。只恢复受信任的同版应用备份；SHA 只检查损坏，不是签名。工具不提供自动加密、轮换、跨 PostgreSQL 大版本升级、角色/ACL 或 PITR。

```bash
dc stop api worker
# 确认所有其他连接此库的调用者也已停止。
dc run --rm -T db-tools backup before-change.dump \
  --confirm-services-stopped --confirm-database tracerag_application
```

归档和 hash 权限为 600，不覆盖同名文件。只向**新空库**恢复，不覆盖或删除源库：

```bash
dc exec -T db psql -X -v ON_ERROR_STOP=1 -U tracerag -d postgres \
  -c 'CREATE DATABASE tracerag_restored'
dc run --rm -T -e PGDATABASE=tracerag_restored db-tools \
  restore before-change.dump --confirm-services-stopped \
  --confirm-database tracerag_restored --trust-local-backup
```

工具先完整解码，再将恢复 SQL 与固定隔离 SQL 放在同一事务。恢复后撤销会话、冻结预算、暂停待处理任务、清理已到期 trace，API/worker 拒绝正常使用。**不自动退款、重跑任务或解冻额度。**

人工核对外部供应商账单、在途请求、备份后发生的隐私删除及保留规则；无法核对时保持隔离。恢复数据库不能撤销供应商收费或浏览器下载。源服务必须保持关闭，避免新旧库同时付费。

审查后，换用一份指向恢复库的私有 env 文件，再显式 release：

```bash
APP_ENV=/absolute/private/restored.env
# restored.env 内 TRACERAG_DB_NAME=tracerag_restored。
dc run --rm tools release --apply \
  --database-url postgresql+psycopg://tracerag@db:5432/tracerag_restored \
  --confirm-database tracerag_restored --embedding-dim 1536 \
  --confirm-services-stopped --confirm-privacy-reviewed --confirm-billing-reviewed
dc up -d --no-deps --wait api worker
```

release 仅解除恢复隔离，旧会话不复活，预算仍 blocked，任务不自动 retry。管理员如需恢复调用，另行核账、调额和解封。

## 6. 保留清理与安全责任

到期 trace 从 API 隐藏不等于已物理删除。管理员应安排显式清理、监控失败，并独立管理备份销毁：

```bash
# 默认 plan；实际执行必须明确应用库。
uv run --frozen python -m scripts.manage_application_traces
# Compose 内沿用容器 PGPASSWORD，不把密码写进 URL。
dc run --rm --entrypoint /app/.venv/bin/python tools \
  -m scripts.manage_application_traces --apply \
  --database-url postgresql+psycopg://tracerag@db:5432/tracerag_application
```

仓库不安装自动备份/清理调度。公网 TLS、可信代理、通用 HTTP 滥用防护、生产最小权限角色、磁盘/备份加密、密钥轮换和原件配额仍由部署方补齐；不要因本机启动成功而直接公网开放。
