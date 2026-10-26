# BudgetRAG

基于 **FastAPI + PostgreSQL/pgvector** 的文档问答应用，提供知识库隔离、异步文档入库、带证据的问答与跨进程估算预算管理。仓库内部分应用命令和配置沿用 **TraceRAG** 名称。

## 功能

- **文档管理**：上传 TXT、Markdown、PDF，后台解析、切块、向量化；支持去重、失败重试、版本更新、原文件和 PDF 物理页定位。
- **检索问答**：向量检索与全文检索经 RRF 融合，可选 LLM 重排；返回上下文、引用编号和生成状态，生成失败时仍可核对已取得的证据。
- **账号与知识库**：管理员 CLI 开户、Cookie 会话、CSRF 校验，以及基于数据库查询的 owner 隔离。
- **调用控制**：有限重试、账号速率/并发和全局模型槽位；每次供应商尝试前原子预占全局与账号累计人民币估算额度。
- **追踪与反馈**：默认只保存问答元数据；正文须逐次同意，最多 7 天可见。支持固定分类反馈和人工脱敏导出。
- **运行维护**：独立 API、worker 和数据库；显式 schema 初始化/迁移、readiness、备份与恢复隔离。

> 这是本地或受控私有环境项目，不应直接开放公网。引用编号合法不代表语义正确，预算是估算而非供应商实扣保证。默认价目未经核验，会拒绝真实模型调用。

## 快速体验：不调用真实模型

需要 **Python 3.12.13、uv 0.12.16、Docker Compose**。首次安装依赖、下载 tokenizer 缓存和数据库镜像需要网络；演示中的模型和向量是确定性替身。

```bash
uv sync --frozen --group dev
uv run --frozen python -m scripts.prepare_test_cache

docker compose --env-file /dev/null -p tracerag-app-test \
  -f compose.application-test.yml up -d --wait
uv run --frozen python -m tests.manual.tracerag_application_demo setup
```

在项目根目录分别打开两个终端：

```bash
# 终端一：API
uv run --frozen python -m tests.manual.tracerag_application_demo api
```

```bash
# 终端二：后台入库 worker
uv run --frozen python -m tests.manual.tracerag_application_demo worker
```

访问 **http://127.0.0.1:18089/**，使用演示账号 `alice` 或 `bob`，密码 `TraceRAG-demo-123`。上传 `tests/fixtures/tracerag-app/` 中的示例并提问。上述公开凭据仅用于临时演示，不能用于真实部署。

演示使用专用临时 PostgreSQL 和独立 schema，不读取开发模型密钥。替身回答故意含无效引用，用于验证界面告警，不能评价模型质量。重复 setup 前须先清理，不能通过删除开发库解决冲突。

结束时先停止两个终端中的 API 和 worker，再执行：

```bash
uv run --frozen python -m tests.manual.tracerag_application_demo cleanup
docker compose --env-file /dev/null -p tracerag-app-test \
  -f compose.application-test.yml down
```

## 文档

| 文档 | 内容 |
| --- | --- |
| [架构与边界](docs/ARCHITECTURE.md) | 数据流、模块职责、隔离、版本与一致性 |
| [配置说明](docs/CONFIGURATION.md) | 模型路由、检索参数、预算和隐私规则 |
| [部署与维护](docs/OPERATIONS.md) | 真实应用部署、开户、迁移、备份和恢复 |
| [开发与测试](docs/DEVELOPMENT.md) | 环境、禁网测试、集成测试与实验工具 |

真实模型模式请从部署说明开始：使用专用应用数据库，先核验价格和完整请求计费上界，再显式分配额度。不要把旧 `make eval`、种子或实验采集脚本当成无费用演示入口。

## 项目结构

```text
app/                         FastAPI、鉴权、路由、RAG 服务与静态界面
  services/                  入库、检索、重排、生成、预算与追踪
  worker.py                  独立后台入库进程
scripts/                     数据库维护、账号/预算管理与实验工具
configs/                     价目示例、检索评测配置与提示词
deploy/                      schema、校验清单和恢复隔离 SQL
tests/                       单元/集成测试、合成夹具和手动演示
examples/                    可公开的示例文档
docs/                        核心说明及工具必需的冻结协议/校验资源
compose.application.yml      受控应用部署
compose.application-test.yml 临时测试数据库
compose.baseline.yml         隔离实验数据库
pyproject.toml / uv.lock      依赖定义与锁文件
```
