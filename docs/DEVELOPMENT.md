# 开发与测试

[返回 README](../README.md) · [架构](ARCHITECTURE.md) · [部署与维护](OPERATIONS.md)

## 环境与依赖

使用 Python **3.12.13**、uv **0.12.16**。`pyproject.toml` 与 `uv.lock` 是依赖真值；`requirements.txt` 仅为兼容平台的运行依赖导出，不手动编辑。

```bash
uv sync --frozen --group dev
# 仅显式准备步骤下载真实 tokenizer，包含 SHA-256 校验。
uv run --frozen python -m scripts.prepare_test_cache
uv run --frozen ruff check .
uv run --frozen pytest -q
```

默认测试不读取 `.env` 密钥，不需数据库或模型 API。TCP/UDP 禁用，ASGI 客户端使用 Unix socket；缺失/损坏 tokenizer 缓存会失败，不在测试中隐式下载。带集成标记的用例默认跳过；跳过不代表已经验证。

重新导出运行依赖：

```bash
make export-requirements
```

## PostgreSQL 集成测试

只使用专用临时测试库，不连接开发或实验数据：

```bash
docker compose --env-file /dev/null -p tracerag-app-test \
  -f compose.application-test.yml up -d --wait
TRACERAG_INTEGRATION=1 uv run --frozen pytest -q
# 执行完且确认没有演示/测试仍使用该容器后：
docker compose --env-file /dev/null -p tracerag-app-test \
  -f compose.application-test.yml down
```

测试数据库为 `127.0.0.1:55439/tracerag_application_test`，使用 tmpfs；容器删除后数据消失。测试使用固定 schema，部署 bootstrap 测试还会创建/删除固定临时数据库。**必须串行执行**，不支持 xdist 或同时运行两套同名测试。中断残留先确认无人使用，再重建本测试容器；不要删除开发库。

## 手动演示与结构回归

基础演示见 README。`tests/manual/tracerag_application_demo.py` 还提供隔离的 `--budget-demo` 场景；setup、api、worker 三个命令必须使用一致选项，不能和普通/故障夹具混跑。模型、价格和 usage 都是虚构测试数据，不是供应商实测。

`tests/manual/` 内保留浏览器脚本，执行需额外安装 Playwright CLI/浏览器及初始化相应临时场景；它们会操作测试账号、文献及截图。生成的 `docs/assets/`、浏览器缓存和 JSON 报告不提交，不能只凭脚本退出码判断全部交互通过。

无需模型的脱敏反馈样例检查：

```bash
uv run --frozen python -m scripts.check_feedback_regression \
  --case tests/fixtures/tracerag-feedback-sanitized.json
```

该检查只验证结构和引用编号范围，不判断语义支持或答案正确性。

部署/恢复演练默认仅打印计划；显式 `--apply` 会创建固定隔离 Docker 项目、操作合成数据并清理自身资源，应先阅读脚本并确认没有同名资源：

```bash
uv run --frozen python -m tests.manual.tracerag_deployment_check
```

## 实验工具的边界

`scripts/` 中的 SciFact、baseline、provider 和重排工具属于独立实验路径，**不是应用部署步骤，也不默认受应用预算托管**。部分工具可调用收费模型、下载数据或写入专用实验库；运行前必须审查参数与输入，不能把 `make eval` 当作禁网测试。

历史 `runs/`、语料、人工评审 JSONL、阶段结果和反馈报告不随仓库发布。依赖它们的分析/重放工具需要自行准备相应本地输入；干净 checkout 不承诺重现历史成绩。

仅保留这些被实验工具直接读取的冻结协议：

- [回答评测协议](refactor/SCIFACT_ANSWER_PROTOCOL.md)
- [重排回退实验协议](refactor/SCIFACT_FALLBACK_PROTOCOL.md)
- [生成 Pilot 协议](refactor/SCIFACT_GENERATION_PILOT_PROTOCOL.md)
- [六指标评测协议](refactor/SCIFACT_SIX_METRICS_PROTOCOL.md)

这些文件保持原字节，含历史计划措辞，**不是当前功能/执行状态说明**。`docs/refactor/baseline/source-manifest.json` 和 `source.tar.gz` 是 baseline 工具及测试依赖的源码/示例哈希夹具，不是运行结果。基线 tag 为 `baseline/pre-refactor-v1`。上下文审计等历史分析仍需本地保存的报告输入，不纳入日常应用运行契约。

## 文件提交规则

- 保留应用源码、维护脚本、测试、合成夹具、示例、配置模板及锁文件。
- `docs/` 使用 `.gitignore` 白名单；新增公开说明须同步更新白名单和 README 导航。
- `interview_materials/`、旧求职/任务书/进展报告、截图、人工评分、原始运行记录留本地并忽略。已跟踪旧文档需从 Git 索引移除，单写忽略规则无效。
- 不提交 `.env`、私有部署配置、模型密钥、原始用户文献或数据库备份；配置示例中的公开测试凭据不能用于真实部署。
- `tests/fixtures/` 与 `configs/evaluation/` 中用于程序验证的合成/固定输入不是运行日志，不因忽略记录而一并删除。
- 不把历史测试计数、上游 CI 徽章或离线替身演示写成当前线上质量/性能保证。
