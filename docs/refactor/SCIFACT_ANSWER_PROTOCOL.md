# SciFact 最终回答评测协议 v1（生成前冻结）

## 状态与范围

用户选择：**先建待核验金标与完整评测工具，人工核验后才生成200份回答**；采用下列门槛。本轮不得把上游标签/自动摘录说成人工核验，不调用生成/评审模型。test 不参与。不能声称回答质量已提高。

固定 prepared development 原顺序前100题（非随机、已经用于重排选参）。旧版 `scifact-nonthinking-development807-v1` 对新版 `scifact-fallback-development807-v1` 的 `relevance_v1`，直接复用保存的排序。每组使用前4个完整 chunks（应用默认 top_k=4），不是重排时的600字符前缀；不补入金标文档，不按相关文档过滤。历史重排模型仍是硅基版本；生成统一使用当前官方 DeepSeek `deepseek-flash`。不重新调用检索/重排，不改应用默认配置。

这是**冻结检索结果→回答生成的离线配对回放**，不是在线 HTTP 全链路测试：不重跑数据库、embedding、search confidence gate、流式传输；未度量在线端到端延迟。无证据且仍生成时，由生成器自行克制。先保留这个边界，不把旧 RRF 分数当 cosine confidence。

## 金标来源与人工核验门禁

BEIR qrels 仅用于显示参考文档，不产生答案标签。使用官方 SciFact archive 的 `claims_train.jsonl` 与 `corpus.jsonl`；仅提取 train/corpus，不提取或使用 dev/test claim 文件。记录下载URL、archive SHA256、选中文件 SHA256。上游 `latest` 是可变 URL，本地哈希才是本次版本标识；未取得发布者 archive 校验和，不宣称发布者签名验证。

逐题 claim 文本/ID与 development 精确匹配；全部5183文档 title、合并空白后的 abstract 与 BEIR 对齐。原文不做修改；证据句索引从0开始。上游 `SUPPORT`→`SUPPORT`，`CONTRADICT`→`REFUTE`，空 evidence→`INSUFFICIENT` **仅作为待审候选标签**。空 evidence 反映原数据标注范围，不保证全corpus绝无其他证据。

审核者必须：
1. 阅读claim、官方依据句及参考文档，确认/纠正三分类结论；证据冲突不自动多数表决。
2. 将摘录改为简明的必要回答要点；每个要点明确ID及依据。官方多个证据组可能是替代充分理由，不应自动全部列为同时必须覆盖的要点，人工须合并/删去冗余。NEI题必须明确“缺少什么证据/为什么不能下结论”，不能把没有标注等同断言为假。
3. 阅读两个匿名上下文 A/B，分别标 `context_sufficient`。它判断**提供的4个chunks**是否足以回答，不等同全局标签，也不因金标文档出现就自动为真。
4. 对每题填写 `review.status=approved`、真实审核者、审核时间、核验说明；全部100题通过后 `seal` 生成独立不可覆盖的金标快照。工具只能校验声明/字段，不能替代人的实际科学核验。

准备包的 `gold.draft.jsonl` 可修改，或在 `review.html` 内逐题编辑JSON后导出 `gold.reviewed.jsonl`（不联网、不自动保存/批准）；`cases.jsonl`（不含标签）、协议、脚本及来源受 manifest 哈希保护，不能修改。`seal` 前后使用相同匿名映射。人工金标不进入生成请求。

## 生成配置与执行约束

- 官方端点 `https://api.deepseek.com/v1`，模型 `deepseek-flash`。
- 复用应用 `generation.SYSTEM_PROMPT` 与 `_messages()` 组装真实上下文和 `[n]` 引用。
- Question固定模板：`Assess this scientific claim using only the provided context. State whether it is supported, refuted, or there is insufficient evidence; explain the essential evidence and cite sources inline. Claim: {claim}`。
- `temperature=0.2`、`max_tokens=2048`、`stream=false`；不添加供应商特有思考参数；SDK retries=0、timeout=60s。模型默认隐藏行为不能保证固定，temperature低也不保证逐字确定。
- 两组完全相同设置，单线程、按query配对，奇偶交替先后顺序（非随机严格盲法；匿名化只减少审核者显式版本偏好）。同上下文也各生成一次，不偷换成复用答案；恰好200次计划请求，无自动重试/补跑。失败和截断均保留在分母中。
- `generate` 默认只计划；必须人工 seal 后再 `--execute`，新目录防覆盖。原始请求/响应、引用位置、上下文、usage、耗时、started/finished写入本轮日志，不记密钥与异常消息。
- 官方DeepSeek费用不能套用硅基价。若没有经过核对的独立官方CNY单价表，估价与实扣为null/unknown，保存全部usage；不能写0。可在生成前提供冻结的 `--pricing`（本地 JSON：currency=CNY，input_cache_hit_per_million、input_cache_miss_per_million、output_per_million、source_url、verified_at）。必须有输入cache拆分且相加等于prompt_tokens，才估算该次费用；reasoning如已含completion不可重复加。不追溯伪造实扣。生成费用与人工/自动评审费用分开；本工具不调用自动judge。

## 答案人工评分规则（生成前固定）

每个答案独立、先不看版本/另一答案，用实际发送上下文核验。填写 `answer_reviews.draft.jsonl`，绑定answer、gold的SHA256，记录审核者/时间/说明。每题每组均需审核，不可只挑成功题。必要时使用第二审核者抽样复核（至少固定query序号0,10,…,90的两组共20份，并覆盖所有生成失败及全部初审错误/拒答/疑难条目）；记录独立复核者与裁决，未复核不作正式达标结论。无需自动评审，避免同模型自评偏差；若以后加自动judge，须另行冻结、单列成本并人工复核。

- **结论正确率**：人工提取答案最终立场 SUPPORT/REFUTE/INSUFFICIENT，与核验金标一致记1；含糊、矛盾、错误或失败记0。INSUFFICIENT要求真的克制，不能先断言真/假再加免责声明。`INVALID`用作无法提取/请求失败，不是第四种金标。
- **事实支持率**：将答案拆成可核验的原子事实（带答案原文片段），由提供上下文蕴含才记supported。仅有常识、仅与相关主题相近、上下文外金标证据不算支持。NEI的“上下文不足”也是一个可核验断言，不能靠空事实列表得满分。报告支持事实数/全部事实数（micro）及逐题均值；失败不添加虚构事实，单列完整请求成功率，且达标要求200次全部成功、非截断、都有至少一个事实记录。
- **引用准确率**：每个inline `[n]`出现为一个引用实例，原子陈述关联到引用原文位置offset；编号越界、无对应陈述、未蕴含均为错。分母是所有引用实例（micro），零引用为NA而非100%。必须给非NEI事实记录引用offset列表；未引用的事实不可因别处有引用而算引用完整。另报事实引用覆盖率（有至少一个有效支持引用的事实/全部事实）。
- **完整性**：覆盖的必要要点数/金标必要要点数，按题macro。要点覆盖必须由审核者确认，不能字符串命中；无要点不是满分。
- **拒答率**：答案明确表示证据不足的比例（拒答不等同科学反驳）。**适当拒答率**仅在相应组 `context_sufficient=false` 中统计；**误拒答率**仅在 `context_sufficient=true` 中统计。全局INSUFFICIENT题结论准确率另报；未检出证据但全局SUPPORT/REFUTE而拒答，可同时是安全拒答与任务结论失败，不能混为一个标签。分母为0报NA，不能补0。

## 预设门槛与结论

每组：结论正确率≥85%；事实支持率≥95%；引用准确率≥95%；完整性≥90%；证据充分题误拒答率≤5%。新版结论正确率不得低于旧版。无分母指标为NA时结论为 `not_evaluable`，不可视为通过；恰好200份、全成功非截断、有事实记录和所需复核是正式门禁。额外报告适当拒答率/引用覆盖率，不新增事后门槛。

`pass` 只说明预设绝对门槛达标且配对不退步；**不自动证明改善**。报告逐题结论差值、改善/下降/持平及query级配对bootstrap 95%区间（10000次，seed=20261020）；只有差值>0且区间下界>0才写本开发cohort“有配对改善证据”，否则写“未证明改善”。这100题已有选参偏差，不写独立test/线上/泛化结论。

报告需同时列旧/新完整质量指标、生成成本（未知保留null）、generation-only P50/P95、配对逐题回答及引用、成功与失败案例。历史检索/重排费用不重复计为新增花费；历史耗时不与新生成拼成严格E2E测量。

## 命令

```bash
# 本轮已执行，完全离线，输出初审包；不调用模型/数据库
.venv/bin/python -m scripts.evaluate_scifact_answers prepare \
  --output runs/scifact-answer-review-v1
# 人工编辑 gold.draft.jsonl 后（本轮尚未进行）
.venv/bin/python -m scripts.evaluate_scifact_answers seal \
  --pack runs/scifact-answer-review-v1 \
  --gold runs/scifact-answer-review-v1/gold.draft.jsonl
# 只验证/写执行计划，仍不请求模型
.venv/bin/python -m scripts.evaluate_scifact_answers generate \
  --pack runs/scifact-answer-review-v1 --output runs/scifact-answers-paired-v1
# 人工确认准备就绪后，使用另一新目录，加 --execute 才发200次请求
# 填写生成目录 answer_reviews.draft.jsonl 后：
.venv/bin/python -m scripts.evaluate_scifact_answers score \
  --pack runs/scifact-answer-review-v1 --answers runs/scifact-answers-paired-v1 \
  --reviews runs/scifact-answers-paired-v1/answer_reviews.draft.jsonl \
  --output runs/scifact-answer-scores-v1
```
