# SciFact 六指标补齐协议 v1

## 任务与边界

目标：在**已有固定100题、200份回答**上补齐检索、上下文与回答评价，形成可追溯的故障分层，不新增生成/embedding/重排/judge调用，不写数据库，不执行Pilot B，不改应用默认或旧证据。

这是看过历史报告后的补充协议，不是所有分析前预注册或独立盲评。主助手直接负责AI评价，不要求用户全面审核，也不委派子代理。评分规则在本次新增评价前冻结；不为漂亮结果回改原gold/review。当前处于补测，不预定候选一定胜出，不将未评项算0。

## 推进步骤与完成条件

1. **协议与离线检索补算**：确认100题/两组绑定，计算文档Hit/Precision/Recall/MRR/nDCG，建立带来源hash的待评包；不把模板当完成的评分。
2. **上下文评价**：按冻结query顺序每批10题，评价两组实际top4。相同query＋相同chunk只判一次并复用，记录各位置。全部完成后汇总Context Relevance和充分性，不用样本外推100题。
3. **回答评价与汇总**：按同一顺序分批评价200份回答；复核事实片段、引用、结论/问题范围和拒答，保留uncertain；输出配对总表、逐题案例、时延/tokens/费用边界。

每批记录覆盖query、来源hash、理由、证据quote及offset、AI评审者和非盲状态；未完成行保持pending。疑难项由主助手再读实际上下文复核，不强迫二元裁决。不得只评“容易判断”或旧错误题。

## Cohort与数据绑定

- development为官方train隔离2条与test规范化文本重复query后的807条；本次固定原顺序前100，与原配对生成完全一致。
- A＝原非思考重排，B＝relevance_v1重排；这不是Pilot A中的两个生成prompt。两组生成均用原system prompt、官方deepseek-flash、temp=.2、max_tokens=2048、实际top4完整chunks。
- 读取`runs/scifact-answers-paired-v1/plan.json`与`answers.jsonl`，使用实际请求与上下文；验证其query、arm、chunk ID/文本hash、与保存排名前4的对应关系。
- 只读取development查询/qrels、固定chunk映射及已有两组排名。**不读取test内容、不访问真实.env。** 不把qrels注入语义评分理由。
- 所有输入文件及本协议/新工具记录SHA-256；修改工具或评分规则须新版本，不静默更新冻结包。

## 一、检索指标：文档层，与真实4个chunks分开

将保存的chunk排名按稳定document_id首次出现去重，再在文档排名上截K。所有100条query宏平均；包含空排名/失败；固定K的Precision分母不因短排名缩小。

- Hit@K：前K唯一文档是否至少命中一个正qrel，取0/1，报告K=4/10。
- Precision@K：前K唯一文档中正qrel文档数/K，K=4/10。未标注文档按未命中计，但不等于人工证明无关，称**qrels口径文档Precision**。
- Recall@K：前K唯一文档命中的正qrel数/该query所有正qrel数，K=4/10/100。
- MRR@10：前10首次正qrel的倒数，无命中为0。
- nDCG@10：沿用仓库线性gain与log2折扣，不切换增益定义。

纯离线结果与旧analysis.screen100重算校对；同题报告差值，后续总报告添加query级配对不确定性。100题已用于选参，区间也不构成独立测试。

**不以此推断chunk Recall。** 只评价已返回top4时，不知道全库所有有效chunk，不存在可靠的完整chunk召回分母。

## 二、Context Relevance与上下文充分性

评价对象＝(query,实际chunk)，不是整个文档、qrels或模型自己的rerank分数。只看本地发送的文本，可同时支持或反驳claim；相关不等于充分。

### 片段等级与不确定

- 0：无关；不帮助判断目标实体、关系或必要条件。
- 1：相关背景或局部关联，但不提供目标关系/条件的实质证据。
- 2：有效证据；对支持、反驳、限定或判断目标命题所缺条件有实质帮助。**不是必须独立足以回答整题**；仍需记录支持范围。
- null：合理歧义/无法可靠判定；保留理由，不算0或2。

每条记录grade、reason、能支持的范围、关键quote及在chunk文本中的[start,end)位置。0级可记录最接近但无关的主题quote；不能伪造“无证据”quote。跨query即使chunk相同也不复用相关性标签。

### 汇总

- Related Context Precision@4：grade≥1的chunk数/4。
- **Useful Context Precision@4（主指标）**：grade=2的chunk数/4。
- 等级均分：sum(grade)/(2×4)，另列不确定上下界；它是自定义AI rubric指标，不冒称标准库分数。
- 对null给下界（不计命中）与上界（允许其命中），报告全100题宏平均。每组必须4个实际chunks；不符则停下调查，不删题。
- **Context Sufficiency**：按整组4个chunks判sufficient/insufficient/uncertain。充分是当前文本足以对完整目标给出有依据的结论（可合理支持或反驳），不是出现主题就算充分，也不是实际回答写得好才算充分。
- Sufficiency Rate下界=sufficient/100，上界=(sufficient+uncertain)/100；两端是判定不确定性界，不是统计置信区间。
- 无已知完整chunk正例集，不报告伪造的chunk Recall；必要时另立已标注证据覆盖指标。

## 三、生成指标：非盲AI评价，不是真值

### Faithfulness / Groundedness

已有上传AI初审仅作辅助，不能直接说本轮完成全量重判。逐份确认答案中被认可的事实与推断，剔出仅转述但未认可的题目claim，尽量按最小可判定的实体/关系/时间/终点拆分，避免把整段当一个“事实”。

- 每个断言supported/unsupported/uncertain，绑定答案span及实际上下文证据。
- 事实支持下界=S/N，上界=(S+U)/N，分别报告片段micro和每份答案macro，不让长答案支配全部结果。
- 单列最终结论是否由证据建立；局部数字正确不覆盖最终范围越界。
- unsupported/N只称“未获支持断言比例”，不自动等于现实事实为假或客观幻觉率；uncertain不强判。
- 引用沿用组合展开v2，另记多来源joint support与覆盖率；单来源partial不冒充完全支持，也不把合理联合支持全部判幻觉。

### Answer Relevance

评价是否切中**完整目标命题**，不是与参考标签一致、也不是文字相似度。

- 0：答非所问、替换关键实体/操作/终点/时间/量词范围，或核心结论与自身限定冲突。
- 1：回应部分目标，但遗漏关键条件/限制，尚不能完整理解它如何回答题目。
- 2：直接回答目标，范围清楚，必要限制与结论一致；恰当、有解释的证据不足回答也可为2。
- null：命题解读等有合理分歧，记录原因。

报告等级0/1/2/null分布、完全相关比例grade2/100及不确定上界；0..1归一化均分仅作自定义rubric分数，不冒称RAGAS结果。与Faithfulness分开：切题但无据、忠实但没答完整均可能发生。

### 拒答与交付

- 有充分上下文时的误拒答率，分母为sufficient题数；上下文uncertain单列，不强行归类。
- 不足上下文时的适当拒答率，分母为insufficient题数；零分母报NA。
- 空答/请求失败/截断必须在全100题交付指标中计失败；不能因没有断言就拿100% Faithfulness或算成功拒答。其事实级比率NA，并单列coverage；不能删失败只报条件均分。
- 保存生成时延、usage、缓存/推理字段；reasoning已包含completion，不重复计。原生成金额未知，不推导为0。

## 结论与防止伪提升

不将全部指标混成事后调权的总分。分别报告检索、上下文、回答、交付及成本。按query配对统计改善/下降/相同/uncertain，不把uncertain判胜；如给bootstrap，固定seed=20261020、10000次、query级抽样，并强调development/非盲边界。

仅补齐评分不代表系统质量新提升。本轮不选新prompt、不用test、不续跑失败Pilot、不更改历史参考。后续是否实验，取决于补齐结果显示的真实瓶颈，须另冻结变量与授权。
