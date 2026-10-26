async page => {
  // Only local disposable fixture --resilience-demo. Failures are injected at
  // generation boundary; real SDK retry behavior is covered by offline tests.
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', msg => {
    if (msg.type() === 'error' && !/401|429/.test(msg.text())) errors.push(msg.text());
  });
  await page.goto('http://127.0.0.1:18089/');
  await page.setViewportSize({width:1440,height:1050});
  await page.locator('#username').fill('alice');
  await page.locator('#password').fill('TraceRAG-demo-123');
  await page.locator('#loginBtn').click();
  await page.waitForFunction(() => document.querySelector('#librarySelect').options.length > 0);
  await page.evaluate(() => { document.querySelector('#addform').parentElement.open = true; });
  await page.locator('#docTitle').fill('P2.3 故障演示文献');
  await page.locator('#docContent').fill('RESILIENCE_EVIDENCE_MARKER: Scientists measured liver cells in a controlled study.');
  await page.locator('#addBtn').click();
  await page.waitForFunction(() => document.querySelector('#docList').textContent.includes('P2.3 故障演示文献'));
  async function ask(text) {
    await page.locator('#q').fill(text);
    const pending = page.waitForResponse(res => res.url().includes('/query/evidence') && res.request().method() === 'POST');
    await page.locator('#ask').click();
    const response = await pending;
    await page.waitForFunction(() => !document.querySelector('#ask').disabled);
    return response;
  }
  const failed = await (await ask('模拟生成超时：What did the scientists measure?')).json();
  if (failed.generation_status !== 'failed' || !failed.warnings.includes('generation_provider_timeout')) throw new Error('Failure metadata missing');
  const failedCard = page.locator('#chat .bot').last();
  if (!(await failedCard.textContent()).includes('回答生成失败')) throw new Error('UI presented failed generation as a valid answer');
  if (!(await failedCard.textContent()).includes('RESILIENCE_EVIDENCE_MARKER')) throw new Error('Evidence was lost on generation failure');
  await failedCard.getByRole('button',{name:'查看库中原文切块'}).click();
  await page.locator('#evidenceText').filter({hasText:'RESILIENCE_EVIDENCE_MARKER'}).waitFor();
  await page.locator('#closeEvidence').click();
  await page.screenshot({path:'docs/assets/tracerag-resilience-generation.png',fullPage:true});
  const degraded = await (await ask('模拟重排超时：What is the evidence?')).json();
  if (degraded.generation_status !== 'completed' || !degraded.warnings.includes('rerank_retrieval_order')) throw new Error('Rerank fallback missing');
  if (!(await page.locator('#chat .bot').last().textContent()).includes('rerank_provider_timeout')) throw new Error('UI lost fallback reason');
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:'docs/assets/tracerag-resilience-mobile.png',fullPage:true});
  let limited = false;
  for (let i=0; i<12; i++) {
    const response = await ask('Rate-limit check ' + i);
    if (response.status() === 429) { limited = true; break; }
    if (response.status() !== 200) throw new Error('Unexpected query status ' + response.status());
  }
  if (!limited) throw new Error('Real account rate limiter did not reject repeated queries');
  if (!(await page.locator('#chat .bot').last().textContent()).includes('Account call limit exceeded')) throw new Error('429 not actionable in UI');
  await page.locator('#logoutBtn').click();
  await page.locator('#loginPanel').waitFor({state:'visible'});
  if (errors.length) throw new Error(JSON.stringify(errors));
  return {generationFailureKeepsEvidence:true,originalEvidenceOpens:true,rerankReasonVisible:true,mobile:true,realAccount429:true,logout:true,unexpectedBrowserErrors:errors.length};
}
