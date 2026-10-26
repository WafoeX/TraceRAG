async page => {
  // Synthetic model responses and rates; actual API, worker, PG trace and budget code.
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', msg => { if (msg.type() === 'error' && !/401/.test(msg.text())) errors.push(msg.text()); });
  await page.goto('http://127.0.0.1:18089/');
  await page.setViewportSize({width:1440,height:1100});
  async function login(name) {
    await page.locator('#username').fill(name);
    await page.locator('#password').fill('TraceRAG-demo-123');
    await page.locator('#loginBtn').click();
    await page.waitForFunction(name => document.querySelector('#authName').textContent.includes(name) && document.querySelector('#librarySelect').options.length > 0, name);
  }
  async function ask(text, consent) {
    await page.locator('#retainTrace').setChecked(consent);
    await page.locator('#q').fill(text);
    const pending = page.waitForResponse(res => res.url().includes('/query/evidence') && res.request().method() === 'POST');
    await page.locator('#ask').click();
    const response = await pending;
    if (response.status() !== 200) throw new Error('Query failed');
    await page.waitForFunction(() => !document.querySelector('#ask').disabled);
    if (await page.locator('#retainTrace').isChecked()) throw new Error('Consent was reused');
    const id = response.headers()['x-trace-id'];
    const detail = await (await page.request.get('http://127.0.0.1:18089/traces/' + id)).json();
    return {id,detail};
  }
  await login('alice');
  await page.evaluate(() => { document.querySelector('#addform').parentElement.open = true; });
  await page.locator('#docTitle').fill('请求追踪隐私演示');
  await page.locator('#docContent').fill('TRACE_PRIVATE_EVIDENCE: Researchers measured liver cells in a controlled study.');
  await page.locator('#addBtn').click();
  await page.waitForFunction(() => document.querySelector('#docList').textContent.includes('请求追踪隐私演示'));
  const normal = await ask('TRACE_PRIVATE_QUESTION: What was measured?', false);
  if (normal.detail.snapshot !== null || JSON.stringify(normal.detail.metadata).includes('TRACE_PRIVATE')) throw new Error('Metadata leaked body');
  if (normal.detail.metadata.calls.length !== 2 || normal.detail.metadata.calls.some(c => c.status !== 'settled')) throw new Error('Missing real budget events');
  const consented = await ask('TRACE_PRIVATE_QUESTION: Keep this snapshot?', true);
  if (!consented.detail.snapshot.contexts[0].content.includes('TRACE_PRIVATE_EVIDENCE')) throw new Error('Consented actual context missing');
  await page.locator('#chat .bot').last().getByRole('button',{name:'查看本次追踪'}).click();
  await page.locator('#traceText').filter({hasText:'TRACE_PRIVATE_EVIDENCE'}).waitFor();
  await page.screenshot({path:'docs/assets/tracerag-trace-detail.png',fullPage:false});
  await page.locator('#closeTrace').click();
  const unknown = await ask('模拟费用未知 What is the evidence?', false);
  if (!unknown.detail.metadata.calls.some(c => c.status === 'unknown' && c.estimated_cny === null)) throw new Error('Unknown represented as zero');
  await page.locator('#loadTraces').click();
  await page.locator('#traceList').filter({hasText:consented.id}).waitFor();
  await page.setViewportSize({width:390,height:844});
  await page.locator('#loadTraces').scrollIntoViewIfNeeded();
  if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw new Error('Mobile horizontal overflow');
  await page.screenshot({path:'docs/assets/tracerag-trace-mobile.png',fullPage:false});
  await page.evaluate(() => { window.confirm = () => true; });
  await page.locator('#docList').getByRole('button',{name:'删除文献',exact:true}).click();
  await page.waitForFunction(() => !document.querySelector('#docList').textContent.includes('请求追踪隐私演示'));
  const erased = await (await page.request.get('http://127.0.0.1:18089/traces/' + consented.id)).json();
  if (erased.snapshot !== null) throw new Error('Document deletion retained snapshot');
  await page.locator('#loadTraces').click();
  const traceCard = page.locator('#traceList .card').filter({hasText:normal.id});
  await traceCard.getByRole('button',{name:'删除追踪',exact:true}).click();
  await page.waitForFunction(id => !document.querySelector('#traceList').textContent.includes(id), normal.id);
  // Hold an authorized Alice detail until after Bob logs in; private epoch must discard it.
  let signalHeld, releaseHeld;
  const held = new Promise(resolve => { signalHeld = resolve; });
  const release = new Promise(resolve => { releaseHeld = resolve; });
  await page.route('**/traces/' + consented.id, async route => {
    const response = await route.fetch(); signalHeld(); await release; await route.fulfill({response});
  });
  await page.evaluate(id => { window.__staleTrace = showTrace(id); }, consented.id);
  await held;
  await page.evaluate(() => document.querySelector('#traceDialog').close());
  await page.locator('#logoutBtn').click();
  await page.locator('#loginPanel').waitFor({state:'visible'});
  if (await page.locator('#traceText').textContent() || await page.locator('#traceList').textContent()) throw new Error('Logout retained traces');
  await login('bob');
  releaseHeld(); await page.unrouteAll({behavior:'wait'});
  await page.evaluate(() => window.__staleTrace);
  if ((await page.locator('#traceText').textContent()).includes('TRACE_PRIVATE') || (await page.locator('#traceText').textContent()).includes(consented.id)) throw new Error('Late detail leaked');
  await page.locator('#loadTraces').click();
  await page.locator('#traceList').filter({hasText:'暂无7天内'}).waitFor();
  await page.locator('#logoutBtn').click();
  if (errors.length) throw new Error(JSON.stringify(errors));
  return {metadataNoBody:true,perRequestConsent:true,actualContextSnapshot:true,budgetEvents:true,unknownIsNull:true,mobile:true,documentDeletionErasesContent:true,manualTraceDeletion:true,accountIsolation:true,lateTraceDropped:true,logoutClearsTraces:true,unexpectedBrowserErrors:errors.length};
}
