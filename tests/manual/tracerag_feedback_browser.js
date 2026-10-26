async page => {
  // Actual API, independent worker, PG and budget; model/rates/usage are synthetic.
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
  async function ask(consent) {
    await page.locator('#retainTrace').setChecked(consent);
    await page.locator('#q').fill('PRIVATE_FEEDBACK_QUESTION: What was measured?');
    const pending = page.waitForResponse(r => r.url().includes('/query/evidence') && r.request().method() === 'POST');
    await page.locator('#ask').click();
    const result = await pending;
    if (result.status() !== 200) throw new Error('Query failed');
    await page.waitForFunction(() => !document.querySelector('#ask').disabled);
    return result.headers()['x-trace-id'];
  }
  async function openLatest() {
    await page.locator('#chat .bot').last().getByRole('button',{name:'反馈 / 导出',exact:true}).click();
    await page.locator('#feedbackStatus').filter({hasText:'尚无反馈'}).waitFor();
  }
  async function save() {
    await page.locator('#saveFeedback').click();
    await page.locator('#feedbackStatus').filter({hasText:'反馈已保存'}).waitFor();
  }
  async function download(button, filename) {
    const pendingResponse = page.waitForResponse(r => r.url().endsWith('/export') && r.request().method() === 'POST');
    const pendingDownload = page.waitForEvent('download');
    await page.locator(button).click();
    const response = await pendingResponse;
    if (response.status() !== 200) throw new Error('Export failed: ' + response.status());
    const data = await response.json();
    const file = await pendingDownload;
    await file.saveAs(filename);
    return data;
  }
  await login('alice');
  await page.evaluate(() => { document.querySelector('#addform').parentElement.open = true; });
  await page.locator('#docTitle').fill('反馈与脱敏演示');
  await page.locator('#docContent').fill('PRIVATE_FEEDBACK_SOURCE research@example.invalid: Researchers measured cells. <img src=x onerror=window.__feedbackXSS=1>');
  await page.locator('#addBtn').click();
  await page.waitForFunction(() => document.querySelector('#docList').textContent.includes('反馈与脱敏演示'));
  const defaultId = await ask(false);
  await openLatest();
  if (!(await page.locator('#loadFeedbackSnapshot').isDisabled())) throw new Error('Default trace retroactively acquired content');
  await page.locator('#feedbackForm [value="unhelpful"]').check();
  await save();
  const metadata = await download('#exportMetadata','docs/assets/tracerag-feedback-metadata.json');
  if (JSON.stringify(metadata).includes('PRIVATE') || JSON.stringify(metadata).includes(defaultId)) throw new Error('Default export leaked source');
  await page.locator('#closeFeedback').click();
  const retainedId = await ask(true);
  await openLatest();
  await page.locator('#feedbackForm [value="unsupported_citation"]').check();
  await page.locator('#feedbackForm [value="outdated"]').check();
  await page.locator('#feedbackCitation').fill('1');
  await save();
  const before = await (await page.request.get('http://127.0.0.1:18089/budget')).json();
  await page.evaluate(() => { document.querySelector('#regressionEditor').open = true; });
  await page.locator('#loadFeedbackSnapshot').click();
  if (!(await page.locator('#caseQuestion').inputValue()).includes('PRIVATE_FEEDBACK')) throw new Error('Opt-in original unavailable for explicit editing');
  await page.locator('#caseQuestion').fill('What did Study A measure?');
  await page.locator('#caseAnswer').fill('Study A measured cells [99].');
  await page.locator('#caseContexts').fill('["Study A measured cells without establishing causation."]');
  await page.locator('#caseObserved').selectOption('completed');
  await page.locator('#caseSemantics').fill('A human reviewer must verify that no causal conclusion is asserted.');
  await page.locator('#caseReviewed').check();
  // Changing any draft after review must require another explicit confirmation.
  await page.locator('#caseQuestion').fill('What did the synthetic Study A measure?');
  if (await page.locator('#caseReviewed').isChecked()) throw new Error('Edited draft reused confirmation');
  await page.locator('#caseReviewed').check();
  await page.locator('#feedbackDialog').evaluate(el => { el.scrollTop = 0; });
  await page.screenshot({path:'docs/assets/tracerag-feedback-editor.png',fullPage:false});
  const regression = await download('#exportRegression','docs/assets/tracerag-feedback-regression.json');
  if (JSON.stringify(regression).includes('PRIVATE') || JSON.stringify(regression).includes(retainedId)) throw new Error('Reviewed export merged raw source');
  if (await page.locator('#caseReviewed').isChecked()) throw new Error('Export reused confirmation');
  if (regression.case.answer !== 'Study A measured cells [99].' || regression.case.expected.citation_range_valid !== true) throw new Error('Regression failure input was altered');
  const original = await (await page.request.get('http://127.0.0.1:18089/traces/' + retainedId)).json();
  if (!original.snapshot.question.includes('PRIVATE_FEEDBACK')) throw new Error('Export mutated original trace');
  const after = await (await page.request.get('http://127.0.0.1:18089/budget')).json();
  if (JSON.stringify(before.account) !== JSON.stringify(after.account)) throw new Error('Feedback/export changed budget');
  if (await page.evaluate(() => !!window.__feedbackXSS)) throw new Error('Source interpreted as HTML');
  await page.setViewportSize({width:390,height:844});
  await page.locator('#feedbackDialog').evaluate(el => { el.scrollTop = 0; });
  if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth || document.querySelector('#feedbackDialog').scrollWidth > document.querySelector('#feedbackDialog').clientWidth)) throw new Error('Mobile overflow');
  await page.screenshot({path:'docs/assets/tracerag-feedback-mobile.png',fullPage:false});
  await page.locator('#removeFeedback').click();
  await page.locator('#feedbackStatus').filter({hasText:'反馈已撤回'}).waitFor();
  const removed = await (await page.request.get('http://127.0.0.1:18089/traces/' + retainedId + '/feedback')).json();
  if (removed.feedback !== null) throw new Error('Feedback not withdrawn');
  await page.locator('#feedbackForm [value="outdated"]').check(); await save();
  await page.locator('#closeFeedback').click();
  await page.waitForFunction(() => document.querySelector('#caseQuestion').value === '');
  // Deliver two authorized Alice responses only after Bob logs in.
  let detailHeld, exportHeld, releaseHeld;
  const heldDetail = new Promise(resolve => { detailHeld = resolve; });
  const heldExport = new Promise(resolve => { exportHeld = resolve; });
  const release = new Promise(resolve => { releaseHeld = resolve; });
  await page.route('**/traces/' + retainedId, async route => { const response = await route.fetch(); detailHeld(); await release; await route.fulfill({response}); });
  await page.route('**/traces/' + retainedId + '/export', async route => { const response = await route.fetch(); exportHeld(); await release; await route.fulfill({response}); });
  let lateDownloads = 0; page.on('download', () => lateDownloads++);
  await page.evaluate(({id, reviewed_case}) => { window.__lateFeedback = openFeedback(id); window.__lateFeedbackExport = feedbackExport({mode:'regression',reviewed_case},'should-not-download.json'); }, {id:retainedId,reviewed_case:regression.case});
  await Promise.all([heldDetail,heldExport]);
  await page.evaluate(() => document.querySelector('#feedbackDialog').close());
  await page.locator('#logoutBtn').click(); await page.locator('#loginPanel').waitFor({state:'visible'});
  await login('bob');
  releaseHeld(); await page.unrouteAll({behavior:'wait'});
  await page.evaluate(() => Promise.all([window.__lateFeedback,window.__lateFeedbackExport]));
  if (lateDownloads) throw new Error('Late private export downloaded after account change');
  if (await page.locator('#caseQuestion').inputValue() || await page.locator('#caseAnswer').inputValue() || await page.locator('#caseSemantics').inputValue()) throw new Error('Late response restored private draft');
  if (await page.evaluate(() => feedbackSnapshot !== null || feedbackTrace !== '')) throw new Error('Account retained feedback state');
  const denied = await page.request.get('http://127.0.0.1:18089/traces/' + retainedId + '/feedback');
  if (denied.status() !== 404) throw new Error('Cross-owner feedback readable');
  await page.locator('#logoutBtn').click();
  if (errors.length) throw new Error(JSON.stringify(errors));
  return {feedbackSaves:true,defaultMetadataOnly:true,noRetroactiveSnapshot:true,manualReviewResets:true,reviewedDownload:true,originalTraceUnchanged:true,budgetUnchanged:true,withdrawal:true,mobile:true,textOnly:true,accountIsolation:true,lateFeedbackDropped:true,lateExportNotDownloaded:true,logoutClearsDraft:true,unexpectedBrowserErrors:errors.length};
}
