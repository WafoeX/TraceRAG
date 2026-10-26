async page => {
  // Fixed local test DB only. Model outputs, rates and usage are synthetic;
  // reservations/settlements/owner boundaries are the actual application code.
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
    await page.locator('#budgetSummary').filter({hasText:'累计额度 ¥0.05'}).waitFor();
  }
  async function budget() { return (await page.request.get('http://127.0.0.1:18089/budget')).json(); }
  async function ask(text) {
    await page.locator('#q').fill(text);
    const pending = page.waitForResponse(res => res.url().includes('/query/evidence') && res.request().method() === 'POST');
    await page.locator('#ask').click();
    const response = await pending;
    await page.waitForFunction(() => !document.querySelector('#ask').disabled);
    if (response.status() !== 200) throw new Error('Unexpected query status ' + response.status());
    return response.json();
  }
  await login('alice');
  await page.evaluate(() => { document.querySelector('#addform').parentElement.open = true; });
  await page.locator('#docTitle').fill('预算与证据演示');
  await page.locator('#docContent').fill('BUDGET_PRIVATE_EVIDENCE: Researchers measured liver cells in a controlled study.');
  await page.locator('#addBtn').click();
  await page.waitForFunction(() => document.querySelector('#docList').textContent.includes('预算与证据演示'));
  const normal = await ask('What did the researchers measure?');
  if (normal.generation_status !== 'completed') throw new Error('Known-usage call failed');
  const settled = await budget();
  if (!(Number(settled.account.estimated_spent_cny) > 0) || Number(settled.account.held_cny) !== 0) throw new Error('Known usage did not settle/release');
  for (let i=0; i<2; i++) {
    const unknown = await ask('模拟费用未知 ' + i + ': What is the evidence?');
    if (!unknown.warnings.includes('budget_charge_unknown')) throw new Error('Missing unknown-cost warning');
  }
  const pending = await budget();
  if (pending.reservations.filter(row => row.status === 'unknown').length !== 2) throw new Error('Unknown reservations were lost');
  if (!(Number(pending.account.held_cny) > 0)) throw new Error('Unknown hold was incorrectly refunded');
  const denied = await ask('预算不足时仍核对证据');
  if (denied.generation_status !== 'failed' || !denied.warnings.includes('generation_budget_exhausted')) throw new Error('Generation budget not enforced');
  const card = page.locator('#chat .bot').last();
  if (!(await card.textContent()).includes('BUDGET_PRIVATE_EVIDENCE')) throw new Error('Budget denial lost evidence');
  await card.getByRole('button',{name:'查看库中原文切块'}).click();
  await page.locator('#evidenceText').filter({hasText:'BUDGET_PRIVATE_EVIDENCE'}).waitFor();
  await page.locator('#closeEvidence').click();
  await page.locator('#budgetSummary').filter({hasText:'2 笔费用未知'}).waitFor();
  await page.screenshot({path:'docs/assets/tracerag-budget-denied.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await page.evaluate(() => window.scrollTo(0,0));
  if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw new Error('Mobile horizontal overflow');
  await page.screenshot({path:'docs/assets/tracerag-budget-mobile.png',fullPage:false});
  // Deliver a previously authorized Alice budget snapshot after Bob signs in.
  let holdNext = true, releaseHeld, signalHeld;
  const held = new Promise(resolve => { signalHeld = resolve; });
  const release = new Promise(resolve => { releaseHeld = resolve; });
  await page.route('**/budget', async route => {
    if (holdNext) { holdNext = false; const response = await route.fetch(); signalHeld(); await release; await route.fulfill({response}); }
    else await route.continue();
  });
  await page.evaluate(() => { window.__staleBudget = loadBudget(); });
  await held;
  await page.locator('#logoutBtn').click();
  await page.locator('#loginPanel').waitFor({state:'visible'});
  if (await page.locator('#budgetSummary').textContent()) throw new Error('Logout retained private budget');
  await login('bob');
  releaseHeld(); await page.unrouteAll({behavior:'wait'});
  await page.evaluate(() => window.__staleBudget);
  const bob = await budget();
  if (bob.reservations.length || Number(bob.account.held_cny) || Number(bob.account.estimated_spent_cny)) throw new Error('Account budget mixed');
  if ((await page.locator('#budgetSummary').textContent()).includes('费用未知')) throw new Error('Stale Alice budget rendered for Bob');
  await page.locator('#logoutBtn').click();
  await page.locator('#loginPanel').waitFor({state:'visible'});
  if (errors.length) throw new Error(JSON.stringify(errors));
  return {knownUsageSettles:true,unknownHoldsRetained:true,budgetDenialKeepsEvidence:true,evidenceOpens:true,mobile:true,accountBudgetIsolation:true,lateBudgetResponseDropped:true,logoutClearsBudget:true,unexpectedBrowserErrors:errors.length};
}
