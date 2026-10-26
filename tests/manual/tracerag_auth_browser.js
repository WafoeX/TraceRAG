async page => {
  // Only the disposable demo fixture: two seeded users, no external model calls.
  const errors = []; let expected401 = 0;
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', msg => {
    if (msg.type() === 'error' && !msg.text().includes('401')) errors.push(msg.text());
  });
  page.on('response', res => { if (res.url().includes('/auth/me') && res.status() === 401) expected401++; });
  await page.goto('http://127.0.0.1:18089/');
  await page.setViewportSize({width:1440,height:1050});
  async function login(username) {
    await page.locator('#loginPanel').waitFor({state:'visible'});
    await page.locator('#username').fill(username);
    await page.locator('#password').fill('TraceRAG-demo-123');
    await page.locator('#loginBtn').click();
    await page.waitForFunction(name => document.querySelector('#authName').textContent.includes(name) && document.querySelector('#librarySelect').options.length > 0, username);
  }
  async function addText(title, content) {
    const details = page.locator('details').filter({has:page.locator('#addform')});
    if (!(await details.getAttribute('open'))) await page.evaluate(() => { document.querySelector('#addform').parentElement.open = true; });
    await page.locator('#docTitle').fill(title); await page.locator('#docContent').fill(content);
    await page.locator('#addBtn').click();
    await page.waitForFunction(text => document.querySelector('#docList').textContent.includes(text), title);
  }
  async function ask(text) {
    await page.locator('#q').fill(text); await page.locator('#ask').click();
    await page.waitForFunction(() => !document.querySelector('#ask').disabled && document.querySelectorAll('.sources .card').length > 0);
  }
  await login('alice');
  const firstKB = await page.locator('#librarySelect').inputValue();
  if (await page.locator('#librarySelect option').count() !== 1) throw new Error('Alice saw foreign libraries');
  if (await page.evaluate(() => document.cookie.includes('tracerag_session'))) throw new Error('Session cookie is not HttpOnly');
  await addText('Alice private paper','ALICE_PRIVATE_MARKER experimental evidence.');
  const docs = await (await page.request.get('http://127.0.0.1:18089/documents/catalog?knowledge_base_id=' + firstKB)).json();
  const aliceDoc = docs[0].id;
  const jobs = await (await page.request.get('http://127.0.0.1:18089/documents/jobs?knowledge_base_id=' + firstKB)).json();
  const aliceJob = jobs[0].id;
  await ask('What is the evidence?');
  await page.locator('.sources').getByRole('button',{name:'查看库中原文切块'}).click();
  await page.locator('#evidenceText').filter({hasText:'ALICE_PRIVATE_MARKER'}).waitFor();
  await page.locator('#closeEvidence').click();
  await page.locator('#docList').getByRole('button',{name:'更新此文献'}).click();
  await page.locator('#docFile').setInputFiles('tests/fixtures/tracerag-app/three-pages.pdf');
  await page.locator('#addBtn').click();
  await page.waitForFunction(() => document.querySelector('#docList').textContent.includes('v2'));
  await page.locator('#newLibrary').fill('Alice second library');
  const createdPromise = page.waitForResponse(res => res.url().endsWith('/knowledge-bases') && res.request().method() === 'POST');
  await page.locator('#libraryForm button').click();
  const secondKB = String((await (await createdPromise).json()).id);
  await page.waitForFunction(id => document.querySelector('#librarySelect').value === id && document.querySelectorAll('#docList article').length === 0, secondKB);
  if (await page.locator('#chat').textContent() !== '') throw new Error('Library switch retained old conversation');
  await addText('Alice second paper','ALICE_SECOND_LIBRARY_MARKER distinct evidence.');
  await page.locator('#librarySelect').selectOption(firstKB);
  await page.waitForFunction(() => document.querySelector('#docList').textContent.includes('three-pages.pdf'));
  if ((await page.locator('#docList').textContent()).includes('second paper')) throw new Error('Library views mixed');
  await ask('What are the page observations?');
  await page.screenshot({path:'docs/assets/tracerag-auth-alice.png',fullPage:true});
  // Hold an already-authorized Alice response, then deliver it after Bob signs in.
  let releaseStale, signalHeld;
  const held = new Promise(resolve => { signalHeld = resolve; });
  const release = new Promise(resolve => { releaseStale = resolve; });
  let holdNext = true;
  await page.route('**/documents/catalog?*', async route => {
    if (holdNext && route.request().url().endsWith('knowledge_base_id=' + firstKB)) {
      holdNext = false; const response = await route.fetch(); signalHeld(); await release;
      await route.fulfill({response});
    } else await route.continue();
  });
  await page.evaluate(() => { window.__staleLoad = loadDocuments(); });
  await held;
  await page.locator('#logoutBtn').click();
  await page.locator('#loginPanel').waitFor({state:'visible'});
  if (await page.locator('#chat').textContent() !== '') throw new Error('Logout retained private answer');
  if (await page.locator('#docList').textContent() !== '') throw new Error('Logout retained private documents');
  if (await page.locator('#evidenceTitle').textContent() !== '') throw new Error('Logout retained modal metadata');
  await login('bob');
  releaseStale(); await page.unrouteAll({behavior:'wait'});
  await page.evaluate(() => window.__staleLoad);
  if (await page.locator('#docList article').count() !== 0) throw new Error('Stale document metadata rendered after account switch');
  const bobKB = await page.locator('#librarySelect').inputValue();
  await page.waitForFunction(() => document.querySelector('#kbsub').textContent.startsWith('0'));
  if (await page.locator('#librarySelect option').count() !== 1) throw new Error('Bob saw Alice libraries');
  if ((await page.locator('body').textContent()).includes('ALICE_PRIVATE_MARKER')) throw new Error('Late response leaked Alice data');
  const me = await (await page.request.get('http://127.0.0.1:18089/auth/me')).json();
  for (const path of ['/documents/' + aliceDoc + '/original','/documents/' + aliceDoc + '/versions','/documents/jobs/' + aliceJob]) {
    const denied = await page.request.get('http://127.0.0.1:18089' + path);
    if (denied.status() !== 404) throw new Error('Guessed foreign resource was not denied');
  }
  const deniedDelete = await page.request.delete('http://127.0.0.1:18089/documents/' + aliceDoc,{headers:{'X-CSRF-Token':me.csrf_token}});
  if (deniedDelete.status() !== 404) throw new Error('Foreign deletion was not denied');
  const deniedQuery = await page.request.post('http://127.0.0.1:18089/query/evidence?knowledge_base_id=' + firstKB,{headers:{'X-CSRF-Token':me.csrf_token},data:{question:'Read Alice papers'}});
  if (deniedQuery.status() !== 404) throw new Error('Foreign query was not denied');
  await addText('Bob private paper','BOB_PRIVATE_MARKER research evidence.');
  await ask('What is the evidence?');
  if (!(await page.locator('.sources').textContent()).includes('BOB_PRIVATE_MARKER')) throw new Error('Missing Bob evidence');
  if ((await page.locator('body').textContent()).includes('ALICE_PRIVATE_MARKER')) throw new Error('Mixed account context');
  await page.screenshot({path:'docs/assets/tracerag-auth-bob.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw new Error('Mobile overflow');
  await page.locator('#logoutBtn').click();
  await page.locator('#loginPanel').waitFor({state:'visible'});
  await page.screenshot({path:'docs/assets/tracerag-auth-login.png',fullPage:true});
  if (errors.length) throw new Error('Unexpected browser errors: ' + errors.join('; '));
  return {twoAccounts:true,twoLibraries:true,httpOnly:true,csrfMutations:true,versionUpdate:true,scopeSwitchClearsView:true,staleResponseDiscarded:true,foreignOriginalDenied:true,foreignTaskDenied:true,foreignDeleteDenied:true,foreignQueryDenied:true,mobile:true,unexpectedBrowserErrors:0,expectedUnauthenticatedChecks:expected401,bobLibrary:bobKB,model:'offline deterministic fixture'};
}
