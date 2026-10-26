async page => {
  // Offline demo API + independent worker + disposable PostgreSQL only.
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', msg => { if (msg.type() === 'error') errors.push(msg.text()); });
  await page.goto('http://127.0.0.1:18089/');
  await page.setViewportSize({width:1440,height:1050});
  await page.locator('summary').filter({hasText:'上传 / 添加文献'}).click();
  async function upload(name) {
    await page.locator('#docFile').setInputFiles('tests/fixtures/tracerag-app/' + name);
    await page.locator('#addBtn').click();
    await page.locator('#uploadStatus').filter({hasText:'任务 #'}).waitFor();
  }
  async function catalog() { return (await page.request.get('http://127.0.0.1:18089/documents/catalog')).json(); }
  async function ask(text, expectedSources) {
    const previous = await page.locator('.message.bot').count();
    await page.locator('#q').fill(text); await page.locator('#ask').click();
    await page.waitForFunction(n => document.querySelectorAll('.message.bot').length === n + 1 && !document.querySelector('#ask').disabled, previous);
    const block = page.locator('.message.bot').last();
    if (await block.locator('.sources .card').count() !== expectedSources) throw new Error('Unexpected actual-context count');
    return block;
  }
  await upload('paper-a.txt');
  await page.waitForFunction(() => document.querySelector('#docList').textContent.includes('paper-a.txt'));
  const v1 = (await catalog())[0];
  const firstAnswer = await ask('What does the document describe?', 1);
  if (!(await firstAnswer.textContent()).includes('v1')) throw new Error('Missing v1 citation');
  await page.locator('#docList').getByRole('button', {name:'更新此文献'}).click();
  await upload('invalid.pdf');
  await page.waitForFunction(() => document.querySelector('#jobs').textContent.includes('invalid_document'));
  if ((await catalog())[0].id !== v1.id) throw new Error('Failed update replaced the active document');
  await page.screenshot({path:'docs/assets/tracerag-version-failure.png',fullPage:true});
  await page.locator('#docList').getByRole('button', {name:'更新此文献'}).click();
  await upload('three-pages.pdf');
  await page.waitForFunction(() => document.querySelector('#docList').textContent.includes('v2'));
  const v2 = (await catalog())[0];
  if (v2.id === v1.id || v2.logical_id !== v1.logical_id) throw new Error('Identity/version invariant failed');
  // Old answer evidence keeps its original immutable URL and contents.
  await firstAnswer.getByRole('button', {name:'查看库中原文切块'}).click();
  await page.locator('#evidenceLocation').filter({hasText:'v1'}).waitFor();
  if (!(await page.locator('#evidenceText').textContent()).includes('END_EVIDENCE')) throw new Error('Old evidence changed');
  await page.locator('#closeEvidence').click();
  const secondAnswer = await ask('What are the physical page observations?', 2);
  if ((await secondAnswer.textContent()).includes('END_EVIDENCE')) throw new Error('New query mixed archived evidence');
  const thirdPage = secondAnswer.locator('.sources .card').filter({hasText:'Third physical page evidence.'});
  await thirdPage.getByRole('button', {name:'查看库中原文切块'}).click();
  await page.locator('#evidenceLocation').filter({hasText:'PDF物理页 3'}).waitFor();
  const originalUrl = await page.locator('#originalLink').getAttribute('href');
  if (originalUrl !== '/documents/' + v2.id + '/original#page=3') throw new Error('Wrong PDF page anchor');
  const original = await page.request.get('http://127.0.0.1:18089' + originalUrl.split('#')[0]);
  if (original.status() !== 200 || original.headers()['content-type'] !== 'application/pdf') throw new Error('Original PDF unavailable');
  const popupPromise = page.waitForEvent('popup');
  await page.locator('#originalLink').click();
  const popup = await popupPromise;
  await popup.waitForLoadState('domcontentloaded');
  if (!popup.url().endsWith('/original#page=3')) throw new Error('PDF navigation lost page anchor');
  await popup.close();
  await page.screenshot({path:'docs/assets/tracerag-version-evidence.png',fullPage:true});
  await page.locator('#closeEvidence').click();
  await page.locator('#docList').getByRole('button', {name:'版本历史'}).click();
  await page.locator('#historyList').getByText('v2 · 当前使用', {exact:true}).waitFor();
  await page.locator('#historyList').getByText('v1 · 已归档', {exact:true}).waitFor();
  if (await page.locator('#historyList a').count() !== 2) throw new Error('Missing immutable original links');
  await page.screenshot({path:'docs/assets/tracerag-version-history.png',fullPage:true});
  await page.locator('#closeHistory').click();
  await page.setViewportSize({width:390,height:844});
  if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw new Error('Mobile overflow');
  await page.evaluate(() => { window.confirm = () => true; });
  await page.locator('#docList').getByRole('button', {name:'删除文献'}).click();
  await page.waitForFunction(() => document.querySelectorAll('#docList article').length === 0);
  for (const doc of [v1,v2]) {
    if ((await page.request.get('http://127.0.0.1:18089/documents/' + doc.id + '/original')).status() !== 404) throw new Error('Deleted original still readable');
  }
  const refused = await ask('What remains?', 0);
  if (!(await refused.textContent()).includes("couldn't find")) throw new Error('Empty-library query did not refuse');
  if (errors.length) throw new Error('Browser errors: ' + errors.join('; '));
  return {failureKeepsOld:true,logicalIdentityStable:true,immutableEvidence:true,versionHistory:true,pdfPhysicalPage3:true,originalPopup:true,mobile:true,allVersionsDeleted:true,emptyLibraryRefused:true,browserErrors:errors.length,model:'deterministic offline fixture'};
}
