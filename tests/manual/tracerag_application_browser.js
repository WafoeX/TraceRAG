async page => {
  // Run only against tests.manual.tracerag_application_demo (synthetic models).
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', msg => { if (msg.type() === 'error') errors.push(msg.text()); });
  await page.goto('http://127.0.0.1:18089/');
  await page.setViewportSize({width: 1440, height: 1050});
  await page.locator('summary').filter({hasText: '上传 / 添加文献'}).click();
  const textA = 'Scientists observed liver cells in a controlled experiment. Methods and observations are limited to the reported experimental conditions. This fixture contains deliberately long evidence so that the browser test can verify the final text is not truncated to a 240-character preview. This is not a scientific conclusion. END_EVIDENCE <img src=x onerror="window.__injected=1">';
  async function upload(name) {
    await page.locator('#docFile').setInputFiles('tests/fixtures/tracerag-app/' + name);
    await page.locator('#addBtn').click();
    await page.locator('#uploadStatus').filter({hasText:'任务 #'}).waitFor();
  }
  await upload('paper-a.txt');
  await page.waitForFunction(() => document.querySelector('#docList').textContent.includes('paper-a.txt'));
  await upload('paper-b.txt');
  await page.waitForFunction(() => document.querySelectorAll('#docList article').length === 2);
  const before = await (await page.request.get('http://127.0.0.1:18089/documents')).json();
  const a = before.find(d => d.title === 'paper-a.txt');
  await upload('renamed-a.txt');
  const after = await (await page.request.get('http://127.0.0.1:18089/documents')).json();
  if (after.length !== 2) throw new Error('Duplicate upload created another document');
  await page.locator('#q').fill('What did the studies observe?');
  await page.locator('#ask').click();
  await page.getByText('警告：回答含无效引用编号 99；不得视为有效来源。', {exact:true}).waitFor();
  if (await page.locator('.sources .card').count() !== 2) throw new Error('Expected two source snapshots');
  const sourceA = page.locator('.sources .card').filter({hasText:'paper-a.txt'});
  await sourceA.getByRole('button', {name:'查看库中原文切块'}).click();
  await page.locator('#evidenceText').filter({hasText:'END_EVIDENCE'}).waitFor();
  if (await page.locator('#evidenceText').textContent() !== textA) throw new Error('Evidence was truncated or modified');
  if (await page.evaluate(() => Boolean(window.__injected))) throw new Error('XSS executed');
  await page.screenshot({path:'docs/assets/tracerag-evidence-demo.png', fullPage:true});
  await page.locator('#closeEvidence').click();
  await page.screenshot({path:'docs/assets/tracerag-application-demo.png', fullPage:true});
  // CLI intercepts native dialogs; explicitly accept confirmation in this fixture.
  await page.evaluate(() => { window.confirm = () => true; });
  await page.locator('#docList article').filter({hasText:'paper-a.txt'}).getByRole('button', {name:'删除文献'}).click();
  await page.waitForFunction(() => document.querySelectorAll('#docList article').length === 1);
  if ((await page.request.get('http://127.0.0.1:18089/documents/' + a.id + '/chunks/0')).status() !== 404) throw new Error('Deleted evidence still accessible');
  await page.locator('#q').fill('What is left in the knowledge base?');
  await page.locator('#ask').click();
  await page.waitForFunction(() => document.querySelectorAll('.sources').length === 2);
  const last = page.locator('.sources').last();
  if (await last.textContent().then(t => t.includes('paper-a.txt'))) throw new Error('New query used deleted evidence');
  // Malformed PDF fails in the worker; UI exposes retry and data-discard actions.
  await upload('invalid.pdf');
  const failedCard = page.locator('#jobs article').filter({hasText:'invalid.pdf'});
  await failedCard.getByRole('button', {name:'重试', exact:true}).waitFor();
  const retried = page.waitForResponse(res => res.url().endsWith('/retry') && res.request().method() === 'POST');
  await failedCard.getByRole('button', {name:'重试', exact:true}).click();
  if ((await retried).status() !== 202) throw new Error('Retry was not queued');
  await page.waitForFunction(async () => {
    const jobs = await (await fetch('/documents/jobs')).json();
    return jobs.some(j => j.title === 'invalid.pdf' && j.state === 'failed');
  });
  await failedCard.getByRole('button', {name:'丢弃原始数据'}).waitFor();
  await failedCard.getByRole('button', {name:'丢弃原始数据'}).click();
  await page.waitForFunction(() => !document.querySelector('#jobs').textContent.includes('invalid.pdf'));
  await page.locator('#docTitle').fill('Pasted methods');
  await page.locator('#docContent').fill('This pasted text describes the experimental methods.');
  await page.locator('#addBtn').click();
  await page.waitForFunction(() => document.querySelector('#docList').textContent.includes('Pasted methods'));
  await page.setViewportSize({width:390, height:844});
  await page.locator('aside').scrollIntoViewIfNeeded();
  await page.screenshot({path:'docs/assets/tracerag-mobile-demo.png', fullPage:true});
  if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw new Error('Mobile horizontal overflow');
  if (errors.length) throw new Error('Browser errors: ' + errors.join('; '));
  return {documentsUploaded:2, dedup:true, completeEvidence:true, xssBlocked:true, citationWarning:true, deletionVerified:true, failedTaskDiscarded:true, retry:true, textSubmission:true, mobile:true, browserErrors:errors.length, model:'deterministic offline fixture'};
}
