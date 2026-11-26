const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const b = await chromium.launch();
  const pg = await b.newPage();
  await pg.goto('file://' + path.resolve('paper/oll_paper.html'));
  await pg.pdf({ path: 'paper/oll_paper.pdf', format: 'A4', margin: { top: '18mm', bottom: '18mm', left: '16mm', right: '16mm' }, printBackground: true });
  await b.close();
  console.log('pdf ok');
})();
