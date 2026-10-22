// Headless UI regression with synthetic metrics; never starts a model or job.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {chromium} = require('playwright');

(async () => {
  const browser = await chromium.launch({headless: true, channel: 'msedge'});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1080}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const web = path.resolve(__dirname, '../../src/openpi/bpx_studio/web');
    const dataset = '/scratch/demo/lerobot/bpx/bpx_blocks_multitask_v2';
    const split = '/scratch/demo/reports/validation_split_v1.json';
    const job = {id: 'train-test', kind: 'train', status: 'running', progress: {current: 2000, total: 10000}, metrics: {loss: .03}, command: 'synthetic test only'};
    const metrics = [0, 1000, 2000].map((current, index) => ({current, metrics: {
      loss: index === 1 ? 10 : .03, train_eval_loss: .1-index*.01, val_loss: .15+index*.02,
      generalization_gap: .05+index*.03, val_stderr: .01, best_validation_step: 0,
      overfit_warning: index === 2 ? 1 : 0,
    }, display: {loss_trend: .03, loss_outlier: index === 1}}));
    let generated = false;
    let splitRequest;
    const exportRequests = [];
    const profileQueries = [];
    await page.route('http://studio.test/**', async route => {
      const url = new URL(route.request().url());
      let data;
      if (url.pathname === '/api/config') data = {
        default_openpi_root: '/code/openpi', default_config_name: 'pi05_bpx_full_low_mem_multitask_v2',
        default_experiment_name: 'test_v001', known_config_names: [], hf_lerobot_home: '/scratch/demo/lerobot',
        clean_dataset: dataset, raw_dataset: dataset+'_raw', checkpoint_root: '/scratch/demo/checkpoints',
        validation_split: split,
      };
      else if (url.pathname === '/api/config-profile') {
        profileQueries.push(url.searchParams.get('validation_manifest'));
        data = {dataset_path: dataset, repo_id: 'bpx/bpx_blocks_multitask_v2', dataset_exists: true,
          stats_paths_match: true, computed_stats_path: '/code/assets/test/norm_stats.json',
          loaded_stats_path: '/code/assets/test/norm_stats.json', computed_stats_exists: false, purpose: 'test'};
      } else if (url.pathname === '/api/jobs/validation-split') {
        splitRequest = route.request().postDataJSON(); generated = true;
        data = {id: 'split-test', kind: 'validation_split', status: 'running'};
      } else if (url.pathname === '/api/jobs/export/hdf5') {
        exportRequests.push(route.request().postDataJSON());
        data = {id: 'export-test', kind: 'hdf5_export', status: 'running'};
      } else if (url.pathname === '/api/jobs') data = generated ? [{id:'split-test',kind:'validation_split',status:'succeeded',progress:{}},job] : [job];
      else if (url.pathname.endsWith('/metrics')) data = metrics;
      else if (url.pathname.endsWith('/log')) data = {text:'synthetic metrics'};
      else if (url.pathname === '/api/checkpoints') data = [];
      else if (url.pathname === '/api/next-experiment') data = {experiment_name:'test_v002'};
      else if (url.pathname === '/api/system') data = {gpus: [], disk: {free: 1e9,total:2e9,used:1e9}};
      if (data !== undefined) return route.fulfill({json:data});
      const file = path.join(web, url.pathname === '/' ? 'index.html' : url.pathname.replace('/static/',''));
      if (!file.startsWith(web) || !fs.existsSync(file)) return route.fulfill({status:404,body:'not found'});
      return route.fulfill({path:file,contentType:file.endsWith('.js')?'text/javascript':file.endsWith('.css')?'text/css':'text/html'});
    });
    await page.goto('http://studio.test/');
    await page.locator('[data-tab="train"]').click();
    await page.locator('.job').first().click();
    await page.waitForFunction(() => document.querySelectorAll('#validationChart path').length === 2);
    assert.equal(await page.locator('#metricBestStep').innerText(), '0');
    assert((await page.locator('#overfitHint').innerText()).includes('疑似过拟合'));
    const trendPath = await page.locator('#lossChart path').getAttribute('d');
    await page.selectOption('#lossView', 'raw');
    const rawPath = await page.locator('#lossChart path').getAttribute('d');
    assert.notEqual(trendPath,rawPath);
    await page.locator('#validationGrouping').selectOption('manifest');
    await page.locator('#validationSource').fill('/scratch/demo/manifests/source.json');
    await page.locator('#createValidationSplit').click();
    await page.waitForFunction(expected => document.getElementById('validationManifest').value === expected, split);
    assert.equal(splitRequest.dataset_root, dataset);
    assert.equal(splitRequest.independent_episodes, false);
    assert.equal(splitRequest.fraction, .1);
    await page.waitForTimeout(100);
    assert(profileQueries.includes(split));
    await page.locator('[data-tab="export"]').click();
    await page.locator('#exportSource').fill(dataset);
    await page.locator('#exportOutput').fill('/scratch/demo/exports/check_one');
    await page.locator('#exportEpisodes').fill('0，2');
    await page.locator('#startExport').click();
    await page.waitForFunction(() => document.querySelector('#exportEpisodes').value === '0，2');
    await page.waitForTimeout(100);
    assert.deepEqual(exportRequests.at(-1).episode_indices, [0, 2]);
    await page.locator('#exportEpisodes').fill('-1');
    const count = exportRequests.length;
    await page.locator('#startExport').click();
    await page.waitForTimeout(100);
    assert.equal(exportRequests.length, count);
    await page.locator('#exportEpisodes').fill('');
    await page.locator('#startExport').click();
    await page.waitForTimeout(100);
    assert.equal(exportRequests.at(-1).episode_indices, null);
    if (process.env.BPX_UI_SCREENSHOT) await page.screenshot({path:process.env.BPX_UI_SCREENSHOT,fullPage:true});
    assert.deepEqual(errors, []);
    console.log('PASS: evaluation chart, raw/trend switch, grouped split, normalization binding, episode export selection, no JS errors');
  } finally { await browser.close(); }
})().catch(error => {console.error(error);process.exitCode=1;});
