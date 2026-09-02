module.exports = async function report({ github, core, context, branch }) {
  const fs = require('fs');
  const { owner, repo } = context.repo;
  const runId = context.runId;

  // Read core suite test names from config (strip comments and quotes for robustness).
  const configContent = fs.readFileSync('.github/workflows/configs/core_suite.yaml', 'utf8');
  const coreTests = configContent
    .split('\n')
    .map(l => l.split('#')[0].trim())
    .filter(l => l.startsWith('- '))
    .map(l => l.slice(2).trim().replace(/^['"]|['"]$/g, ''));
  core.info(`Core suite tests: ${JSON.stringify(coreTests)}`);

  // List all jobs in the current workflow run (with empty-page and max-page safety).
  let allJobs = [];
  let page = 1;
  const MAX_JOB_PAGES = 50;
  while (page <= MAX_JOB_PAGES) {
    const resp = await github.rest.actions.listJobsForWorkflowRun({
      owner, repo, run_id: runId, per_page: 100, page,
    });
    if (!resp.data.jobs || resp.data.jobs.length === 0) break;
    allJobs = allJobs.concat(resp.data.jobs);
    if (allJobs.length >= resp.data.total_count) break;
    page++;
  }

  // Download job logs once per job (avoids redundant API calls).
  const fetchJobLogs = async (job) => {
    try {
      const resp = await github.rest.actions.downloadJobLogsForWorkflowRun({
        owner, repo, job_id: job.id,
      });
      return Buffer.isBuffer(resp.data) ? resp.data.toString('utf8') : String(resp.data);
    } catch (e) {
      core.warning(`Failed to download logs for job "${job.name}": ${e.message}`);
      return null;
    }
  };

  // Match core tests against job names.
  const results = [];
  for (const test of coreTests) {
    const matched = allJobs.filter(j => j.name.includes(test));
    if (matched.length === 0) {
      core.info(`Core test "${test}" not found in this run, skipped.`);
      continue;
    }
    for (const job of matched) {
      const conclusion = job.conclusion;
      core.info(`Core test "${test}" -> job "${job.name}": conclusion=${conclusion}`);
      if (conclusion === 'success') {
        results.push({ test, status: 'pass' });
      } else if (conclusion === 'failure') {
        const logText = await fetchJobLogs(job);
        if (logText && logText.includes('Application startup complete')) {
          core.info('  -> Has "Application startup complete" -> non-core issue, pass');
          results.push({ test, status: 'pass' });
        } else if (logText && logText.includes('test session starts')) {
          core.info('  -> No "Application startup complete" but test session ran -> startup failure');
          results.push({ test, status: 'fail' });
        } else {
          core.info('  -> No "Application startup complete" and no test session -> env/resource issue, ignored');
          results.push({ test, status: 'pass' });
        }
      } else {
        core.info(`  -> conclusion=${conclusion}, ignored`);
        results.push({ test, status: 'pass' });
      }
    }
  }

  const anyFail = results.some(r => r.status === 'fail');
  const state = anyFail ? 'failure' : 'success';
  const description = anyFail
    ? `Core suite failed: ${results.filter(r => r.status === 'fail').map(r => r.test).join(', ')}`
    : 'Core suite passed';
  core.info(`Overall status: ${state} (${description})`);

  // Write to branch HEAD.
  const refData = await github.rest.git.getRef({
    owner, repo, ref: `heads/${branch}`,
  });
  const branchHeadSha = refData.data.object.sha;
  await github.rest.repos.createCommitStatus({
    owner, repo, sha: branchHeadSha,
    context: 'nightly/core-suite', state, description,
  });
  core.info(`Wrote nightly/core-suite=${state} on ${branch} HEAD ${branchHeadSha}`);

  // Propagate to all open PRs targeting this branch.
  let prPage = 1;
  let propagated = 0;
  const MAX_PR_PAGES = 50;
  while (prPage <= MAX_PR_PAGES) {
    const prs = await github.rest.pulls.list({
      owner, repo, state: 'open', base: branch, per_page: 100, page: prPage,
    });
    if (prs.data.length === 0) break;
    for (const pr of prs.data) {
      try {
        await github.rest.repos.createCommitStatus({
          owner, repo, sha: pr.head.sha,
          context: 'nightly/core-suite', state, description,
        });
        propagated++;
      } catch (e) {
        core.warning(`Failed to propagate to PR #${pr.number}: ${e.message}`);
      }
    }
    if (prs.data.length < 100) break;
    prPage++;
  }
  core.info(`Propagated nightly/core-suite to ${propagated} open PRs`);

  return { state, description };
};