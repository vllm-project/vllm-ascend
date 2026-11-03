// Maintain one preview-link comment using live PR metadata.
// Call only from trusted workflow code, under the shared per-PR concurrency group.
module.exports = async ({ github, context, pr: prInput }) => {
  const marker = '<!-- recipes-preview:vllm-ascend -->';
  const prNumber = String(prInput ?? '');
  if (!/^[1-9][0-9]*$/.test(prNumber) || !Number.isSafeInteger(Number(prNumber))) {
    throw new Error('Expected a positive integer PR number without leading zeros.');
  }
  const repo = context.repo;
  const issue_number = Number(prNumber);
  // Read current state even on reruns or delayed events.
  const { data: pr } = await github.rest.pulls.get({ ...repo, pull_number: issue_number });
  if (pr.base.repo.full_name !== 'vllm-project/vllm-ascend') {
    throw new Error('Unexpected PR base repository.');
  }
  const comments = await github.paginate(github.rest.issues.listComments, {
    ...repo, issue_number, per_page: 100,
  });
  const existing = comments.find(comment =>
    comment.user?.login === 'github-actions[bot]' &&
    comment.user?.type === 'Bot' &&
    comment.body?.startsWith(marker + '\n')
  );
  let status;
  let active = false;
  if (pr.state === 'closed') {
    if (!existing) return;
    status = pr.merged
      ? 'This PR has been merged. Preview cleanup and production publication are handled separately by the recipes repository.'
      : 'This PR has been closed. Preview cleanup is handled separately by the recipes repository.';
  } else if (pr.base.ref !== 'main') {
    if (!existing) return;
    status = 'This PR no longer targets `main`, so it is not eligible for a recipe preview.';
  } else if (!pr.labels.some(label => label.name === 'recipe')) {
    // Recheck the live label after selection or waiting for concurrency.
    if (!existing) return;
    status = 'This PR no longer has the `recipe` label, so it is not eligible for a recipe preview.';
  } else {
    const files = await github.paginate(github.rest.pulls.listFiles, {
      ...repo, pull_number: issue_number, per_page: 100,
    });
    const isRecipePath = name =>
      typeof name === 'string' && name.startsWith('recipes/') && name.endsWith('.yaml');
    const hasRecipes = files.some(file =>
      isRecipePath(file.filename) || isRecipePath(file.previous_filename)
    );
    if (!hasRecipes && pr.changed_files > files.length) {
      throw new Error('Incomplete PR file list; leaving the preview comment unchanged.');
    }
    if (!hasRecipes && !existing) return;
    active = hasRecipes;
    status = hasRecipes
      ? 'The preview builds asynchronously and may not be ready or may still show an earlier commit. Check the preview banner for the rendered revision; this comment does not report deployment success.'
      : 'This PR no longer changes recipe files eligible for a preview, so this preview link is no longer applicable.';
  }
  const url = `https://vllm-recipes-git-preview-vllm-ascend-pr-${issue_number}-inferact-inc.vercel.app`;
  const body = [
    marker,
    '### Recipe preview',
    '',
    ...(active ? [`[Open recipe preview](${url})`, ''] : []),
    `Current PR commit: \`${pr.head.sha}\` (not a deployment status).`,
    '',
    status,
    ...(active ? [
      '',
      'The recipes scheduled scanner requires the `recipe` label. Builds and deployment status are available in the [recipes preview workflow](https://github.com/vllm-project/recipes/actions/workflows/preview-prs.yml) and its preview branch checks.',
    ] : []),
  ].join('\n');
  if (existing) {
    if (existing.body === body) return;
    await github.rest.issues.updateComment({ ...repo, comment_id: existing.id, body });
  } else {
    await github.rest.issues.createComment({ ...repo, issue_number, body });
  }
};
