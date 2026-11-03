const assert = require('node:assert/strict');
const test = require('node:test');
const updatePreviewComment = require('./recipes-preview-comment.cjs');

const PR_NUMBER = 20096;
const COMMENT_ID = 400;
const MARKER = '<!-- recipes-preview:vllm-ascend -->';
const PREVIEW_URL = `https://vllm-recipes-git-preview-vllm-ascend-pr-${PR_NUMBER}-inferact-inc.vercel.app`;
const LIVE_SHA = 'a'.repeat(40);

function ownedComment(body, id = COMMENT_ID) {
  return { id, body, user: { login: 'github-actions[bot]', type: 'Bot' } };
}

function fixture({ pr = {}, files, comments = [] } = {}) {
  const state = {
    files: files ?? [{ filename: 'recipes/manifest.yaml' }],
    comments: structuredClone(comments),
  };
  state.pr = {
    number: PR_NUMBER,
    state: 'open',
    merged: false,
    labels: [{ name: 'recipe' }],
    base: { ref: 'main', repo: { full_name: 'vllm-project/vllm-ascend' } },
    head: { sha: LIVE_SHA, repo: { full_name: 'contributor/fork' } },
    changed_files: state.files.length,
    ...pr,
  };
  const calls = [];
  const context = {
    repo: { owner: 'vllm-project', repo: 'vllm-ascend' },
    eventName: 'pull_request_target',
    payload: {
      action: 'opened',
      pull_request: { number: PR_NUMBER, labels: [], head: { sha: 'stale-event-sha' } },
    },
  };
  const record = (method, args) => {
    assert.equal(args.owner, context.repo.owner);
    assert.equal(args.repo, context.repo.repo);
    calls.push({ method, ...structuredClone(args) });
  };
  const github = {
    rest: {
      pulls: {
        get: async args => {
          record('get', args);
          assert.equal(args.pull_number, PR_NUMBER);
          return { data: structuredClone(state.pr) };
        },
        listFiles: Symbol('files'),
      },
      issues: {
        listComments: Symbol('comments'),
        createComment: async args => {
          record('create', args);
          assert.equal(args.issue_number, PR_NUMBER);
          const comment = ownedComment(args.body);
          state.comments.push(comment);
          return { data: structuredClone(comment) };
        },
        updateComment: async args => {
          record('update', args);
          const comment = state.comments.find(item => item.id === args.comment_id);
          assert.ok(comment, 'Only an existing comment may be updated');
          comment.body = args.body;
          return { data: structuredClone(comment) };
        },
      },
    },
    paginate: async (method, args) => {
      assert.equal(args.per_page, 100);
      if (method === github.rest.issues.listComments) {
        record('comments', args);
        assert.equal(args.issue_number, PR_NUMBER);
        return structuredClone(state.comments);
      }
      assert.equal(method, github.rest.pulls.listFiles);
      record('files', args);
      assert.equal(args.pull_number, PR_NUMBER);
      return structuredClone(state.files);
    },
  };
  return {
    state, calls, github, context,
    run: (pr = PR_NUMBER) => updatePreviewComment({ github, context, pr }),
    writes: () => calls.filter(call => call.method === 'create' || call.method === 'update'),
  };
}

test('labeler completion creates a preview despite labels missing from the opened event', async () => {
  const f = fixture();
  await f.run();
  const [comment] = f.state.comments;
  assert.equal(f.writes().length, 1);
  assert.ok(comment.body.startsWith(`${MARKER}\n`));
  assert.ok(comment.body.includes(PREVIEW_URL));
  assert.ok(comment.body.includes(LIVE_SHA));
  assert.ok(!comment.body.includes('stale-event-sha'));
  assert.match(comment.body, /does not report deployment success/);
});

test('the serialized workflow_run fallback does not duplicate the immediate comment', async () => {
  const f = fixture();
  await f.run(PR_NUMBER);
  f.context.eventName = 'workflow_run';
  f.context.payload = { workflow_run: { pull_requests: [] } };
  await f.run(String(PR_NUMBER));
  assert.equal(f.state.comments.length, 1);
  assert.deepEqual(f.writes().map(call => call.method), ['create']);
});

test('a delayed event updates the owned comment to the current live SHA', async () => {
  const f = fixture();
  await f.run();
  f.state.pr.head.sha = 'b'.repeat(40);
  await f.run();
  assert.equal(f.state.comments.length, 1);
  assert.equal(f.writes()[1].comment_id, COMMENT_ID);
  assert.ok(f.state.comments[0].body.includes('b'.repeat(40)));
  assert.ok(!f.state.comments[0].body.includes(LIVE_SHA));
});

const CLEANUP_CASES = [
  ['unlabeled', f => { f.state.pr.labels = []; }, /no longer has the `recipe` label/],
  ['closed', f => { f.state.pr.state = 'closed'; }, /has been closed/],
  ['merged', f => { f.state.pr.state = 'closed'; f.state.pr.merged = true; }, /production publication are handled separately/],
  ['retargeted', f => { f.state.pr.base.ref = 'release'; }, /no longer targets `main`/],
  ['without YAML changes', f => { f.state.files = [{ filename: 'recipes/README.md' }]; }, /no longer changes recipe files eligible for a preview/],
];

for (const [name, change, reason] of CLEANUP_CASES) {
  test(`live ${name} state deactivates the existing link after the initial writer`, async () => {
    const f = fixture();
    await f.run();
    change(f);
    await f.run();
    assert.equal(f.state.comments.length, 1);
    assert.equal(f.writes()[1].comment_id, COMMENT_ID);
    assert.match(f.state.comments[0].body, reason);
    assert.ok(!f.state.comments[0].body.includes(PREVIEW_URL));
  });

  test(`live ${name} state without an owned comment creates nothing`, async () => {
    const f = fixture();
    change(f);
    await f.run();
    assert.deepEqual(f.writes(), []);
  });
}

test('reopening restores the link in the same owned comment', async () => {
  const f = fixture();
  await f.run();
  f.state.pr.state = 'closed';
  await f.run();
  f.state.pr.state = 'open';
  await f.run();
  assert.equal(f.state.comments.length, 1);
  assert.ok(f.state.comments[0].body.includes(PREVIEW_URL));
  assert.deepEqual(f.writes().map(call => call.method), ['create', 'update', 'update']);
});

for (const filename of [
  'recipes/README.md',
  'recipes/overlays/README.md',
  'recipes/overlays/model.yml',
  'docs/recipes/model.yaml',
]) {
  test(`${filename} alone does not create a preview comment`, async () => {
    const f = fixture({ files: [{ filename }] });
    await f.run();
    assert.deepEqual(f.writes(), []);
  });
}

for (const files of [
  [{ filename: 'recipes/README.md' }, { filename: 'recipes/manifest.yaml' }],
  [{ filename: 'recipes/overlays/model/model.yaml' }],
  [{ filename: 'docs/old.yaml', previous_filename: 'recipes/overlays/model.yaml' }],
  [{ filename: 'recipes/overlays/model.yaml', previous_filename: 'docs/old.yaml' }],
  [{ filename: 'recipes/manifest.yaml', status: 'removed' }],
]) {
  test(`recipe YAML changes remain eligible: ${JSON.stringify(files)}`, async () => {
    const f = fixture({ files });
    await f.run();
    assert.equal(f.writes().length, 1);
    assert.ok(f.state.comments[0].body.includes(PREVIEW_URL));
  });
}

test('missing or non-string rename metadata does not make non-YAML files eligible', async () => {
  const f = fixture({ files: [
    { filename: 'recipes/README.md' },
    { filename: 'recipes/notes.txt', previous_filename: null },
    { filename: 'recipes/example.yml', previous_filename: 123 },
    { filename: undefined, previous_filename: {} },
  ] });
  await f.run();
  assert.deepEqual(f.writes(), []);
});

test('a truncated file list preserves an existing comment instead of deactivating it', async () => {
  const f = fixture();
  await f.run();
  const before = f.state.comments[0].body;
  f.state.files = [{ filename: 'recipes/README.md' }];
  f.state.pr.changed_files = 2;
  await assert.rejects(f.run, /Incomplete PR file list/);
  assert.equal(f.state.comments[0].body, before);
  assert.equal(f.writes().length, 1);
});

test('a truncated file list without known YAML cannot create a new comment', async () => {
  const f = fixture({ pr: { changed_files: 2 }, files: [{ filename: 'README.md' }] });
  await assert.rejects(f.run, /Incomplete PR file list/);
  assert.deepEqual(f.writes(), []);
});

test('known YAML eligibility does not depend on unseen files in a truncated list', async () => {
  const f = fixture({ pr: { changed_files: 2 } });
  await f.run();
  assert.equal(f.writes().length, 1);
});

test('only the correctly marked GitHub Actions bot comment can be updated', async () => {
  const unrelated = [
    { id: 1, body: `${MARKER}\nhuman`, user: { login: 'contributor', type: 'User' } },
    { id: 2, body: `${MARKER}\nother bot`, user: { login: 'other[bot]', type: 'Bot' } },
    { id: 3, body: `${MARKER}\nwrong type`, user: { login: 'github-actions[bot]', type: 'User' } },
    ownedComment('Welcome', 4),
    ownedComment(`Quoted\n${MARKER}\n`, 5),
    ownedComment(`${MARKER}without newline`, 6),
  ];
  const f = fixture({ comments: [...unrelated, ownedComment(`${MARKER}\nold`)] });
  await f.run();
  assert.deepEqual(f.state.comments.slice(0, unrelated.length), unrelated);
  assert.equal(f.writes().length, 1);
  assert.equal(f.writes()[0].method, 'update');
  assert.equal(f.writes()[0].comment_id, COMMENT_ID);
});

test('an unowned marker comment does not suppress creating the bot comment', async () => {
  const human = { id: 1, body: `${MARKER}\nspoof`, user: { login: 'contributor', type: 'User' } };
  const f = fixture({ comments: [human] });
  await f.run();
  assert.deepEqual(f.state.comments[0], human);
  assert.equal(f.writes()[0].method, 'create');
  assert.equal(f.state.comments.length, 2);
});

test('an unexpected base repository fails before comment discovery or mutation', async () => {
  const f = fixture({ pr: { base: { ref: 'main', repo: { full_name: 'contributor/fork' } } } });
  await assert.rejects(f.run, /Unexpected PR base repository/);
  assert.deepEqual(f.calls.map(call => call.method), ['get']);
});

for (const pr of [undefined, null, '', '01', 'abc', '1;echo unsafe', 0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1]) {
  test(`invalid PR number ${String(pr)} fails before any API call`, async () => {
    const f = fixture();
    await assert.rejects(
      () => updatePreviewComment({ github: f.github, context: f.context, pr }),
      /Expected a positive integer PR number/,
    );
    assert.deepEqual(f.calls, []);
  });
}
