$ErrorActionPreference = 'Stop'
$originalRoot = 'D:\code\vllm-ascend'
$v30Root = Join-Path $originalRoot 'v30-workspace'
$v30Branch = 'integrate/dsnext-megamoe-v0.30.0'
$expectedV30Head = '59b8a8abde2ab4ad857ea0480e1fa6123f43ff3d'
$expectedReleaseHead = 'acd91d31e476f4b40666f5e2682d537a1319fb5e'
Set-Location -LiteralPath $originalRoot

function Invoke-CheckedGit {
    param([string[]]$GitArguments)
    $gitOutput = @(& git @GitArguments)
    if ($LASTEXITCODE -ne 0) {
        throw ('Git command failed: git ' + ($GitArguments -join ' '))
    }
    return $gitOutput
}

$dirtyTrackedFiles = @(Invoke-CheckedGit -GitArguments @('status', '--porcelain', '--untracked-files=no'))
if ($dirtyTrackedFiles.Count -gt 0) {
    throw 'Tracked files have uncommitted changes. Commit or stash them before running this script.'
}
$sourceStatus = @(Invoke-CheckedGit -GitArguments @('-C', $v30Root, 'status', '--porcelain'))
if ($sourceStatus.Count -gt 0) { throw 'The v30 workspace has uncommitted files.' }
$sourceHead = @(Invoke-CheckedGit -GitArguments @('-C', $v30Root, 'rev-parse', "refs/heads/$v30Branch"))[0]
if ($sourceHead -ne $expectedV30Head) { throw 'The v30 source branch has changed; no branches were deleted.' }
$baseHead = @(Invoke-CheckedGit -GitArguments @('-C', $v30Root, 'merge-base', $v30Branch, 'releases/v0.30.0'))[0]
if ($baseHead -ne $expectedReleaseHead) { throw 'Unexpected v30 release baseline.' }

$oldBranches = @(
    @{ Name = 'backup/dsnext-megamoe-squashed'; ExpectedPrefix = 'c0997b791' },
    @{ Name = 'integrate/dsnext-megamoe'; ExpectedPrefix = 'c0997b791' },
    @{ Name = 'integrate/dsnext-megamoe-history'; ExpectedPrefix = 'a40fda41b' },
    @{ Name = 'integrate/dsnext-megamoe-pr'; ExpectedPrefix = 'ef4530f34' },
    @{ Name = 'main'; ExpectedPrefix = 'd280bb693' }
)
foreach ($oldBranch in $oldBranches) {
    $tip = @(& git rev-parse --verify --quiet "refs/heads/$($oldBranch.Name)")
    if ($LASTEXITCODE -eq 0 -and -not $tip[0].StartsWith($oldBranch.ExpectedPrefix)) {
        throw ('Branch changed since inspection; refusing to delete: ' + $oldBranch.Name)
    }
}

Invoke-CheckedGit -GitArguments @('fetch', '--no-tags', $v30Root, "refs/heads/${v30Branch}:refs/heads/$v30Branch")
$importedHead = @(Invoke-CheckedGit -GitArguments @('rev-parse', "refs/heads/$v30Branch"))[0]
if ($importedHead -ne $expectedV30Head) { throw 'Imported v30 head does not match.' }
Invoke-CheckedGit -GitArguments @('switch', $v30Branch)
$commitCount = @(Invoke-CheckedGit -GitArguments @('rev-list', '--count', "$expectedReleaseHead..HEAD"))[0]
if ($commitCount -ne '4') { throw 'Expected exactly three source commits and one v30 adaptation commit.' }

foreach ($oldBranch in $oldBranches) {
    & git show-ref --verify --quiet "refs/heads/$($oldBranch.Name)"
    if ($LASTEXITCODE -eq 0) { Invoke-CheckedGit -GitArguments @('branch', '-D', $oldBranch.Name) }
}
& git show-ref --verify --quiet 'refs/remotes/zhangy/integrate/dsnext-megamoe-pr'
if ($LASTEXITCODE -eq 0) {
    Invoke-CheckedGit -GitArguments @('branch', '-dr', 'zhangy/integrate/dsnext-megamoe-pr')
}
Write-Output 'Done: the original repository is now on the v30 integration branch.'
Invoke-CheckedGit -GitArguments @('branch', '-vv')
