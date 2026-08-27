$dir1 = "D:\vllm-ascend-rfc-vllm_cann"
$dir2 = "C:\Users\qigs\Downloads\vllm-ascend-rfc-vllm_cann (3)\vllm-ascend-rfc-vllm_cann"

Write-Host "Scanning D drive..."
$files1 = Get-ChildItem -Path $dir1 -Recurse -File | Where-Object { $_.FullName -notmatch "\\\.git\\" }

Write-Host "Scanning Downloads drive..."
$files2 = Get-ChildItem -Path $dir2 -Recurse -File | Where-Object { $_.FullName -notmatch "\\\.git\\" }

Write-Host "Hashing D drive..."
$hash1 = @{}
foreach ($f in $files1) {
    $rel = $f.FullName.Substring($dir1.Length)
    $hash1[$rel] = (Get-FileHash -Path $f.FullName -Algorithm MD5).Hash
}

Write-Host "Hashing Downloads drive..."
$hash2 = @{}
foreach ($f in $files2) {
    $rel = $f.FullName.Substring($dir2.Length)
    $hash2[$rel] = (Get-FileHash -Path $f.FullName -Algorithm MD5).Hash
}

Write-Host ""
Write-Host "=== Modified or New files in D:\vllm-ascend-rfc-vllm_cann ==="
foreach ($key in $hash1.Keys) {
    if (-not $hash2.ContainsKey($key)) {
        Write-Host "[NEW] $key"
    } elseif ($hash1[$key] -ne $hash2[$key]) {
        Write-Host "[MOD] $key"
    }
}

</contents>