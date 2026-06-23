param(
    [string]$RuntimeDir = $env:IMAGE_ALIGNER_RUNTIME_DIR,
    [int]$KeepLatest = 5,
    [switch]$ConfirmDelete
)

$ErrorActionPreference = "Stop"
if (-not $RuntimeDir) {
    $RuntimeDir = "D:\auto_clip_runtime\image_aligner\runs"
}

if (-not (Test-Path -LiteralPath $RuntimeDir -PathType Container)) {
    New-Item -ItemType Directory -Path $RuntimeDir | Out-Null
}

$RuntimePath = (Resolve-Path -LiteralPath $RuntimeDir).Path.TrimEnd('\')
$RuntimePrefix = $RuntimePath + "\"

function Get-ItemSizeBytes {
    param([System.IO.FileSystemInfo]$Item)

    if ($Item.PSIsContainer) {
        $sum = Get-ChildItem -LiteralPath $Item.FullName -Recurse -Force -File -ErrorAction SilentlyContinue |
            Measure-Object Length -Sum
        return [int64]($sum.Sum)
    }
    return [int64]$Item.Length
}

$allItems = Get-ChildItem -LiteralPath $RuntimePath -Force |
    Sort-Object LastWriteTime -Descending

$keepItems = $allItems | Select-Object -First $KeepLatest
$deleteItems = $allItems | Select-Object -Skip $KeepLatest

$deleteBytes = 0
$deleteRows = foreach ($item in $deleteItems) {
    $resolved = (Resolve-Path -LiteralPath $item.FullName).Path
    if ($resolved -ne $RuntimePath -and -not $resolved.StartsWith($RuntimePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to delete outside runtime: $resolved"
    }

    $bytes = Get-ItemSizeBytes -Item $item
    $deleteBytes += $bytes
    [PSCustomObject]@{
        LastWriteTime = $item.LastWriteTime
        MB            = [math]::Round($bytes / 1MB, 2)
        Name          = $item.Name
    }
}

Write-Host "Runtime: $RuntimePath"
Write-Host "Keep latest items: $KeepLatest"
Write-Host "Will delete: $($deleteItems.Count) item(s), $([math]::Round($deleteBytes / 1MB, 2)) MB"

if ($deleteRows) {
    $deleteRows | Sort-Object LastWriteTime | Format-Table -AutoSize
}

if (-not $ConfirmDelete) {
    Write-Host "Preview only. Re-run with -ConfirmDelete to delete."
    exit 0
}

foreach ($item in $deleteItems) {
    $resolved = (Resolve-Path -LiteralPath $item.FullName).Path
    if ($resolved -eq $RuntimePath -or -not $resolved.StartsWith($RuntimePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to delete unsafe path: $resolved"
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}

Write-Host "Cleanup done."
