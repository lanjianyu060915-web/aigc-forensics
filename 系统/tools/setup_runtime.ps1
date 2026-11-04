[CmdletBinding()]
param(
    [string]$Root,
    [ValidateSet('cu128', 'cpu')][string]$EngineBackend,
    [string[]]$Components = @('app', 'vit', 'engine', 'ffmpeg', 'frozen'),
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
# 用 -File 启动时，参数默认值里的 $PSScriptRoot 是空的，所以在这里取。
$ToolsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Root) { $Root = Split-Path -Parent $ToolsDir }
$exitCode = 1
$bootstrap = $null
$oldOutputEncoding = $OutputEncoding
$oldConsoleEncoding = [Console]::OutputEncoding
$OutputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $OutputEncoding

try {
    $Root = [System.IO.Path]::GetFullPath($Root)
    $selected = @()
    foreach ($entry in $Components) {
        foreach ($component in $entry.Split(',')) {
            $component = $component.Trim()
            if ($component -notin @('app', 'vit', 'engine', 'ffmpeg', 'frozen')) {
                throw "未知组件：$component；仅支持 app、vit、engine、ffmpeg、frozen。"
            }
            if ($selected -notcontains $component) { $selected += $component }
        }
    }
    if ($selected.Count -eq 0) { throw '至少需要指定一个组件。' }
    if (-not [Environment]::Is64BitOperatingSystem) { throw '仅支持 Windows x64。' }
    [System.IO.Directory]::CreateDirectory($Root) | Out-Null
    $python = $null
    $candidates = @()
    if ($env:AIGC_BOOTSTRAP_PYTHON) { $candidates += $env:AIGC_BOOTSTRAP_PYTHON }
    foreach ($name in @('python.exe', 'python3.exe')) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) { $candidates += $command.Source }
    }
    foreach ($relative in @('python_runtime\python.exe', 'engine_runtime\python.exe', 'packaging_v0.4\resources\portable_python_v2\python.exe')) {
        $candidates += Join-Path $Root $relative
    }
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
        if ($candidate -like '*\WindowsApps\*') { continue }
        try {
            & $candidate -I -B -c 'import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)' 2>$null
            if ($LASTEXITCODE -eq 0) { $python = $candidate; break }
        } catch { }
    }
    if (-not $python) {
        Write-Host '[引导] 未找到 Python 3.12+，使用 Windows HTTPS 引导固定 Python；正常下载由 Python 完成。'
        $bootstrap = Join-Path $Root ('.bootstrap-' + [Guid]::NewGuid().ToString('N').Substring(0,8))
        [System.IO.Directory]::CreateDirectory($bootstrap) | Out-Null
        $manifestPath = Join-Path (Split-Path -Parent $ToolsDir) 'requirements\runtime-sources.json'
        $source = (Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json).python
        $archive = Join-Path $bootstrap 'python.nupkg'
        $cacheDirectory = Join-Path $Root '.runtime-cache'
        if ($env:AIGC_RUNTIME_CACHE) { $cacheDirectory = $env:AIGC_RUNTIME_CACHE }
        $cachedArchive = Join-Path $cacheDirectory $source.filename
        if ((Test-Path -LiteralPath $cachedArchive -PathType Leaf) -and ((Get-FileHash -LiteralPath $cachedArchive -Algorithm SHA256).Hash.ToLowerInvariant() -eq $source.sha256)) {
            Write-Host '[引导] 使用已校验 SHA-256 的基础 Python 缓存。'
            Copy-Item -LiteralPath $cachedArchive -Destination $archive
        } else {
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            $client = New-Object Net.WebClient
            try {
                $proxyAddress = $env:HTTPS_PROXY
                if (-not $proxyAddress) { $proxyAddress = $env:HTTP_PROXY }
                if ($proxyAddress) { $client.Proxy = New-Object Net.WebProxy($proxyAddress) }
                $client.DownloadFile($source.url, $archive)
            } finally { $client.Dispose() }
        }
        if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $source.sha256) {
            throw '基础 Python 下载 SHA-256 不匹配。'
        }
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [System.IO.Compression.ZipFile]::ExtractToDirectory($archive, (Join-Path $bootstrap 'expanded'))
        $python = Join-Path $bootstrap 'expanded\tools\python.exe'
    }
    Write-Host "[引导] 下载与构建助手：$python（不会改动该环境）"
    $arguments = @('-I', '-B', (Join-Path $ToolsDir 'runtime_builder.py'), '--root', $Root, '--components', ($selected -join ','))
    if ($EngineBackend) { $arguments += @('--engine-backend', $EngineBackend) }
    if ($Force) { $arguments += '--force' }
    & $python @arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) { Write-Host "[失败] 运行环境安装未完成，退出码：$exitCode。请查看上方中文原因。" }
} catch {
    Write-Host "[失败] $($_.Exception.Message)"
    $exitCode = 1
} finally {
    if ($bootstrap -and (Test-Path -LiteralPath $bootstrap)) {
        $absolute = [System.IO.Path]::GetFullPath($bootstrap)
        $allowed = $Root.TrimEnd('\') + '\'
        if ($absolute.StartsWith($allowed, [StringComparison]::OrdinalIgnoreCase)) {
            Remove-Item -LiteralPath $absolute -Recurse -Force
        }
    }
    $OutputEncoding = $oldOutputEncoding
    [Console]::OutputEncoding = $oldConsoleEncoding
}
exit $exitCode
