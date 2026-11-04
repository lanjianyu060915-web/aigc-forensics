[CmdletBinding()]
param(
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$ProductRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ConfigPath = Join-Path $ProductRoot "config.json"
$FrontendPath = Join-Path $ProductRoot "frontend\app.py"

function Test-FrontendPython {
    param([Parameter(Mandatory = $true)][string]$Executable)

    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
        return $false
    }
    & $Executable -B -c "import streamlit" *> $null
    return $LASTEXITCODE -eq 0
}

function Resolve-FrontendPython {
    $candidates = @(
        (Join-Path $ProductRoot "python_runtime\python.exe"),
        (Join-Path $ProductRoot "frontend_runtime\python.exe"),
        (Join-Path $ProductRoot "packaging\.venv_frontend\Scripts\python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-FrontendPython -Executable $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $pythonCommand -and (Test-FrontendPython -Executable $pythonCommand.Source)) {
        return $pythonCommand.Source
    }
    throw "未找到可导入 Streamlit 的前端 Python。请确认 python_runtime 目录完整。"
}

function Test-PortFree {
    param([Parameter(Mandatory = $true)][int]$Port)

    $listener = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, $Port)
    try {
        $listener.ExclusiveAddressUse = $true
        $listener.Start()
        return $true
    } catch {
        return $false
    } finally {
        try { $listener.Stop() } catch { }
    }
}

function Select-FreePort {
    param([Parameter(Mandatory = $true)][int]$Preferred)

    foreach ($candidate in $Preferred..($Preferred + 40)) {
        if (Test-PortFree -Port $candidate) {
            return $candidate
        }
    }
    throw "端口 $Preferred 起的 40 个端口都被占用，无法启动。"
}

function Resolve-DataRoot {
    if ($env:FSTS_DATA_ROOT) {
        return $env:FSTS_DATA_ROOT
    }
    $portable = Join-Path (Split-Path -Parent $ProductRoot) "用户数据"
    try {
        New-Item -ItemType Directory -Force -Path $portable | Out-Null
        $probe = Join-Path $portable ".write_probe"
        Set-Content -LiteralPath $probe -Value "ok" -Encoding ascii
        Remove-Item -LiteralPath $probe -Force
        return $portable
    } catch {
        # 目录不可写（例如放在只读介质上）时回到系统默认的用户数据目录。
        return (Join-Path $env:LOCALAPPDATA "FaceSourceTrace")
    }
}

if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
    throw "配置文件不存在：$ConfigPath"
}
if (-not (Test-Path -LiteralPath $FrontendPath -PathType Leaf)) {
    throw "前端入口不存在：$FrontendPath"
}

# 不让本机其他 Python 环境影响随包运行环境。
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
$env:PYTHONNOUSERSITE = "1"
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"

$InstallRoot = Split-Path -Parent $ProductRoot
if ($InstallRoot.Length -gt 60) {
    # Windows 默认路径上限 260 字符，目录太深时部分证据图会写不出来。
    Write-Warning "所在目录路径 $($InstallRoot.Length) 字符，超过 60 字符，部分证据图可能写不出来。建议把整个目录移到短路径，例如 D:\换脸寻源。"
}

$FrontendPython = Resolve-FrontendPython
$env:FSTS_CONFIG_PATH = $ConfigPath
$env:FSTS_DATA_ROOT = Resolve-DataRoot

Write-Host "[1/4] 运行环境检查" -ForegroundColor Cyan
$EnvironmentChecker = Join-Path $ProductRoot "check_environment.py"
if (Test-Path -LiteralPath $EnvironmentChecker -PathType Leaf) {
    & $FrontendPython -B $EnvironmentChecker --config $ConfigPath
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "部分环境检查未通过。界面仍会启动，并禁用不可用模块。"
    }
}

Write-Host "[2/4] 检查模型与数据库" -ForegroundColor Cyan
$StatusChecker = Join-Path $ProductRoot "system_status.py"
if (Test-Path -LiteralPath $StatusChecker -PathType Leaf) {
    & $FrontendPython -B $StatusChecker --config $ConfigPath *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "存在不可用模块；请在界面的系统状态页查看原因。"
    }
} else {
    Write-Warning "system_status.py 尚不存在。"
}

$Config = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
$Address = if ($Config.frontend.address) { [string]$Config.frontend.address } else { "127.0.0.1" }
$PreferredPort = if ($Config.frontend.port) { [int]$Config.frontend.port } else { 8767 }
$Port = Select-FreePort -Preferred $PreferredPort
if ($Port -ne $PreferredPort) {
    Write-Host "端口 $PreferredPort 已被占用，改用 $Port。" -ForegroundColor Yellow
}

Write-Host "[3/4] 启动本地系统：http://$Address`:$Port" -ForegroundColor Cyan
Write-Host "      用户数据目录：$($env:FSTS_DATA_ROOT)" -ForegroundColor DarkGray
Write-Host "[4/4] 服务就绪后会自动打开浏览器；关闭这个窗口或在界面点“关闭系统”即可退出。" -ForegroundColor Cyan

if (-not $NoBrowser -and -not $env:FSTS_NO_BROWSER) {
    # 模型服务初始化期间页面也能打开；这里只等 Streamlit 自身的健康检查通过。
    $opener = @"
`$deadline = (Get-Date).AddSeconds(120)
while ((Get-Date) -lt `$deadline) {
    try {
        `$response = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri 'http://$Address`:$Port/_stcore/health'
        if (`$response.StatusCode -eq 200) { Start-Process 'http://$Address`:$Port'; break }
    } catch { }
    Start-Sleep -Milliseconds 500
}
"@
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($opener))
    Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @("-NoLogo", "-NoProfile", "-EncodedCommand", $encoded) | Out-Null
}

Set-Location -LiteralPath $ProductRoot
& $FrontendPython -B -m streamlit run $FrontendPath `
    --server.address=$Address `
    --server.port=$Port `
    --server.headless=true `
    --server.fileWatcherType=none `
    --browser.gatherUsageStats=false `
    --global.developmentMode=false
$StreamlitExit = $LASTEXITCODE
# 界面上的“关闭系统”用 SIGTERM 结束进程，退出码 15 属于正常关闭。
if ($StreamlitExit -eq 15) {
    Write-Host "系统已关闭。" -ForegroundColor Cyan
    exit 0
}
exit $StreamlitExit
