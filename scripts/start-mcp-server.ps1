<#
.SYNOPSIS
    Start a Local Context Engine MCP server for a repository.

.DESCRIPTION
    Launches the MCP server in HTTP mode so it stays warm between AI coding
    sessions. If the port is already in use the script exits cleanly.

.PARAMETER RepoPath
    Path to the indexed repository (must contain a .context/ folder).

.PARAMETER Port
    HTTP port to listen on (default: 8765).

.EXAMPLE
    .\start-mcp-server.ps1 D:\MIS\mis-admin
    .\start-mcp-server.ps1 D:\MIS\mis-backend -Port 8766
#>

param(
    [Parameter(Mandatory, Position = 0)]
    [string]$RepoPath,

    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"

# -- Resolve paths -----------------------------------------------
$RepoPath = (Resolve-Path $RepoPath).Path

if (-not (Test-Path "$RepoPath\.context")) {
    Write-Host "[ERROR] No .context/ folder in $RepoPath - run 'context index' first." -ForegroundColor Red
    exit 1
}

# -- Locate context executable ------------------------------------
$contextExe = Get-Command "context" -ErrorAction SilentlyContinue |
              Select-Object -ExpandProperty Source

if (-not $contextExe) {
    $contextExe = "$env:APPDATA\Python\Python313\Scripts\context.exe"
}

if (-not (Test-Path $contextExe)) {
    Write-Host "[ERROR] context executable not found. Install with: pip install local-context-engine" -ForegroundColor Red
    exit 1
}

# -- Check if port is already taken -------------------------------
$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    Write-Host "[OK] Port $Port is already in use - server may already be running." -ForegroundColor Yellow
    exit 0
}

# -- Environment --------------------------------------------------
$env:PYTHONUTF8 = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

# -- Launch -------------------------------------------------------
$repoName = Split-Path $RepoPath -Leaf

Write-Host ""
Write-Host "  Local Context Engine - MCP Server" -ForegroundColor Cyan
Write-Host "  Repo : $RepoPath"
Write-Host "  Port : $Port"
Write-Host "  URL  : http://127.0.0.1:$Port/mcp"
Write-Host ""
Write-Host "  Press Ctrl+C to stop."
Write-Host ""

& $contextExe mcp $RepoPath --transport streamable-http
