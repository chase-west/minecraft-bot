# Supervisor loop for the bot. Each iteration spawns a fresh node process so
# in-process leaks (bedrock-protocol internals on reconnect) cannot accumulate.
# Bot exit code 10 = death/disconnect, restart. Exit code 0 = clean shutdown.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

while ($true) {
    Write-Host "[$(Get-Date -Format HH:mm:ss)] starting bot session" -ForegroundColor Cyan
    npm run dev
    $code = $LASTEXITCODE
    Write-Host "[$(Get-Date -Format HH:mm:ss)] bot exited with code $code" -ForegroundColor Yellow
    if ($code -eq 0) {
        Write-Host "clean shutdown - supervisor done"
        break
    }
    Start-Sleep -Seconds 3
}
