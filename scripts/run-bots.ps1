# Parallel multi-bot supervisor for Bedrock AI training.
#
# Spawns N independent bot processes, each with:
#   - unique username (AIBot1, AIBot2, ...)
#   - distinct ONLINE_EPSILON for behavioral diversity in the replay buffer
#   - independent restart loop (one bot dying does not affect the others)
#   - dedicated log file at logs/bot<N>.log (stdout) + logs/bot<N>.err (stderr)
#
# Each child is the raw `node ... src/index.ts` process (NOT `npm run dev`)
# so we get a real PID we can kill cleanly on Windows. .env is loaded by
# node's --env-file flag, matching package.json's "dev" script.
#
# Exit codes from the bot:
#   10 = disconnect / death  -> restart
#    0 = clean shutdown      -> do not restart (treat as user intent)
#  other = crash             -> restart with crash-loop tracking
#
# Crash-loop guard: if a single bot exits non-zero more than 5 times within
# 60 seconds, back off for 30s before its next restart. Other bots are
# unaffected.
#
# Graceful shutdown: Ctrl+C (or any terminating exception) triggers the
# finally block which kills every child process tree we spawned. No orphans.
#
# Usage:
#   .\scripts\run-bots.ps1                  # 4 bots, default epsilons
#   .\scripts\run-bots.ps1 -Count 8         # 8 bots
#   .\scripts\run-bots.ps1 -Count 2 -Epsilons @(0.1, 0.3)

[CmdletBinding()]
param(
    [int]$Count = 4,
    [double[]]$Epsilons = $null,
    [int]$HealthIntervalSeconds = 30,
    [int]$CrashWindowSeconds = 60,
    [int]$CrashThreshold = 5,
    [int]$BackoffSeconds = 30,
    [int]$RestartDelaySeconds = 3
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

$logDir = Join-Path $root "logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

$dataDir = Join-Path $root "data\online"
$modelPath = Join-Path $root "models\policy.onnx"

# Default epsilon distribution: one exploit-heavy, mostly balanced, one
# explore-heavy. For Count != 4 we tile/extend the pattern so every bot
# gets a value, and you still get a mix.
function Get-DefaultEpsilons {
    param([int]$n)
    $base = @(0.05, 0.15, 0.15, 0.40)
    $out = @()
    for ($i = 0; $i -lt $n; $i++) {
        $out += $base[$i % $base.Length]
    }
    return ,$out
}

if ($null -eq $Epsilons -or $Epsilons.Count -eq 0) {
    $Epsilons = Get-DefaultEpsilons -n $Count
} elseif ($Epsilons.Count -lt $Count) {
    # Pad by recycling the supplied values.
    $padded = @()
    for ($i = 0; $i -lt $Count; $i++) {
        $padded += $Epsilons[$i % $Epsilons.Count]
    }
    $Epsilons = $padded
}

# Locate node.exe. Fall back to PATH lookup.
$nodeExe = (Get-Command node -ErrorAction SilentlyContinue).Source
if (-not $nodeExe) {
    Write-Host "ERROR: node.exe not found on PATH" -ForegroundColor Red
    exit 1
}

# Arguments to spawn one bot. Mirrors `npm run dev` in package.json:
#   node --env-file=.env --max-old-space-size=8192 --import tsx src/index.ts
$nodeArgs = @(
    "--env-file=.env",
    "--max-old-space-size=8192",
    "--import", "tsx",
    "src/index.ts"
)

# ---------------------------------------------------------------------------
# Per-bot state
# ---------------------------------------------------------------------------

# Each entry: @{
#   Index, Username, Epsilon, Process, LogFile, ErrFile,
#   Restarts, CrashTimes (queue of [DateTime]), BackoffUntil ([DateTime] or $null)
# }
$bots = @()
for ($i = 1; $i -le $Count; $i++) {
    $bots += [pscustomobject]@{
        Index        = $i
        Username     = "AIBot$i"
        Epsilon      = [double]$Epsilons[$i - 1]
        Process      = $null
        LogFile      = Join-Path $logDir "bot$i.log"
        ErrFile      = Join-Path $logDir "bot$i.err"
        Restarts     = 0
        CrashTimes   = New-Object System.Collections.Generic.Queue[datetime]
        BackoffUntil = $null
    }
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-Stamp {
    param([string]$msg, [ConsoleColor]$Color = [ConsoleColor]::Gray)
    Write-Host "[$(Get-Date -Format HH:mm:ss)] $msg" -ForegroundColor $Color
}

function Start-Bot {
    param($bot)

    # Persist env vars via process-scoped variables. Start-Process inherits
    # the current PowerShell process environment, so set then spawn then unset.
    $prevUser = $env:BEDROCK_USERNAME
    $prevMode = $env:POLICY_MODE
    $prevEps  = $env:ONLINE_EPSILON
    try {
        $env:BEDROCK_USERNAME = $bot.Username
        $env:POLICY_MODE      = "online"
        $env:ONLINE_EPSILON   = ([string]$bot.Epsilon)

        $proc = Start-Process -FilePath $nodeExe `
            -ArgumentList $nodeArgs `
            -WorkingDirectory $root `
            -RedirectStandardOutput $bot.LogFile `
            -RedirectStandardError  $bot.ErrFile `
            -WindowStyle Hidden `
            -PassThru
        $bot.Process = $proc
        Write-Stamp ("spawn bot{0} '{1}' eps={2:N2} pid={3}" -f $bot.Index, $bot.Username, $bot.Epsilon, $proc.Id) Cyan
    } finally {
        $env:BEDROCK_USERNAME = $prevUser
        $env:POLICY_MODE      = $prevMode
        $env:ONLINE_EPSILON   = $prevEps
    }
}

function Stop-BotProcess {
    param($proc)
    if ($null -eq $proc) { return }
    try {
        if (-not $proc.HasExited) {
            # Kill the whole tree (node may have spawned children).
            try {
                taskkill.exe /PID $proc.Id /T /F 2>$null | Out-Null
            } catch {}
            try { $proc.WaitForExit(5000) | Out-Null } catch {}
        }
    } catch {}
}

function Get-SampleCount {
    if (-not (Test-Path $dataDir)) { return 0 }
    $total = 0
    Get-ChildItem -Path $dataDir -Filter "*.jsonl" -ErrorAction SilentlyContinue | ForEach-Object {
        try {
            $m = Get-Content -LiteralPath $_.FullName -ReadCount 0 -ErrorAction Stop |
                 Measure-Object -Line
            $total += $m.Lines
        } catch {
            # File may be locked mid-write; skip this tick.
        }
    }
    return $total
}

function Get-ModelAgeText {
    if (-not (Test-Path $modelPath)) { return "n/a" }
    try {
        $mt = (Get-Item -LiteralPath $modelPath).LastWriteTime
        $age = [int]((Get-Date) - $mt).TotalSeconds
        if ($age -lt 60) { return "${age}s" }
        if ($age -lt 3600) { return ("{0}m" -f [int]($age / 60)) }
        return ("{0}h" -f [int]($age / 3600))
    } catch { return "n/a" }
}

function Should-Backoff {
    param($bot)

    # Drop crash timestamps older than the window.
    $cutoff = (Get-Date).AddSeconds(-$CrashWindowSeconds)
    while ($bot.CrashTimes.Count -gt 0 -and $bot.CrashTimes.Peek() -lt $cutoff) {
        [void]$bot.CrashTimes.Dequeue()
    }
    return ($bot.CrashTimes.Count -ge $CrashThreshold)
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

Write-Stamp ("supervisor starting: count={0} epsilons=[{1}]" -f $Count, ($Epsilons -join ", ")) Green
Write-Stamp ("logs at {0}\bot<N>.log  |  data at {1}" -f $logDir, $dataDir) Green
Write-Stamp "press Ctrl+C to shut down all bots cleanly" Green

# Initial spawn.
foreach ($b in $bots) { Start-Bot -bot $b }

$lastHealth = Get-Date
$shuttingDown = $false

try {
    while (-not $shuttingDown) {
        Start-Sleep -Seconds 1
        $now = Get-Date

        foreach ($b in $bots) {
            # Honor backoff window.
            if ($null -ne $b.BackoffUntil) {
                if ($now -lt $b.BackoffUntil) { continue }
                $b.BackoffUntil = $null
                Write-Stamp ("bot{0} backoff complete, resuming" -f $b.Index) Yellow
            }

            $proc = $b.Process
            if ($null -eq $proc) {
                Start-Bot -bot $b
                continue
            }
            if (-not $proc.HasExited) { continue }

            $code = $proc.ExitCode
            $b.Process = $null

            if ($code -eq 0) {
                Write-Stamp ("bot{0} clean shutdown (exit 0), not restarting" -f $b.Index) Magenta
                continue
            }

            $label = "disconnect"
            if ($code -ne 10) { $label = "crash" }
            Write-Stamp ("bot{0} exited code={1} ({2})" -f $b.Index, $code, $label) Yellow
            $b.Restarts++
            $b.CrashTimes.Enqueue($now)

            if (Should-Backoff -bot $b) {
                $b.BackoffUntil = $now.AddSeconds($BackoffSeconds)
                Write-Stamp ("bot{0} crash-loop detected ({1} crashes in {2}s), backing off until {3}" -f `
                    $b.Index, $b.CrashTimes.Count, $CrashWindowSeconds, $b.BackoffUntil.ToString("HH:mm:ss")) Red
                continue
            }

            Start-Sleep -Seconds $RestartDelaySeconds
            Start-Bot -bot $b
        }

        # Periodic health line.
        if (($now - $lastHealth).TotalSeconds -ge $HealthIntervalSeconds) {
            $lastHealth = $now
            $alive = 0
            foreach ($b in $bots) {
                if ($null -ne $b.Process -and -not $b.Process.HasExited) { $alive++ }
            }
            $samples = Get-SampleCount
            $modelAge = Get-ModelAgeText
            $restartSummary = ($bots | ForEach-Object { "b$($_.Index)=$($_.Restarts)" }) -join " "
            Write-Stamp ("alive: {0}/{1} bots | samples: {2} | model age: {3} | restarts: {4}" -f `
                $alive, $Count, $samples, $modelAge, $restartSummary) Green
        }

        # If every bot has cleanly shut down (no process and no backoff), exit.
        $anyActive = $false
        foreach ($b in $bots) {
            if ($null -ne $b.Process) { $anyActive = $true; break }
            if ($null -ne $b.BackoffUntil) { $anyActive = $true; break }
        }
        if (-not $anyActive) {
            Write-Stamp "all bots exited cleanly, supervisor done" Green
            $shuttingDown = $true
        }
    }
} finally {
    Write-Stamp "shutting down: terminating all child bots..." Yellow
    foreach ($b in $bots) {
        Stop-BotProcess -proc $b.Process
        $b.Process = $null
    }
    Write-Stamp "supervisor exit" Yellow
}
