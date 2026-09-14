param(
    [switch]$Discover,
    [string]$EventId
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project virtual environment is missing: $python"
}

if ($EventId) {
    if ($EventId -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Invalid event ID' }
    $logDir = Join-Path $projectRoot 'reports\phase46\scheduler_logs'
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    Push-Location $projectRoot
    try {
        & $python -m nfl_td_model.phase46_scheduler --event-id $EventId *>&1 |
            Out-File -LiteralPath (Join-Path $logDir "$EventId.log") -Append
        if ($LASTEXITCODE -ne 0) { throw "T-60 capture failed for $EventId" }
    } finally {
        Pop-Location
    }
    exit 0
}

if ($Discover) {
    Push-Location $projectRoot
    try {
        $eventJson = & $python -m nfl_td_model.phase46_scheduler --list-events
        if ($LASTEXITCODE -ne 0) { throw 'Event discovery failed' }
    } finally {
        Pop-Location
    }
    $events = @($eventJson | ConvertFrom-Json)
    foreach ($event in $events) {
        if ($event.event_id -notmatch '^[a-zA-Z0-9_-]+$') { continue }
        # ConvertFrom-Json has already parsed the ISO string to a UTC DateTime.
        # Parsing that DateTime again as text would silently treat UTC as local.
        $cutoff = ([DateTimeOffset]$event.kickoff).AddMinutes(-60).LocalDateTime
        $expectedUtc = ([DateTimeOffset]$event.kickoff).AddMinutes(-60).UtcDateTime
        if ($cutoff.ToUniversalTime() -ne $expectedUtc) { throw "T-60 timezone conversion failed for $($event.event_id)" }
        if ($cutoff -le (Get-Date).AddSeconds(15)) { continue }
        $name = "NFLTD-Phase46-$($event.event_id)"
        $argument = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -EventId $($event.event_id)"
        $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $argument -WorkingDirectory $projectRoot
        $trigger = New-ScheduledTaskTrigger -Once -At $cutoff
        # New-ScheduledTaskTrigger serializes local input as Z; Task Scheduler
        # then treats its clock time as local and runs hours late. Store the
        # local wall-clock time without a zone suffix for this Windows task.
        $trigger.StartBoundary = $cutoff.ToString('yyyy-MM-ddTHH:mm:ss')
        $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 3) `
            -MultipleInstances IgnoreNew -WakeToRun `
            -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
        Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
        # Register-ScheduledTask reserializes a newly constructed trigger.
        # Read it back and set the exact local boundary as a second operation.
        $registered = Get-ScheduledTask -TaskName $name
        $registered.Triggers[0].StartBoundary = $cutoff.ToString('yyyy-MM-ddTHH:mm:ss')
        Set-ScheduledTask -TaskName $name -Trigger $registered.Triggers | Out-Null
        $nextRun = (Get-ScheduledTaskInfo -TaskName $name).NextRunTime
        if ([math]::Abs(($nextRun - $cutoff).TotalSeconds) -gt 1) {
            throw "Registered T-60 task has incorrect next-run time for $($event.event_id)"
        }
    }
    exit 0
}

$argument = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Discover"
$dailyAction = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $argument -WorkingDirectory $projectRoot
$dailyTrigger = New-ScheduledTaskTrigger -Daily -At '09:00'
$dailySettings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 5) `
    -MultipleInstances IgnoreNew -WakeToRun `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName 'NFLTD-Phase46-Discover' -Action $dailyAction -Trigger $dailyTrigger -Settings $dailySettings -Force | Out-Null
& $PSCommandPath -Discover
