# Windows Task Scheduler Setup for VLA ArxivTracker
# Run this script as Administrator

$TaskName = "VLA-ArxivTracker"
$TaskDescription = "Daily automatic download of VLA/robot manipulation papers from arXiv"
$ScriptPath = "D:\code\vla-flow\scripts\run_tracker.bat"
$WorkDir = "D:\code\vla-flow"

# Check if task exists
$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($ExistingTask) {
    Write-Host "Task exists, updating..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Create trigger: Daily at 8:00 AM
$Trigger = New-ScheduledTaskTrigger -Daily -At 8am

# Create action
$Action = New-ScheduledTaskAction -Execute $ScriptPath -WorkingDirectory $WorkDir

# Create settings
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopOnIdleEnd `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

# Register task
Register-ScheduledTask `
    -TaskName $TaskName `
    -Description $TaskDescription `
    -Trigger $Trigger `
    -Action $Action `
    -Settings $Settings

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Task created successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Task Name: $TaskName"
Write-Host "Run Time:  Daily at 8:00 AM"
Write-Host "Script:    $ScriptPath"
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To test manually, run: $ScriptPath" -ForegroundColor Yellow
Write-Host "To view task: Task Scheduler -> Task Name: $TaskName" -ForegroundColor Yellow

# Show task info
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName, State | Format-Table