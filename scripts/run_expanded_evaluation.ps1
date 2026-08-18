[CmdletBinding()]
param(
    [switch]$PreflightOnly,
    [switch]$BenchmarkOnly,
    [switch]$ForceBenchmark,
    [ValidateSet(0, 1, 2)]
    [int]$ParallelismOverride = 0,
    [string]$OutputDir = "data/results/evaluation_300_seed42_20260817"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:WorkspaceRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$script:BaseUrl = "http://127.0.0.1:11435"
$script:PilotDir = [IO.Path]::GetFullPath((Join-Path $script:WorkspaceRoot "data/results/evaluation"))
$script:ArchiveDir = [IO.Path]::GetFullPath((Join-Path $script:WorkspaceRoot "data/results/archived_evaluations/pilot_30_before_expansion_20260817"))
$script:RunDir = if ([IO.Path]::IsPathRooted($OutputDir)) {
    [IO.Path]::GetFullPath($OutputDir)
} else {
    [IO.Path]::GetFullPath((Join-Path $script:WorkspaceRoot $OutputDir))
}
$script:RuntimeDir = [IO.Path]::GetFullPath((Join-Path $script:WorkspaceRoot "data/results/evaluation_runtime_300_seed42_20260817"))
$script:BenchmarkDir = Join-Path $script:RuntimeDir "benchmark"
$script:CandidateOne = Join-Path $script:BenchmarkDir "candidate_parallel_1.json"
$script:CandidateTwo = Join-Path $script:BenchmarkDir "candidate_parallel_2.json"
$script:SelectedFile = Join-Path $script:BenchmarkDir "selected_parallelism.json"

function Test-IsInsideWorkspace {
    param([Parameter(Mandatory = $true)][string]$Path)
    $resolved = [IO.Path]::GetFullPath($Path)
    $root = $script:WorkspaceRoot.TrimEnd([IO.Path]::DirectorySeparatorChar)
    $prefix = $root + [IO.Path]::DirectorySeparatorChar
    return $resolved.Equals($root, [StringComparison]::OrdinalIgnoreCase) -or
        $resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
}

function Assert-SafePath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (-not (Test-IsInsideWorkspace -Path $Path)) {
        throw "$Label is outside the workspace: $Path"
    }
}

function Test-PortOpen {
    param([int]$Port)
    $client = [Net.Sockets.TcpClient]::new()
    try {
        $task = $client.ConnectAsync("127.0.0.1", $Port)
        return $task.Wait(250) -and $client.Connected
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

function Get-ToolPath {
    param([Parameter(Mandatory = $true)][string]$Name)
    $command = Get-Command $Name -ErrorAction Stop | Select-Object -First 1
    if (-not $command.Source) {
        throw "Could not resolve executable: $Name"
    }
    return $command.Source
}

function Write-Preflight {
    Assert-SafePath -Path $script:PilotDir -Label "Pilot directory"
    Assert-SafePath -Path $script:ArchiveDir -Label "Archive directory"
    Assert-SafePath -Path $script:RunDir -Label "Run directory"
    Assert-SafePath -Path $script:RuntimeDir -Label "Runtime directory"

    if ($script:RunDir.Equals($script:PilotDir, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Run directory must differ from the pilot directory."
    }
    $pilotCases = Join-Path $script:PilotDir "test_cases.jsonl"
    if (-not (Test-Path -LiteralPath $pilotCases -PathType Leaf)) {
        throw "Pilot cases not found: $pilotCases"
    }
    $pilotCount = @(Get-Content -LiteralPath $pilotCases).Count
    if ($pilotCount -ne 30) {
        throw "Expected 30 pilot cases, found $pilotCount."
    }
    if (Test-Path -LiteralPath $script:RunDir -PathType Container) {
        $entries = @(Get-ChildItem -LiteralPath $script:RunDir -Force)
        $checkpoint = Join-Path $script:RunDir "checkpoint"
        if ($entries.Count -gt 0 -and -not (Test-Path -LiteralPath $checkpoint -PathType Container)) {
            throw "Run directory is non-empty and has no checkpoint: $script:RunDir"
        }
    }

    $script:OllamaExe = Get-ToolPath -Name "ollama.exe"
    $script:NvidiaSmiExe = Get-ToolPath -Name "nvidia-smi.exe"
    $script:PythonExe = Get-ToolPath -Name "python.exe"
    $modelManifest = Join-Path $env:USERPROFILE ".ollama/models/manifests/registry.ollama.ai/library/gemma4/e4b"
    if (-not (Test-Path -LiteralPath $modelManifest -PathType Leaf)) {
        throw "Local gemma4:e4b manifest was not found: $modelManifest"
    }

    $gpuLines = @(& $script:NvidiaSmiExe --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader,nounits)
    if ($LASTEXITCODE -ne 0 -or $gpuLines.Count -eq 0) {
        throw "nvidia-smi could not read the GPU."
    }
    $gpuParts = @($gpuLines[0] -split ',' | ForEach-Object { $_.Trim() })
    if ($gpuParts.Count -lt 4) {
        throw "Unexpected nvidia-smi output: $($gpuLines[0])"
    }
    $script:GpuName = $gpuParts[0]
    $script:GpuTotalMiB = $gpuParts[1]
    $script:GpuFreeMiB = $gpuParts[2]
    $script:GpuDriver = $gpuParts[3]

    $operatingSystem = Get-CimInstance Win32_OperatingSystem
    $freeRamGiB = [math]::Round($operatingSystem.FreePhysicalMemory / 1MB, 2)
    $driveName = (Split-Path -Qualifier $script:WorkspaceRoot).TrimEnd(':')
    $freeDiskGiB = [math]::Round((Get-PSDrive -Name $driveName).Free / 1GB, 2)

    Write-Host "Preflight OK"
    Write-Host "  Workspace: $script:WorkspaceRoot"
    Write-Host "  Pilot: $pilotCount cases at $script:PilotDir"
    Write-Host "  Output: $script:RunDir"
    Write-Host "  GPU: $script:GpuName; total=$script:GpuTotalMiB MiB; free=$script:GpuFreeMiB MiB; driver=$script:GpuDriver"
    Write-Host "  Free RAM: $freeRamGiB GiB; free disk: $freeDiskGiB GiB"
    Write-Host "  Model: gemma4:e4b (local manifest present)"
    Write-Host "  MTP: false (checkpoint has no compatible draft layer on Windows/CUDA)"
    Write-Host "  Dedicated Ollama endpoint: $script:BaseUrl"
}

function Set-RuntimeEnvironment {
    $env:CUDA_VISIBLE_DEVICES = "0"
    $env:OLLAMA_HOST = "127.0.0.1:11435"
    $env:OLLAMA_FLASH_ATTENTION = "1"
    $env:OLLAMA_KV_CACHE_TYPE = "q8_0"
    $env:OLLAMA_MAX_LOADED_MODELS = "1"
    $env:OLLAMA_KEEP_ALIVE = "-1"
    $env:OLLAMA_GPU_OVERHEAD = "268435456"
    $env:VIRTUAL_REPORTER_RUN_LLM = "1"
    $env:VIRTUAL_REPORTER_LLM_PROVIDER = "ollama"
    $env:VIRTUAL_REPORTER_LLM_MODEL = "gemma4:e4b"
    $env:VIRTUAL_REPORTER_LLM_BASE_URL = $script:BaseUrl
    $env:VIRTUAL_REPORTER_OLLAMA_NUM_GPU = "999"
    $env:VIRTUAL_REPORTER_OLLAMA_NUM_CTX = "4096"
    $env:VIRTUAL_REPORTER_OLLAMA_NUM_THREAD = "14"
    $env:VIRTUAL_REPORTER_OLLAMA_NUM_PREDICT = "512"
    $env:VIRTUAL_REPORTER_OLLAMA_FORMAT = "json"
    $env:VIRTUAL_REPORTER_OLLAMA_THINK = "false"
    $env:VIRTUAL_REPORTER_LLM_RETRIES = "2"
    $env:VIRTUAL_REPORTER_PARSE_RETRIES = "2"
    $env:VIRTUAL_REPORTER_LLM_COOLDOWN_SECONDS = "6"
    $env:VIRTUAL_REPORTER_CASE_COOLDOWN_SECONDS = "3"
    $env:VIRTUAL_REPORTER_GPU_NAME = $script:GpuName
    $env:VIRTUAL_REPORTER_GPU_TOTAL_MIB = $script:GpuTotalMiB
    $env:VIRTUAL_REPORTER_GPU_DRIVER = $script:GpuDriver
    $env:PYTHONUNBUFFERED = "1"
}

function Ensure-PilotArchive {
    $sourceCases = Join-Path $script:PilotDir "test_cases.jsonl"
    $archiveCases = Join-Path $script:ArchiveDir "test_cases.jsonl"
    $sourceHash = (Get-FileHash -LiteralPath $sourceCases -Algorithm SHA256).Hash
    if (Test-Path -LiteralPath $script:ArchiveDir) {
        if (-not (Test-Path -LiteralPath $archiveCases -PathType Leaf)) {
            throw "Existing pilot archive is incomplete: $script:ArchiveDir"
        }
        $archiveHash = (Get-FileHash -LiteralPath $archiveCases -Algorithm SHA256).Hash
        if ($archiveHash -ne $sourceHash) {
            throw "Existing pilot archive hash does not match the source pilot."
        }
        Write-Host "Pilot archive already exists and its test-case hash matches."
        return
    }
    $archiveParent = Split-Path -Parent $script:ArchiveDir
    New-Item -ItemType Directory -Path $archiveParent -Force | Out-Null
    Copy-Item -LiteralPath $script:PilotDir -Destination $script:ArchiveDir -Recurse
    $archiveHash = (Get-FileHash -LiteralPath $archiveCases -Algorithm SHA256).Hash
    if ($archiveHash -ne $sourceHash) {
        throw "Pilot archive verification failed after copying."
    }
    Write-Host "Pilot archived without modifying the source: $script:ArchiveDir"
}

function Start-DedicatedServer {
    param(
        [Parameter(Mandatory = $true)][int]$Parallelism,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (Test-PortOpen -Port 11435) {
        throw "Dedicated port 11435 is already in use; refusing to touch that process."
    }
    $env:OLLAMA_NUM_PARALLEL = "$Parallelism"
    $stdout = Join-Path $script:RuntimeDir "ollama_${Label}_stdout.log"
    $stderr = Join-Path $script:RuntimeDir "ollama_${Label}_stderr.log"
    $process = Start-Process -FilePath $script:OllamaExe -ArgumentList @("serve") -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    Write-Host "Started dedicated Ollama PID $($process.Id) with parallelism=$Parallelism"
    for ($attempt = 0; $attempt -lt 90; $attempt++) {
        if ($process.HasExited) {
            throw "Dedicated Ollama exited during startup; inspect $stderr"
        }
        try {
            Invoke-RestMethod -Uri "$script:BaseUrl/api/tags" -Method Get -TimeoutSec 2 | Out-Null
            return $process
        } catch {
            Start-Sleep -Milliseconds 1000
        }
    }
    throw "Dedicated Ollama did not become ready within 90 seconds."
}

function Start-GpuMonitor {
    param(
        [Parameter(Mandatory = $true)][string]$OutputPath,
        [Parameter(Mandatory = $true)][string]$ErrorPath
    )
    $arguments = @(
        "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv",
        "-l",
        "2"
    )
    $process = Start-Process -FilePath $script:NvidiaSmiExe -ArgumentList $arguments -WindowStyle Hidden -PassThru -RedirectStandardOutput $OutputPath -RedirectStandardError $ErrorPath
    Write-Host "Started GPU monitor PID $($process.Id)"
    return $process
}

function Get-DescendantProcessIds {
    param([int]$RootProcessId)
    $found = [Collections.Generic.List[int]]::new()
    $queue = [Collections.Generic.Queue[int]]::new()
    $queue.Enqueue($RootProcessId)
    while ($queue.Count -gt 0) {
        $parentId = $queue.Dequeue()
        $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$parentId" -ErrorAction SilentlyContinue)
        foreach ($child in $children) {
            $childId = [int]$child.ProcessId
            $found.Add($childId)
            $queue.Enqueue($childId)
        }
    }
    return @($found)
}

function Stop-OwnedProcess {
    param([AllowNull()][Diagnostics.Process]$Process)
    if ($null -eq $Process) {
        return
    }
    $ownedIds = @(Get-DescendantProcessIds -RootProcessId $Process.Id)
    [array]::Reverse($ownedIds)
    foreach ($ownedId in $ownedIds) {
        Stop-Process -Id $ownedId -Force -ErrorAction SilentlyContinue
    }
    Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
    try {
        $Process.WaitForExit(5000) | Out-Null
    } catch {
    }
}

function Invoke-BenchmarkCandidate {
    param([Parameter(Mandatory = $true)][int]$Parallelism)
    $label = "benchmark_${Parallelism}_$([DateTime]::Now.ToString('yyyyMMdd_HHmmss'))"
    $gpuCsv = Join-Path $script:BenchmarkDir "gpu_parallel_${Parallelism}.csv"
    $gpuError = Join-Path $script:BenchmarkDir "gpu_parallel_${Parallelism}_stderr.log"
    $output = if ($Parallelism -eq 1) { $script:CandidateOne } else { $script:CandidateTwo }
    $server = $null
    $monitor = $null
    try {
        $server = Start-DedicatedServer -Parallelism $Parallelism -Label $label
        $monitor = Start-GpuMonitor -OutputPath $gpuCsv -ErrorPath $gpuError
        & $script:PythonExe -m ifmt_models.ollama_benchmark run --output $output --parallelism $Parallelism --base-url $script:BaseUrl --gpu-log $gpuCsv --pilot-dir $script:PilotDir
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            throw "Benchmark candidate $Parallelism failed with exit code $exitCode."
        }
    } finally {
        Stop-OwnedProcess -Process $monitor
        Stop-OwnedProcess -Process $server
    }
}

function Ensure-BenchmarkSelection {
    $canReuse = (-not $ForceBenchmark) -and
        (Test-Path -LiteralPath $script:CandidateOne -PathType Leaf) -and
        (Test-Path -LiteralPath $script:CandidateTwo -PathType Leaf) -and
        (Test-Path -LiteralPath $script:SelectedFile -PathType Leaf)
    if (-not $canReuse) {
        Invoke-BenchmarkCandidate -Parallelism 1
        Invoke-BenchmarkCandidate -Parallelism 2
        & $script:PythonExe -m ifmt_models.ollama_benchmark select --one $script:CandidateOne --two $script:CandidateTwo --output $script:SelectedFile --max-vram-mib 7800
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            throw "Benchmark selection failed with exit code $exitCode."
        }
    } else {
        Write-Host "Reusing verified benchmark files in $script:BenchmarkDir"
    }
    $selection = Get-Content -Raw -LiteralPath $script:SelectedFile | ConvertFrom-Json
    $selected = [int]$selection.selected_parallelism
    if ($selected -notin @(1, 2)) {
        throw "Invalid selected parallelism: $selected"
    }
    Write-Host "Selected parallelism: $selected"
    return $selected
}

function Invoke-ExpandedRun {
    param([Parameter(Mandatory = $true)][int]$Parallelism)
    $env:VIRTUAL_REPORTER_CASE_WORKERS = "$Parallelism"
    $label = "evaluation_$([DateTime]::Now.ToString('yyyyMMdd_HHmmss'))"
    $gpuCsv = Join-Path $script:RuntimeDir "gpu_${label}.csv"
    $gpuError = Join-Path $script:RuntimeDir "gpu_${label}_stderr.log"
    $server = $null
    $monitor = $null
    try {
        $server = Start-DedicatedServer -Parallelism $Parallelism -Label $label
        $monitor = Start-GpuMonitor -OutputPath $gpuCsv -ErrorPath $gpuError
        & $script:PythonExe -m ifmt_models.cli run-expanded-evaluation --case-count 300 --seed 42 --output-dir $script:RunDir --pilot-dir $script:PilotDir --resume
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            throw "Expanded evaluation stopped with exit code $exitCode. Checkpoints were preserved."
        }
    } finally {
        Stop-OwnedProcess -Process $monitor
        Stop-OwnedProcess -Process $server
    }
}

Push-Location $script:WorkspaceRoot
try {
    Write-Preflight
    if ($PreflightOnly) {
        return
    }
    New-Item -ItemType Directory -Path $script:RuntimeDir -Force | Out-Null
    New-Item -ItemType Directory -Path $script:BenchmarkDir -Force | Out-Null
    Set-RuntimeEnvironment
    Ensure-PilotArchive
    $selectedParallelism = Ensure-BenchmarkSelection
    if ($BenchmarkOnly) {
        return
    }
    if ($ParallelismOverride -ne 0) {
        Write-Host "Operational safety override: parallelism $selectedParallelism -> $ParallelismOverride"
        $selectedParallelism = $ParallelismOverride
    }
    Invoke-ExpandedRun -Parallelism $selectedParallelism
} finally {
    Pop-Location
}
