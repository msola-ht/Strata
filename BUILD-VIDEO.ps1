# Build the optional Responses video encoder. -Update runs the existing installer update first.
[CmdletBinding()]
param(
    [switch]$Update,
    [switch]$Cpu,
    [ValidateRange(1, 64)][int]$Jobs = 4,
    [string]$LlamaDir,
    [string]$VsPath
)

$ErrorActionPreference = 'Stop'
$pinnedCommit = '3cf03257f219afbe7334045ff7c6a06ac68c627d'
$originalLocation = Get-Location

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Program failed (exit $LASTEXITCODE). See the first error above."
    }
}

try {
    if ($env:OS -ne 'Windows_NT') { throw 'Run this script on Windows.' }
    Set-Location $PSScriptRoot

    if ($Update) {
        # Never reset a feature branch to origin/main on a failed pull.
        $localChanges = & git.exe status --porcelain --untracked-files=no
        if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect the current Git working tree.' }
        if ($localChanges) {
            throw 'Tracked files have local changes. Save your fixes in a commit or stash before updating. Nothing was updated.'
        }
        Invoke-Checked 'git.exe' @('pull', '--ff-only')
        Invoke-Checked (Join-Path $PSScriptRoot 'START-HERE.bat') @('--update')
        Write-Host 'Installer update command returned successfully; this does not confirm a model is installed.'
        Write-Host 'Starting the video encoder build. Model configuration is a separate step.'
    }

    # Load the complete MSVC + Windows SDK environment, not only cl.exe's PATH.
    if (-not $VsPath) {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
        if (-not (Test-Path $vswhere)) {
            throw 'Install Visual Studio 2022 Build Tools with Desktop development with C++ and a Windows SDK.'
        }
        $VsPath = & $vswhere -latest -products '*' -version '[17.0,18.0)' `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($LASTEXITCODE -ne 0 -or -not $VsPath) {
            throw 'Visual Studio 2022 C++ tools not found. Use -VsPath to select another compatible installation.'
        }
    }
    $devShell = Join-Path $VsPath 'Common7/Tools/Launch-VsDevShell.ps1'
    if (-not (Test-Path $devShell)) { throw "Developer PowerShell script not found: $devShell" }
    & $devShell -Arch amd64 -HostArch amd64
    Set-Location $PSScriptRoot

    $required = @('git.exe', 'cl.exe', 'rc.exe', 'mt.exe', 'cmake.exe', 'ninja.exe')
    if (-not $Cpu) { $required += 'nvcc.exe' }
    foreach ($tool in $required) {
        if (-not (Get-Command $tool -CommandType Application -ErrorAction SilentlyContinue)) {
            throw "Missing $tool. Install the required build tools / Windows SDK / CUDA Toolkit and retry."
        }
    }

    $buildName = if ($Cpu) { 'build-vision-video-cpu' } else { 'build-vision-video' }
    $buildDir = Join-Path $PSScriptRoot $buildName
    if (-not $LlamaDir) {
        $existing = Join-Path $PSScriptRoot 'third_party/llama-video'
        # Reuse the dependency from the manual instructions; otherwise keep it in the ignored build directory.
        $LlamaDir = if (Test-Path $existing) { $existing } else { Join-Path $buildDir 'deps/llama.cpp' }
    }
    if (-not [System.IO.Path]::IsPathRooted($LlamaDir)) {
        $LlamaDir = Join-Path $PSScriptRoot $LlamaDir
    }
    $LlamaDir = [System.IO.Path]::GetFullPath($LlamaDir)
    if (-not (Test-Path $LlamaDir)) {
        New-Item -ItemType Directory -Force -Path (Split-Path $LlamaDir -Parent) | Out-Null
        Invoke-Checked 'git.exe' @('clone', '--no-checkout', 'https://github.com/ggml-org/llama.cpp.git', $LlamaDir)
        Invoke-Checked 'git.exe' @('-C', $LlamaDir, 'checkout', '--detach', $pinnedCommit)
    }
    $actualCommit = & git.exe -C $LlamaDir rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $pinnedCommit) {
        throw "Dependency must be at $pinnedCommit. Existing directory was not changed: $LlamaDir"
    }
    $dirty = & git.exe -C $LlamaDir status --porcelain --untracked-files=no
    if ($LASTEXITCODE -ne 0 -or $dirty) {
        throw "Dependency has local changes or cannot be inspected. Existing files were preserved: $LlamaDir"
    }

    $cuda = if ($Cpu) { 'OFF' } else { 'ON' }
    $configure = @('--fresh', '-S', 'tools/vision', '-B', $buildDir, '-G', 'Ninja',
        '-DCMAKE_BUILD_TYPE=Release', "-DLLAMA_DIR=$LlamaDir", "-DSTRATA_VISION_CUDA=$cuda")
    if (-not $Cpu) { $configure += '-DCMAKE_CUDA_ARCHITECTURES=native' }
    Invoke-Checked 'cmake.exe' $configure
    Invoke-Checked 'cmake.exe' @('--build', $buildDir, '--target', 'strata-vision', '--parallel', "$Jobs")

    $encoder = Join-Path $buildDir 'bin/strata-vision.exe'
    if (-not (Test-Path $encoder -PathType Leaf)) { throw "Build output not found: $encoder" }
    Write-Host "`nBuild succeeded: $encoder"
    Write-Host ('Set vision.exe in your model config to: ' + $encoder.Replace('\', '/'))
    Write-Host 'Enable vision.video.enabled, configure FFmpeg/FFprobe, and disable engine batching.'
    Write-Host 'The build did not select this encoder in your config or restart the service.'
    Write-Host 'Configure it before restarting Strata; build success alone does not enable video.'
} catch {
    Write-Host "`nVideo encoder build failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally {
    Set-Location $originalLocation
}
