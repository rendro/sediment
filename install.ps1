#Requires -Version 5.1
<#
.SYNOPSIS
    Install sediment on Windows.
.DESCRIPTION
    Downloads the latest sediment release from GitHub and installs it to
    $env:LOCALAPPDATA\sediment\bin (or $env:SEDIMENT_INSTALL_DIR if set).
#>
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Repo = 'rendro/sediment'
$InstallDir = if ($env:SEDIMENT_INSTALL_DIR) { $env:SEDIMENT_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA 'sediment\bin' }

# Detect architecture
$Arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
if ($Arch -ne 'X64') {
    Write-Error "Unsupported architecture: $Arch. Only x86_64 is supported."
    exit 1
}

$Target = 'x86_64-pc-windows-msvc'

# Get latest version
Write-Host 'Fetching latest release...'
$Release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest"
$Version = $Release.tag_name -replace '^v', ''
if (-not $Version) {
    Write-Error 'Failed to determine latest version'
    exit 1
}
Write-Host "Installing sediment v$Version ($Target)..."

$ZipName = "sediment-$Target.zip"
$Url = "https://github.com/$Repo/releases/download/v$Version/$ZipName"

# Download
$TmpDir = Join-Path ([System.IO.Path]::GetTempPath()) "sediment-install-$([guid]::NewGuid())"
New-Item -ItemType Directory -Path $TmpDir -Force | Out-Null

try {
    $ZipPath = Join-Path $TmpDir $ZipName
    Invoke-WebRequest -Uri $Url -OutFile $ZipPath -UseBasicParsing

    # Verify checksum
    $ChecksumsUrl = "https://github.com/$Repo/releases/download/v$Version/checksums.txt"
    try {
        $ChecksumsPath = Join-Path $TmpDir 'checksums.txt'
        Invoke-WebRequest -Uri $ChecksumsUrl -OutFile $ChecksumsPath -UseBasicParsing
        $Expected = (Get-Content $ChecksumsPath | Where-Object { $_ -match $ZipName } | ForEach-Object { ($_ -split '\s+')[0] })
        if ($Expected) {
            $Actual = (Get-FileHash $ZipPath -Algorithm SHA256).Hash.ToLower()
            if ($Actual -ne $Expected) {
                Write-Error "Checksum verification failed!`n  Expected: $Expected`n  Actual:   $Actual"
                exit 1
            }
            Write-Host 'Checksum verified.'
        }
    } catch {
        Write-Warning 'No checksums.txt available, binary integrity could not be verified.'
    }

    # Extract
    $ExtractDir = Join-Path $TmpDir 'extract'
    Expand-Archive -Path $ZipPath -DestinationPath $ExtractDir -Force

    # Install
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    Copy-Item -Path (Join-Path $ExtractDir 'sediment.exe') -Destination (Join-Path $InstallDir 'sediment.exe') -Force

    Write-Host "Installed sediment to $InstallDir\sediment.exe"

    # Check PATH
    if (-not ($env:PATH -split ';' | Where-Object { $_ -eq $InstallDir })) {
        Write-Host ''
        Write-Host "Add $InstallDir to your PATH:"
        Write-Host "  [Environment]::SetEnvironmentVariable('PATH', `"$InstallDir;`$env:PATH`", 'User')"
    }
} finally {
    Remove-Item -Path $TmpDir -Recurse -Force -ErrorAction SilentlyContinue
}
