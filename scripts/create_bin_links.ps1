# scripts/create_bin_links.ps1
$binDir = Join-Path $PSScriptRoot "..\bin"
if (-not (Test-Path $binDir)) {
    New-Item -ItemType Directory -Path $binDir | Out-Null
}

$terraformSource = "C:\Users\rahul\AppData\Local\Microsoft\WinGet\Packages\Hashicorp.Terraform_Microsoft.Winget.Source_8wekyb3d8bbwe\terraform.exe"
$trivySource = "C:\Users\rahul\AppData\Local\Microsoft\WinGet\Packages\AquaSecurity.Trivy_Microsoft.Winget.Source_8wekyb3d8bbwe\trivy.exe"
$awsSource = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"

# Copy or create cmd wrappers in bin
@'
@echo off
"C:\Program Files\Amazon\AWSCLIV2\aws.exe" %*
'@ | Set-Content -Path (Join-Path $binDir "aws.cmd")

@'
@echo off
"C:\Users\rahul\AppData\Local\Microsoft\WinGet\Packages\Hashicorp.Terraform_Microsoft.Winget.Source_8wekyb3d8bbwe\terraform.exe" %*
'@ | Set-Content -Path (Join-Path $binDir "terraform.cmd")

@'
@echo off
"C:\Users\rahul\AppData\Local\Microsoft\WinGet\Packages\AquaSecurity.Trivy_Microsoft.Winget.Source_8wekyb3d8bbwe\trivy.exe" %*
'@ | Set-Content -Path (Join-Path $binDir "trivy.cmd")

# Also copy into .venv\Scripts which is already active in python virtualenv
$venvScripts = Join-Path $PSScriptRoot "..\.venv\Scripts"
if (Test-Path $venvScripts) {
    Copy-Item (Join-Path $binDir "aws.cmd") $venvScripts -Force
    Copy-Item (Join-Path $binDir "terraform.cmd") $venvScripts -Force
    Copy-Item (Join-Path $binDir "trivy.cmd") $venvScripts -Force
}

Write-Host "Created binary wrappers in bin/ and .venv/Scripts/ successfully."
