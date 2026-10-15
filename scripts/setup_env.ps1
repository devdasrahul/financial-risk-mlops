# scripts/setup_env.ps1 - Environment Verification & Path Configuration

$userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
$targets = @(
    "C:\Program Files\Amazon\AWSCLIV2",
    "C:\Users\rahul\AppData\Local\Microsoft\WinGet\Packages\Hashicorp.Terraform_Microsoft.Winget.Source_8wekyb3d8bbwe",
    "C:\Users\rahul\AppData\Local\Microsoft\WinGet\Packages\AquaSecurity.Trivy_Microsoft.Winget.Source_8wekyb3d8bbwe"
)

$splitPaths = $userPath.Split(';')
$updatedPaths = [System.Collections.Generic.List[string]]::new()
foreach ($entry in $splitPaths) {
    if (-not [string]::IsNullOrWhiteSpace($entry) -and -not $updatedPaths.Contains($entry)) {
        $updatedPaths.Add($entry)
    }
}

foreach ($target in $targets) {
    if (-not $updatedPaths.Contains($target) -and (Test-Path $target)) {
        $updatedPaths.Add($target)
        Write-Host "Added to User PATH: $target"
    }
}

$newPathString = [string]::Join(';', $updatedPaths)
[System.Environment]::SetEnvironmentVariable("Path", $newPathString, "User")
$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + $newPathString

Write-Host "=== VERIFYING INSTALLED TOOLS IN CURRENT SESSION ==="
aws --version
terraform --version
trivy --version
git --version
python --version
docker --version
