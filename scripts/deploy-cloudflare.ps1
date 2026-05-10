param(
    [string]$ProjectName = "plain-band-13be",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

if (-not $SkipBuild) {
    & "$PSScriptRoot\build-web.ps1"
}

npx wrangler deploy --name $ProjectName
