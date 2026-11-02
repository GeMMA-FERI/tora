param(
    [Parameter(Mandatory = $true)][string]$Cli,
    [ValidateSet(0, 1)][int]$Jit = 0
)

$ErrorActionPreference = "Stop"
$cliPath = (Resolve-Path -LiteralPath $Cli).Path
Push-Location "$PSScriptRoot/.."
try {
    & npm.cmd run build
    if ($LASTEXITCODE -ne 0) { throw "TypeScript build failed." }
    & "_build/capture-venv/Scripts/python.exe" examples/capture_conv2d_module_input.py
    if ($LASTEXITCODE -ne 0) { throw "Module capture failed." }
    & node examples/conv2d-module-cli.mjs
    if ($LASTEXITCODE -ne 0) { throw "CLI graph generation failed." }

    $outputPath = Join-Path (Get-Location) "_build/conv2d-module-cli-output.json"
    $reportPath = Join-Path (Get-Location) "_build/conv2d-module-cli-report.json"
    if (Test-Path -LiteralPath $outputPath) { Remove-Item -LiteralPath $outputPath }
    # Factory serialization overwrites the result even if it already exists;
    # otherwise the CLI interprets an existing file as a destination template.
    & $cliPath node execute --graph (Join-Path (Get-Location) "_build/conv2d-module-cli.nodegraph.json") `
        --inputs-inline "{}" --outputs $outputPath --report $reportPath `
        --only-factory-serialize --output-names-type short --jit $Jit
    if ($LASTEXITCODE -ne 0) { throw "NodeEngine CLI execution failed." }
    $result = Get-Content -LiteralPath $outputPath -Raw | ConvertFrom-Json
    if ([string]::IsNullOrWhiteSpace($result.Text) -or $result.Text -cne $result.Expected) {
        throw "NodeEngine output text is missing or differs from the eager PyTorch reference. See $outputPath"
    }
    Write-Output "NodeEngine output matches the formatted eager PyTorch reference (JIT $Jit)."
    Write-Output $outputPath
} finally {
    Pop-Location
}
