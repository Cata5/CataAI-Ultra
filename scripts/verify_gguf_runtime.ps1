# Load a GGUF export in a real llama.cpp build. This is a smoke test, not a
# benchmark: it proves the file parses, tokenizes and generates without error.
# Usage:
#   .\scripts\verify_gguf_runtime.ps1 -LlamaCli C:\tools\llama-cli.exe -Gguf .\models\run\model.gguf
param(
    [Parameter(Mandatory = $true)][string]$LlamaCli,
    [Parameter(Mandatory = $true)][string]$Gguf,
    [string]$Prompt = "Hello",
    [int]$Tokens = 4,
    [int]$Context = 64
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path $LlamaCli)) { throw "llama-cli not found: $LlamaCli" }
if (-not (Test-Path $Gguf)) { throw "GGUF not found: $Gguf" }

Write-Host "Loading $Gguf in $LlamaCli"
& $LlamaCli -m $Gguf -p $Prompt -n $Tokens --temp 0 -t 2 -c $Context -st --no-warmup
if ($LASTEXITCODE -ne 0) { throw "llama-cli failed with exit code $LASTEXITCODE" }
Write-Host "GGUF runtime smoke test passed"
