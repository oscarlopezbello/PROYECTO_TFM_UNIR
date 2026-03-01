# Script para ejecutar MCP Inspector con tu servidor
# Asegúrate de tener Node.js instalado

Write-Host " Iniciando MCP Inspector para TFM Match..." -ForegroundColor Green
Write-Host ""

# Configurar variables de entorno desde .env
$envFile = Join-Path $PSScriptRoot ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]*?)\s*=\s*(.*?)\s*$') {
            $name = $matches[1]
            $value = $matches[2]
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
            Write-Host "✓ Variable cargada: $name" -ForegroundColor Gray
        }
    }
}

Write-Host ""
Write-Host "Ejecutando MCP Inspector..." -ForegroundColor Cyan
Write-Host "URL Inspector: http://localhost:5173" -ForegroundColor Yellow
Write-Host "Tu servidor MCP se conectará automáticamente" -ForegroundColor Yellow
Write-Host ""

# Ejecutar inspector con la configuración del servidor
npx @modelcontextprotocol/inspector poetry run python -m tfm_match.mcp.server
