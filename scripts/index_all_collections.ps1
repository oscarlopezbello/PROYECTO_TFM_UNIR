# Script para indexar todas las colecciones Chroma
# Ejecutar: .\scripts\index_all_collections.ps1

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$collections = @(
    "index_skills",
    "index_education", 
    "index_language",
    "index_sector",
    "index_experience",
    "index_job_title"
)

Write-Host "=== Indexando colecciones Chroma ===" -ForegroundColor Cyan
Write-Host "CHROMA_DIR debe apuntar a data/chroma en .env" -ForegroundColor Yellow
Write-Host ""

foreach ($mod in $collections) {
    Write-Host ">>> Ejecutando $mod..." -ForegroundColor Green
    poetry run python -m "tfm_match.embeddings.$mod"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR en $mod" -ForegroundColor Red
        exit 1
    }
    Write-Host "OK $mod" -ForegroundColor Green
    Write-Host ""
}

Write-Host "=== Todas las colecciones indexadas ===" -ForegroundColor Cyan
Write-Host "Verificar con: poetry run python src/tfm_match/embeddings/test_chroma.py" -ForegroundColor Yellow
