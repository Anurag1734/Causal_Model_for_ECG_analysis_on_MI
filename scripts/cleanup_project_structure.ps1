# Project Structure Cleanup Script
# Run from: C:\Users\Acer\Desktop\Capstone\Capstone
# Usage: .\scripts\cleanup_project_structure.ps1

Write-Host "🧹 Cleaning up Capstone project structure..." -ForegroundColor Cyan

# 1. Move misplaced Python scripts to legacy folder
Write-Host "`n1. Moving legacy scripts..." -ForegroundColor Yellow
Move-Item -Path ".\1.py" -Destination ".\scripts\legacy\1.py" -Force
Move-Item -Path ".\2.py" -Destination ".\scripts\legacy\2.py" -Force
Write-Host "   ✓ Moved 1.py and 2.py to scripts/legacy/" -ForegroundColor Green

# 2. Move analyze_adjudication.py to scripts
Write-Host "`n2. Moving analyze_adjudication.py..." -ForegroundColor Yellow
Move-Item -Path ".\analyze_adjudication.py" -Destination ".\scripts\analyze_adjudication.py" -Force
Write-Host "   ✓ Moved to scripts/" -ForegroundColor Green

# 3. Remove duplicate database (keep the one in data/)
Write-Host "`n3. Removing duplicate database..." -ForegroundColor Yellow
if (Test-Path ".\mimic_database.duckdb") {
    Remove-Item ".\mimic_database.duckdb" -Force
    Write-Host "   ✓ Removed duplicate mimic_database.duckdb from root" -ForegroundColor Green
} else {
    Write-Host "   ⚠ File already removed" -ForegroundColor Gray
}

# 4. Archive installation files
Write-Host "`n4. Archiving installation files..." -ForegroundColor Yellow
New-Item -Path ".\archive" -ItemType Directory -Force | Out-Null
Move-Item -Path ".\INSTALL_NOW.txt" -Destination ".\archive\INSTALL_NOW.txt" -Force -ErrorAction SilentlyContinue
Move-Item -Path ".\INSTALL_REQUIREMENTS.txt" -Destination ".\archive\INSTALL_REQUIREMENTS.txt" -Force -ErrorAction SilentlyContinue
Move-Item -Path ".\SETUP_CUDA_ENV.txt" -Destination ".\archive\SETUP_CUDA_ENV.txt" -Force -ErrorAction SilentlyContinue
Move-Item -Path ".\RUN_NOW.txt" -Destination ".\archive\RUN_NOW.txt" -Force -ErrorAction SilentlyContinue
Write-Host "   ✓ Moved installation guides to archive/" -ForegroundColor Green

# 5. Remove miniconda installer
Write-Host "`n5. Removing installer..." -ForegroundColor Yellow
if (Test-Path ".\miniconda.exe") {
    Remove-Item ".\miniconda.exe" -Force
    Write-Host "   ✓ Removed miniconda.exe (311 MB freed)" -ForegroundColor Green
} else {
    Write-Host "   ⚠ Installer already removed" -ForegroundColor Gray
}

# 6. Consolidate redundant documentation
Write-Host "`n6. Consolidating documentation..." -ForegroundColor Yellow
New-Item -Path ".\docs" -ItemType Directory -Force | Out-Null
Move-Item -Path ".\READY_TO_RUN.md" -Destination ".\docs\READY_TO_RUN.md" -Force -ErrorAction SilentlyContinue
Move-Item -Path ".\IMPLEMENTATION_SUMMARY.md" -Destination ".\docs\IMPLEMENTATION_SUMMARY.md" -Force -ErrorAction SilentlyContinue
Move-Item -Path ".\PHASE_D_SUMMARY.md" -Destination ".\docs\PHASE_D_SUMMARY.md" -Force -ErrorAction SilentlyContinue
Write-Host "   ✓ Moved legacy docs to docs/" -ForegroundColor Green

# Final summary
Write-Host "`n" + ("="*60) -ForegroundColor Cyan
Write-Host "✅ CLEANUP COMPLETE!" -ForegroundColor Green
Write-Host ("="*60) -ForegroundColor Cyan

Write-Host "`nChanges made:" -ForegroundColor Yellow
Write-Host "  • Moved 1.py, 2.py → scripts/legacy/"
Write-Host "  • Moved analyze_adjudication.py → scripts/"
Write-Host "  • Removed duplicate mimic_database.duckdb from root"
Write-Host "  • Archived installation guides → archive/"
Write-Host "  • Removed miniconda.exe installer"
Write-Host "  • Consolidated legacy docs → docs/"

Write-Host "`nKeep these core files in root:" -ForegroundColor Yellow
Write-Host "  • README.md                    (Main documentation)"
Write-Host "  • PHASE_D3_D5_README.md        (Technical guide)"
Write-Host "  • QUICKSTART_VAE_TRAINING.md   (User guide)"
Write-Host "  • COMMAND_REFERENCE.md         (Quick reference)"
Write-Host "  • FINAL_VERIFICATION_REPORT.md (Status report)"

Write-Host "`n📁 Project structure is now clean and organized!" -ForegroundColor Green
