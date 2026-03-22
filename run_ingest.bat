@echo off
REM Script pour démarrer l'ingestion des documents
echo ====================================
echo INGESTION DES DOCUMENTS
echo ====================================
echo.

call venv\Scripts\activate.bat
python ingest_documents.py %*

if errorlevel 1 (
    echo.
    echo ERREUR lors de l'ingestion
    pause
    exit /b 1
)

echo.
echo ====================================
echo INGESTION TERMINEE AVEC SUCCES
echo ====================================
echo.
echo Vous pouvez maintenant demarrer l'API avec: run_api.bat
echo.
pause
