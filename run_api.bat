@echo off
REM Script pour démarrer l'API
echo ====================================
echo DEMARRAGE DE L'API RAG
echo ====================================
echo.

call venv\Scripts\activate.bat

echo L'API sera disponible sur: http://localhost:8000
echo Documentation interactive: http://localhost:8000/docs
echo.
echo Appuyez sur Ctrl+C pour arreter le serveur
echo.

python api.py

pause
