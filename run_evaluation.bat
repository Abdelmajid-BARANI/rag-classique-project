@echo off
echo ============================================
echo   Évaluation RAGAS du pipeline RAG
echo ============================================
echo.

call venv\Scripts\activate.bat

echo Lancement de l'évaluation...
echo.
python run_evaluation.py %*

echo.
echo Évaluation terminée. Voir logs\evaluation_report.json
pause
