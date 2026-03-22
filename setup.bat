@echo off
REM Script de démarrage rapide pour le projet RAG Benchmark
echo ====================================
echo RAG BENCHMARK - DEMARRAGE RAPIDE
echo ====================================
echo.

REM Vérifier si l'environnement virtuel existe
if not exist "venv\" (
    echo [1/5] Creation de l'environnement virtuel...
    python -m venv venv
    if errorlevel 1 (
        echo ERREUR: Impossible de creer l'environnement virtuel
        pause
        exit /b 1
    )
) else (
    echo [1/5] Environnement virtuel deja present
)

REM Activer l'environnement virtuel
echo [2/5] Activation de l'environnement virtuel...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERREUR: Impossible d'activer l'environnement virtuel
    pause
    exit /b 1
)

REM Installer les dépendances
echo [3/5] Installation des dependances...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERREUR: Probleme lors de l'installation des dependances
    pause
    exit /b 1
)

REM Tester le système
echo [4/5] Test du systeme...
python test_system.py
if errorlevel 1 (
    echo AVERTISSEMENT: Certains tests ont echoue
)

echo.
echo [5/5] Systeme pret!
echo.
echo ====================================
echo PROCHAINES ETAPES:
echo ====================================
echo.
echo 1. Verifier qu'Ollama est lance avec: ollama list
echo 2. Telecharger le modele: ollama pull llama3.1:8b
echo 3. Ingerer les documents: python ingest_documents.py
echo 4. Demarrer l'API: python api.py
echo 5. Tester avec Postman sur http://localhost:8000
echo.
pause
