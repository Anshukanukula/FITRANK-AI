@echo off
echo ===================================================
echo FitRank AI — Hugging Face Space Deployment Script
echo ===================================================
echo.
echo Attempting to push local project files to Hugging Face Spaces...
echo.

git remote remove origin 2>nul
git remote add origin https://huggingface.co/spaces/anshukanukula03/fitrank-ai

git push origin main --force
if %errorlevel% neq 0 (
    echo.
    echo [INFO] Git push failed. This usually means you need to authenticate.
    echo Please generate a Write Access Token at https://huggingface.co/settings/tokens
    set /p token="Enter your Hugging Face Write Access Token: "
    if not "%token%"=="" (
        echo.
        echo Attempting to push using the access token...
        git push https://anshukanukula03:%token%@huggingface.co/spaces/anshukanukula03/fitrank-ai main --force
    ) else (
        echo No token provided. Exiting.
    )
)

echo.
echo Done! Press any key to exit.
pause >nul
