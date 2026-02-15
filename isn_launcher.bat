@echo off
chcp 65001 >nul 2>&1
title ISn v4.0 SUPREME — openbdf
color 0A

echo.
echo  ================================================================
echo   ISn v4.0 SUPREME — Motor CMD Descentralizado
echo   Modelo de Datos Disparejos + Mercado de Prompts
echo   Operador: openbdf
echo  ================================================================
echo.

:: Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python no encontrado en PATH.
    echo  Instale Python 3.10+ desde https://python.org 
    pause
    exit /b 1
)

echo  [OK] Python detectado
echo.

:: Crear carpeta exports si no existe
if not exist "%~dp0exports" mkdir "%~dp0exports"

:: Lanzar el motor Python directamente
python "%~dp0isn_bit_calculator.pyw"

echo.
echo  [ISn] Sesion terminada.
pause