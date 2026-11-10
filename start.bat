@echo off
chcp 65001 >nul
title TrainKey
cd /d "%~dp0"

where py >nul 2>nul
if not errorlevel 1 goto use_py
where python >nul 2>nul
if not errorlevel 1 goto use_python

echo Python не найден. Установите его с https://www.python.org/downloads/
echo При установке отметьте галочку "Add python.exe to PATH" и запустите start.bat ещё раз.
pause
exit /b 1

:use_py
py -3 run.py
goto end

:use_python
python run.py

:end
pause
