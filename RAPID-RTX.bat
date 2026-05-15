::[Bat To Exe Converter]
::
::YAwzoRdxOk+EWAjk
::fBw5plQjdCqDJG6L5klwbltnWBGGJVe1C7sV/u3p/O+7kGwtfcZySrvjX2hupQ/iYos08EvY6v+hRk3M6ydqXyKCSkIGnVYMv2eKVw==
::YAwzuBVtJxjWCl3EqQJgSA==
::ZR4luwNxJguZRRnk
::Yhs/ulQjdF+5
::cxAkpRVqdFKZSDk=
::cBs/ulQjdF+5
::ZR41oxFsdFKZSDk=
::eBoioBt6dFKZSDk=
::cRo6pxp7LAbNWATEpCI=
::egkzugNsPRvcWATEpCI=
::dAsiuh18IRvcCxnZtBJQ
::cRYluBh/LU+EWAnk
::YxY4rhs+aU+JeA==
::cxY6rQJ7JhzQF1fEqQJQ
::ZQ05rAF9IBncCkqN+0xwdVs0
::ZQ05rAF9IAHYFVzEqQJQ
::eg0/rx1wNQPfEVWB+kM9LVsJDGQ=
::fBEirQZwNQPfEVWB+kM9LVsJDGQ=
::cRolqwZ3JBvQF1fEqQJQ
::dhA7uBVwLU+EWDk=
::YQ03rBFzNR3SWATElA==
::dhAmsQZ3MwfNWATElA==
::ZQ0/vhVqMQ3MEVWAtB9wSA==
::Zg8zqx1/OA3MEVWAtB9wSA==
::dhA7pRFwIByZRRnk
::Zh4grVQjdCqDJG6L5klwbltnWBGGJVe1C7sV/u3p/O+7kGwtfcZySrvjX2hupQ/iYos08EvY6v+hRk3M6ydqXyKCSkIGnVYMs3yAVw==
::YB416Ek+ZG8=
::
::
::978f952a14a936cc963da21a135fa983
@echo off
setlocal
:: 启用 UTF-8
set PYTHONUTF8=1
:: 切换目录
cd /d "%~dp0"
set SCRIPT_DIR=%~dp0

echo ====================================================
echo   RAPID-RTX bootloader (Environment Check)
echo ====================================================

REM --- 1. 检 ?python.bat 是否存在 ---
if not exist "%SCRIPT_DIR%python.bat" goto :NoPython

REM --- 2. 检查并运行安装脚本 ---
if not exist "%SCRIPT_DIR%setup_RAPID-RTX.py" goto :NoSetup
echo [Step 1/2] Checking Python dependencies...
call "%SCRIPT_DIR%python.bat" "%SCRIPT_DIR%setup_RAPID-RTX.py"

REM --- 3. 验证安装是否成功 (检查哨兵文 ? ---
if not exist "%SCRIPT_DIR%.deps_installed" goto :InstallError

REM --- 4. 启动程序 (避开 IF 括号 ? ---
echo [Step 2/2] [Launch] Starting RAPID-RTX Platform...
:: 注意：这里直接使用变量，不要包裹 ?if  ?() 内部
call "%SCRIPT_DIR%kit\kit.exe" "%SCRIPT_DIR%apps\rapid_rtx.kit" %*
goto :End

:NoPython
echo [Error] python.bat not found in: "%SCRIPT_DIR%"
pause
goto :End

:NoSetup
echo [Note] setup_RAPID-RTX.py not found, skipping dependency checks.
goto :LaunchAnyway

:InstallError
echo [Error] Environment setup failed. Please check the logs above.
pause
goto :End

:LaunchAnyway
echo [Launch] Starting RAPID-RTX Platform (without check)...
call "%SCRIPT_DIR%kit\kit.exe" "%SCRIPT_DIR%apps\rapid_rtx.kit" %*
goto :End

:End
echo [Status] RAPID-RTX App is closed.