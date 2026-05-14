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
:: Configure Python to use UTF-8 mode
set PYTHONUTF8=1
cd /d "%~dp0"
set SCRIPT_DIR=%~dp0

echo ====================================================
echo   RAPID-RTX bootloader (Environment Check)
echo ====================================================

REM Run the installation script
if exist "%SCRIPT_DIR%python.bat" (
    call "%SCRIPT_DIR%python.bat" "%SCRIPT_DIR%setup_RAPID-RTX.py"
)

REM Start the main program.
if exist "%SCRIPT_DIR%.deps_installed" (
    echo [Launch] Starting RAPID-RTX Platform...
    call "%SCRIPT_DIR%kit\kit.exe" "%%~dp0apps/rapid_rtx.kit" %*
) else (
    echo [Error] Environment setup failed.
    pause
)