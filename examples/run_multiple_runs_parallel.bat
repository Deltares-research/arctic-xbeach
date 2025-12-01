@echo off
setlocal
REM ============================================================
REM  Arctic-XBeach batch runner (parallel, new windows)
REM  Each run is started in its own CMD window and runs in parallel
REM ============================================================

REM ---- SETTINGS ------------------------------------------------
REM Name of your conda environment
set "CONDA_ENV=arctic-xbeach"

REM Path to conda activate script (adjust if needed!)
REM For Anaconda:
set "CONDA_ACTIVATE=%USERPROFILE%\anaconda3\Scripts\activate.bat"
REM For Miniconda, you might need something like:
REM set "CONDA_ACTIVATE=%USERPROFILE%\miniconda3\Scripts\activate.bat"

REM Path to main Python script
set "MAIN_PY=D:\Git\thermo-morphological-model\main.py"

REM Run directories (add/remove as needed)
set "RUN1=D:\Git\thermo-morphological-model\runs\coupled_calibration_runs\run003_my_estimate_lower_critical_slope"
set "RUN2=d:\Git\thermo-morphological-model\runs\coupled_calibration_runs\run003_my_estimate_lower_facua"
set "RUN3=d:\Git\thermo-morphological-model\runs\coupled_validation_runs\run003_my_estimate_validation1_2012"
set "RUN4=d:\Git\thermo-morphological-model\runs\coupled_validation_runs\run003_my_estimate_validation2_2019"
set "RUN5=d:\Git\thermo-morphological-model\runs\coupled_calibration_runs\run003_my_estimate_revised_settings"

REM --------------------------------------------------------------

echo.
echo ============================================================
echo   Launching Arctic-XBeach runs in parallel...
echo ============================================================
echo.

REM ---- LAUNCH RUNS IN PARALLEL --------------------------------
REM For each RUN*, call :launch_run with a nice window title

call :launch_run "Run 1" "%RUN1%"
call :launch_run "Run 2" "%RUN2%"
call :launch_run "Run 3" "%RUN3%"
call :launch_run "Run 4" "%RUN4%"
call :launch_run "Run 5" "%RUN5%"

echo.
echo All runs have been launched in separate windows.
echo (They will continue running in parallel.)
echo.
pause
goto :eof


:launch_run
REM %~1 = window title, %~2 = run directory
set "TITLE=%~1"
set "RUNDIR=%~2"

if "%RUNDIR%"=="" goto :eof

echo Starting %TITLE%
echo   Directory: %RUNDIR%
echo.

REM /k keeps the window open after the run finishes (for logs).
REM Change to /c if you want the window to close automatically.
start "%TITLE%" cmd /k ^
    call "%CONDA_ACTIVATE%" %CONDA_ENV% ^&^& ^
    python "%MAIN_PY%" "%RUNDIR%"

goto :eof
