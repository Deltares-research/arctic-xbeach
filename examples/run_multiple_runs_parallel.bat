@echo off
setlocal
REM ============================================================
REM  Arctic-XBeach batch runner (parallel, new windows)
REM  Each run is started in its own CMD window and runs in parallel
REM ============================================================

REM ---- SETTINGS ------------------------------------------------
REM Name of your conda environment
set "CONDA_ENV=arctic-xbeach-dev"

REM Path to conda activate script (adjust if needed!)
REM For Anaconda:
set "CONDA_ACTIVATE=%USERPROFILE%\anaconda3\Scripts\activate.bat"
REM For Miniconda, you might need something like:
REM set "CONDA_ACTIVATE=%USERPROFILE%\miniconda3\Scripts\activate.bat"

REM Path to main Python script
set "MAIN_PY=d:\Git\arctic-xbeach\main.py"

REM Run directories (add/remove as needed)
set "RUN1=d:\Git\arctic-xbeach\examples\analytical\01_dirichlet_warming"
set "RUN2=d:\Git\arctic-xbeach\examples\analytical\02_dirichlet_cooling"
set "RUN3=d:\Git\arctic-xbeach\examples\analytical\03_neumann_constant"

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
