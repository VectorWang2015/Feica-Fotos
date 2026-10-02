@echo off
setlocal
rem Offline launcher. No administrator rights, installation, or downloads.
pushd "%~dp0"
if errorlevel 1 exit /b 1
set "PYTHON=%CD%\.venv\Scripts\python.exe"
if not exist "%PYTHON%" goto setup
"%PYTHON%" -E -s -c "import sys; assert sys.version_info[:2] == (3, 12), 'Python 3.12 required'; import numpy; from PIL import Image, ImageCms; from PySide6.QtWidgets import QApplication"
if errorlevel 1 goto setup
"%PYTHON%" -E -s -m apps.local_looks %*
set "RESULT=%ERRORLEVEL%"
popd
exit /b %RESULT%

:setup
echo Local Looks needs this project's .venv with Python 3.12 and runtime dependencies. 1>&2
echo Open a terminal in "%CD%" and run these commands yourself: 1>&2
echo   py -3.12 -m venv .venv 1>&2
echo   .venv\Scripts\python.exe -m pip install -r requirements-app.txt 1>&2
echo The install command may use package indexes; this launcher never installs or downloads. 1>&2
echo Offline setup: use pip --no-index --find-links with a trusted wheel directory. 1>&2
echo If dependencies are present, review the Python/Qt system-library error above. 1>&2
popd
exit /b 1
