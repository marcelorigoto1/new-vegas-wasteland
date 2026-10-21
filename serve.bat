@echo off
rem Serves the 3D site at http://localhost:8080 using the Python bundled with QGIS.
call "C:\Program Files\QGIS 3.34.11\bin\o4w_env.bat"
cd /d "%~dp0site"
echo Open http://localhost:8080 in your browser. Ctrl+C to stop.
python -m http.server 8080
