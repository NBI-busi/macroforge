@echo off
cd /d "%~dp0"
python estat_census_labor_force.py --out-dir ./output
pause