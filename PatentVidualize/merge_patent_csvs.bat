@echo off
cd /d "%~dp0"
python merge_patent_csvs.py --input-dir ./data --output merged_patents.csv.gz
pause