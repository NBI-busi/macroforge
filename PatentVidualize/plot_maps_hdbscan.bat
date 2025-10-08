@echo off
cd /d "%~dp0"
python plot_maps_hdbscan.py --input-mapped mapped.parquet --outdir ./figs_hdbscan --period-width 5 --year-min 2006 --year-max 2025 --min-cluster-size 1600 --label-keywords 5 --text-col Title --hide-axes --show-context --cluster-outline density --outline-level 0.25
pause