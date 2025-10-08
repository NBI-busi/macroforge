@echo off
cd /d "%~dp0"
python fit_axes.py --input merged_patents.csv.gz --output_mapped mapped.parquet --model_out axes_model.joblib --assume_english --max_features 50000 --svd_dim 256 --n_neighbors 30

pause