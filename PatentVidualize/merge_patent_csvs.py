# merge_patent_csvs.py
# -*- coding: utf-8 -*-
import argparse, os, re, glob, sys
import pandas as pd

PATTERN = re.compile(r"(?P<theme>engine|ev|hybrid|fuelcell)-(?P<jurisdiction>us|ep|wo)-\d{4}to\d{4}\.csv$", re.I)

DEFAULT_COLS = [
    "Lens ID", "Title", "Abstract", "Applicants", "Publication Date",
    "Jurisdiction", "CPC Classifications", "IPCR Classifications", "URL"
]

def infer_from_filename(path):
    m = PATTERN.search(os.path.basename(path))
    if not m:
        return None, None
    theme = m.group("theme").lower()
    juris = m.group("jurisdiction").upper()
    return theme, juris

def parse_args():
    p = argparse.ArgumentParser(description="Merge CSVs under a directory into one file (streaming).")
    p.add_argument("--input-dir", default="./data", help="Directory containing CSVs")
    p.add_argument("--pattern", default="*.csv", help="Glob pattern under input-dir")
    p.add_argument("--output", default="merged_patents.csv.gz", help="Output .csv.gz or .parquet")
    p.add_argument("--select-cols", default=",".join(DEFAULT_COLS), help="Columns to keep (comma-separated)")
    p.add_argument("--chunksize", type=int, default=200_000, help="Rows per read_csv chunk")
    p.add_argument("--encoding", default="utf-8", help="Input CSV encoding")
    p.add_argument("--low-memory", action="store_true", help="Enable pandas low_memory")
    p.add_argument("--add-source-col", default="SourceFile", help="Column to store source filename (blank to disable)")
    return p.parse_args()

def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        print(f"[ERROR] No files found in {args.input_dir}/{args.pattern}", file=sys.stderr)
        sys.exit(1)

    use_cols = [c.strip() for c in args.select_cols.split(",") if c.strip()]
    out_is_parquet = args.output.lower().endswith(".parquet")
    header_written = False
    total = 0

    # Parquetは逐次追記が難しいため、まずはCSV.gz推奨
    if out_is_parquet:
        tmp_csv = args.output + ".tmp.csv.gz"
        out_path = tmp_csv
        print(f"[INFO] Writing stream to {tmp_csv} and will convert to Parquet at the end.")
    else:
        out_path = args.output

    for fp in files:
        theme, juris = infer_from_filename(fp)
        if theme is None:
            print(f"[WARN] Filename not matched, skipping Theme/Jurisdiction auto-tag: {fp}")

        try:
            reader = pd.read_csv(fp, chunksize=args.chunksize, encoding=args.encoding, low_memory=args.low_memory)
        except Exception as e:
            print(f"[WARN] Failed to open {fp}: {e}")
            continue

        for chunk in reader:
            # 主要列が無ければ空で追加して揃える
            for col in use_cols:
                if col not in chunk.columns:
                    chunk[col] = pd.NA
            df = chunk[use_cols].copy()

            # ファイル名から Theme / Jurisdiction を補完
            if "Theme" not in df.columns:
                df["Theme"] = theme
            else:
                df["Theme"] = df["Theme"].fillna(theme)

            if "Jurisdiction" not in df.columns:
                df["Jurisdiction"] = juris
            else:
                df["Jurisdiction"] = df["Jurisdiction"].fillna(juris)

            # ソースファイル名
            if args.add_source_col:
                df[args.add_source_col] = os.path.basename(fp)

            # write
            df.to_csv(out_path,
                      mode="a",
                      index=False,
                      header=(not header_written),
                      encoding="utf-8",
                      compression="gzip" if out_path.endswith(".gz") else None)
            header_written = True
            total += len(df)

        print(f"[INFO] Merged {os.path.basename(fp)}")

    print(f"[DONE] rows={total}, output={out_path}")

    # convert to Parquet if requested
    if out_is_parquet:
        print("[INFO] Converting to Parquet...")
        full = pd.read_csv(out_path, low_memory=False)
        full.to_parquet(args.output, index=False)
        os.remove(out_path)
        print(f"[DONE] Parquet written: {args.output} (rows={len(full)})")

if __name__ == "__main__":
    main()
