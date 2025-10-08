# fit_axes.py
# -*- coding: utf-8 -*-
"""
英語（または英語相当のサブセット）で軸をfitし、全件を同座標系にtransformする。
TF-IDF -> TruncatedSVD(圧縮) -> UMAP の省メモリ構成。
- fit は英語サブセットから最大 N サンプル（--fit-sample-size）で実施
- 全件を transform
- 座標は Parquet で出力、ベクトライザ/SVD/UMAP は joblib 保存

依存:
  pip install pandas scikit-learn umap-learn joblib
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import umap
import joblib

def parse_args():
    p = argparse.ArgumentParser(description="Fit UMAP axes on English subset (or assume-English) then transform all.")
    p.add_argument("--input", required=True, help="入力: CSV/CSV.GZ/Parquet")
    p.add_argument("--output_mapped", default="mapped.parquet", help="出力: 座標付きParquet")
    p.add_argument("--model_out", default="axes_model.joblib", help="出力: 学習済みモデル(joblib)")

    p.add_argument("--id_col", default="Lens ID", help="ID列名（なければ行番号）")
    p.add_argument("--title_col", default="Title", help="タイトル列（任意）")
    p.add_argument("--text_cols", default="Abstract,Title",
                   help="結合して学習に使うテキスト列（カンマ区切り）")
    p.add_argument("--assume_english", action="store_true",
                   help="全件を英語として扱う（検索で既に英語絞り込み済みの想定）")
    p.add_argument("--lang_col", default=None, help="言語列（en/eng/english を英語として扱う）")

    # TF-IDF
    p.add_argument("--max_features", type=int, default=50000)
    p.add_argument("--min_df", type=float, default=2)
    p.add_argument("--max_df", type=float, default=0.95)
    p.add_argument("--ngram_min", type=int, default=1)
    p.add_argument("--ngram_max", type=int, default=2)

    # SVD
    p.add_argument("--svd_dim", type=int, default=256)

    # UMAP
    p.add_argument("--n_components", type=int, default=2)
    p.add_argument("--n_neighbors", type=int, default=30)
    p.add_argument("--min_dist", type=float, default=0.05)
    p.add_argument("--metric", default="cosine")
    p.add_argument("--random_state", type=int, default=42)

    # Fit サンプル数（None なら英語サブセット全件）
    p.add_argument("--fit_sample_size", type=int, default=20000)

    return p.parse_args()

def read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)

def build_text(df: pd.DataFrame, text_cols):
    cols = [c for c in text_cols if c in df.columns]
    if not cols:
        raise ValueError(f"テキスト列が見つかりません: 指定={text_cols} / 実列={list(df.columns)[:30]}")
    # 空→""にして結合
    series_list = [df[c].fillna("").astype(str) for c in cols]
    text = series_list[0]
    for s in series_list[1:]:
        text = text.str.cat(s, sep=" ")
    return text

def main():
    args = parse_args()
    try:
        df = read_any(args.input)
    except Exception as e:
        print("[ERROR] 入力ファイル読み込みに失敗:", e, file=sys.stderr)
        sys.exit(1)

    # テキスト準備
    text_cols = [c.strip() for c in args.text_cols.split(",") if c.strip()]
    text = build_text(df, text_cols)

    # 英語サブセット判定
    if args.assume_english:
        is_en = np.ones(len(df), dtype=bool)
    elif args.lang_col and args.lang_col in df.columns:
        lang_norm = df[args.lang_col].astype(str).str.strip().str.lower()
        is_en = lang_norm.isin({"en", "eng", "english"})
    else:
        # 明示列が無い場合は全件をfit対象に（ただし sample で絞る想定）
        is_en = np.ones(len(df), dtype=bool)

    en_idx = np.where(is_en)[0]
    if len(en_idx) == 0:
        print("[ERROR] 英語サブセットが空です。--assume_english か --lang_col を見直してください。", file=sys.stderr)
        sys.exit(1)

    # Fit 用サンプル
    if args.fit_sample_size and len(en_idx) > args.fit_sample_size:
        rng = np.random.RandomState(args.random_state)
        fit_idx = rng.choice(en_idx, size=args.fit_sample_size, replace=False)
    else:
        fit_idx = en_idx

    print(f"[INFO] Fit rows: {len(fit_idx)} / All rows: {len(df)}")

    # ベクトル化 & 次元圧縮
    vect = TfidfVectorizer(
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_range=(args.ngram_min, args.ngram_max),
        lowercase=True,
        stop_words="english"
    )
    svd = TruncatedSVD(n_components=args.svd_dim, random_state=args.random_state)

    # Fit (TF-IDF, SVD)
    X_fit = vect.fit_transform(text.iloc[fit_idx].values)
    Z_fit = svd.fit_transform(X_fit)

    # Fit UMAP
    reducer = umap.UMAP(
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state,
        low_memory=True,
        verbose=False
    )
    _ = reducer.fit_transform(Z_fit)

    # Transform ALL
    X_all = vect.transform(text.values)
    Z_all = svd.transform(X_all)
    emb_all = reducer.transform(Z_all)

    # 出力DF
    if args.id_col in df.columns:
        id_series = df[args.id_col]
        id_name = args.id_col
    else:
        id_series = pd.Series(df.index, name="RowID")
        id_name = "RowID"

    out = pd.DataFrame({
        id_name: id_series,
        "x": emb_all[:, 0],
        "y": emb_all[:, 1] if emb_all.shape[1] > 1 else np.zeros(len(df))
    })

    # 可視化に使う列を持ち出し
    for col in ["Theme", "Jurisdiction", "Applicants", "Publication Date", args.title_col]:
        if col and col in df.columns:
            out[col] = df[col]

    # 保存
    out.to_parquet(args.output_mapped, index=False)
    joblib.dump({"vectorizer": vect, "svd": svd, "umap": reducer}, args.model_out)

    print(f"[DONE] mapped -> {args.output_mapped}")
    print(f"[DONE] model  -> {args.model_out}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR]", e, file=sys.stderr)
        sys.exit(1)
