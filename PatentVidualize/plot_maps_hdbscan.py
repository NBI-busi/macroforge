# plot_maps_hdbscan.py
# -*- coding: utf-8 -*-
"""
全データで HDBSCAN によりクラスタリング（配色固定）→ 5年ごとに同じ配色で出力。
- 軸は非表示オプション (--hide-axes)
- 各クラスタの代表キーワードを凡例/図中ラベルに表示 (--label-keywords)
- 密度アウトライン（等高線）で“有機的な境界”を描画 (--cluster-outline density)

使い方（例）:
  python plot_maps_hdbscan.py \
    --input-mapped mapped.parquet \
    --outdir ./figs_hdbscan \
    --period-width 5 --year-min 2006 --year-max 2025 \
    --min-cluster-size 80 \
    --label-keywords 5 --text-col Title \
    --hide-axes --show-context \
    --cluster-outline density --outline-level 0.25
"""

import argparse, os, re, sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import hdbscan
except Exception:
    print("[ERROR] hdbscan が見つかりません。`pip install hdbscan` を実行してください。", file=sys.stderr)
    sys.exit(1)

try:
    from scipy.stats import gaussian_kde
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False

# --------- CLI ---------
def parse_args():
    p = argparse.ArgumentParser(description="HDBSCAN-based global clustering, split by 5-year periods.")
    p.add_argument("--input-mapped", default="mapped.parquet")
    p.add_argument("--outdir", default="./figs_hdbscan")
    p.add_argument("--dot-size", type=float, default=6.0)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--dpi", type=int, default=200)

    # 期間設定
    p.add_argument("--period-width", type=int, default=5)
    p.add_argument("--year-min", type=int, default=None)
    p.add_argument("--year-max", type=int, default=None)

    # 表示
    p.add_argument("--hide-axes", action="store_true")
    p.add_argument("--show-context", action="store_true", help="背景に全データを薄く描く")

    # HDBSCAN
    p.add_argument("--min-cluster-size", type=int, default=80)
    p.add_argument("--min-samples", type=int, default=None, help="未指定なら自動（None）")
    p.add_argument("--metric", default="euclidean")

    # 代表キーワード
    p.add_argument("--label-keywords", type=int, default=5, help="各クラスタの代表語数（0で無効）")
    p.add_argument("--text-col", default="Title", help="キーワード抽出に使う列（Title/Abstract など）")

    # アウトライン
    p.add_argument("--cluster-outline", choices=["none", "density"], default="none")
    p.add_argument("--outline-level", type=float, default=0.25, help="密度最大値に対する比率(0~1)")
    p.add_argument("--outline-min-points", type=int, default=120)
    p.add_argument("--outline-grid", type=int, default=200)

    return p.parse_args()

# --------- IO ---------
def read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)

def ensure_year_series(df: pd.DataFrame, col="Publication Date") -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce").dt.year

def build_periods(years: pd.Series, width: int, y_min: int=None, y_max: int=None) -> List[Tuple[int,int]]:
    yrs = years.dropna().astype(int)
    if yrs.empty:
        return []
    start = y_min if y_min is not None else int(yrs.min())
    end   = y_max if y_max is not None else int(yrs.max())
    if start > end:
        start, end = end, start
    out = []
    a = start
    while a <= end:
        b = min(a + width - 1, end)
        out.append((a, b))
        a = b + 1
    return out

def pick_palette(n: int):
    cmap = plt.cm.tab20(np.linspace(0,1,20))
    return [cmap[i % 20] for i in range(n)]

# --------- Keywords ---------
def extract_cluster_keywords(df: pd.DataFrame, labels: np.ndarray, text_col: str, topk: int) -> Dict[int, List[str]]:
    if topk <= 0 or text_col not in df.columns:
        return {}
    texts = df[text_col].fillna("").astype(str).values
    if not any(len(t.strip()) > 0 for t in texts):
        return {}
    vect = TfidfVectorizer(stop_words="english", lowercase=True,
                           max_features=20000, ngram_range=(1,2), min_df=2, max_df=0.98)
    try:
        X = vect.fit_transform(texts)
    except Exception:
        return {}
    vocab = np.array(vect.get_feature_names_out())
    labs = np.array(labels)
    out: Dict[int, List[str]] = {}
    for cl in sorted(np.unique(labs)):
        if cl == -1:
            continue
        m = (labs == cl)
        if not np.any(m):
            continue
        mean_vec = X[m].mean(axis=0).A1
        idx = np.argsort(mean_vec)[::-1][:topk]
        words = [vocab[i] for i in idx if mean_vec[i] > 0]
        out[cl] = words
    return out

def annotate_cluster_labels(ax, sub: pd.DataFrame, labels: np.ndarray, keywords: Dict[int, List[str]], topk: int):
    labs = np.array(labels)
    for cl in sorted(np.unique(labs)):
        if cl == -1:  # ノイズはスキップ
            continue
        m = (labs == cl)
        if not np.any(m):
            continue
        cx = float(sub.loc[m, "x"].mean())
        cy = float(sub.loc[m, "y"].mean())
        kws = keywords.get(cl, [])
        text = f"C{cl}" + (": " + ", ".join(kws[:topk]) if kws else "")
        ax.text(cx, cy, text, fontsize=8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85))

# --------- Outline ---------
def draw_density_outline(ax, pts_xy: np.ndarray, level_ratio: float = 0.25, grid: int = 200, color=(0,0,0,0.35)):
    if not _SCIPY_OK or len(pts_xy) < 5:
        return
    try:
        kde = gaussian_kde(pts_xy.T)
    except Exception:
        return
    xmin, ymin = pts_xy.min(axis=0)
    xmax, ymax = pts_xy.max(axis=0)
    pad_x = (xmax - xmin) * 0.1 if xmax > xmin else 1.0
    pad_y = (ymax - ymin) * 0.1 if ymax > ymin else 1.0
    xmin -= pad_x; xmax += pad_x; ymin -= pad_y; ymax += pad_y
    xs = np.linspace(xmin, xmax, grid)
    ys = np.linspace(ymin, ymax, grid)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    level = ZZ.max() * max(min(level_ratio, 0.95), 0.01)
    ax.contour(XX, YY, ZZ, levels=[level], colors=[color], linewidths=1.2)

# --------- main ---------
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 読み込み
    try:
        df = read_any(args.input_mapped)
    except Exception as e:
        print("[ERROR] Failed to read input:", e, file=sys.stderr)
        sys.exit(1)
    if not {"x","y"}.issubset(df.columns):
        print("[ERROR] 'x' and 'y' are required in mapped file.", file=sys.stderr)
        sys.exit(1)
    if "Publication Date" not in df.columns:
        print("[ERROR] 'Publication Date' column is required for period split.", file=sys.stderr)
        sys.exit(1)

    # 統一軸範囲
    x_min, x_max = float(df["x"].min()), float(df["x"].max())
    y_min, y_max = float(df["y"].min()), float(df["y"].max())
    x_pad = (x_max - x_min) * 0.02 if x_max > x_min else 1.0
    y_pad = (y_max - y_min) * 0.02 if y_max > y_min else 1.0
    x_min_plot, x_max_plot = x_min - x_pad, x_max + x_pad
    y_min_plot, y_max_plot = y_min - y_pad, y_max + y_pad

    # 全体 HDBSCAN（配色固定）
    X2 = df[["x","y"]].values.astype(float)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size,
                                min_samples=args.min_samples,
                                metric=args.metric)
    labels_all = clusterer.fit_predict(X2)

    uniq = sorted(set(labels_all))
    palette = pick_palette(len([u for u in uniq if u != -1]))
    cmap = {}
    pi = 0
    for u in uniq:
        if u == -1:
            cmap[u] = (0.7,0.7,0.7,0.35)
        else:
            cmap[u] = palette[pi]; pi += 1

    # 代表キーワード（全体で計算 → 各期間で表示）
    keywords = extract_cluster_keywords(df, labels_all, args.text_col, args.label_keywords)

    # 5年ごとのグループを作成
    years = ensure_year_series(df)
    periods = build_periods(years, args.period_width, args.year_min, args.year_max)
    if not periods:
        print("[WARN] No valid periods.", file=sys.stderr)
        sys.exit(0)

    # 各期間図を出力（配色/キーワードは全体に固定）
    for (ys, ye) in periods:
        mask = (years >= ys) & (years <= ye)
        sub = df.loc[mask].copy()
        if sub.empty:
            continue
        labs = labels_all[mask.values]  # 同じ順序で対応

        fig, ax = plt.subplots(figsize=(10,8))
        if args.show_context:
            ax.scatter(df["x"], df["y"], s=max(args.dot_size-2,1), alpha=0.1, c=[(0.6,0.6,0.6,0.1)])

        # 各クラスタを同じ色で描く
        for u in sorted(set(labs)):
            m = (labs == u)
            ax.scatter(sub.loc[m, "x"], sub.loc[m, "y"], s=args.dot_size, alpha=args.alpha,
                       c=[cmap[u]], label=(f"C{u}" if u!=-1 else "Noise"))

            # 密度アウトライン
            if args.cluster_outline == "density" and u != -1 and np.count_nonzero(m) >= args.outline_min_points:
                pts = sub.loc[m, ["x","y"]].values.astype(float)
                draw_density_outline(ax, pts, level_ratio=args.outline_level,
                                     grid=args.outline_grid, color=cmap[u])

        # ラベル（クラスタ中心＋代表語）
        if args.label_keywords > 0:
            for u in sorted(set(labs)):
                if u == -1: 
                    continue
                m = (labs == u)
                if not np.any(m):
                    continue
                cx = float(sub.loc[m, "x"].mean())
                cy = float(sub.loc[m, "y"].mean())
                kws = keywords.get(u, [])
                txt = f"C{u}" + (": " + ", ".join(kws[:args.label_keywords]) if kws else "")
                ax.text(cx, cy, txt, fontsize=8, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85))

        # 凡例（キーワード入り）
        handles, labels = ax.get_legend_handles_labels()
        newlabels = []
        for L in labels:
            if L.startswith("C"):
                try:
                    cid = int(L[1:])
                    kws = keywords.get(cid, [])
                    if kws:
                        L = f"{L}: {', '.join(kws[:args.label_keywords])}"
                except Exception:
                    pass
            newlabels.append(L)
        ax.legend(handles, newlabels, fontsize=8, ncol=2, frameon=True, title="Clusters")

        # 軸
        ax.set_xlim(x_min_plot, x_max_plot)
        ax.set_ylim(y_min_plot, y_max_plot)
        if args.hide_axes:
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

        ax.set_title(f"UMAP  HDBSCAN (global)  Years {ys}-{ye}")
        ax.set_xlabel("" if args.hide_axes else "x")
        ax.set_ylabel("" if args.hide_axes else "y")

        safe = f"{ys}-{ye}"
        out = os.path.join(args.outdir, f"period_{safe}__hdbscan_global.png")
        fig.tight_layout(); fig.savefig(out, dpi=args.dpi); plt.close(fig)
        print(f"[DONE] {out}  (n={len(sub)})")

    print("[ALL DONE]")

if __name__ == "__main__":
    main()
