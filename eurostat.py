# pip install eurostat pandas matplotlib
import re
import argparse
import os
import eurostat as es
import pandas as pd
import matplotlib.pyplot as plt

# 主要国 + 中東欧の自動車工場国を初期セット
DEFAULT_GEOS = [
    "EU27_2020", "EA20",  # EU27（現行構成）とユーロ圏20
    "DE", "FR", "IT", "ES", "PL", "CZ", "SK", "HU", "RO"
]

def to_tidy(df, start_year=2000, agg="sum"):
    """
    Eurostatの '年が横持ち' / '縦持ち' の両方に対応し、geo×year/time で重複を畳み込む。
    """
    if df is None or df.empty:
        return pd.DataFrame()

    geo_col = next((c for c in df.columns if "geo" in c.lower()), None)
    if geo_col is None:
        raise ValueError("geo列が見つかりません。返り値の列名を一度 print して確認してください。")

    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]

    if year_cols:
        g = df.melt(id_vars=[geo_col], value_vars=year_cols,
                    var_name="year", value_name="values")
        g["year"] = pd.to_numeric(g["year"], errors="coerce")
        g["values"] = pd.to_numeric(g["values"], errors="coerce")
        g = g.dropna(subset=["year", "values"])
        g = g[g["year"] >= start_year]
        g = g.groupby([geo_col, "year"], as_index=False)["values"].agg(agg)
        g["time"] = pd.to_datetime(g["year"].astype(int), format="%Y")
        out = g.pivot(index="time", columns=geo_col, values="values").sort_index()
        return out

    elif "time" in df.columns:
        g = df[[geo_col, "time", "values"]].copy()
        g["time"] = pd.to_datetime(g["time"])
        g["values"] = pd.to_numeric(g["values"], errors="coerce")
        g = g.dropna(subset=["time", "values"])
        g = g[g["time"].dt.year >= start_year]
        g = g.groupby([geo_col, "time"], as_index=False)["values"].agg(agg)
        out = g.pivot(index="time", columns=geo_col, values="values").sort_index()
        return out

    else:
        raise ValueError("年列（YYYY）も 'time' 列も見つかりません。")

def fetch_total_employment(geos, start_year=2000):
    """総就業者（千人, 年次）"""
    df = es.get_data_df(
        "nama_10_pe",
        filter_pars={"geo": geos, "na_item": "EMP_DC", "unit": "THS_PER"},
        flags=False
    )
    return to_tidy(df, start_year)

def fetch_industry_employment(nace_code, geos, start_year=2000):
    """産業別就業者（千人, 年次, NACE A*64）"""
    def _get(nace):
        return es.get_data_df(
            "nama_10_a64_e",
            filter_pars={"geo": geos, "nace_r2": nace, "unit": "THS_PER", "na_item": "EMP_DC"},
            flags=False
        )

    df = pd.DataFrame()
    # 1) まず複合コードを試す
    try:
        if nace_code in ("C29-30", "C29_C30"):
            df = _get("C29_C30")
        else:
            df = _get(nace_code)
    except Exception:
        df = pd.DataFrame()

    # 2) 取れない場合は C29 と C30 の和で代替
    if (df is None) or df.empty:
        if nace_code in ("C29-30", "C29_C30"):
            d29 = _get("C29")
            d30 = _get("C30")
            df29 = to_tidy(d29, start_year) if d29 is not None and not d29.empty else pd.DataFrame()
            df30 = to_tidy(d30, start_year) if d30 is not None and not d30.empty else pd.DataFrame()
            if df29.empty and df30.empty:
                raise ValueError("C29_C30 も C29/C30 もデータが見当たりません。")
            out = df29.add(df30, fill_value=0)
            return out.sort_index()
        else:
            raise ValueError(f"No data for nace_r2={nace_code}")

    return to_tidy(df, start_year)

def plot_and_save(df, title, ylabel, outpath, dpi=180):
    """df: index=time, columns=geo。複数系列を1枚に描画して保存。"""
    if df is None or df.empty:
        print(f"[WARN] No data to plot for: {title}")
        return
    plt.figure(figsize=(11, 6))
    # 欠損が多い系列は自動で飛ばさず、そのまま描く（ギャップは切れ目）
    for col in df.columns:
        try:
            plt.plot(df.index, df[col], label=col)
        except Exception as e:
            print(f"[WARN] Skip {col} due to {e}")
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=dpi)
    plt.close()
    print(f"[SAVE] {outpath}")

def main():
    parser = argparse.ArgumentParser(description="EU雇用（総計/製造業/自動車・輸送機械）を取得・保存")
    parser.add_argument("--geos", type=str, default=",".join(DEFAULT_GEOS),
                        help="カンマ区切りの地域コード（例：EU27_2020,EA20,DE,FR,IT,ES,PL,CZ,SK,HU,RO）")
    parser.add_argument("--start-year", type=int, default=2000, help="抽出開始年（デフォルト: 2000）")
    parser.add_argument("--outdir", type=str, default="figs_eu_jobs", help="画像の保存先ディレクトリ")
    parser.add_argument("--dpi", type=int, default=180, help="保存DPI（デフォルト: 180）")
    args = parser.parse_args()

    geos = [g.strip() for g in args.geos.split(",") if g.strip()]
    start_year = args.start_year
    outdir = args.outdir
    dpi = args.dpi

    # 取得
    emp_total = fetch_total_employment(geos, start_year)
    emp_manu  = fetch_industry_employment("C", geos, start_year)          # 製造業
    emp_auto  = fetch_industry_employment("C29_C30", geos, start_year)    # 自動車・その他輸送機械

    # 3枚の図を保存
    plot_and_save(emp_total,
                  title="Employment (Total) — thousand persons",
                  ylabel="Thousand persons",
                  outpath=os.path.join(outdir, "employment_total.png"),
                  dpi=dpi)

    plot_and_save(emp_manu,
                  title="Employment (Manufacturing: NACE C) — thousand persons",
                  ylabel="Thousand persons",
                  outpath=os.path.join(outdir, "employment_manufacturing.png"),
                  dpi=dpi)

    plot_and_save(emp_auto,
                  title="Employment (Motor vehicles & other transport: NACE C29–C30) — thousand persons",
                  ylabel="Thousand persons",
                  outpath=os.path.join(outdir, "employment_motor_transport.png"),
                  dpi=dpi)

    # 進捗出力（末尾3行）
    def tail_print(name, d):
        try:
            print(f"\n[{name}] tail:\n{d.tail(3)}")
        except Exception:
            pass

    tail_print("Total", emp_total)
    tail_print("Manufacturing (C)", emp_manu)
    tail_print("Motor & Transport (C29–C30)", emp_auto)

if __name__ == "__main__":
    main()
