import os
import json
import requests
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ====== 設定 ======
# 環境変数 BLS_API_KEY があれば自動で使います。未設定なら下の"**********"を書き換え。
API_KEY = os.getenv("BLS_API_KEY", "**********")

# 取得対象シリーズID（季節調整済み / CES 系列）
SERIES_MAP = {
    "CES0000000001": "total_nonfarm_sa_thousands",        # 全体（非農業雇用者数）
    "CES3000000001": "manufacturing_sa_thousands",        # 製造業
    "CES3133600001": "transport_equip_mfg_sa_thousands",  # 輸送用機器製造
    "CES3133600101": "motor_vehicles_parts_sa_thousands", # 自動車・同部品
    # 任意: 労働力人口（CPS）
    # "LNS11000000": "civilian_labor_force_sa_thousands",
}

# 何年分取得するか（20年が基本）
YEARS_BACK = 20
# さらにさかのぼれる限り（APIの20年制限をまたいでフルヒストリーを取りに行く）
FULL_HISTORY = False  # True にすると可能な限りさかのぼって収集

# ====== BLS API 呼び出し ======
def fetch_bls_window(series_ids, start_year, end_year):
    """start_year〜end_year の1窓分（最大20年）を取得"""
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    headers = {"Content-type": "application/json"}
    payload = {
        "seriesid": list(series_ids),
        "startyear": str(start_year),
        "endyear": str(end_year),
    }
    # APIキーを設定していれば付与
    if API_KEY and API_KEY != "**********":
        payload["registrationkey"] = API_KEY

    r = requests.post(url, data=json.dumps(payload), headers=headers, timeout=60)
    r.raise_for_status()
    j = r.json()
    if j.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS API error: {j.get('message')}")
    return j["Results"]["series"]

def normalize_to_long(series_blocks):
    """BLS応答を tidy (long) 形式に整形"""
    rows = []
    for s in series_blocks:
        sid = s["seriesID"]
        for obs in s["data"]:
            period = obs.get("period", "")
            if not period.startswith("M"):  # 年平均(M13)などは除外
                continue
            rows.append({
                "seriesID": sid,
                "year": int(obs["year"]),
                "period": period,  # "M01" .. "M12"
                "value": float(obs["value"]),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["year"].astype(str) + df["period"].str[1:], format="%Y%m")
    df = df.loc[:, ["date", "seriesID", "value"]].sort_values("date")
    return df

def fetch_bls(series_ids, end_year, years_back=20, full_history=False):
    """
    series_ids: 取得したい系列IDのリスト
    end_year:   取得の最終年（例: 2025）
    years_back: 直近 N 年（既定20）
    full_history=True で可能な限り過去まで 20年窓を繰り返し取得
    """
    all_long = []

    # まず直近 years_back 年（API上限に収まるよう inclusiveに20年）
    start_year = max(1939, end_year - (years_back - 1))
    blocks = fetch_bls_window(series_ids, start_year, end_year)
    df_long = normalize_to_long(blocks)
    all_long.append(df_long)

    if full_history:
        # さらに過去へ 20年ずつ窓をずらして取得
        current_end = start_year - 1
        while current_end >= 1939:  # CESの多くは1939年頃まで遡及
            current_start = max(1939, current_end - 19)  # inclusive 20年
            blocks = fetch_bls_window(series_ids, current_start, current_end)
            df_chunk = normalize_to_long(blocks)
            if df_chunk.empty:
                break
            all_long.append(df_chunk)
            current_end = current_start - 1

    # 結合 & 重複排除
    df_all = pd.concat(all_long, ignore_index=True) if len(all_long) > 1 else df_long
    if df_all.empty:
        return df_all
    df_all = df_all.drop_duplicates(subset=["date", "seriesID"]).sort_values("date")
    return df_all

def wide_pivot(df_long, rename_map=None):
    if df_long.empty:
        return df_long
    wide = df_long.pivot_table(index="date", columns="seriesID", values="value", aggfunc="first").sort_index()
    if rename_map:
        wide = wide.rename(columns=rename_map)
    return wide.reset_index()

if __name__ == "__main__":
    end_year = date.today().year  # 実行年に自動追随
    df_long = fetch_bls(SERIES_MAP.keys(), end_year=end_year, years_back=YEARS_BACK, full_history=FULL_HISTORY)
    df_wide = wide_pivot(df_long, rename_map=SERIES_MAP)

    out_csv = f"bls_employment_{'full' if FULL_HISTORY else f'{YEARS_BACK}y'}.csv"
    df_wide.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_csv}")
    print(df_wide.tail())

OUTPUT_DIR = "charts"
PLOT_NORMALIZED = True   # 各系列を最初の有効値=100に正規化して重ね描き
PLOT_YOY = True          # 各系列の前年比(%)も出力

os.makedirs(OUTPUT_DIR, exist_ok=True)

# date を index に
plot_df = df_wide.copy()
plot_df["date"] = pd.to_datetime(plot_df["date"])
plot_df = plot_df.set_index("date").sort_index()

# ====== 直近N年だけにウィンドウを絞る ======
LAST_YEARS = 20  # ← ここを変えれば年数を調整できます

if LAST_YEARS is not None:
    latest = plot_df.index.max()
    # 例: 最新が 2025-09 のとき start_date は 2005-09
    start_date = pd.Timestamp(latest.year - LAST_YEARS, latest.month, 1)
    plot_df = plot_df.loc[plot_df.index >= start_date].copy()
# ==========================================


# 系列名一覧（date除く）
series_cols = [c for c in plot_df.columns if c != "date"]

def style_year_axis(ax, start, end, major_every=5):
    """X軸を年表示に統一（5年おきの主目盛り、年ラベル、範囲固定）"""
    ax.set_xlim(start, end)
    ax.set_xlabel("Year")  # 軸タイトルを Year に
    ax.xaxis.set_major_locator(mdates.YearLocator(base=major_every))  # 5年おき
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))          # 西暦表示
    ax.xaxis.set_minor_locator(mdates.YearLocator())                  # 任意：毎年を副目盛り


def save_single_series_plot(s, title, ylabel, fname):
    fig, ax = plt.subplots(figsize=(10, 5))

    s = s.copy()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None)
    s = s.sort_index()

    s.plot(ax=ax)
    start = s.index.min()
    end   = s.index.max()

    ax.set_title(f"{title}  [{start.strftime('%Y-%m')} – {end.strftime('%Y-%m')}]")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", linestyle="--", alpha=0.4)

    # ← ここを追加
    style_year_axis(ax, start, end, major_every=5)

    fig.tight_layout()
    outpath_png = os.path.join(OUTPUT_DIR, f"{fname}.png")
    outpath_svg = os.path.join(OUTPUT_DIR, f"{fname}.svg")
    fig.savefig(outpath_png, dpi=150, bbox_inches="tight")
    fig.savefig(outpath_svg, bbox_inches="tight")
    plt.close(fig)
    return [outpath_png, outpath_svg]

saved_files = []

# 1) 各系列の「水準」グラフ（千人）
for col in series_cols:
    s = plot_df[col].dropna()
    if s.empty:
        continue
    saved_files += save_single_series_plot(
        s=s,
        title=f"{col} (Level)",
        ylabel="Thousands of persons",
        fname=f"{col}_level"
    )

# 2) 各系列の前年比(%)グラフ（任意）
if PLOT_YOY:
    for col in series_cols:
        s = plot_df[col].dropna()
        if s.empty or len(s) < 13:
            continue
        yoy = s.pct_change(12) * 100.0
        yoy = yoy.dropna()
        if yoy.empty:
            continue
        saved_files += save_single_series_plot(
            s=yoy,
            title=f"{col} (YoY %)",
            ylabel="Percent",
            fname=f"{col}_yoy"
        )

# 3) 正規化（最初の有効値=100）での重ね描き（任意）
if PLOT_NORMALIZED and len(series_cols) >= 2:
    # 各列ごとに最初の有効値で割って *100
    norm_df = plot_df.copy()
    for col in series_cols:
        s = norm_df[col].dropna()
        if s.empty:
            norm_df[col] = pd.NA
            continue
        base = s.iloc[0]
        norm_df[col] = (norm_df[col] / base) * 100.0
    norm_df = norm_df[series_cols].dropna(how="all")

    if not norm_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        norm_df.plot(ax=ax)  # 複数系列の重ね描き（色指定なし）
        style_year_axis(ax, norm_df.index.min(), norm_df.index.max(), major_every=5)
        ax.set_xlim(norm_df.index.min(), norm_df.index.max())
        #ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
        #ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_title("Normalized to 100 (first available value)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Index = 100")
        ax.grid(True, which="major", linestyle="--", alpha=0.4)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        fig.tight_layout()
        outpath_png = os.path.join(OUTPUT_DIR, "normalized_overlay.png")
        outpath_svg = os.path.join(OUTPUT_DIR, "normalized_overlay.svg")
        fig.savefig(outpath_png, dpi=150, bbox_inches="tight")
        fig.savefig(outpath_svg, bbox_inches="tight")
        plt.close(fig)
        saved_files += [outpath_png, outpath_svg]

print("Saved charts:")
for f in saved_files:
    print(" -", f)
# ====== グラフ生成 ここまで ======
