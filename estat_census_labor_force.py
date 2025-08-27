# -*- coding: utf-8 -*-
"""
estat_census_labor_force.py  (robust v3)

e-Stat「国勢調査 時系列データ（人口の労働力状態，就業者の産業・職業）」から、
最近30年（1995-2020）の労働力関連データを取得してグラフ化。

出力:
  - output/census_overall_labor_force.csv, .png
  - output/census_age_labor_force_5yr.csv, .png
  - output/census_industry_employment.csv, .png
"""
from __future__ import annotations
import os, sys, re, time, argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import requests
import pandas as pd
import matplotlib.pyplot as plt

# ====== 設定 ======
API_BASE = "https://api.e-stat.go.jp/rest/3.0/app/json"

# 国勢調査 時系列（人口の労働力状態，就業者の産業・職業）
SID_OVERALL = "0003412175"  # 表1：労働力状態（3区分）など
SID_AGE     = "0003410394"  # 表3：年齢（5歳階級）
SID_IND     = "0003410395"  # 表4：産業（大分類）

TARGET_YEARS = [1995, 2000, 2005, 2010, 2015, 2020]
EXCLUDE_TIME_SUBSTR = ["不詳補完値"]

# 可視化
JP_FONTS = ["Meiryo", "Yu Gothic", "Noto Sans CJK JP", "IPAexGothic", "TakaoPGothic"]
DEBUG_PRINT = True  # Trueで年コードや系列の検出結果を出力

# ====== ユーティリティ ======
def ensure_outdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_app_id() -> str:
    env_keys = ["ESTAT_APP_ID", "ESTAT_API_KEY", "ESTAT_APPKEY"]
    for k in env_keys:
        v = os.getenv(k)
        if v:
            return v.strip()
    # .streamlit/secrets.toml / secrets.toml から拾う
    for fp in (Path(".streamlit/secrets.toml"), Path("secrets.toml")):
        if fp.exists():
            try:
                try:
                    import tomllib as toml  # py311+
                except ModuleNotFoundError:
                    import tomli as toml    # pip install tomli
                with open(fp, "rb") as f:
                    cfg = toml.load(f)
                for sec in ("estat_api", "e_stat", "default"):
                    d = cfg.get(sec, {})
                    for k in ("app_id", "api_key", "appId", "estat_app_id", "estat_api_key"):
                        if isinstance(d, dict) and d.get(k):
                            return str(d[k]).strip()
                for k in ("app_id", "api_key", "appId", "estat_app_id", "estat_api_key"):
                    if cfg.get(k):
                        return str(cfg[k]).strip()
            except Exception:
                pass
    raise RuntimeError("e-Stat API appId が見つかりません。ESTAT_APP_ID を設定するか secrets.toml へ記載してください。")

def estat_request(endpoint: str, params: Dict[str, Any], max_retries: int = 3, sleep_sec: float = 1.0) -> Dict[str, Any]:
    url = f"{API_BASE}/{endpoint}"
    last = None
    for i in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            env = data.get("GET_STATS_DATA") or data.get("GET_META_INFO") or data.get("GET_STATS_LIST") or {}
            res = env.get("RESULT", {})
            if res.get("STATUS", 0) != 0:
                raise RuntimeError(f"e-Stat API error STATUS={res.get('STATUS')} MSG={res.get('ERROR_MSG')}")
            return data
        except Exception as e:
            last = e
            time.sleep(sleep_sec * (i + 1))
    raise RuntimeError(f"e-Stat API request failed: {last}")

def get_meta(app_id: str, sid: str) -> Dict[str, Any]:
    data = estat_request("getMetaInfo", {"appId": app_id, "statsDataId": sid, "explanationGetFlg": "N"})
    return data["GET_META_INFO"]["METADATA_INF"]

def get_values(app_id: str, sid: str, qp: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    p = {"appId": app_id, "statsDataId": sid, "metaGetFlg": "N", "cntGetFlg": "N", "limit": 100000, **qp}
    data = estat_request("getStatsData", p)
    payload = data["GET_STATS_DATA"]
    next_key = payload.get("NEXT_KEY")
    values = payload.get("STATISTICAL_DATA", {}).get("DATA_INF", {}).get("VALUE", [])
    if isinstance(values, dict):
        values = [values]
    return values, next_key

def get_all_values(app_id: str, sid: str, qp: Dict[str, Any]) -> List[Dict[str, Any]]:
    out, start = [], 1
    while True:
        vals, next_key = get_values(app_id, sid, {**qp, "startPosition": start})
        out.extend(vals)
        if not next_key:
            break
        start = int(next_key)
    return out

# ====== メタ処理 ======
def _collect_classes(meta: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    obj = meta.get("CLASS_INF", {}).get("CLASS_OBJ", [])
    if isinstance(obj, dict):
        obj = [obj]
    return {o.get("@id"): o for o in obj if o.get("@id")}

def _class_items(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = obj.get("CLASS", [])
    if isinstance(items, dict):
        items = [items]
    return items

def find_item_code_by_name_contains(obj: Dict[str, Any], substrs: List[str]) -> Optional[str]:
    for it in _class_items(obj):
        name = it.get("@name", "")
        if all(s in name for s in substrs):
            return it.get("@code")
    return None

def find_codes_by_predicate(obj: Dict[str, Any], pred) -> Dict[str, str]:
    out = {}
    for it in _class_items(obj):
        nm = it.get("@name", "")
        key = pred(nm)
        if key:
            out[key] = it.get("@code")
    return out

# ---- 年コード抽出（まず CLASS_OBJ('time')、次に TIME_INF） ----
ERA_BASE = {"昭和": 1925, "平成": 1988, "令和": 2018}

def _era_to_year(s: str) -> Optional[int]:
    m = re.search(r"(昭和|平成|令和)\s*(\d+)\s*年", s)
    if not m: return None
    return ERA_BASE[m.group(1)] + int(m.group(2))

def _western_year(s: str) -> Optional[int]:
    m = re.search(r"(19|20)\d{2}", s)
    return int(m.group(0)) if m else None

def _pick_time_from_classobj(meta: Dict[str, Any], targets: List[int], excludes: List[str]) -> Dict[str, str]:
    classes = _collect_classes(meta)
    t = classes.get("time")
    if not t:
        return {}
    out, wanted = {}, set(targets)
    for it in _class_items(t):
        nm = it.get("@name", "") or ""
        if any(x in nm for x in excludes):
            continue
        code = it.get("@code")
        if not code:
            continue
        y = _western_year(nm) or _era_to_year(nm)
        if y is None:
            # code から先頭4桁を拾う（例: 2015000000）
            m = re.match(r"^(19|20)\d{2}", str(code))
            y = int(m.group(0)) if m else None
        if y in wanted:
            out[f"{y}年"] = code
    return out

def _pick_time_from_timeinf(meta: Dict[str, Any], targets: List[int], excludes: List[str]) -> Dict[str, str]:
    tinf = meta.get("TIME_INF", {}).get("TIME", [])
    if isinstance(tinf, dict):
        tinf = [tinf]
    out, wanted = {}, set(targets)
    for t in tinf:
        nm = t.get("@name") or t.get("$") or ""
        if any(x in nm for x in excludes):
            continue
        code = t.get("@time") or t.get("@code")
        if not code:
            continue
        y = _western_year(nm) or _era_to_year(nm)
        if y is None:
            m = re.match(r"^(19|20)\d{2}", str(code))
            y = int(m.group(0)) if m else None
        if y in wanted:
            out[f"{y}年"] = code
    return out

def pick_time_codes_auto(meta: Dict[str, Any], targets: List[int], excludes: List[str]) -> Dict[str, str]:
    m = _pick_time_from_classobj(meta, targets, excludes)
    if not m:
        m = _pick_time_from_timeinf(meta, targets, excludes)
    missing = [y for y in targets if f"{y}年" not in m]
    if missing:
        raise RuntimeError(f"時間コード抽出に失敗: {missing} 年が見つかりませんでした。")
    return m

def set_japanese_font():
    for f in JP_FONTS:
        try:
            plt.rcParams["font.family"] = f
            return
        except Exception:
            pass

# ====== 取得 ======
def fetch_overall(app_id: str, out_dir: Path) -> pd.DataFrame:
    meta = get_meta(app_id, SID_OVERALL)
    classes = _collect_classes(meta)

    tab = classes.get("tab")
    area = classes.get("area")
    if not tab or not area:
        raise RuntimeError("表1: tab/area が見つかりません。")

    # 表章項目: 人口（率は除外）
    tab_code = find_item_code_by_name_contains(tab, ["人口"])
    if not tab_code:
        for it in _class_items(tab):
            if "率" not in it.get("@name", ""):
                tab_code = it.get("@code"); break
    if not tab_code:
        raise RuntimeError("表1: 人口系 表章項目コードが特定できません。")

    # 地域: 全国/日本
    cd_area = find_item_code_by_name_contains(area, ["全国"]) or find_item_code_by_name_contains(area, ["日本"])
    if not cd_area:
        raise RuntimeError("表1: 地域『全国/日本』コードが見つかりません。")

    # 性別: 男女計/総数 があれば固定
    sex_cat_id, sex_total = None, None
    for cid, obj in classes.items():
        if cid.startswith("cat"):
            code = find_item_code_by_name_contains(obj, ["男女"]) or \
                   find_item_code_by_name_contains(obj, ["男女計"]) or \
                   find_item_code_by_name_contains(obj, ["総数"]) or \
                   find_item_code_by_name_contains(obj, ["計"])
            if code:
                sex_cat_id, sex_total = cid, code
                break

    # 労働力状態の分類（就業者/完全失業者/労働力人口/非労働力/15歳以上人口 を拾う）
    status_cat_id = None
    for cid, obj in classes.items():
        if cid.startswith("cat"):
            hits = 0
            for it in _class_items(obj):
                nm = it.get("@name", "")
                if any(k in nm for k in ["就業者", "完全失業者", "失業者", "労働力人口", "非労働力", "15歳以上人口"]):
                    hits += 1
            if hits >= 2:
                status_cat_id = cid; break
    if not status_cat_id:
        raise RuntimeError("表1: 労働力状態の分類(catXX)が見つかりません。")
    status_obj = classes[status_cat_id]

    def status_key(nm: str) -> Optional[str]:
        if "15歳以上人口" in nm: return "15歳以上人口"
        if "労働力人口" in nm and "率" not in nm: return "労働力人口"
        if "就業者" in nm and "非就業" not in nm: return "就業者"
        if ("完全失業者" in nm) or ("失業者" in nm and "率" not in nm): return "完全失業者"
        if ("非労働力人口" in nm) or ("非労働力" in nm): return "非労働力人口"
        return None

    status_map = find_codes_by_predicate(status_obj, status_key)
    if DEBUG_PRINT:
        print("[DEBUG] 表1: 検出された系列:", list(status_map.keys()))

    # ---- 追加の分類軸を「1値に固定」する（ここが改善点） ----
    fixed_extra = {}
    for cid, obj in classes.items():
        if not cid.startswith("cat"): 
            continue
        if cid in {status_cat_id, sex_cat_id}:
            continue
        code, name = pick_default_code_and_name(obj)
        # 複数候補があり選べない場合は None のまま -> データが膨張するので明示エラーにする
        if code is None:
            items = _class_items(obj)
            if len(items) > 1:
                raise RuntimeError(f"表1: 追加軸 {cid} に複数候補があり固定できません。例: {[it.get('@name') for it in items][:5]}")
        else:
            fixed_extra[cid] = (code, name)

    if DEBUG_PRINT and fixed_extra:
        show = {cid: name for cid, (code, name) in fixed_extra.items()}
        print("[DEBUG] 表1: 固定した追加軸（catID: 名称）:", show)

    # ---- 時間コード ----
    time_map = pick_time_codes_auto(meta, TARGET_YEARS, EXCLUDE_TIME_SUBSTR)
    if DEBUG_PRINT:
        print("[DEBUG] 表1: 年コード:", time_map)

    # ---- 取得条件 ----
    params = {"cdTab": tab_code, "cdArea": cd_area, "cdTime": ",".join(time_map.values())}
    if sex_cat_id and sex_total:
        params[f"cd{sex_cat_id.title()}"] = sex_total
    if status_map:
        params[f"cd{status_cat_id.title()}"] = ",".join(status_map.values())
    # 追加軸の固定を反映
    for cid, (code, _) in fixed_extra.items():
        params[f"cd{cid.title()}"] = code

    values = get_all_values(app_id, SID_OVERALL, params)

    # ---- 整形 ----
    code2year = {v: k for k, v in time_map.items()}
    code2status = {v: k for k, v in status_map.items()}
    rows = []
    for v in values:
        s = v.get("$")
        if s in (None, "", "-"): 
            continue
        try:
            num = float(s)
        except:
            continue
        y = code2year.get(v.get("@time"), v.get("@time"))
        st = code2status.get(v.get(f"@{status_cat_id}"), v.get(f"@{status_cat_id}"))
        rows.append({"年": y, "系列": st, "値": num})

    df_rows = pd.DataFrame(rows)
    if df_rows.empty:
        raise RuntimeError("表1: データが取得できませんでした。抽出条件をご確認ください。")

    # 同一（年×系列）の重複があれば first で潰す（安全弁）
    if df_rows.duplicated(subset=["年", "系列"]).any():
        if DEBUG_PRINT:
            dups = df_rows[df_rows.duplicated(subset=["年", "系列"], keep=False)]
            print("[DEBUG] 表1: 重複検出 → 件数:", len(dups))
        df_rows = df_rows.sort_values(["年", "系列"]).drop_duplicates(subset=["年", "系列"], keep="first")

    df = df_rows.pivot(index="年", columns="系列", values="値")

    # 欠け列の補完
    if ("労働力人口" not in df.columns) and all(c in df.columns for c in ("就業者", "完全失業者")):
        df["労働力人口"] = df["就業者"] + df["完全失業者"]
    if ("非労働力人口" not in df.columns) and all(c in df.columns for c in ("15歳以上人口", "労働力人口")):
        df["非労働力人口(推計)"] = df["15歳以上人口"] - df["労働力人口"]

    # 年順
    df = df.reindex([f"{y}年" for y in TARGET_YEARS])

    out_csv = out_dir / "census_overall_labor_force.csv"

    # --- 安全弁: 労働力人口は定義で再計算して CSV も上書き ---
    if all(c in df.columns for c in ("就業者", "完全失業者")):
        lf_calc = df["就業者"].astype(float) + df["完全失業者"].astype(float)
        # 「労働力人口」が無い/ズレている場合は置き換え
        if ("労働力人口" not in df.columns) or ((df["労働力人口"] - lf_calc).abs().max() > 1):
            diff = (df.get("労働力人口", lf_calc) - lf_calc).abs().max()
            df["労働力人口"] = lf_calc
            print(f"[FIX] 労働力人口を再計算で上書きしました（最大差={diff:.0f}）。")
    else:
        raise RuntimeError("表1: 就業者/完全失業者が不足しており、整合チェックができません。")

    df.to_csv(out_csv, encoding="utf-8-sig")
    print(f"[OK] overall CSV: {out_csv}")

    # --- 安全弁: 労働力人口は定義で再計算して強制整合 ---
    if all(c in df.columns for c in ("就業者", "完全失業者")):
        lf_calc = (df["就業者"] + df["完全失業者"]).astype(float)
        if "労働力人口" not in df.columns:
            df["労働力人口"] = lf_calc
            if DEBUG_PRINT:
                print("[FIX] 労働力人口列が無かったため再計算で新規作成しました。")
        else:
            # 就業者を下回る・または再計算とズレがある場合は上書き
            bad = (df["労働力人口"] < df["就業者"]).any() \
                  or (df["労働力人口"] - lf_calc).abs().max() > 1
            if bad:
                df["労働力人口"] = lf_calc
                print("[FIX] 労働力人口列を再計算で上書きしました。")
    else:
        raise RuntimeError("表1: 就業者/完全失業者が揃っていないため整合チェックができません。")

    # グラフ（順序固定＋右端に系列名を注記）
    set_japanese_font()
    cols = [c for c in ["労働力人口", "就業者", "完全失業者"] if c in df.columns] \
           + [c for c in df.columns if c not in ["労働力人口", "就業者", "完全失業者"]]

    plt.figure(figsize=(9, 5))
    for col in cols:
        plt.plot(df.index, df[col], marker="o", label=str(col))
        x = len(df.index) - 1
        y = df[col].iloc[-1]
        plt.annotate(str(col), xy=(x, y), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=9)

    plt.title("国全体の労働力推移（国勢調査・1995–2020）")
    plt.xlabel("年")
    plt.ylabel("人数（百万人）")  # ← ラベルも合わせておく

    # ★ここで“百万人”フォーマッタを適用（描画後〜保存前）
    from matplotlib.ticker import FuncFormatter
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/1e6:.1f}"))

    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    out_png = out_dir / "census_overall_labor_force.png"
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

    # グラフ（失業率）※後付け
    df = pd.read_csv("output/census_overall_labor_force.csv", index_col=0)

    # 失業率（＝完全失業者 / 労働力人口 ×100）
    df["失業率(%)"] = (df["完全失業者"] / df["労働力人口"]) * 100

    # 労働力率（＝労働力人口 / 15歳以上人口 ×100）※列がある時だけ
    if "15歳以上人口" in df.columns:
        df["労働力率(%)"] = (df["労働力人口"] / df["15歳以上人口"]) * 100

    plt.figure(figsize=(9,5))
    for c in [c for c in ["失業率(%)","労働力率(%)"] if c in df.columns]:
        plt.plot(df.index, df[c], marker="o", label=c)
    plt.title("失業率・労働力率（国勢調査）"); plt.xlabel("年"); plt.ylabel("％")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig("output/census_rates.png", dpi=200); plt.close()

    # ---- 1995年=1 の比率グラフ（overall）----
    base_idx = "1995年" if "1995年" in df.index else df.index[0]
    cols = [c for c in ["労働力人口", "就業者", "完全失業者"] if c in df.columns]

    # 比率テーブルを作成（無限大は NaN へ）
    idx = df[cols].astype(float).div(df.loc[base_idx, cols].astype(float))
    idx = idx.replace([float("inf"), -float("inf")], float("nan"))

    # 任意：比率CSVも保存
    idx.to_csv(out_dir / "census_overall_labor_force_index1995.csv", encoding="utf-8-sig")

    # 図を作成
    set_japanese_font()
    plt.figure(figsize=(9, 5))
    for c in cols:
        plt.plot(idx.index, idx[c], marker="o", label=c)
        # 右端に系列名を注記
        x = len(idx.index) - 1
        y = idx[c].iloc[-1]
        plt.annotate(c, xy=(x, y), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=9)

    plt.title(f"国全体の労働力推移（{base_idx}=1）")
    plt.xlabel("年"); plt.ylabel(f"{base_idx}=1 比率")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    out_png_idx = out_dir / "census_overall_labor_force_index1995.png"
    plt.tight_layout(); plt.savefig(out_png_idx, dpi=200); plt.close()
    print(f"[OK] overall index PNG: {out_png_idx}")

def fetch_age_labor_force(app_id: str, out_dir: Path) -> pd.DataFrame:
    """
    表3: 年齢（5歳階級）× 労働力状態（8区分）
      → 「労働力人口」×（男女計/総数）×（全国/日本, もし軸があれば）× 1995-2020
    - tab（表章項目）が無い表でも動く（= cdTab を省略）
    - area（地域）軸が無い表でも動く（= cdArea を省略）
    - 余剰の cat 軸は pick_default_code_and_name() で1値に固定
    """
    meta = get_meta(app_id, SID_AGE)
    classes = _collect_classes(meta)

    # ---- 表章項目(tab)：任意（ないこともある）
    tab_obj = classes.get("tab")
    tab_code_population = None
    if tab_obj:
        tab_code_population = find_item_code_by_name_contains(tab_obj, ["人口"])
        if not tab_code_population:
            # 「率」を含まない最初の項目
            for it in _class_items(tab_obj):
                if "率" not in it.get("@name", ""):
                    tab_code_population = it.get("@code")
                    break

    # ---- 地域(area)：任意（全国固定できる場合のみ付与）
    area_obj = classes.get("area")
    code_zenkoku = None
    if area_obj:
        code_zenkoku = find_item_code_by_name_contains(area_obj, ["全国"]) \
                    or find_item_code_by_name_contains(area_obj, ["日本"])

    # ---- 性別（男女計/総数）も任意
    sex_cat_id, sex_total_code = None, None
    for cid, obj in classes.items():
        if cid.startswith("cat"):
            code = (find_item_code_by_name_contains(obj, ["男女計"]) or
                    find_item_code_by_name_contains(obj, ["男女"]) or
                    find_item_code_by_name_contains(obj, ["総数"]) or
                    find_item_code_by_name_contains(obj, ["計"]))
            if code:
                sex_cat_id, sex_total_code = cid, code
                break

    # ---- 労働力状態（8区分）から「労働力人口」コード
    labour_cat_id, labour_code = None, None
    for cid, obj in classes.items():
        if cid.startswith("cat"):
            code = find_item_code_by_name_contains(obj, ["労働力人口"])
            if code:
                labour_cat_id, labour_code = cid, code
                break
    if not labour_code:
        raise RuntimeError("表3: 労働力人口コードが見つかりません。")

    # ---- 年齢（5歳階級）
    age_cat_id, age_obj = None, None
    for cid, obj in classes.items():
        if cid.startswith("cat"):
            if find_item_code_by_name_contains(obj, ["歳"]):
                age_cat_id, age_obj = cid, obj
                break
    if not age_obj:
        raise RuntimeError("表3: 年齢（5歳階級）分類が見つかりません。")
    age_map = {it.get("@name"): it.get("@code")
               for it in _class_items(age_obj)
               if "歳" in it.get("@name", "")}

    # ---- 余剰の分類軸を 1 値に固定（表1と同様）
    fixed_extra = {}
    for cid, obj in classes.items():
        if not cid.startswith("cat"):
            continue
        if cid in {sex_cat_id, labour_cat_id, age_cat_id}:
            continue
        code, name = pick_default_code_and_name(obj)
        # 複数候補があり固定不能ならエラーにして明示
        if code is None:
            items = _class_items(obj)
            if len(items) > 1:
                raise RuntimeError(
                    f"表3: 追加軸 {cid} に複数候補があり固定できません。例: {[it.get('@name') for it in items][:5]}"
                )
        else:
            fixed_extra[cid] = (code, name)

    if DEBUG_PRINT:
        show = {cid: name for cid, (code, name) in fixed_extra.items()}
        print("[DEBUG] 表3: 固定した追加軸（catID: 名称）:", show)

    # ---- 時間コード
    time_map = pick_time_codes_auto(meta, TARGET_YEARS, EXCLUDE_TIME_SUBSTR)
    if DEBUG_PRINT:
        print("[DEBUG] 表3: 年コード:", time_map)

    # ---- API パラメータ
    params = {
        "cdTime": ",".join(time_map.values()),
        f"cd{labour_cat_id.title()}": labour_code,
        f"cd{age_cat_id.title()}": ",".join(age_map.values()),
    }
    if tab_code_population:
        params["cdTab"] = tab_code_population
    if code_zenkoku:
        params["cdArea"] = code_zenkoku
    if sex_cat_id and sex_total_code:
        params[f"cd{sex_cat_id.title()}"] = sex_total_code
    for cid, (code, _) in fixed_extra.items():
        params[f"cd{cid.title()}"] = code

    values = get_all_values(app_id, SID_AGE, params)

    # ---- 整形
    code_to_year = {v: k for k, v in time_map.items()}
    code_to_age = {v: k for k, v in age_map.items()}
    rows = []
    for v in values:
        s = v.get("$")
        if s in (None, "", "-"):
            continue
        try:
            num = float(s)
        except Exception:
            continue
        y = code_to_year.get(v.get("@time"), v.get("@time"))
        a = code_to_age.get(v.get(f"@{age_cat_id}"), v.get(f"@{age_cat_id}"))
        rows.append({"年": y, "年齢階級": a, "値": num})

    df_rows = pd.DataFrame(rows)
    if df_rows.empty:
        raise RuntimeError("表3: データが取得できませんでした。抽出条件をご確認ください。")

    # 重複（年×年齢階級）があれば first で潰す
    if df_rows.duplicated(subset=["年", "年齢階級"]).any():
        if DEBUG_PRINT:
            dups = df_rows[df_rows.duplicated(subset=["年", "年齢階級"], keep=False)]
            print("[DEBUG] 表3: 重複検出 → 件数:", len(dups))
        df_rows = df_rows.sort_values(["年", "年齢階級"]).drop_duplicates(subset=["年", "年齢階級"], keep="first")

    df = df_rows.pivot(index="年", columns="年齢階級", values="値")
    df = df.reindex([f"{y}年" for y in TARGET_YEARS])

    out_csv = out_dir / "census_age_labor_force_5yr.csv"
    df.to_csv(out_csv, encoding="utf-8-sig")
    print(f"[OK] age CSV: {out_csv}")

    # ---- グラフ：年齢別 労働力人口（15–64歳・積み上げ）人数（百万人）----
    set_japanese_font()

    import re
    from matplotlib.ticker import FuncFormatter

    def is_under65(name: str) -> bool:
        if "不詳" in name: return False
        nums = re.findall(r"\d+", name)
        if not nums: return False
        first = int(nums[0])
        if "以上" in name: return first < 65
        if "～" in name:
            upper = int(nums[-1])
            return upper <= 64
        return first < 65

    cols = [c for c in df.columns if is_under65(c)]
    plot_df = df[cols].fillna(0)

    # 大きい系列を下に積む
    cols_order = plot_df.mean().sort_values(ascending=False).index.tolist()
    plot_df = plot_df[cols_order]

    x = list(range(len(plot_df.index)))
    ys = [plot_df[c].values for c in plot_df.columns]

    plt.figure(figsize=(11, 6))
    plt.stackplot(x, *ys, labels=plot_df.columns)
    plt.xticks(x, plot_df.index)
    plt.title("年齢別の労働力人口（15–64歳・積み上げ）1995–2020")
    plt.xlabel("年"); plt.ylabel("人数（百万人）")

    # ★Y軸を百万人表示
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v/1e6:.1f}"))

    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3, fontsize=8, loc="upper left")
    out_png = out_dir / "census_age_labor_force_5yr.png"
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()
    print(f"[OK] age stacked PNG: {out_png}")

    # ---- 年齢構成（各年=100%）の積み上げ ----
    # 既存の df は「年×年齢階級」のピボット（値=人数）
    from matplotlib.ticker import FuncFormatter
    import re

    # 不詳は除外。65歳以上を含めるかは切替可能（True=含める→合計100％）
    INCLUDE_65PLUS_FOR_SHARE = False

    def is_age_bin(name: str) -> bool:
        return ("歳" in name) and ("不詳" not in name)

    def age_key(label: str) -> int:
        nums = re.findall(r"\d+", label)
        return int(nums[0]) if nums else 999

    # 年齢列を抽出して若い順に並べる
    age_cols_all = sorted([c for c in df.columns if is_age_bin(c)], key=age_key)

    if INCLUDE_65PLUS_FOR_SHARE:
        cols_share = age_cols_all
    else:
        def under65(name: str) -> bool:
            nums = re.findall(r"\d+", name)
            if not nums: return False
            first = int(nums[0])
            if "以上" in name: return first < 65
            if "～" in name:   return int(nums[-1]) <= 64
            return first < 65
        cols_share = [c for c in age_cols_all if under65(c)]

    share_df = df[cols_share].fillna(0)

    # 各年の合計を100％に正規化
    denom = share_df.sum(axis=1)
    share_df = share_df.div(denom, axis=0) * 100

    # 積み上げ描画（％）
    x = list(range(len(share_df.index)))
    ys = [share_df[c].values for c in share_df.columns]

    plt.figure(figsize=(11, 6))
    plt.stackplot(x, *ys, labels=share_df.columns)
    plt.xticks(x, share_df.index)
    plt.title("年齢別 労働力構成（各年=100%）1995–2020")
    plt.xlabel("年"); plt.ylabel("構成比（％）")
    ax = plt.gca()
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.0f}"))
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3, fontsize=8, loc="upper left")

    out_png_share = out_dir / "census_age_labor_force_share_stacked.png"
    plt.tight_layout(); plt.savefig(out_png_share, dpi=200); plt.close()
    print(f"[OK] age share stacked PNG: {out_png_share}")

    # 任意：CSVも残す場合
    # share_df.to_csv(out_dir / "census_age_labor_force_share_stacked.csv", encoding="utf-8-sig")

    # ---- 1995年=1 の比率グラフ（年齢別・上位10系列）----
    base_idx = "1995年" if "1995年" in plot_df.index else plot_df.index[0]
    idx = plot_df.div(plot_df.loc[base_idx]).replace([float("inf"), -float("inf")], float("nan"))

    top10 = plot_df.mean().sort_values(ascending=False).head(10).index.tolist()
    idx_top = idx[top10]

    plt.figure(figsize=(11, 6))
    for c in idx_top.columns:
        plt.plot(idx_top.index, idx_top[c], marker="o", label=c)
    plt.title(f"年齢別 労働力人口の推移（{base_idx}=1）")
    plt.xlabel("年"); plt.ylabel(f"{base_idx}=1 比率")
    plt.grid(True, alpha=0.3); plt.legend(ncol=2, fontsize=8)
    out_png_idx = out_dir / "census_age_labor_force_5yr_index1995.png"
    plt.tight_layout(); plt.savefig(out_png_idx, dpi=200); plt.close()
    print(f"[OK] age index PNG: {out_png_idx}")

# ===== 産業（大分類）名の正規化とホワイトリスト =====
BASE_INDUSTRIES = [
    "農業，林業",
    "漁業",
    "鉱業，採石業，砂利採取業",
    "建設業",
    "製造業",
    "電気・ガス・熱供給・水道業",
    "情報通信業",
    "運輸業，郵便業",
    "卸売業，小売業",
    "金融業，保険業",
    "不動産業，物品賃貸業",
    "学術研究，専門・技術サービス業",
    "宿泊業，飲食サービス業",
    "生活関連サービス業，娯楽業",
    "教育，学習支援業",
    "医療，福祉",
    "複合サービス事業",
    "公務（他に分類されるものを除く）",
    "サービス業（他に分類されないもの）",
    # "分類不能の産業" は意図的に除外
]

def _normalize_ind_name(s: str) -> str:
    """産業名の表記ゆれを吸収：先頭の記号/英字・半角/全角カンマ等を正規化"""
    if s is None:
        return ""
    s = str(s)
    # 先頭のコードや英字（例: 'A ', 'Ｈ '）を除去
    import re
    s = re.sub(r"^[A-Za-zＡ-Ｚａ-ｚ]\s*", "", s)
    # 半角カンマ/読点を全角カンマに統一
    s = s.replace(",", "，").replace("、", "，")
    # 余分な空白を削る
    s = re.sub(r"\s+", "", s)
    return s

def fetch_industry_employment(app_id: str, out_dir: Path) -> pd.DataFrame:
    """
    表4: 産業（大分類）× 就業者数（各産業の値のみ）
    - 総数/合計/第1～3次/分類不能は除外
    - 産業コードの桁揺れ（ゼロ埋め等）を吸収してマッチ
    - cd指定が効かない場合に備え、cd無し再取得のフォールバック
    """
    import re
    from collections import Counter

    meta = get_meta(app_id, SID_IND)
    classes = _collect_classes(meta)

    # ---- tab（任意：就業者“数”優先）
    tab_obj = classes.get("tab")
    tab_code_workers = None
    if tab_obj:
        tab_code_workers = find_item_code_by_name_contains(tab_obj, ["就業者数"])
        if not tab_code_workers:
            for it in _class_items(tab_obj):
                nm = it.get("@name", "")
                if all(x not in nm for x in ["率", "構成比", "割合"]):
                    tab_code_workers = it.get("@code"); break

    # ---- area（任意：全国/日本）
    area_obj = classes.get("area")
    cd_area = None
    if area_obj:
        cd_area = find_item_code_by_name_contains(area_obj, ["全国"]) \
               or find_item_code_by_name_contains(area_obj, ["日本"])

    # ---- 性別（任意：男女計/総数/計）
    sex_cat_id, sex_total = None, None
    for cid, obj in classes.items():
        if cid.startswith("cat"):
            code = (find_item_code_by_name_contains(obj, ["男女計"]) or
                    find_item_code_by_name_contains(obj, ["男女"]) or
                    find_item_code_by_name_contains(obj, ["総数"]) or
                    find_item_code_by_name_contains(obj, ["計"]))
            if code:
                sex_cat_id, sex_total = cid, code; break

    # ---- 産業（大分類）cat をホワイトリスト一致で検出
    wl_norm = { _normalize_ind_name(n) for n in BASE_INDUSTRIES }
    ind_cat_id, ind_obj, best_hits = None, None, -1
    for cid, obj in classes.items():
        if not cid.startswith("cat"): continue
        names = [it.get("@name", "") for it in _class_items(obj)]
        norm = {_normalize_ind_name(n) for n in names}
        hits = len(norm & wl_norm)
        if hits > best_hits:
            best_hits = hits; ind_cat_id, ind_obj = cid, obj
    if not ind_obj or best_hits < 8:
        raise RuntimeError(f"表4: 産業（大分類）が検出できません（ホワイトリスト一致={best_hits}）。")

    # ---- code -> 正式名（集計カテゴリは除外）
    drop_patterns = [r"^総数$", r"総計", r"合計", r"小計", r"計$", r"（再掲）", r"分類不能", r"第.?次産業"]
    def is_aggregate(name: str) -> bool:
        return any(re.search(p, name) for p in drop_patterns)

    code_items = []
    for it in _class_items(ind_obj):
        raw_name = it.get("@name", ""); code = it.get("@code"); level = it.get("@level")
        if not code: continue
        if is_aggregate(raw_name): continue
        if _normalize_ind_name(raw_name) in wl_norm:
            code_items.append((code, raw_name, level))

    if not code_items:
        raise RuntimeError("表4: 産業（大分類）の有効項目を抽出できません。")

    # 最頻レベルを lvCat01 に指定（大分類の階層を固定）
    level_counts = Counter([lv for _,_,lv in code_items if lv is not None])
    common_lv = level_counts.most_common(1)[0][0] if level_counts else None

    # マッピング（メタコード -> 正式名）
    meta_code2name = {code: name for code, name, _ in code_items}

    if DEBUG_PRINT:
        print("[DEBUG] 表4: 産業cat:", ind_cat_id, "件数:", len(meta_code2name), "最頻level:", common_lv)
        print("[DEBUG] 表4: サンプル:", list(meta_code2name.items())[:5])

    # ---- 年コード
    time_map = pick_time_codes_auto(meta, TARGET_YEARS, EXCLUDE_TIME_SUBSTR)

    # ---- 取得（まず cd 指定アリでトライ）
    base_params = {"cdTime": ",".join(time_map.values())}
    if tab_code_workers: base_params["cdTab"] = tab_code_workers
    if cd_area: base_params["cdArea"] = cd_area
    if sex_cat_id and sex_total: base_params[f"cd{sex_cat_id.title()}"] = sex_total
    if common_lv: base_params[f"lv{ind_cat_id.title()}"] = common_lv

    # 1) cd{ind_cat}で19業を指定
    params1 = dict(base_params)
    params1[f"cd{ind_cat_id.title()}"] = ",".join(meta_code2name.keys())
    values = get_all_values(app_id, SID_IND, params1)

    # フォールバック: 取得ゼロ/マッチゼロなら cd 指定を外す
    if not values:
        params2 = dict(base_params)
        values = get_all_values(app_id, SID_IND, params2)

    # ---- VALUE側のコード正規化関数
    def norm_code(s: str) -> str:
        if s is None: return ""
        s = str(s)
        s = re.sub(r"\D", "", s)         # 非数字は除去（念のため）
        s = re.sub(r"0+$", "", s) or s   # 末尾ゼロを削る（全部ゼロなら元のまま）
        return s

    meta_norm2name = { norm_code(k): v for k, v in meta_code2name.items() }

    # ---- 整形：年×産業
    code2year = {v: k for k, v in time_map.items()}
    rows, unmatched = [], 0

    for v in values:
        s = v.get("$")
        if s in (None, "", "-"): continue
        try: num = float(s)
        except: continue

        raw_code = v.get(f"@{ind_cat_id}")
        if not raw_code:
            unmatched += 1
            continue

        c_norm = norm_code(raw_code)
        name = meta_norm2name.get(c_norm)

        # prefix一致（VALUEコードが長い場合）も許容
        if name is None:
            for mcode_norm, mname in meta_norm2name.items():
                if c_norm.startswith(mcode_norm):
                    name = mname; break

        if name is None:
            unmatched += 1
            continue

        y = code2year.get(v.get("@time"), v.get("@time"))
        rows.append({"年": y, "産業": name, "値": num})

    if DEBUG_PRINT:
        print(f"[DEBUG] 表4: 行総数={len(values)} / マッチ={len(rows)} / 非マッチ={unmatched}")

    df_rows = pd.DataFrame(rows)
    if df_rows.empty:
        raise RuntimeError("表4: データが取得できませんでした（産業コードのマッチに失敗）。")

    # 重複（年×産業）は first で潰す
    if df_rows.duplicated(subset=["年", "産業"]).any():
        if DEBUG_PRINT:
            dups = df_rows[df_rows.duplicated(subset=["年", "産業"], keep=False)]
            print("[DEBUG] 表4: 重複検出 → 件数:", len(dups))
        df_rows = df_rows.sort_values(["年", "産業"]).drop_duplicates(subset=["年", "産業"], keep="first")

    df = df_rows.pivot(index="年", columns="産業", values="値").reindex([f"{y}年" for y in TARGET_YEARS])

    # ---- 出力
    out_csv = out_dir / "census_industry_employment.csv"
    df.to_csv(out_csv, encoding="utf-8-sig"); print(f"[OK] industry CSV: {out_csv}")

    # ---- グラフ：産業別 就業者数（各産業の値を積み上げ）人数（百万人）----
    set_japanese_font()
    from matplotlib.ticker import FuncFormatter

    plot_df = df.fillna(0)

    # 2020年の大きい順（なければ平均）
    if "2020年" in plot_df.index:
        cols_order = plot_df.loc["2020年"].sort_values(ascending=False).index.tolist()
    else:
        cols_order = plot_df.mean().sort_values(ascending=False).index.tolist()
    plot_df = plot_df[cols_order]

    x = list(range(len(plot_df.index)))
    ys = [plot_df[c].values for c in plot_df.columns]

    plt.figure(figsize=(11.5, 6.5))
    plt.stackplot(x, *ys, labels=plot_df.columns)
    plt.xticks(x, plot_df.index)
    plt.title("産業別 就業者数（積み上げ）1995–2020")
    plt.xlabel("年"); plt.ylabel("人数（百万人）")

    # ★Y軸を百万人表示
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v/1e6:.1f}"))

    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3, fontsize=8, loc="upper left")
    out_png = out_dir / "census_industry_employment.png"
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()
    print(f"[OK] industry stacked PNG: {out_png}")

    # ---- 産業構成（各年=100%）の積み上げ ----
    from matplotlib.ticker import FuncFormatter
    share_src = df.fillna(0)

    # 見やすさのため、2020年のシェアが大きい順に並べ替え（2020年が無ければ平均順）
    if "2020年" in share_src.index:
        cols_order = share_src.loc["2020年"].sort_values(ascending=False).index.tolist()
    else:
        cols_order = share_src.mean().sort_values(ascending=False).index.tolist()
    share_src = share_src[cols_order]

    # 各年の合計を100%に正規化（ゼロ割回避）
    denom = share_src.sum(axis=1).replace(0, pd.NA)
    share_pct = (share_src.div(denom, axis=0) * 100).fillna(0)

    # 積み上げ描画（％）
    x = list(range(len(share_pct.index)))
    ys = [share_pct[c].values for c in share_pct.columns]

    plt.figure(figsize=(11.5, 6.5))
    plt.stackplot(x, *ys, labels=share_pct.columns)
    plt.xticks(x, share_pct.index)
    plt.title("産業別 就業者 構成比（各年=100%）1995–2020")
    plt.xlabel("年"); plt.ylabel("構成比（％）")
    ax = plt.gca()
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.0f}"))
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3, fontsize=8, loc="upper left")

    out_png_share = out_dir / "census_industry_share_stacked.png"
    plt.tight_layout(); plt.savefig(out_png_share, dpi=200); plt.close()
    print(f"[OK] industry share stacked PNG: {out_png_share}")

    # 任意：数値も残したい場合（各年=100%のテーブル）
    # share_pct.to_csv(out_dir / "census_industry_share_stacked.csv", encoding="utf-8-sig")

    # ---- 1995年=1 の比率グラフ（産業別・上位10系列）----
    base_idx = "1995年" if "1995年" in plot_df.index else plot_df.index[0]
    idx = plot_df.div(plot_df.loc[base_idx]).replace([float("inf"), -float("inf")], float("nan"))

    top10 = plot_df.mean().sort_values(ascending=False).head(10).index.tolist()
    idx_top = idx[top10]

    plt.figure(figsize=(11.5, 6.5))
    for c in idx_top.columns:
        plt.plot(idx_top.index, idx_top[c], marker="o", label=c)
    plt.title(f"産業別 就業者の推移（{base_idx}=1）")
    plt.xlabel("年"); plt.ylabel(f"{base_idx}=1 比率")
    plt.grid(True, alpha=0.3); plt.legend(ncol=2, fontsize=8)
    out_png_idx = out_dir / "census_industry_employment_index1995.png"
    plt.tight_layout(); plt.savefig(out_png_idx, dpi=200); plt.close()
    print(f"[OK] industry index PNG: {out_png_idx}")

def _class_first_name_by_code(obj: Dict[str, Any], code: str) -> str:
    for it in _class_items(obj):
        if it.get("@code") == code:
            return it.get("@name", "")
    return ""

def pick_default_code_and_name(obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    追加分類軸（catXX）で、総数など1値に固定するためのデフォルトコードを選ぶ。
    優先順位: 総数/計/男女計/男女 → 15歳以上/15歳以上人口 → 単一候補しか無ければそれ
    """
    # まず「総数/計/男女計/男女」
    for kw in (["総数"], ["計"], ["男女計"], ["男女"]):
        code = find_item_code_by_name_contains(obj, kw)
        if code:
            return code, _class_first_name_by_code(obj, code)
    # 次に「15歳以上」系
    for kw in (["15歳以上人口"], ["15歳以上"]):
        code = find_item_code_by_name_contains(obj, kw)
        if code:
            return code, _class_first_name_by_code(obj, code)
    # 候補が1つならそれ
    items = _class_items(obj)
    if len(items) == 1:
        code = items[0].get("@code")
        return code, items[0].get("@name", "")
    return None, None


# ====== CLI ======
def main():
    parser = argparse.ArgumentParser(description="e-Stat 国勢調査（最近30年）労働力・産業データ取得")
    parser.add_argument("--out-dir", type=str, default="output", help="出力ディレクトリ")
    args = parser.parse_args()

    out_dir = Path(args.out_dir); ensure_outdir(out_dir)
    app_id = load_app_id()

    fetch_overall(app_id, out_dir)
    fetch_age_labor_force(app_id, out_dir)
    fetch_industry_employment(app_id, out_dir)
    print("\n[DONE] すべての取得と出力が完了しました。")

if __name__ == "__main__":
    main()
