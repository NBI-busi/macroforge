# macroforge
A toolbox to automate retrieval, analysis, and visualization of regional fundamentals (People / Production / Capital).  
地域別のファンダメンタルズ（ヒト／モノ／カネ）の取得・分析・可視化を自動化するツールボックス。

---

## Planned directory structure (draft) / 今後想定しているディレクトリ構成（仮）
※現状、functionsフォルダ内の***.pyを積み上げています。

macroforge/
├─ README.md
├─ pyproject.toml            # 依存管理（poetry/uv/hatchいずれか） | Dependency management (choose poetry/uv/hatch)
├─ src/
│  └─ macroforge/
│     ├─ common/             # 共有ユーティリティ | Shared utilities
│     │  ├─ io/estat.py, bls.py, eurostat.py, edinet.py   # 公的データI/O | Public data I/O
│     │  ├─ viz/plotting.py  # 可視化共通 | Common plotting helpers
│     │  └─ etl/base.py      # ETL基盤 | ETL base classes
│     └─ cli.py              # コマンド入口（例: mf run jp hito census） | CLI entry point (e.g., mf run jp hito census)
├─ pipelines/
│  ├─ jp/
│  │  ├─ hito/   # 労働人口など | People (workforce/labor)
│  │  ├─ mono/   # AI+ロボット / 特許 | Production (AI & robotics / patents)
│  │  └─ kane/   # 設備投資 など | Capital (capex, etc.)
│  ├─ us/
│  │  ├─ hito/   # BLS 等 | People (BLS, etc.)
│  │  ├─ mono/   # Production
│  │  └─ kane/   # BEA 等 | Capital (BEA, etc.)
│  └─ eu/
│     ├─ hito/   # Eurostat | People (Eurostat)
│     ├─ mono/   # Production
│     └─ kane/   # Capital
├─ data/
│  ├─ raw/{region}/{source}/       # 取得データ（生）| Raw inputs
│  └─ processed/{region}/{topic}/  # 整形後データ | Processed outputs
├─ charts/{region}/{topic}/        # 図の出力先 | Chart outputs
├─ notebooks/                      # 探索用（成果物は pipelines へ昇格）| Exploratory; promote stable work to pipelines
├─ configs/
│  └─ datasets.yaml                # 統計表ID・パラメタのカタログ | Catalog of dataset/table IDs & params
├─ tests/                          # テスト | Tests
└─ .github/workflows/ci.yml        # Lint/テスト/図の生成チェック | Lint/tests/chart-generation checks
