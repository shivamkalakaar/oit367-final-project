# Do Ethics Pay?

**Evaluating the Stock Market Performance of Ethisphere's World's Most Ethical Companies (2021–2025)**

OIT 367: Business Intelligence with Big Data — Final Project  
Stanford Graduate School of Business | March 2026

**Team:** Shivam Kalkar, Kevin Stephen, Andy Parker, Serena Davis

## Overview

This project partners with Ethisphere Institute to evaluate whether companies on their World's Most Ethical Companies (WMEC) list exhibit superior risk-adjusted stock market performance relative to a broad market benchmark. We analyze 5 years of daily price data (Jan 2021 – Dec 2025) for 113 publicly traded WMEC honorees across six analyses: downside resilience, consistency, upside/downside capture, tail-risk behavior, cross-sectional segmentation, and a predictive model using LASSO, random forest, and gradient boosting.

**Key finding:** Honorees provide statistically significant downside protection in the worst market months (+1.39 pp excess return, 59.3% hit rate), but show no consistent alpha in normal conditions. Firmographic features cannot predict which honorees outperform in downturns, suggesting the tail-risk benefit is broadly distributed across the WMEC universe.

## Repository Structure

```
├── Data/                          # Source datasets
│   ├── CAPIQ Stocks and Daily Movement.csv    # Daily prices for 113 honoree stocks
│   ├── Solactive_Ethic_Backtest_20260130.xlsx # Ethics Premium GTR & benchmark index
│   ├── WMEC Stocks and Info.csv               # Firmographic metadata (sector, country, etc.)
│   ├── 2026 WMEC Public Symbols and Firmographics.xlsx
│   └── OIT 367 CAPIQ Investigation v9_Hardcodes.xlsx
│
├── Model/
│   └── model_vF.ipynb             # Main analysis notebook (run this)
│
├── output/                        # Pre-generated outputs from model_vF.ipynb
│   ├── master_summary.csv         # Summary of all analyses with CIs
│   ├── stock_level_results.csv    # Per-stock performance metrics
│   ├── segment_results.csv        # Segmentation by sector, country, etc.
│   └── fig0–fig6 (*.png)          # All figures used in the report
│
├── Project Guidelines/            # Course instructions and reference materials
└── README.md
```

## How to Run

1. Open `Model/model_vF.ipynb` in Jupyter.
2. The notebook expects data files in the working directory. Either:
   - Copy all files from `Data/` into `Model/`, or
   - Run from the repo root and adjust paths accordingly.
3. Run all cells. Outputs (CSVs and figures) are saved to `output/`.

**Dependencies:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `openpyxl`

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn openpyxl
```

## Data Quality Fixes

The notebook programmatically corrects four data issues in the firmographics file (see Appendix A of the report for details):

| Issue | Fix |
|-------|-----|
| LabCorp assigned wrong ticker (LRCX → LH) | Corrected in code |
| Cemex case mismatch (CEMEXCPO → CemexCPO) | Normalized |
| SK Hynix quoted ticker ("000660" → 000660) | Stripped quotes |
| Allstate (ALL) missing from firmographics | Added programmatically |

## Analyses

1. **Downside Resilience** — Max drawdown, time underwater, recovery speed vs. benchmark
2. **Consistency** — Monthly, rolling 3-month/6-month, and down-month hit rates
3. **Upside/Downside Capture** — Nested bootstrap (60-day windows × stock resampling)
4. **Tail-Risk Behavior** — Performance in worst 5% of benchmark months, plus 10th-percentile robustness check
5. **Segmentation** — Kruskal-Wallis tests across sector, industry, country, workforce size, and revenue
6. **Predictive Model** — LASSO logistic regression, random forest, and gradient boosting to predict tail-month outperformance from firmographic features (5-fold stratified CV)
