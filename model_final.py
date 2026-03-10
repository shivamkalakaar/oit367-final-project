# =============================================================================
# model_final.py  ·  OIT 367 – Ethical Premium Analysis  ·  March 2026
# Team: Shivam Kalkar · Kevin Stephen · Andy Parker · Serena Davis
#
# This is the FINAL consolidated analysis script.  It merges:
#   • model.py          – full 5-analysis pipeline + enhanced E1-E4 analytics
#   • model_march10_9am.ipynb – 5 critical data-quality bug fixes:
#       Bug 1 (CRITICAL): LabCorp wrong ticker LRCX → corrected to LH
#       Bug 2: Cemex case mismatch CEMEXCPO → CemexCPO
#       Bug 3: SK Hynix quoted ticker "000660" → 000660
#       Bug 4: Allstate (ALL) missing from firmographics → added
#       Bug 5: Recovery denominator excluded never-recovered stocks → fixed
#     Net effect: 109 → 113/113 stocks matched in firmographics.
#
# All outputs saved to output/:
#   CSVs   : master_summary.csv, stock_level_results.csv, segment_results.csv,
#            year_by_year.csv, tail_risk_thresholds.csv
#   Figures: fig0–fig11 (12 figures total)
# =============================================================================

import pandas as pd
import numpy as np
import os
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

np.random.seed(42)
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
N_BOOTSTRAP = 10_000

# ── Plot style ────────────────────────────────────────────────────────────────
BRAND_BLUE   = "#1a3a5c"
BRAND_GOLD   = "#c9a84c"
BRAND_RED    = "#c0392b"
BRAND_GREY   = "#7f8c8d"
BRAND_GREEN  = "#27ae60"
BRAND_LIGHT  = "#ecf0f1"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#cccccc",
    "axes.grid":        True,
    "grid.color":       "#eeeeee",
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
})


def sig_stars(p):
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    if p < 0.10:   return "."
    return "ns"


def bootstrap_pvalue(bootstrap_dist, null_value, alternative="two-sided"):
    if alternative == "greater":
        return np.mean(bootstrap_dist <= null_value)
    if alternative == "less":
        return np.mean(bootstrap_dist >= null_value)
    centered = bootstrap_dist - np.mean(bootstrap_dist)
    observed = np.mean(bootstrap_dist) - null_value
    return np.mean(np.abs(centered) >= np.abs(observed))


def save_fig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved figure: {path}")


def max_drawdown(cum_series):
    peak = cum_series.cummax()
    return ((cum_series - peak) / peak).min()


def time_underwater(cum_series):
    """Return (max_underwater_days, recovery_days_after_mdd).
    max_underwater_days = longest streak where cum < previous peak.
    recovery_days = trading days from max-drawdown trough back to a new peak.
                    np.nan if never recovered within the series.
    """
    peak = cum_series.cummax()
    underwater = cum_series < peak

    # Longest underwater streak
    streaks = []
    current = 0
    for uw in underwater:
        if uw:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    max_uw = max(streaks) if streaks else 0

    # Recovery time from the max-drawdown trough
    dd = (cum_series - peak) / peak
    trough_idx = dd.idxmin()
    trough_loc = cum_series.index.get_loc(trough_idx)
    peak_at_trough = peak.iloc[trough_loc]
    # Find first date after trough where cumulative value >= pre-trough peak
    post_trough = cum_series.iloc[trough_loc:]
    recovered = post_trough[post_trough >= peak_at_trough]
    if len(recovered) > 0:
        recovery_loc = cum_series.index.get_loc(recovered.index[0])
        recovery_days = recovery_loc - trough_loc
    else:
        recovery_days = np.nan

    return max_uw, recovery_days


# =============================================================================
# DATA LOADING
# =============================================================================
print("=" * 70)
print("DATA LOADING")
print("=" * 70)

raw = pd.read_csv("CAPIQ Stocks and Daily Movement.csv", header=None)
tickers       = raw.iloc[1, 2:].values
exchanges     = raw.iloc[0, 2:].values
company_names = raw.iloc[2, 2:].values

prices = raw.iloc[8:, 2:].copy()
prices.columns = tickers
prices.index   = pd.to_datetime(raw.iloc[8:, 1], format="%m/%d/%Y")
prices.index.name = "Date"
prices = prices.replace(",", "", regex=True).apply(pd.to_numeric, errors="coerce")
prices = prices.sort_index()
prices = prices[~prices.index.duplicated(keep="first")]
print(f"Stock prices: {prices.shape[0]} days x {prices.shape[1]} stocks")

perf = pd.read_excel("Solactive_Ethic_Backtest_20260130.xlsx",
                     sheet_name="Performance")
perf["Date"] = pd.to_datetime(perf["Date"])
perf = perf.set_index("Date")[["Ethics Premium GTR", "SGMACUT Index"]].copy()
perf = perf.sort_index()
perf = perf[~perf.index.duplicated(keep="first")]
perf.columns = ["EthicsPremium_GTR", "Benchmark_GTR"]
print(f"Benchmark data: {perf.shape[0]} days")

weights = pd.read_excel("Solactive_Ethic_Backtest_20260130.xlsx",
                        sheet_name="Composition_Weights")
weights = weights.rename(columns={"Ticker": "Ticker_Exch", "Name": "Company"})
weights["Ticker"] = weights["Ticker_Exch"].str.split("-").str[0]

firmographics   = pd.read_csv("WMEC Stocks and Info.csv")

# ── DATA QUALITY FIXES (5 bugs corrected; net effect: 109 → 113/113 matched) ─
# Fix 1 (CRITICAL): LabCorp incorrectly assigned ticker LRCX (Lam Research).
#                   LabCorp's correct NYSE ticker is LH.
firmographics.loc[firmographics["Symbol"] == "LRCX", "Symbol"] = firmographics.loc[
    firmographics["Symbol"] == "LRCX", "Company"
].apply(lambda c: "LH" if "Lab" in c else "LRCX")

# Fix 2: Cemex case mismatch — firmographics has "CEMEXCPO", CAPIQ has "CemexCPO"
firmographics["Symbol"] = firmographics["Symbol"].replace("CEMEXCPO", "CemexCPO")

# Fix 3: SK Hynix quoted ticker — firmographics has '"000660"', CAPIQ has '000660'
firmographics["Symbol"] = firmographics["Symbol"].str.strip('"')

# Fix 4: Allstate (ALL) present in CAPIQ prices but absent from firmographics entirely.
#         Public data: NYSE: ALL, Financials / Insurance, ~$61B revenue, ~54,000 employees.
allstate_row = {
    "Company": "The Allstate Corporation",
    "Symbol": "ALL",
    "Exchange": "NYS",
    "Structure": "Public",
    "Sectors": "Financials",
    "Industry Groups": "Insurance",
    "Industries": "Property & Casualty Insurance",
    "Sub-Industries": "Property & Casualty Insurance",
    "Domicile Country": "United States",
    "Workforce Size": "50,000 - 100,000 employees",
    "Annual Revenue": "$50,000,000,001 - $75,000,000,000",
    "Workforce Representation": None,
}
firmographics = pd.concat(
    [firmographics, pd.DataFrame([allstate_row])], ignore_index=True
)

ticker_to_firm  = firmographics.set_index("Symbol")
matched_tickers = [t for t in tickers if t in ticker_to_firm.index]
print(f"Firmographics: {firmographics.shape[0]} companies | "
      f"Matched: {len(matched_tickers)}")

# =============================================================================
# COMPUTE RETURNS
# =============================================================================
print("\n" + "=" * 70)
print("COMPUTING RETURNS")
print("=" * 70)

stock_returns = prices.pct_change().iloc[1:]
bench_returns = perf.pct_change().iloc[1:]
common_dates  = stock_returns.index.intersection(bench_returns.index)
stock_returns = stock_returns.loc[common_dates]
bench_returns = bench_returns.loc[common_dates]

bm_daily = bench_returns["Benchmark_GTR"]
ep_daily = bench_returns["EthicsPremium_GTR"]

stock_monthly = stock_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
bm_monthly    = bm_daily.resample("ME").apply(lambda x: (1 + x).prod() - 1)

stock_cum = (1 + stock_returns).cumprod()
bm_cum    = (1 + bm_daily).cumprod()

print(f"Aligned: {len(common_dates)} trading days | {len(bm_monthly)} months")

# ── Figure 0: Cumulative index performance ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(perf.index, perf["EthicsPremium_GTR"], color=BRAND_BLUE,
        linewidth=2, label="Ethics Premium GTR")
ax.plot(perf.index, perf["Benchmark_GTR"], color=BRAND_GREY,
        linewidth=2, linestyle="--", label="Benchmark (SGMACUT)")
ax.fill_between(perf.index,
                perf["EthicsPremium_GTR"], perf["Benchmark_GTR"],
                where=perf["EthicsPremium_GTR"] >= perf["Benchmark_GTR"],
                alpha=0.15, color=BRAND_GREEN, label="Outperformance")
ax.fill_between(perf.index,
                perf["EthicsPremium_GTR"], perf["Benchmark_GTR"],
                where=perf["EthicsPremium_GTR"] < perf["Benchmark_GTR"],
                alpha=0.15, color=BRAND_RED, label="Underperformance")
ax.set_title("Ethics Premium GTR vs. Benchmark (2021-2025)")
ax.set_ylabel("Index Level (Base = 1,000)")
ax.legend(loc="upper left")
final_ep = perf["EthicsPremium_GTR"].iloc[-1]
final_bm = perf["Benchmark_GTR"].iloc[-1]
ax.annotate(f"EP: {final_ep:.0f} (+{(final_ep/1000-1)*100:.1f}%)",
            xy=(perf.index[-1], final_ep), xytext=(-90, 10),
            textcoords="offset points", color=BRAND_BLUE, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=BRAND_BLUE))
ax.annotate(f"BM: {final_bm:.0f} (+{(final_bm/1000-1)*100:.1f}%)",
            xy=(perf.index[-1], final_bm), xytext=(-90, -35),
            textcoords="offset points", color=BRAND_GREY, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=BRAND_GREY))
save_fig("fig0_index_performance.png")


# =============================================================================
# ANALYSIS 1: DOWNSIDE RESILIENCE
#   - Max drawdown
#   - Time-to-recover (days from trough to new peak)
#   - Time underwater (longest streak below previous peak)
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 1: DOWNSIDE RESILIENCE")
print("=" * 70)

# ── Max drawdown ──────────────────────────────────────────────────────────────
stock_mdd = stock_cum.apply(max_drawdown)
bm_mdd    = max_drawdown(bm_cum)

valid_stocks   = stock_mdd.dropna()
n_stocks       = len(valid_stocks)
stock_mdd_vals = valid_stocks.values
beats_bm_mdd   = (stock_mdd_vals > bm_mdd).astype(float)  # less severe = better
n_beats_mdd    = int(beats_bm_mdd.sum())

boot_mdd = np.array([beats_bm_mdd[np.random.randint(0, n_stocks, n_stocks)].mean()
                     for _ in range(N_BOOTSTRAP)])
ci_lo_mdd  = np.percentile(boot_mdd, 2.5)
ci_hi_mdd  = np.percentile(boot_mdd, 97.5)
pt_mdd     = beats_bm_mdd.mean()

p_binom_mdd           = stats.binomtest(n_beats_mdd, n_stocks, 0.5).pvalue
tstat_mdd, p_ttest_mdd = stats.ttest_1samp(stock_mdd_vals, bm_mdd)
ws_mdd, p_wilcox_mdd   = stats.wilcoxon(stock_mdd_vals - bm_mdd)
p_boot_mdd             = bootstrap_pvalue(boot_mdd, 0.5)

print(f"Benchmark MDD: {bm_mdd*100:.2f}%")
print(f"Stocks beating BM: {n_beats_mdd}/{n_stocks} ({pt_mdd*100:.1f}%) "
      f"[{ci_lo_mdd*100:.1f}%, {ci_hi_mdd*100:.1f}%]")
print(f"  Binomial p={p_binom_mdd:.4f} {sig_stars(p_binom_mdd)} | "
      f"T-test p={p_ttest_mdd:.4f} {sig_stars(p_ttest_mdd)} | "
      f"Wilcoxon p={p_wilcox_mdd:.4f} {sig_stars(p_wilcox_mdd)}")

# ── Time-to-recover & time-underwater ─────────────────────────────────────────
print("\nComputing time-to-recover and time-underwater...")
bm_uw, bm_recovery = time_underwater(bm_cum)

stock_uw       = {}
stock_recovery = {}
for t in stock_cum.columns:
    uw, rec = time_underwater(stock_cum[t])
    stock_uw[t]       = uw
    stock_recovery[t] = rec

stock_uw_s       = pd.Series(stock_uw)
stock_recovery_s = pd.Series(stock_recovery)

# % of stocks with shorter max-underwater than benchmark
beats_uw = (stock_uw_s < bm_uw).astype(float)
n_beats_uw = int(beats_uw.sum())
p_binom_uw = stats.binomtest(n_beats_uw, len(beats_uw), 0.5).pvalue

# % of stocks with faster recovery than benchmark
# Fix 5: Never-recovered stocks (NaN) counted as slower via fillna(inf).
#         Using only recovered stocks in the denominator would be selection bias —
#         a stock that never returned to its prior peak is unambiguously worse.
valid_rec = stock_recovery_s.dropna()  # kept for t-test (observed values only)
if not np.isnan(bm_recovery):
    beats_rec_all = (stock_recovery_s.fillna(np.inf) < bm_recovery).astype(float)
    n_beats_rec   = int(beats_rec_all.sum())
    n_total_rec   = len(beats_rec_all)
    p_binom_rec   = stats.binomtest(n_beats_rec, n_total_rec, 0.5).pvalue
else:
    beats_rec_all = pd.Series(dtype=float)
    n_beats_rec   = 0
    n_total_rec   = len(stock_recovery_s)
    p_binom_rec   = np.nan

# T-tests
ts_uw, p_uw_t  = stats.ttest_1samp(stock_uw_s.values, bm_uw)
if not np.isnan(bm_recovery) and len(valid_rec) > 1:
    ts_rec, p_rec_t = stats.ttest_1samp(valid_rec.values, bm_recovery)
else:
    p_rec_t = np.nan

boot_uw = np.array([
    beats_uw.values[np.random.randint(0, len(beats_uw), len(beats_uw))].mean()
    for _ in range(N_BOOTSTRAP)])
ci_uw_lo = np.percentile(boot_uw, 2.5)
ci_uw_hi = np.percentile(boot_uw, 97.5)

boot_rec = np.array([
    beats_rec_all.values[np.random.randint(0, len(beats_rec_all), len(beats_rec_all))].mean()
    for _ in range(N_BOOTSTRAP)])
ci_rec_lo = np.percentile(boot_rec, 2.5)
ci_rec_hi = np.percentile(boot_rec, 97.5)

print(f"Benchmark: max underwater={bm_uw} days, "
      f"recovery={bm_recovery if not np.isnan(bm_recovery) else 'never'} days")
print(f"Median stock (recovered only): underwater={stock_uw_s.median():.0f} days, "
      f"recovery={valid_rec.median():.0f} days")
print(f"Stocks with shorter underwater than BM: {n_beats_uw}/{len(beats_uw)} "
      f"({beats_uw.mean()*100:.1f}%) "
      f"[{ci_uw_lo*100:.1f}%, {ci_uw_hi*100:.1f}%]")
print(f"  Binomial p={p_binom_uw:.4f} {sig_stars(p_binom_uw)} | "
      f"T-test p={p_uw_t:.4f} {sig_stars(p_uw_t)}")
if not np.isnan(bm_recovery):
    print(f"Stocks faster recovery than BM (NaN=slower): {n_beats_rec}/{n_total_rec} "
          f"({beats_rec_all.mean()*100:.1f}%) [{ci_rec_lo*100:.1f}%, {ci_rec_hi*100:.1f}%]")
    print(f"  Binomial p={p_binom_rec:.4f} {sig_stars(p_binom_rec)} | "
          f"T-test p={p_rec_t:.4f} {sig_stars(p_rec_t)}")

# ── Figures ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) Max drawdown histogram
ax = axes[0]
ax.hist(stock_mdd_vals * 100, bins=25, color=BRAND_BLUE, alpha=0.75, edgecolor="white")
ax.axvline(bm_mdd * 100, color=BRAND_RED, linewidth=2.5, linestyle="--",
           label=f"BM MDD ({bm_mdd*100:.1f}%)")
ax.set_title("Max Drawdown Distribution")
ax.set_xlabel("Max Drawdown (%)")
ax.set_ylabel("Count")
ax.legend()
ax.text(0.97, 0.97,
        f"{pt_mdd*100:.0f}% beat BM\np={p_binom_mdd:.3f} {sig_stars(p_binom_mdd)}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BRAND_LIGHT, alpha=0.8))

# (b) Time-underwater histogram
ax = axes[1]
ax.hist(stock_uw_s.values, bins=25, color=BRAND_BLUE, alpha=0.75, edgecolor="white")
ax.axvline(bm_uw, color=BRAND_RED, linewidth=2.5, linestyle="--",
           label=f"BM underwater ({bm_uw} days)")
ax.set_title("Max Time Underwater")
ax.set_xlabel("Trading Days Below Peak")
ax.set_ylabel("Count")
ax.legend()
ax.text(0.97, 0.97,
        f"{beats_uw.mean()*100:.0f}% shorter than BM\n"
        f"p={p_binom_uw:.3f} {sig_stars(p_binom_uw)}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BRAND_LIGHT, alpha=0.8))

# (c) Recovery time histogram
ax = axes[2]
rec_vals = valid_rec.values
ax.hist(rec_vals, bins=25, color=BRAND_BLUE, alpha=0.75, edgecolor="white")
if not np.isnan(bm_recovery):
    ax.axvline(bm_recovery, color=BRAND_RED, linewidth=2.5, linestyle="--",
               label=f"BM recovery ({bm_recovery:.0f} days)")
n_never = stock_recovery_s.isna().sum()
ax.set_title(f"Time-to-Recover from Max Drawdown\n({n_never} stocks never recovered)")
ax.set_xlabel("Trading Days to Recover")
ax.set_ylabel("Count")
ax.legend()
if not np.isnan(bm_recovery):
    ax.text(0.97, 0.97,
            f"{beats_rec_all.mean()*100:.0f}% faster than BM\n"
            f"(NaN=slower, n={n_total_rec})\n"
            f"p={p_binom_rec:.3f} {sig_stars(p_binom_rec)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=BRAND_LIGHT, alpha=0.8))
plt.suptitle("Analysis 1: Downside Resilience", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig("fig1_downside_resilience.png")


# =============================================================================
# ANALYSIS 2: CONSISTENCY
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 2: CONSISTENCY")
print("=" * 70)

monthly_outperform = stock_monthly.gt(bm_monthly, axis=0)
stock_hit_rates    = monthly_outperform.mean()
valid_hit_rates    = stock_hit_rates.dropna()
n_stocks_hr        = len(valid_hit_rates)
hr_vals            = valid_hit_rates.values

boot_hr = np.array([hr_vals[np.random.randint(0, n_stocks_hr, n_stocks_hr)].mean()
                    for _ in range(N_BOOTSTRAP)])
ci_lo_hr   = np.percentile(boot_hr, 2.5)
ci_hi_hr   = np.percentile(boot_hr, 97.5)
pt_hr      = hr_vals.mean()
ts_hr, p_hr_t = stats.ttest_1samp(hr_vals, 0.5)
ws_hr, p_hr_w = stats.wilcoxon(hr_vals - 0.5)
p_hr_b        = bootstrap_pvalue(boot_hr, 0.5)

print(f"Monthly Hit Rate: {pt_hr*100:.1f}% [{ci_lo_hr*100:.1f}%, {ci_hi_hr*100:.1f}%]")
print(f"  T-test p={p_hr_t:.4f} {sig_stars(p_hr_t)} | "
      f"Wilcoxon p={p_hr_w:.4f} {sig_stars(p_hr_w)}")


def rolling_hit_rate(sm, bm, window):
    sr = sm.rolling(window).apply(lambda x: (1 + x).prod() - 1, raw=True)
    br = bm.rolling(window).apply(lambda x: (1 + x).prod() - 1, raw=True)
    return sr.gt(br, axis=0).dropna(how="all").mean()


results_rolling = {}
for wname, wsz in [("3-Month", 3), ("6-Month", 6)]:
    rhr    = rolling_hit_rate(stock_monthly, bm_monthly, wsz).dropna()
    n_r    = len(rhr)
    r_vals = rhr.values
    boot_r = np.array([r_vals[np.random.randint(0, n_r, n_r)].mean()
                       for _ in range(N_BOOTSTRAP)])
    ts, pv = stats.ttest_1samp(r_vals, 0.5)
    ws, wp = stats.wilcoxon(r_vals - 0.5)
    results_rolling[wname] = dict(
        pt=r_vals.mean(), ci_lo=np.percentile(boot_r, 2.5),
        ci_hi=np.percentile(boot_r, 97.5), p_t=pv, p_w=wp)
    print(f"{wname} Rolling: {r_vals.mean()*100:.1f}% "
          f"[{np.percentile(boot_r,2.5)*100:.1f}%, {np.percentile(boot_r,97.5)*100:.1f}%] "
          f"T-test p={pv:.4f} {sig_stars(pv)}")

# Down-month hit rate
down_months = bm_monthly[bm_monthly < 0].index
print(f"Down months: {len(down_months)}")
if len(down_months) > 0:
    stock_down    = stock_monthly.loc[down_months]
    bm_down       = bm_monthly.loc[down_months]
    down_op       = stock_down.gt(bm_down, axis=0)
    stock_down_hr = down_op.mean()
    valid_down_hr = stock_down_hr.dropna()
    d_vals        = valid_down_hr.values
    boot_d        = np.array([d_vals[np.random.randint(0, len(d_vals), len(d_vals))].mean()
                              for _ in range(N_BOOTSTRAP)])
    pt_d   = d_vals.mean()
    ci_d_lo = np.percentile(boot_d, 2.5)
    ci_d_hi = np.percentile(boot_d, 97.5)
    ts_d, p_d_t = stats.ttest_1samp(d_vals, 0.5)
    ws_d, p_d_w = stats.wilcoxon(d_vals - 0.5)
    print(f"Down-Month Hit Rate: {pt_d*100:.1f}% [{ci_d_lo*100:.1f}%, {ci_d_hi*100:.1f}%] "
          f"T-test p={p_d_t:.4f} {sig_stars(p_d_t)}")
else:
    pt_d = ci_d_lo = ci_d_hi = p_d_t = p_d_w = np.nan
    stock_down_hr = pd.Series(dtype=float)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(hr_vals * 100, bins=25, color=BRAND_BLUE, alpha=0.75, edgecolor="white")
ax.axvline(50, color=BRAND_RED, linewidth=2, linestyle="--", label="50% (random)")
ax.axvline(pt_hr * 100, color=BRAND_GOLD, linewidth=2,
           label=f"Mean: {pt_hr*100:.1f}%")
ax.set_title("Per-Stock Monthly Hit Rate Distribution")
ax.set_xlabel("Monthly Hit Rate (%)")
ax.set_ylabel("Count")
ax.legend()
ax.text(0.97, 0.97,
        f"T-test vs 50%: p={p_hr_t:.3f} {sig_stars(p_hr_t)}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BRAND_LIGHT, alpha=0.8))

ax = axes[1]
labels = ["Monthly", "3-Month\nRolling", "6-Month\nRolling", "Down-Month"]
pts  = [pt_hr, results_rolling["3-Month"]["pt"],
        results_rolling["6-Month"]["pt"], pt_d]
lows = [ci_lo_hr, results_rolling["3-Month"]["ci_lo"],
        results_rolling["6-Month"]["ci_lo"], ci_d_lo]
highs = [ci_hi_hr, results_rolling["3-Month"]["ci_hi"],
         results_rolling["6-Month"]["ci_hi"], ci_d_hi]
x = np.arange(len(labels))
bars = ax.bar(x, [p * 100 for p in pts],
              color=[BRAND_BLUE, BRAND_BLUE, BRAND_BLUE, BRAND_GOLD],
              alpha=0.8, width=0.5, edgecolor="white")
for j in range(len(x)):
    if not np.isnan(lows[j]):
        ax.errorbar(x[j], pts[j]*100,
                    yerr=[[pts[j]*100-lows[j]*100], [highs[j]*100-pts[j]*100]],
                    fmt="none", color="black", capsize=5, linewidth=1.5)
ax.axhline(50, color=BRAND_RED, linewidth=1.5, linestyle="--", label="50%")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 80)
ax.set_ylabel("Hit Rate (%)")
ax.set_title("Hit Rates Across Horizons (Error bars = 95% CI)")
ax.legend()
for bar, pt in zip(bars, pts):
    if not np.isnan(pt):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1.5,
                f"{pt*100:.1f}%", ha="center", fontsize=9, fontweight="bold")
save_fig("fig2_consistency.png")


# =============================================================================
# ANALYSIS 3: UPSIDE / DOWNSIDE CAPTURE
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 3: UPSIDE / DOWNSIDE CAPTURE")
print("=" * 70)

WINDOW_SIZE = 60
n_days      = len(stock_returns)
max_start   = n_days - WINDOW_SIZE
up_days     = bm_daily > 0
down_days   = bm_daily < 0

stock_up_capture   = {}
stock_down_capture = {}
for t in stock_returns.columns:
    sr = stock_returns[t]
    if up_days.sum() > 0:
        stock_up_capture[t]   = sr[up_days].mean() / bm_daily[up_days].mean()
    if down_days.sum() > 0:
        stock_down_capture[t] = sr[down_days].mean() / bm_daily[down_days].mean()

up_cap_s   = pd.Series(stock_up_capture)
down_cap_s = pd.Series(stock_down_capture)

ts_uc, p_uc_t = stats.ttest_1samp(up_cap_s.dropna().values, 1.0)
ts_dc, p_dc_t = stats.ttest_1samp(down_cap_s.dropna().values, 1.0)
ws_uc, p_uc_w = stats.wilcoxon(up_cap_s.dropna().values - 1.0)
ws_dc, p_dc_w = stats.wilcoxon(down_cap_s.dropna().values - 1.0)

print(f"Avg upside capture  : {up_cap_s.mean()*100:.1f}% "
      f"T-test p={p_uc_t:.4f} {sig_stars(p_uc_t)}")
print(f"Avg downside capture: {down_cap_s.mean()*100:.1f}% "
      f"T-test p={p_dc_t:.4f} {sig_stars(p_dc_t)}")

# Nested bootstrap
stock_ret_arr = stock_returns.values
bm_arr        = bm_daily.values
n_stocks_cap  = stock_ret_arr.shape[1]
boot_up       = np.empty(N_BOOTSTRAP)
boot_dn       = np.empty(N_BOOTSTRAP)

print(f"Running nested bootstrap ({N_BOOTSTRAP:,} iterations)...")
for i in range(N_BOOTSTRAP):
    start  = np.random.randint(0, max_start + 1)
    wstock = stock_ret_arr[start:start + WINDOW_SIZE, :]
    wbm    = bm_arr[start:start + WINDOW_SIZE]
    w_up   = wbm > 0
    w_dn   = wbm < 0
    samp   = np.random.randint(0, n_stocks_cap, size=n_stocks_cap)
    uc = np.full(n_stocks_cap, np.nan)
    dc = np.full(n_stocks_cap, np.nan)
    if w_up.sum() > 0:
        mu = wbm[w_up].mean()
        if mu != 0:
            uc = wstock[w_up, :][:, samp].mean(axis=0) / mu
    if w_dn.sum() > 0:
        md = wbm[w_dn].mean()
        if md != 0:
            dc = wstock[w_dn, :][:, samp].mean(axis=0) / md
    boot_up[i] = np.nanmean(uc > 1)
    boot_dn[i] = np.nanmean(dc < 1)

bpu = boot_up[~np.isnan(boot_up)]
bpd = boot_dn[~np.isnan(boot_dn)]
up_pt, up_ci_lo, up_ci_hi = bpu.mean(), np.percentile(bpu, 2.5), np.percentile(bpu, 97.5)
dn_pt, dn_ci_lo, dn_ci_hi = bpd.mean(), np.percentile(bpd, 2.5), np.percentile(bpd, 97.5)
p_boot_uc = bootstrap_pvalue(bpu, 0.5, "greater")
p_boot_dc = bootstrap_pvalue(bpd, 0.5, "greater")

print(f"Upside capture > 100% : {up_pt*100:.1f}% "
      f"[{up_ci_lo*100:.1f}%, {up_ci_hi*100:.1f}%] p={p_boot_uc:.4f} {sig_stars(p_boot_uc)}")
print(f"Downside capture < 100%: {dn_pt*100:.1f}% "
      f"[{dn_ci_lo*100:.1f}%, {dn_ci_hi*100:.1f}%] p={p_boot_dc:.4f} {sig_stars(p_boot_dc)}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
uc_v = up_cap_s.values * 100
dc_v = down_cap_s.reindex(up_cap_s.index).values * 100
sector_map = firmographics.set_index("Symbol")["Sectors"]
sector_palette = dict(zip(
    firmographics["Sectors"].dropna().unique(),
    sns.color_palette("tab10", n_colors=12)))
colors_s = []
for t in up_cap_s.index:
    sv = sector_map.get(t, None)
    if isinstance(sv, pd.Series):
        sv = sv.iloc[0]
    colors_s.append(sector_palette.get(sv, BRAND_GREY))

ax.scatter(uc_v, dc_v, c=colors_s, alpha=0.75, s=60, edgecolors="white")
ax.axhline(100, color="black", linewidth=1, linestyle="--", alpha=0.5)
ax.axvline(100, color="black", linewidth=1, linestyle="--", alpha=0.5)
ax.set_xlabel("Upside Capture (%)")
ax.set_ylabel("Downside Capture (%)")
ax.set_title("Upside vs. Downside Capture\n(Ideal: right of 100%, below 100%)")
sector_counts = firmographics.groupby("Sectors").size()
for sec, color in sector_palette.items():
    if sector_counts.get(sec, 0) >= 3:
        ax.scatter([], [], c=[color], label=sec, s=40, alpha=0.8)
ax.legend(fontsize=7, loc="upper left", ncol=2)
ax.text(0.97, 0.03,
        f"Upside T-test vs 100%: p={p_uc_t:.3f} {sig_stars(p_uc_t)}\n"
        f"Downside T-test vs 100%: p={p_dc_t:.3f} {sig_stars(p_dc_t)}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BRAND_LIGHT, alpha=0.8))

ax = axes[1]
ax.hist(bpu * 100, bins=50, color=BRAND_GREEN, alpha=0.6,
        label=f"Up capture > 100% ({up_pt*100:.1f}%)", edgecolor="white")
ax.hist(bpd * 100, bins=50, color=BRAND_BLUE, alpha=0.6,
        label=f"Down capture < 100% ({dn_pt*100:.1f}%)", edgecolor="white")
ax.axvline(50, color=BRAND_RED, linewidth=1.5, linestyle="--", label="50% (null)")
ax.set_xlabel("% of Stocks with Superior Capture")
ax.set_ylabel("Bootstrap Frequency")
ax.set_title("Nested Bootstrap — Capture Profiles (95% CI shaded)")
ax.legend()
save_fig("fig3_capture_ratios.png")


# =============================================================================
# ANALYSIS 4: TAIL-RISK STORY
#   - Worst months (bottom 5%)
#   - Drawdown clustering: do honorees' drawdowns correlate?
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 4: TAIL-RISK STORY")
print("=" * 70)

# ── Worst months ──────────────────────────────────────────────────────────────
threshold_pct = 5
threshold_val = np.percentile(bm_monthly.dropna(), threshold_pct)
worst_months  = bm_monthly[bm_monthly <= threshold_val].index
stock_worst   = stock_monthly.loc[worst_months]
bm_worst      = bm_monthly.loc[worst_months]

avg_stock_worst = stock_worst.mean(axis=1).mean()
avg_bm_worst    = bm_worst.mean()
avg_excess      = avg_stock_worst - avg_bm_worst

worst_outperform = stock_worst.gt(bm_worst, axis=0)
worst_hit_rates  = worst_outperform.mean()
valid_worst_hr   = worst_hit_rates.dropna()
breadth          = worst_outperform.sum(axis=1)
total_stocks_w   = stock_worst.shape[1]

stock_worst_means = stock_worst.mean(axis=0).dropna()
excess_per_stock  = stock_worst_means.values - avg_bm_worst
ts_tail, p_tail_t   = stats.ttest_1samp(excess_per_stock, 0)
ws_tail, p_tail_w    = stats.wilcoxon(excess_per_stock)
n_beats_tail         = int((valid_worst_hr.values > 0.5).sum())
p_binom_tail         = stats.binomtest(n_beats_tail, len(valid_worst_hr), 0.5,
                                        alternative="greater").pvalue

boot_excess  = np.array([
    stock_worst_means.values[np.random.randint(0, len(stock_worst_means),
                                                len(stock_worst_means))].mean()
    - avg_bm_worst for _ in range(N_BOOTSTRAP)])
boot_whr     = np.array([
    valid_worst_hr.values[np.random.randint(0, len(valid_worst_hr),
                                             len(valid_worst_hr))].mean()
    for _ in range(N_BOOTSTRAP)])
boot_breadth = np.array([
    breadth.values[np.random.randint(0, len(breadth), len(breadth))].mean()
    for _ in range(N_BOOTSTRAP)])

ci_ex_lo  = np.percentile(boot_excess, 2.5)
ci_ex_hi  = np.percentile(boot_excess, 97.5)
ci_whr_lo = np.percentile(boot_whr, 2.5)
ci_whr_hi = np.percentile(boot_whr, 97.5)
ci_br_lo  = np.percentile(boot_breadth, 2.5)
ci_br_hi  = np.percentile(boot_breadth, 97.5)

print(f"Worst months ({threshold_pct}%): "
      f"{[d.strftime('%Y-%m') for d in worst_months]}")
print(f"Avg excess: {avg_excess*100:.2f}% [{ci_ex_lo*100:.2f}%, {ci_ex_hi*100:.2f}%] "
      f"T-test p={p_tail_t:.4f} {sig_stars(p_tail_t)}")
print(f"Hit rate: {valid_worst_hr.mean()*100:.1f}% "
      f"[{ci_whr_lo*100:.1f}%, {ci_whr_hi*100:.1f}%] "
      f"Binomial p={p_binom_tail:.4f} {sig_stars(p_binom_tail)}")
print(f"Breadth: {breadth.mean():.0f} [{ci_br_lo:.0f}, {ci_br_hi:.0f}] "
      f"/ {total_stocks_w}")

# ── Drawdown clustering ──────────────────────────────────────────────────────
# Do honoree drawdowns correlate?  High correlation = they drop together
# Compute pairwise correlation of daily returns during down-market periods
print("\nDrawdown Clustering:")
down_market_days = bm_daily[bm_daily < 0].index
stock_down_daily = stock_returns.loc[down_market_days]

# Average pairwise correlation during down days
corr_matrix = stock_down_daily.corr()
# Extract upper triangle (exclude diagonal)
mask_tri = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
pairwise_corrs = corr_matrix.values[mask_tri]
avg_corr_down = np.nanmean(pairwise_corrs)

# Compare with full-period average correlation
corr_matrix_all = stock_returns.corr()
pairwise_corrs_all = corr_matrix_all.values[mask_tri]
avg_corr_all = np.nanmean(pairwise_corrs_all)

# Are correlations higher during down markets? (paired t-test)
ts_corr, p_corr = stats.ttest_rel(pairwise_corrs, pairwise_corrs_all)

# What fraction of stocks draw down together?  In worst months, what % have
# negative returns?
pct_neg_worst = (stock_worst < 0).mean(axis=1)

print(f"  Avg pairwise correlation (all days): {avg_corr_all:.3f}")
print(f"  Avg pairwise correlation (down days): {avg_corr_down:.3f}")
print(f"  Paired t-test (down > all): p={p_corr:.4f} {sig_stars(p_corr)}")
print(f"  Avg % of honorees with negative returns in worst months: "
      f"{pct_neg_worst.mean()*100:.1f}%")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) Worst months bar chart
ax = axes[0]
mlabels = [d.strftime("%b %Y") for d in worst_months]
x  = np.arange(len(worst_months))
w  = 0.35
ax.bar(x - w/2, bm_worst.values * 100, w, label="Benchmark",
       color=BRAND_GREY, alpha=0.85, edgecolor="white")
ax.bar(x + w/2, stock_worst.mean(axis=1).values * 100, w, label="Avg Honoree",
       color=BRAND_BLUE, alpha=0.85, edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(mlabels, rotation=15)
ax.set_ylabel("Return (%)")
ax.set_title(f"Worst Months — Excess: {avg_excess*100:.2f}%\n"
             f"T-test p={p_tail_t:.3f} {sig_stars(p_tail_t)}")
ax.axhline(0, color="black", linewidth=0.8)
ax.legend()

# (b) Breadth
ax = axes[1]
bars = ax.bar(mlabels, breadth.values, color=BRAND_BLUE, alpha=0.8, edgecolor="white")
ax.axhline(total_stocks_w / 2, color=BRAND_RED, linewidth=1.5, linestyle="--",
           label="50% of stocks")
ax.set_ylabel("Stocks Beating Benchmark")
ax.set_title(f"Breadth — Avg {breadth.mean():.0f}/{total_stocks_w} per month")
for bar, val in zip(bars, breadth.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.5,
            str(int(val)), ha="center", fontsize=10, fontweight="bold")
ax.legend()

# (c) Drawdown clustering: correlation histogram
ax = axes[2]
ax.hist(pairwise_corrs_all, bins=50, alpha=0.5, color=BRAND_GREY,
        label=f"All days (avg={avg_corr_all:.3f})", edgecolor="white")
ax.hist(pairwise_corrs, bins=50, alpha=0.5, color=BRAND_RED,
        label=f"Down days (avg={avg_corr_down:.3f})", edgecolor="white")
ax.set_xlabel("Pairwise Return Correlation")
ax.set_ylabel("Frequency")
ax.set_title(f"Drawdown Clustering\np={p_corr:.4f} {sig_stars(p_corr)}")
ax.legend()

plt.suptitle("Analysis 4: Tail-Risk Story", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig("fig4_tail_risk.png")


# =============================================================================
# ANALYSIS 5: SEGMENTATION
#   Sector, Industry Group, Country, Workforce, Revenue, Structure
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 5: SEGMENTATION")
print("=" * 70)

stock_summary = pd.DataFrame({
    "Ticker":              list(tickers),
    "Max_Drawdown_Pct":    stock_mdd.reindex(tickers).values * 100,
    "Time_Underwater_Days": stock_uw_s.reindex(tickers).values,
    "Recovery_Days":       stock_recovery_s.reindex(tickers).values,
    "Monthly_Hit_Rate_Pct": stock_hit_rates.reindex(tickers).values * 100,
    "Upside_Capture_Pct":  up_cap_s.reindex(tickers).values * 100,
    "Downside_Capture_Pct": down_cap_s.reindex(tickers).values * 100,
    "Down_Month_HR_Pct":   (stock_down_hr.reindex(tickers).values * 100
                             if len(down_months) > 0 else np.nan),
    "Worst_Month_HR_Pct":  worst_hit_rates.reindex(tickers).values * 100,
}).merge(
    firmographics[["Symbol", "Sectors", "Industry Groups", "Domicile Country",
                   "Workforce Size", "Annual Revenue", "Structure"]],
    left_on="Ticker", right_on="Symbol", how="left"
).drop(columns=["Symbol"])

# ── Output CSV 1: stock_level_results.csv ─────────────────────────────────────
stock_summary.to_csv(os.path.join(OUTPUT_DIR, "stock_level_results.csv"), index=False)

segment_dims = {
    "Sector":         "Sectors",
    "Industry_Group": "Industry Groups",
    "Country":        "Domicile Country",
    "Workforce_Size": "Workforce Size",
    "Revenue":        "Annual Revenue",
    "Structure":      "Structure",
}

seg_frames = []
for seg_label, seg_col in segment_dims.items():
    if seg_col not in stock_summary.columns:
        continue
    grouped = stock_summary.dropna(subset=[seg_col]).groupby(seg_col)
    seg = grouped.agg(
        N=("Ticker", "count"),
        Avg_MDD_Pct=("Max_Drawdown_Pct", "mean"),
        Avg_Underwater_Days=("Time_Underwater_Days", "mean"),
        Avg_Monthly_HR_Pct=("Monthly_Hit_Rate_Pct", "mean"),
        Avg_Up_Capture_Pct=("Upside_Capture_Pct", "mean"),
        Avg_Dn_Capture_Pct=("Downside_Capture_Pct", "mean"),
        Avg_Down_Month_HR_Pct=("Down_Month_HR_Pct", "mean"),
        Avg_Worst_Month_HR_Pct=("Worst_Month_HR_Pct", "mean"),
    ).round(2)
    seg = seg[seg["N"] >= 3].sort_values("Avg_Monthly_HR_Pct", ascending=False)
    seg.insert(0, "Segment_Type", seg_label)
    seg.index.name = "Segment_Value"
    seg = seg.reset_index()
    seg_frames.append(seg)

    # Kruskal-Wallis for monthly hit rate
    groups = [g["Monthly_Hit_Rate_Pct"].dropna().values
              for _, g in stock_summary.dropna(
                  subset=[seg_col, "Monthly_Hit_Rate_Pct"]).groupby(seg_col)
              if len(g) >= 3]
    if len(groups) >= 2:
        kw_stat, kw_p = stats.kruskal(*groups)
        print(f"{seg_label}: KW H={kw_stat:.2f}, p={kw_p:.4f} {sig_stars(kw_p)}")
    print(seg[["Segment_Value", "N", "Avg_Monthly_HR_Pct",
               "Avg_Up_Capture_Pct", "Avg_Dn_Capture_Pct"]].to_string(index=False))
    print()

# ── Output CSV 2: segment_results.csv ─────────────────────────────────────────
all_segments = pd.concat(seg_frames, ignore_index=True)
all_segments.to_csv(os.path.join(OUTPUT_DIR, "segment_results.csv"), index=False)

# ── Figures ───────────────────────────────────────────────────────────────────
# Sector metrics bar chart
sec_seg = all_segments[all_segments["Segment_Type"] == "Sector"].copy()
if len(sec_seg) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = [
        ("Avg_Monthly_HR_Pct",  "Monthly Hit Rate (%)", 50),
        ("Avg_Up_Capture_Pct",  "Upside Capture (%)",   100),
        ("Avg_Dn_Capture_Pct",  "Downside Capture (%)", 100),
    ]
    labels_s = [s.replace(" ", "\n") for s in sec_seg["Segment_Value"]]
    x = np.arange(len(sec_seg))
    for ax, (col, title, ref) in zip(axes, metrics):
        vals = sec_seg[col].values
        colors_b = [BRAND_BLUE if (col != "Avg_Dn_Capture_Pct" and v > ref)
                     or (col == "Avg_Dn_Capture_Pct" and v < ref)
                     else BRAND_GOLD for v in vals]
        ax.barh(x, vals, color=colors_b, alpha=0.85, edgecolor="white")
        ax.axvline(ref, color=BRAND_RED, linewidth=1.5, linestyle="--")
        ax.set_yticks(x)
        ax.set_yticklabels(labels_s, fontsize=8)
        ax.set_title(title, fontweight="bold")
        ax.invert_yaxis()
        for i, v in enumerate(vals):
            ax.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=8)
    plt.suptitle("Sector Segmentation", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig("fig5a_sector_metrics.png")

# Sector heatmap
if len(sec_seg) > 0:
    heat_cols = ["Avg_MDD_Pct", "Avg_Underwater_Days", "Avg_Monthly_HR_Pct",
                 "Avg_Up_Capture_Pct", "Avg_Dn_Capture_Pct",
                 "Avg_Down_Month_HR_Pct", "Avg_Worst_Month_HR_Pct"]
    heat_labels = ["Max DD %", "Underwater\nDays", "Monthly\nHit Rate %",
                   "Up Cap %", "Down Cap %", "Down-Mo\nHR %", "Worst-Mo\nHR %"]
    heat_data = sec_seg.set_index("Segment_Value")[heat_cols]
    heat_data.index = [s.replace(" ", "\n") for s in heat_data.index]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heat_data.values, xticklabels=heat_labels,
                yticklabels=heat_data.index,
                annot=True, fmt=".1f", cmap="RdYlGn",
                linewidths=0.5, ax=ax, annot_kws={"size": 9})
    ax.set_title("Sector Heatmap — All Metrics", fontweight="bold", pad=12)
    plt.tight_layout()
    save_fig("fig5b_sector_heatmap.png")

# Country hit rate bar
cty_seg = all_segments[all_segments["Segment_Type"] == "Country"].copy().head(10)
if len(cty_seg) > 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    x  = np.arange(len(cty_seg))
    hr = cty_seg["Avg_Monthly_HR_Pct"].values
    ax.bar(x, hr,
           color=[BRAND_BLUE if v > 50 else BRAND_GOLD for v in hr],
           alpha=0.85, edgecolor="white")
    ax.axhline(50, color=BRAND_RED, linewidth=1.5, linestyle="--", label="50%")
    ax.set_xticks(x)
    ax.set_xticklabels(cty_seg["Segment_Value"], rotation=20, ha="right")
    ax.set_ylabel("Avg Monthly Hit Rate (%)")
    ax.set_title("Country Segmentation — Monthly Hit Rate")
    for i, (v, n) in enumerate(zip(hr, cty_seg["N"].values)):
        ax.text(i, v + 0.3, f"{v:.1f}%\n(n={n})", ha="center", fontsize=8)
    ax.legend()
    plt.tight_layout()
    save_fig("fig5c_country_hit_rate.png")


# =============================================================================
# ENHANCED ANALYSES
# =============================================================================
print("\n" + "=" * 70)
print("ENHANCED ANALYSES")
print("=" * 70)

# ── E1. Year-by-year performance breakdown ─────────────────────────────────
print("\n--- Year-by-Year Performance ---")
years = sorted(stock_monthly.index.year.unique())
yoy_rows = []
for yr in years:
    mask = stock_monthly.index.year == yr
    sm_yr = stock_monthly[mask]
    bm_yr = bm_monthly[mask]
    if len(bm_yr) == 0:
        continue
    # Hit rate this year
    outperf_yr = sm_yr.gt(bm_yr, axis=0)
    hr_yr      = outperf_yr.mean()
    avg_hr_yr  = hr_yr.mean()
    # Cumulative return this year
    sr_yr  = stock_returns[stock_returns.index.year == yr]
    bm_d_yr = bm_daily[bm_daily.index.year == yr]
    stock_cum_yr = (1 + sr_yr).prod() - 1  # per stock
    bm_cum_yr    = (1 + bm_d_yr).prod() - 1
    pct_beat_cum = (stock_cum_yr > bm_cum_yr).mean()
    avg_stock_ret = stock_cum_yr.mean()
    # Down-month HR this year
    down_yr = bm_yr[bm_yr < 0].index
    if len(down_yr) > 0:
        dm_hr_yr = sm_yr.loc[down_yr].gt(bm_yr.loc[down_yr], axis=0).mean().mean()
    else:
        dm_hr_yr = np.nan
    yoy_rows.append({
        "Year": yr,
        "Months": len(bm_yr),
        "BM_Return_Pct": bm_cum_yr * 100,
        "Avg_Honoree_Return_Pct": avg_stock_ret * 100,
        "Pct_Beat_BM_Cumulative": pct_beat_cum * 100,
        "Avg_Monthly_HR_Pct": avg_hr_yr * 100,
        "Down_Month_HR_Pct": dm_hr_yr * 100 if not np.isnan(dm_hr_yr) else np.nan,
        "N_Down_Months": len(down_yr),
    })
    print(f"  {yr}: BM={bm_cum_yr*100:+.1f}% | Avg Honoree={avg_stock_ret*100:+.1f}% | "
          f"Monthly HR={avg_hr_yr*100:.1f}% | {pct_beat_cum*100:.0f}% beat BM annually")

yoy_df = pd.DataFrame(yoy_rows)

# ── E2. Risk-adjusted metrics ──────────────────────────────────────────────
print("\n--- Risk-Adjusted Metrics ---")
ann_factor    = 252
ann_stock_ret = stock_returns.mean() * ann_factor
ann_stock_vol = stock_returns.std() * np.sqrt(ann_factor)
ann_bm_ret    = bm_daily.mean() * ann_factor
ann_bm_vol    = bm_daily.std() * np.sqrt(ann_factor)

# Sharpe (assuming risk-free = 0 for simplicity)
stock_sharpe  = ann_stock_ret / ann_stock_vol
bm_sharpe     = ann_bm_ret / ann_bm_vol

# Sortino (downside deviation)
stock_down_dev = stock_returns[stock_returns < 0].std() * np.sqrt(ann_factor)
bm_down_dev    = bm_daily[bm_daily < 0].std() * np.sqrt(ann_factor)
stock_sortino  = ann_stock_ret / stock_down_dev
bm_sortino     = ann_bm_ret / bm_down_dev

# Information ratio (excess return / tracking error vs benchmark)
excess_daily  = stock_returns.sub(bm_daily, axis=0)
tracking_err  = excess_daily.std() * np.sqrt(ann_factor)
info_ratio    = (excess_daily.mean() * ann_factor) / tracking_err

# Full-period batting average — % of stocks that beat BM over entire 5 years
stock_total_ret = stock_cum.iloc[-1] - 1
bm_total_ret    = bm_cum.iloc[-1] - 1
pct_beat_full   = (stock_total_ret > bm_total_ret).mean()
binom_full      = stats.binomtest(int((stock_total_ret > bm_total_ret).sum()),
                                   len(stock_total_ret), 0.5)
p_full_bat      = binom_full.pvalue

print(f"Benchmark: ann ret={ann_bm_ret*100:.1f}%, vol={ann_bm_vol*100:.1f}%, "
      f"Sharpe={bm_sharpe:.3f}, Sortino={bm_sortino:.3f}")
print(f"Avg honoree: ann ret={ann_stock_ret.mean()*100:.1f}%, "
      f"vol={ann_stock_vol.mean()*100:.1f}%, "
      f"Sharpe={stock_sharpe.mean():.3f}, Sortino={stock_sortino.mean():.3f}")
print(f"Avg Information Ratio: {info_ratio.mean():.3f}")
print(f"Full-period batting avg: {pct_beat_full*100:.1f}% beat BM over 5 years "
      f"(p={p_full_bat:.4f} {sig_stars(p_full_bat)})")

# Stocks with higher Sharpe than BM
pct_sharpe = (stock_sharpe > bm_sharpe).mean()
p_sharpe   = stats.binomtest(int((stock_sharpe > bm_sharpe).sum()),
                              len(stock_sharpe), 0.5).pvalue
print(f"Stocks with Sharpe > BM: {pct_sharpe*100:.1f}% "
      f"(p={p_sharpe:.4f} {sig_stars(p_sharpe)})")

# Add to stock_summary
stock_summary["Ann_Return_Pct"]    = ann_stock_ret.reindex(tickers).values * 100
stock_summary["Ann_Volatility_Pct"] = ann_stock_vol.reindex(tickers).values * 100
stock_summary["Sharpe_Ratio"]      = stock_sharpe.reindex(tickers).values
stock_summary["Sortino_Ratio"]     = stock_sortino.reindex(tickers).values
stock_summary["Info_Ratio"]        = info_ratio.reindex(tickers).values
stock_summary["Total_Return_Pct"]  = stock_total_ret.reindex(tickers).values * 100

# Re-save updated stock_level_results
stock_summary.to_csv(os.path.join(OUTPUT_DIR, "stock_level_results.csv"), index=False)

# ── E3. Broader tail risk (5%, 10%, 25% thresholds) ───────────────────────
print("\n--- Broader Tail-Risk Analysis ---")
tail_rows = []
for pct in [5, 10, 25]:
    thresh = np.percentile(bm_monthly.dropna(), pct)
    wm     = bm_monthly[bm_monthly <= thresh].index
    sw     = stock_monthly.loc[wm]
    bw     = bm_monthly.loc[wm]
    exc    = sw.mean(axis=1).mean() - bw.mean()
    hr_t   = sw.gt(bw, axis=0).mean().mean()
    br_t   = sw.gt(bw, axis=0).sum(axis=1).mean()
    tail_rows.append({
        "Threshold": f"Bottom {pct}%",
        "N_Months": len(wm),
        "Avg_BM_Return_Pct": bw.mean() * 100,
        "Avg_Excess_Pct": exc * 100,
        "Hit_Rate_Pct": hr_t * 100,
        "Avg_Breadth": br_t,
    })
    print(f"  Bottom {pct}% ({len(wm)} months): "
          f"excess={exc*100:+.2f}%, hit rate={hr_t*100:.1f}%, "
          f"breadth={br_t:.0f}/{total_stocks_w}")
tail_df = pd.DataFrame(tail_rows)

# ── E4. Figures — enhanced ──────────────────────────────────────────────────

# Fig 6: Drawdown timeline
print()
fig, ax = plt.subplots(figsize=(12, 5))
bm_dd_ts = (bm_cum - bm_cum.cummax()) / bm_cum.cummax() * 100
avg_stock_cum = stock_cum.mean(axis=1)
avg_stock_dd = (avg_stock_cum - avg_stock_cum.cummax()) / avg_stock_cum.cummax() * 100
ax.fill_between(bm_dd_ts.index, bm_dd_ts, 0, alpha=0.3, color=BRAND_GREY,
                label="Benchmark drawdown")
ax.plot(avg_stock_dd.index, avg_stock_dd, color=BRAND_BLUE, linewidth=1.5,
        label="Avg honoree drawdown")
ax.set_title("Drawdown Timeline — Benchmark vs. Average Honoree")
ax.set_ylabel("Drawdown (%)")
ax.set_xlabel("")
ax.legend()
ax.set_ylim(top=2)
save_fig("fig6_drawdown_timeline.png")

# Fig 7: Rolling 12-month excess return
fig, ax = plt.subplots(figsize=(12, 5))
avg_stock_monthly = stock_monthly.mean(axis=1)
excess_monthly    = avg_stock_monthly - bm_monthly
rolling_12m_excess = excess_monthly.rolling(12).sum() * 100
ax.plot(rolling_12m_excess.index, rolling_12m_excess, color=BRAND_BLUE, linewidth=1.5)
ax.axhline(0, color=BRAND_RED, linewidth=1, linestyle="--")
ax.fill_between(rolling_12m_excess.index, rolling_12m_excess, 0,
                where=rolling_12m_excess >= 0, alpha=0.3, color=BRAND_GREEN)
ax.fill_between(rolling_12m_excess.index, rolling_12m_excess, 0,
                where=rolling_12m_excess < 0, alpha=0.3, color=BRAND_RED)
ax.set_title("Rolling 12-Month Cumulative Excess Return (Avg Honoree vs Benchmark)")
ax.set_ylabel("Cumulative Excess Return (%)")
save_fig("fig7_rolling_12m_excess.png")

# Fig 8: Year-by-year comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
x = np.arange(len(yoy_df))
w = 0.35
ax.bar(x - w/2, yoy_df["BM_Return_Pct"], w, label="Benchmark",
       color=BRAND_GREY, alpha=0.85, edgecolor="white")
ax.bar(x + w/2, yoy_df["Avg_Honoree_Return_Pct"], w, label="Avg Honoree",
       color=BRAND_BLUE, alpha=0.85, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(yoy_df["Year"].astype(int))
ax.set_ylabel("Annual Return (%)")
ax.set_title("Year-by-Year Returns")
ax.axhline(0, color="black", linewidth=0.8)
ax.legend()

ax = axes[1]
ax.bar(x - w/2, yoy_df["Avg_Monthly_HR_Pct"], w, label="Monthly HR",
       color=BRAND_BLUE, alpha=0.85, edgecolor="white")
ax.bar(x + w/2, yoy_df["Pct_Beat_BM_Cumulative"], w, label="% Beat BM (Annual)",
       color=BRAND_GOLD, alpha=0.85, edgecolor="white")
ax.axhline(50, color=BRAND_RED, linewidth=1.5, linestyle="--", label="50%")
ax.set_xticks(x)
ax.set_xticklabels(yoy_df["Year"].astype(int))
ax.set_ylabel("%")
ax.set_title("Year-by-Year Hit Rate & Batting Average")
ax.legend()
save_fig("fig8_year_by_year.png")

# Fig 9: Monthly excess return heatmap
excess_df = (stock_monthly.mean(axis=1) - bm_monthly) * 100
heatmap_data = excess_df.to_frame("excess")
heatmap_data["Year"]  = heatmap_data.index.year
heatmap_data["Month"] = heatmap_data.index.month
hm_pivot = heatmap_data.pivot(index="Year", columns="Month", values="excess")
hm_pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(hm_pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
            linewidths=0.5, ax=ax, annot_kws={"size": 9},
            cbar_kws={"label": "Excess Return (%)"})
ax.set_title("Monthly Excess Return Heatmap — Avg Honoree vs Benchmark (%)",
             fontweight="bold")
ax.set_ylabel("")
plt.tight_layout()
save_fig("fig9_monthly_excess_heatmap.png")

# Fig 10: Risk-return scatter (annualized)
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(ann_stock_vol.values * 100, ann_stock_ret.values * 100,
           c=colors_s if len(colors_s) == len(ann_stock_vol) else BRAND_BLUE,
           alpha=0.65, s=50, edgecolors="white")
ax.scatter([ann_bm_vol * 100], [ann_bm_ret * 100],
           c=BRAND_RED, s=200, marker="*", zorder=5, label="Benchmark")
ax.set_xlabel("Annualized Volatility (%)")
ax.set_ylabel("Annualized Return (%)")
ax.set_title("Risk-Return Profile — Honorees vs. Benchmark")
ax.legend()
save_fig("fig10_risk_return_scatter.png")

# Fig 11: Broader tail risk bar chart
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col, title in zip(axes,
                           ["Avg_Excess_Pct", "Hit_Rate_Pct", "Avg_Breadth"],
                           ["Avg Excess Return (%)", "Hit Rate (%)", "Avg Breadth"]):
    ax.bar(tail_df["Threshold"], tail_df[col],
           color=[BRAND_BLUE, BRAND_GOLD, BRAND_GREEN], alpha=0.85, edgecolor="white")
    if col == "Hit_Rate_Pct":
        ax.axhline(50, color=BRAND_RED, linewidth=1.5, linestyle="--")
    elif col == "Avg_Breadth":
        ax.axhline(total_stocks_w / 2, color=BRAND_RED, linewidth=1.5, linestyle="--")
    ax.set_title(title)
    for i, v in enumerate(tail_df[col]):
        ax.text(i, v + 0.2, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")
plt.suptitle("Tail-Risk at Different Severity Thresholds",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig("fig11_tail_risk_thresholds.png")

# Save enhanced data
yoy_df.to_csv(os.path.join(OUTPUT_DIR, "year_by_year.csv"), index=False)
tail_df.to_csv(os.path.join(OUTPUT_DIR, "tail_risk_thresholds.csv"), index=False)


# =============================================================================
# MASTER SUMMARY (Output CSV 3)
# =============================================================================
print("\n" + "=" * 70)
print("MASTER SUMMARY")
print("=" * 70)

master = pd.DataFrame({
    "Analysis": [
        "1. Downside: % stocks beating BM max drawdown",
        "1. Downside: % shorter underwater than BM",
        "1. Downside: % faster recovery than BM",
        "2. Monthly Hit Rate",
        "2. 3-Month Rolling Hit Rate",
        "2. 6-Month Rolling Hit Rate",
        "2. Down-Month Hit Rate",
        "3. Upside Capture > 100% (bootstrap)",
        "3. Downside Capture < 100% (bootstrap)",
        "4. Tail-Risk Excess Return",
        "4. Tail-Risk Hit Rate",
        "4. Tail-Risk Breadth",
        "4. Drawdown Clustering (corr down vs all)",
        "E. Full-Period Batting Average (% beat BM 5yr)",
        "E. Avg Sharpe > BM Sharpe",
        "E. Avg Information Ratio",
    ],
    "Point_Estimate": [
        f"{pt_mdd*100:.1f}%",
        f"{beats_uw.mean()*100:.1f}%",
        f"{beats_rec_all.mean()*100:.1f}%" if len(beats_rec_all) > 0 else "N/A",
        f"{pt_hr*100:.1f}%",
        f"{results_rolling['3-Month']['pt']*100:.1f}%",
        f"{results_rolling['6-Month']['pt']*100:.1f}%",
        f"{pt_d*100:.1f}%" if not np.isnan(pt_d) else "N/A",
        f"{up_pt*100:.1f}%",
        f"{dn_pt*100:.1f}%",
        f"{avg_excess*100:.2f}%",
        f"{valid_worst_hr.mean()*100:.1f}%",
        f"{breadth.mean():.0f}/{total_stocks_w}",
        f"{avg_corr_down:.3f} vs {avg_corr_all:.3f}",
        f"{pct_beat_full*100:.1f}%",
        f"{pct_sharpe*100:.1f}%",
        f"{info_ratio.mean():.3f}",
    ],
    "CI_95_Lower": [
        f"{ci_lo_mdd*100:.1f}%",
        f"{ci_uw_lo*100:.1f}%",
        f"{ci_rec_lo*100:.1f}%" if len(beats_rec_all) > 0 else "",
        "", "", "", "",
        f"{up_ci_lo*100:.1f}%",
        f"{dn_ci_lo*100:.1f}%",
        f"{ci_ex_lo*100:.2f}%",
        f"{ci_whr_lo*100:.1f}%",
        f"{ci_br_lo:.0f}",
        "", "", "", "",
    ],
    "CI_95_Upper": [
        f"{ci_hi_mdd*100:.1f}%",
        f"{ci_uw_hi*100:.1f}%",
        f"{ci_rec_hi*100:.1f}%" if len(beats_rec_all) > 0 else "",
        "", "", "", "",
        f"{up_ci_hi*100:.1f}%",
        f"{dn_ci_hi*100:.1f}%",
        f"{ci_ex_hi*100:.2f}%",
        f"{ci_whr_hi*100:.1f}%",
        f"{ci_br_hi:.0f}",
        "", "", "", "",
    ],
    "Primary_p_value": [
        f"{p_binom_mdd:.4f}",
        f"{p_binom_uw:.4f}",
        f"{p_binom_rec:.4f}" if not np.isnan(p_binom_rec) else "N/A",
        f"{p_hr_t:.4f}",
        f"{results_rolling['3-Month']['p_t']:.4f}",
        f"{results_rolling['6-Month']['p_t']:.4f}",
        f"{p_d_t:.4f}" if not np.isnan(p_d_t) else "N/A",
        f"{p_boot_uc:.4f}",
        f"{p_boot_dc:.4f}",
        f"{p_tail_t:.4f}",
        f"{p_binom_tail:.4f}",
        "",
        f"{p_corr:.4f}",
        f"{p_full_bat:.4f}",
        f"{p_sharpe:.4f}",
        "",
    ],
    "Significance": [
        sig_stars(p_binom_mdd),
        sig_stars(p_binom_uw),
        sig_stars(p_binom_rec) if not np.isnan(p_binom_rec) else "N/A",
        sig_stars(p_hr_t),
        sig_stars(results_rolling["3-Month"]["p_t"]),
        sig_stars(results_rolling["6-Month"]["p_t"]),
        sig_stars(p_d_t) if not np.isnan(p_d_t) else "N/A",
        sig_stars(p_boot_uc),
        sig_stars(p_boot_dc),
        sig_stars(p_tail_t),
        sig_stars(p_binom_tail),
        "",
        sig_stars(p_corr),
        sig_stars(p_full_bat),
        sig_stars(p_sharpe),
        "",
    ],
})
master.to_csv(os.path.join(OUTPUT_DIR, "master_summary.csv"), index=False)

print(master.to_string(index=False))
print(f"""
Significance: *** p<0.001 | ** p<0.01 | * p<0.05 | . p<0.10 | ns = not significant

Output ({OUTPUT_DIR}/):
  CSVs   : master_summary.csv, stock_level_results.csv, segment_results.csv
  Figures: fig0 (index), fig1 (resilience), fig2 (consistency),
           fig3 (capture), fig4 (tail risk), fig5a/5b/5c (segmentation)
DONE.
""")
