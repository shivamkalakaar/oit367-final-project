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
    """Return significance stars for a p-value."""
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    if p < 0.10:   return "."
    return "ns"


def bootstrap_pvalue(bootstrap_dist, null_value, alternative="two-sided"):
    """Compute bootstrap p-value against a null value."""
    if alternative == "greater":
        return np.mean(bootstrap_dist <= null_value)
    if alternative == "less":
        return np.mean(bootstrap_dist >= null_value)
    # two-sided: reflect around null
    centered = bootstrap_dist - np.mean(bootstrap_dist)
    observed = np.mean(bootstrap_dist) - null_value
    return np.mean(np.abs(centered) >= np.abs(observed))


def save_fig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved figure: {path}")


# =============================================================================
# DATA LOADING
# =============================================================================
print("=" * 70)
print("DATA LOADING")
print("=" * 70)

# ── 1. Individual stock prices (CAPIQ) ──────────────────────────────────────
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

# ── 2. Solactive benchmark / index data ──────────────────────────────────────
perf = pd.read_excel("Solactive_Ethic_Backtest_20260130.xlsx",
                     sheet_name="Performance")
perf["Date"] = pd.to_datetime(perf["Date"])
perf = perf.set_index("Date")[["Ethics Premium GTR", "SGMACUT Index"]].copy()
perf = perf.sort_index()
perf = perf[~perf.index.duplicated(keep="first")]
perf.columns = ["EthicsPremium_GTR", "Benchmark_GTR"]

print(f"Benchmark data: {perf.shape[0]} days, "
      f"{perf.index.min().date()} to {perf.index.max().date()}")

# ── 3. Composition weights ────────────────────────────────────────────────────
weights = pd.read_excel("Solactive_Ethic_Backtest_20260130.xlsx",
                        sheet_name="Composition_Weights")
weights = weights.rename(columns={"Ticker": "Ticker_Exch", "Name": "Company"})
weights["Ticker"] = weights["Ticker_Exch"].str.split("-").str[0]

# ── 4. Firmographic data ─────────────────────────────────────────────────────
firmographics   = pd.read_csv("WMEC Stocks and Info.csv")
ticker_to_firm  = firmographics.set_index("Symbol")
matched_tickers = [t for t in tickers if t in ticker_to_firm.index]
print(f"Firmographics: {firmographics.shape[0]} companies | "
      f"Matched tickers: {len(matched_tickers)}")

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

bm_daily   = bench_returns["Benchmark_GTR"]
ep_daily   = bench_returns["EthicsPremium_GTR"]

stock_monthly = stock_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
bm_monthly    = bm_daily.resample("ME").apply(lambda x: (1 + x).prod() - 1)
ep_monthly    = ep_daily.resample("ME").apply(lambda x: (1 + x).prod() - 1)

stock_cum = (1 + stock_returns).cumprod()
bm_cum    = (1 + bm_daily).cumprod()
ep_cum    = (1 + ep_daily).cumprod()

print(f"Aligned returns: {len(common_dates)} trading days | "
      f"{len(bm_monthly)} monthly periods")

# ── Figure 0: Cumulative index performance (Ethics Premium vs Benchmark) ─────
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
ax.set_title("Ethics Premium GTR vs. Benchmark — Cumulative Performance (2021–2025)")
ax.set_ylabel("Index Level (Base = 1,000)")
ax.legend(loc="upper left")
final_ep = perf["EthicsPremium_GTR"].iloc[-1]
final_bm = perf["Benchmark_GTR"].iloc[-1]
ax.annotate(f"EP GTR: {final_ep:.0f}\n(+{(final_ep/1000-1)*100:.1f}%)",
            xy=(perf.index[-1], final_ep), xytext=(-90, 10),
            textcoords="offset points", color=BRAND_BLUE, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=BRAND_BLUE))
ax.annotate(f"BM: {final_bm:.0f}\n(+{(final_bm/1000-1)*100:.1f}%)",
            xy=(perf.index[-1], final_bm), xytext=(-90, -35),
            textcoords="offset points", color=BRAND_GREY, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=BRAND_GREY))
save_fig("fig0_index_performance.png")


# =============================================================================
# ANALYSIS 1: DOWNSIDE RESILIENCE
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 1: DOWNSIDE RESILIENCE (Max Drawdown)")
print("=" * 70)


def max_drawdown(cum_series):
    peak = cum_series.cummax()
    return ((cum_series - peak) / peak).min()


stock_mdd = stock_cum.apply(max_drawdown)
bm_mdd    = max_drawdown(bm_cum)

valid_stocks   = stock_mdd.dropna()
n_stocks       = len(valid_stocks)
stock_mdd_vals = valid_stocks.values
beats_bm_mdd   = (stock_mdd_vals > bm_mdd).astype(float)
n_beats_mdd    = int(beats_bm_mdd.sum())

# ── Bootstrap ──
bootstrap_pcts_mdd = np.empty(N_BOOTSTRAP)
for i in range(N_BOOTSTRAP):
    idx = np.random.randint(0, n_stocks, size=n_stocks)
    bootstrap_pcts_mdd[i] = beats_bm_mdd[idx].mean()

ci_lower_mdd = np.percentile(bootstrap_pcts_mdd, 2.5)
ci_upper_mdd = np.percentile(bootstrap_pcts_mdd, 97.5)
point_est_mdd = beats_bm_mdd.mean()

# ── Statistical significance ──
# Binomial test: H0 = 50% of stocks beat benchmark (pure chance)
binom_result_mdd = stats.binomtest(n_beats_mdd, n=n_stocks, p=0.5,
                                    alternative="two-sided")
p_binom_mdd = binom_result_mdd.pvalue

# One-sample t-test on per-stock MDD vs benchmark MDD
tstat_mdd, p_ttest_mdd = stats.ttest_1samp(stock_mdd_vals, bm_mdd)

# Wilcoxon signed-rank: H0 = stock MDD equal to benchmark MDD
wilcox_stat_mdd, p_wilcox_mdd = stats.wilcoxon(stock_mdd_vals - bm_mdd)

# Bootstrap p-value: H0 = point_est = 0.5
p_boot_mdd = bootstrap_pvalue(bootstrap_pcts_mdd, 0.5, "two-sided")

print(f"Benchmark max drawdown : {bm_mdd*100:.2f}%")
print(f"Stocks beating benchmark: {n_beats_mdd}/{n_stocks} "
      f"({point_est_mdd*100:.1f}%)")
print(f"95% CI               : [{ci_lower_mdd*100:.1f}%, {ci_upper_mdd*100:.1f}%]")
print(f"\nStatistical Significance (H0: no different from random 50%):")
print(f"  Binomial test  : p={p_binom_mdd:.4f}  {sig_stars(p_binom_mdd)}")
print(f"  T-test vs BM MDD: p={p_ttest_mdd:.4f}  {sig_stars(p_ttest_mdd)}")
print(f"  Wilcoxon test   : p={p_wilcox_mdd:.4f}  {sig_stars(p_wilcox_mdd)}")
print(f"  Bootstrap p-val : p={p_boot_mdd:.4f}  {sig_stars(p_boot_mdd)}")

# ── Figures ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: histogram of max drawdowns
ax = axes[0]
ax.hist(stock_mdd_vals * 100, bins=25, color=BRAND_BLUE, alpha=0.75,
        edgecolor="white", label="Honoree stocks")
ax.axvline(bm_mdd * 100, color=BRAND_RED, linewidth=2.5,
           linestyle="--", label=f"Benchmark MDD ({bm_mdd*100:.1f}%)")
ax.axvline(np.median(stock_mdd_vals) * 100, color=BRAND_GOLD, linewidth=1.5,
           linestyle=":", label=f"Median stock MDD ({np.median(stock_mdd_vals)*100:.1f}%)")
ax.set_title("Distribution of Max Drawdowns — Honoree Stocks vs. Benchmark")
ax.set_xlabel("Max Drawdown (%)")
ax.set_ylabel("Number of Stocks")
ax.legend()
pct_better = point_est_mdd * 100
ax.text(0.97, 0.97,
        f"{pct_better:.0f}% of stocks\nhave less severe MDD\nthan benchmark\n"
        f"Binomial p={p_binom_mdd:.3f} {sig_stars(p_binom_mdd)}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BRAND_LIGHT, alpha=0.8))

# Right: bootstrap distribution
ax = axes[1]
ax.hist(bootstrap_pcts_mdd * 100, bins=50, color=BRAND_BLUE, alpha=0.7,
        edgecolor="white")
ax.axvline(point_est_mdd * 100, color=BRAND_GOLD, linewidth=2.5,
           label=f"Point estimate: {point_est_mdd*100:.1f}%")
ax.axvline(50, color=BRAND_RED, linewidth=1.5, linestyle="--",
           label="Null (50%)")
ax.axvspan(ci_lower_mdd * 100, ci_upper_mdd * 100, alpha=0.2,
           color=BRAND_GOLD, label=f"95% CI [{ci_lower_mdd*100:.1f}%–{ci_upper_mdd*100:.1f}%]")
ax.set_title("Bootstrap Distribution — % of Stocks Beating Benchmark MDD")
ax.set_xlabel("% of Sampled Stocks with MDD Less Severe than Benchmark")
ax.set_ylabel("Bootstrap Frequency")
ax.legend()
save_fig("fig1_downside_resilience.png")

# Save CSVs
mdd_results = pd.DataFrame({
    "Ticker": valid_stocks.index,
    "Max_Drawdown_Pct": stock_mdd_vals * 100,
    "Beats_Benchmark": beats_bm_mdd.astype(bool)
}).sort_values("Max_Drawdown_Pct", ascending=False)
mdd_results.to_csv(os.path.join(OUTPUT_DIR, "analysis1_max_drawdowns.csv"),
                   index=False)

pd.DataFrame({
    "Metric": ["Benchmark MDD", "Point Est (% beating BM)",
               "95% CI Lower", "95% CI Upper",
               "Binomial p-value", "Binomial sig",
               "T-test p-value", "T-test sig",
               "Wilcoxon p-value", "Wilcoxon sig",
               "Bootstrap p-value", "Bootstrap sig"],
    "Value":  [f"{bm_mdd*100:.2f}%", f"{point_est_mdd*100:.1f}%",
               f"{ci_lower_mdd*100:.1f}%", f"{ci_upper_mdd*100:.1f}%",
               f"{p_binom_mdd:.4f}", sig_stars(p_binom_mdd),
               f"{p_ttest_mdd:.4f}", sig_stars(p_ttest_mdd),
               f"{p_wilcox_mdd:.4f}", sig_stars(p_wilcox_mdd),
               f"{p_boot_mdd:.4f}", sig_stars(p_boot_mdd)]
}).to_csv(os.path.join(OUTPUT_DIR, "analysis1_summary.csv"), index=False)


# =============================================================================
# ANALYSIS 2: CONSISTENCY
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 2: CONSISTENCY (Hit Rates)")
print("=" * 70)

# ── 2a. Monthly hit rate ──────────────────────────────────────────────────────
monthly_outperform = stock_monthly.gt(bm_monthly, axis=0)
stock_hit_rates    = monthly_outperform.mean()
valid_hit_rates    = stock_hit_rates.dropna()
n_stocks_hr        = len(valid_hit_rates)
hr_vals            = valid_hit_rates.values

bootstrap_avg_hr = np.empty(N_BOOTSTRAP)
for i in range(N_BOOTSTRAP):
    idx = np.random.randint(0, n_stocks_hr, size=n_stocks_hr)
    bootstrap_avg_hr[i] = hr_vals[idx].mean()

ci_lower_hr  = np.percentile(bootstrap_avg_hr, 2.5)
ci_upper_hr  = np.percentile(bootstrap_avg_hr, 97.5)
point_est_hr = hr_vals.mean()

# Significance
tstat_hr, p_ttest_hr       = stats.ttest_1samp(hr_vals, 0.5)
wilcox_stat_hr, p_wilcox_hr = stats.wilcoxon(hr_vals - 0.5)
p_boot_hr                   = bootstrap_pvalue(bootstrap_avg_hr, 0.5, "two-sided")

print(f"2a. Monthly Hit Rate: {point_est_hr*100:.1f}% "
      f"[{ci_lower_hr*100:.1f}%, {ci_upper_hr*100:.1f}%]")
print(f"    T-test vs 50%  : p={p_ttest_hr:.4f} {sig_stars(p_ttest_hr)}")
print(f"    Wilcoxon vs 50%: p={p_wilcox_hr:.4f} {sig_stars(p_wilcox_hr)}")
print(f"    Bootstrap p-val: p={p_boot_hr:.4f}  {sig_stars(p_boot_hr)}")

# ── 2b. Rolling hit rates ─────────────────────────────────────────────────────
def rolling_hit_rate(sm, bm, window):
    sr = sm.rolling(window).apply(lambda x: (1 + x).prod() - 1, raw=True)
    br = bm.rolling(window).apply(lambda x: (1 + x).prod() - 1, raw=True)
    return sr.gt(br, axis=0).dropna(how="all").mean()


results_rolling = {}
for wname, wsz in [("3-Month", 3), ("6-Month", 6)]:
    rhr     = rolling_hit_rate(stock_monthly, bm_monthly, wsz).dropna()
    n_r     = len(rhr)
    r_vals  = rhr.values
    boot_r  = np.array([r_vals[np.random.randint(0, n_r, n_r)].mean()
                        for _ in range(N_BOOTSTRAP)])
    ci_lo   = np.percentile(boot_r, 2.5)
    ci_hi   = np.percentile(boot_r, 97.5)
    pt      = r_vals.mean()
    ts, pt_p = stats.ttest_1samp(r_vals, 0.5)
    ws, wp   = stats.wilcoxon(r_vals - 0.5)
    pb       = bootstrap_pvalue(boot_r, 0.5, "two-sided")
    results_rolling[wname] = dict(point_estimate=pt, ci_lower=ci_lo,
                                  ci_upper=ci_hi, n_stocks=n_r,
                                  p_ttest=pt_p, p_wilcox=wp, p_boot=pb)
    print(f"\n2b. {wname} Rolling Hit Rate: {pt*100:.1f}% "
          f"[{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]")
    print(f"    T-test p={pt_p:.4f} {sig_stars(pt_p)} | "
          f"Wilcoxon p={wp:.4f} {sig_stars(wp)}")

# ── 2c. Down-month hit rate ───────────────────────────────────────────────────
down_months = bm_monthly[bm_monthly < 0].index
print(f"\n2c. Down months (benchmark < 0): {len(down_months)}")

if len(down_months) > 0:
    stock_down    = stock_monthly.loc[down_months]
    bm_down       = bm_monthly.loc[down_months]
    down_op       = stock_down.gt(bm_down, axis=0)
    stock_down_hr = down_op.mean()
    valid_down_hr = stock_down_hr.dropna()
    n_d           = len(valid_down_hr)
    d_vals        = valid_down_hr.values
    boot_d        = np.array([d_vals[np.random.randint(0, n_d, n_d)].mean()
                              for _ in range(N_BOOTSTRAP)])
    ci_lo_d       = np.percentile(boot_d, 2.5)
    ci_hi_d       = np.percentile(boot_d, 97.5)
    pt_est_d      = d_vals.mean()
    ts_d, p_tt_d  = stats.ttest_1samp(d_vals, 0.5)
    ws_d, p_wl_d  = stats.wilcoxon(d_vals - 0.5)
    p_boot_d      = bootstrap_pvalue(boot_d, 0.5, "two-sided")
    print(f"    Down-month hit rate: {pt_est_d*100:.1f}% "
          f"[{ci_lo_d*100:.1f}%, {ci_hi_d*100:.1f}%]")
    print(f"    T-test p={p_tt_d:.4f} {sig_stars(p_tt_d)} | "
          f"Wilcoxon p={p_wl_d:.4f} {sig_stars(p_wl_d)}")
else:
    pt_est_d = ci_lo_d = ci_hi_d = np.nan
    p_tt_d = p_wl_d = p_boot_d = np.nan
    n_d = 0

# ── Figures ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: distribution of per-stock monthly hit rates
ax = axes[0]
ax.hist(hr_vals * 100, bins=25, color=BRAND_BLUE, alpha=0.75, edgecolor="white")
ax.axvline(50, color=BRAND_RED, linewidth=2, linestyle="--", label="50% (random)")
ax.axvline(point_est_hr * 100, color=BRAND_GOLD, linewidth=2,
           label=f"Mean: {point_est_hr*100:.1f}%")
ax.set_title("Distribution of Per-Stock Monthly Hit Rates\n(% of months each honoree beats benchmark)")
ax.set_xlabel("Monthly Hit Rate (%)")
ax.set_ylabel("Number of Stocks")
ax.text(0.97, 0.97,
        f"T-test vs 50%: p={p_ttest_hr:.3f} {sig_stars(p_ttest_hr)}\n"
        f"Wilcoxon: p={p_wilcox_hr:.3f} {sig_stars(p_wilcox_hr)}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BRAND_LIGHT, alpha=0.8))
ax.legend()

# Right: hit rate comparison across all windows with CI error bars
ax = axes[1]
labels  = ["Monthly", "3-Month\nRolling", "6-Month\nRolling", "Down-Month"]
pts     = [point_est_hr,
           results_rolling["3-Month"]["point_estimate"],
           results_rolling["6-Month"]["point_estimate"],
           pt_est_d]
ci_los  = [ci_lower_hr,
           results_rolling["3-Month"]["ci_lower"],
           results_rolling["6-Month"]["ci_lower"],
           ci_lo_d]
ci_his  = [ci_upper_hr,
           results_rolling["3-Month"]["ci_upper"],
           results_rolling["6-Month"]["ci_upper"],
           ci_hi_d]
colors  = [BRAND_BLUE, BRAND_BLUE, BRAND_BLUE, BRAND_GOLD]

x = np.arange(len(labels))
bars = ax.bar(x, [p * 100 for p in pts], color=colors, alpha=0.8,
              width=0.5, edgecolor="white")
for j, (lo, hi, pt) in enumerate(zip(ci_los, ci_his, pts)):
    if not np.isnan(lo):
        ax.errorbar(x[j], pt * 100,
                    yerr=[[pt * 100 - lo * 100], [hi * 100 - pt * 100]],
                    fmt="none", color="black", capsize=5, linewidth=1.5)
ax.axhline(50, color=BRAND_RED, linewidth=1.5, linestyle="--",
           label="50% (random benchmark)")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 80)
ax.set_ylabel("Hit Rate (%)")
ax.set_title("Consistency — Hit Rates Across Horizons\n(Error bars = 95% CI)")
ax.legend()
for bar, pt in zip(bars, pts):
    if not np.isnan(pt):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5, f"{pt*100:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
save_fig("fig2_consistency.png")

# Save CSVs
pd.DataFrame({
    "Ticker": valid_hit_rates.index,
    "Monthly_Hit_Rate_Pct": valid_hit_rates.values * 100
}).sort_values("Monthly_Hit_Rate_Pct", ascending=False).to_csv(
    os.path.join(OUTPUT_DIR, "analysis2_monthly_hit_rates.csv"), index=False)

pd.DataFrame({
    "Metric": [
        "Monthly Hit Rate - Point Estimate",
        "Monthly Hit Rate - 95% CI Lower", "Monthly Hit Rate - 95% CI Upper",
        "Monthly Hit Rate - T-test p", "Monthly Hit Rate - T-test sig",
        "Monthly Hit Rate - Wilcoxon p", "Monthly Hit Rate - Wilcoxon sig",
        "Monthly Hit Rate - Bootstrap p", "Monthly Hit Rate - Bootstrap sig",
        "3M Rolling - Point Estimate",
        "3M Rolling - 95% CI Lower", "3M Rolling - 95% CI Upper",
        "3M Rolling - T-test p", "3M Rolling - Sig",
        "6M Rolling - Point Estimate",
        "6M Rolling - 95% CI Lower", "6M Rolling - 95% CI Upper",
        "6M Rolling - T-test p", "6M Rolling - Sig",
        "Down-Month Hit Rate - Point Estimate",
        "Down-Month Hit Rate - 95% CI Lower", "Down-Month Hit Rate - 95% CI Upper",
        "Down-Month Hit Rate - T-test p", "Down-Month Hit Rate - Sig",
        "Number of Down Months"
    ],
    "Value": [
        f"{point_est_hr*100:.1f}%",
        f"{ci_lower_hr*100:.1f}%", f"{ci_upper_hr*100:.1f}%",
        f"{p_ttest_hr:.4f}", sig_stars(p_ttest_hr),
        f"{p_wilcox_hr:.4f}", sig_stars(p_wilcox_hr),
        f"{p_boot_hr:.4f}", sig_stars(p_boot_hr),
        f"{results_rolling['3-Month']['point_estimate']*100:.1f}%",
        f"{results_rolling['3-Month']['ci_lower']*100:.1f}%",
        f"{results_rolling['3-Month']['ci_upper']*100:.1f}%",
        f"{results_rolling['3-Month']['p_ttest']:.4f}",
        sig_stars(results_rolling["3-Month"]["p_ttest"]),
        f"{results_rolling['6-Month']['point_estimate']*100:.1f}%",
        f"{results_rolling['6-Month']['ci_lower']*100:.1f}%",
        f"{results_rolling['6-Month']['ci_upper']*100:.1f}%",
        f"{results_rolling['6-Month']['p_ttest']:.4f}",
        sig_stars(results_rolling["6-Month"]["p_ttest"]),
        f"{pt_est_d*100:.1f}%" if not np.isnan(pt_est_d) else "N/A",
        f"{ci_lo_d*100:.1f}%" if not np.isnan(ci_lo_d) else "N/A",
        f"{ci_hi_d*100:.1f}%" if not np.isnan(ci_hi_d) else "N/A",
        f"{p_tt_d:.4f}" if not np.isnan(p_tt_d) else "N/A",
        sig_stars(p_tt_d) if not np.isnan(p_tt_d) else "N/A",
        len(down_months)
    ]
}).to_csv(os.path.join(OUTPUT_DIR, "analysis2_summary.csv"), index=False)


# =============================================================================
# ANALYSIS 3: UPSIDE / DOWNSIDE CAPTURE
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 3: UPSIDE / DOWNSIDE CAPTURE")
print("=" * 70)

WINDOW_SIZE   = 60
n_days        = len(stock_returns)
max_start     = n_days - WINDOW_SIZE
up_days       = bm_daily > 0
down_days     = bm_daily < 0

stock_up_capture   = {}
stock_down_capture = {}
for t in stock_returns.columns:
    sr = stock_returns[t]
    if up_days.sum() > 0:
        stock_up_capture[t]   = sr[up_days].mean() / bm_daily[up_days].mean()
    if down_days.sum() > 0:
        stock_down_capture[t] = sr[down_days].mean() / bm_daily[down_days].mean()

up_capture_s   = pd.Series(stock_up_capture)
down_capture_s = pd.Series(stock_down_capture)

# Significance on full-period capture ratios
tstat_uc, p_ttest_uc = stats.ttest_1samp(up_capture_s.dropna().values, 1.0)
tstat_dc, p_ttest_dc = stats.ttest_1samp(down_capture_s.dropna().values, 1.0)
ws_uc, p_wl_uc       = stats.wilcoxon(up_capture_s.dropna().values - 1.0)
ws_dc, p_wl_dc       = stats.wilcoxon(down_capture_s.dropna().values - 1.0)

print(f"Avg upside capture  : {up_capture_s.mean()*100:.1f}% | "
      f"T-test vs 100%: p={p_ttest_uc:.4f} {sig_stars(p_ttest_uc)}")
print(f"Avg downside capture: {down_capture_s.mean()*100:.1f}% | "
      f"T-test vs 100%: p={p_ttest_dc:.4f} {sig_stars(p_ttest_dc)}")
print(f"Wilcoxon upside capture  : p={p_wl_uc:.4f} {sig_stars(p_wl_uc)}")
print(f"Wilcoxon downside capture: p={p_wl_dc:.4f} {sig_stars(p_wl_dc)}")

# Nested bootstrap
stock_ret_arr  = stock_returns.values
bm_arr         = bm_daily.values
n_stocks_cap   = stock_ret_arr.shape[1]
bootstrap_pct_up   = np.empty(N_BOOTSTRAP)
bootstrap_pct_down = np.empty(N_BOOTSTRAP)

print(f"Running nested bootstrap ({N_BOOTSTRAP:,} iterations)...")
for i in range(N_BOOTSTRAP):
    start       = np.random.randint(0, max_start + 1)
    wstock      = stock_ret_arr[start:start + WINDOW_SIZE, :]
    wbm         = bm_arr[start:start + WINDOW_SIZE]
    w_up        = wbm > 0
    w_down      = wbm < 0
    sampled     = np.random.randint(0, n_stocks_cap, size=n_stocks_cap)
    uc_arr      = np.empty(n_stocks_cap)
    dc_arr      = np.empty(n_stocks_cap)
    uc_arr[:]   = np.nan
    dc_arr[:]   = np.nan
    if w_up.sum() > 0:
        bm_up_mean = wbm[w_up].mean()
        if bm_up_mean != 0:
            uc_arr = wstock[w_up, :][:, sampled].mean(axis=0) / bm_up_mean
    if w_down.sum() > 0:
        bm_dn_mean = wbm[w_down].mean()
        if bm_dn_mean != 0:
            dc_arr = wstock[w_down, :][:, sampled].mean(axis=0) / bm_dn_mean
    bootstrap_pct_up[i]   = np.nanmean(uc_arr > 1)
    bootstrap_pct_down[i] = np.nanmean(dc_arr < 1)

bpu_clean  = bootstrap_pct_up[~np.isnan(bootstrap_pct_up)]
bpd_clean  = bootstrap_pct_down[~np.isnan(bootstrap_pct_down)]
up_ci_lo   = np.percentile(bpu_clean, 2.5)
up_ci_hi   = np.percentile(bpu_clean, 97.5)
up_pt      = np.mean(bpu_clean)
down_ci_lo = np.percentile(bpd_clean, 2.5)
down_ci_hi = np.percentile(bpd_clean, 97.5)
down_pt    = np.mean(bpd_clean)

p_boot_uc  = bootstrap_pvalue(bpu_clean, 0.5, "greater")
p_boot_dc  = bootstrap_pvalue(bpd_clean, 0.5, "greater")

print(f"\nNested Bootstrap (60-day windows):")
print(f"  Upside capture > 100%  : {up_pt*100:.1f}% "
      f"[{up_ci_lo*100:.1f}%, {up_ci_hi*100:.1f}%] "
      f"p={p_boot_uc:.4f} {sig_stars(p_boot_uc)}")
print(f"  Downside capture < 100%: {down_pt*100:.1f}% "
      f"[{down_ci_lo*100:.1f}%, {down_ci_hi*100:.1f}%] "
      f"p={p_boot_dc:.4f} {sig_stars(p_boot_dc)}")

# ── Figures ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: scatter plot upside vs downside capture
ax = axes[0]
uc_vals = up_capture_s.values * 100
dc_vals = down_capture_s.reindex(up_capture_s.index).values * 100

# Color by sector if available
sector_map = firmographics.set_index("Symbol")["Sectors"]
colors_s   = []
sector_palette = dict(zip(
    firmographics["Sectors"].dropna().unique(),
    sns.color_palette("tab10", n_colors=12)
))
for t in up_capture_s.index:
    sec_val = sector_map.get(t, None)
    if isinstance(sec_val, pd.Series):
        sec_val = sec_val.iloc[0]
    colors_s.append(sector_palette.get(sec_val, BRAND_GREY))

sc = ax.scatter(uc_vals, dc_vals, c=colors_s, alpha=0.75, s=60, edgecolors="white")
ax.axhline(100, color="black", linewidth=1, linestyle="--", alpha=0.5)
ax.axvline(100, color="black", linewidth=1, linestyle="--", alpha=0.5)
ax.fill_between([90, max(uc_vals) + 5], 90, 100,
                color=BRAND_GREEN, alpha=0.06,
                label="Ideal zone (up>100, down<100)")
ax.set_xlabel("Upside Capture Ratio (%)")
ax.set_ylabel("Downside Capture Ratio (%)")
ax.set_title("Upside vs. Downside Capture per Honoree Stock\n(Ideal: right of 100% x-axis, below 100% y-axis)")
# Add legend for sectors with 3+ stocks
sector_counts = stock_summary_for_legend = firmographics.groupby("Sectors").size()
for sec, color in sector_palette.items():
    if sector_counts.get(sec, 0) >= 3:
        ax.scatter([], [], c=[color], label=sec, s=40, alpha=0.8)
ax.legend(fontsize=7, loc="upper left", ncol=2)
ax.text(0.97, 0.03,
        f"T-test upside capture vs 100%: p={p_ttest_uc:.3f} {sig_stars(p_ttest_uc)}\n"
        f"T-test downside capture vs 100%: p={p_ttest_dc:.3f} {sig_stars(p_ttest_dc)}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BRAND_LIGHT, alpha=0.8))

# Right: bootstrap distributions
ax = axes[1]
ax.hist(bpu_clean * 100, bins=50, color=BRAND_GREEN, alpha=0.6,
        label=f"Upside capture > 100% (mean={up_pt*100:.1f}%)", edgecolor="white")
ax.hist(bpd_clean * 100, bins=50, color=BRAND_BLUE, alpha=0.6,
        label=f"Downside capture < 100% (mean={down_pt*100:.1f}%)", edgecolor="white")
ax.axvline(50, color=BRAND_RED, linewidth=1.5, linestyle="--", label="50% (null)")
ax.axvspan(up_ci_lo * 100, up_ci_hi * 100, alpha=0.15, color=BRAND_GREEN)
ax.axvspan(down_ci_lo * 100, down_ci_hi * 100, alpha=0.15, color=BRAND_BLUE)
ax.set_xlabel("% of Sampled Stocks with Superior Capture")
ax.set_ylabel("Bootstrap Frequency")
ax.set_title("Nested Bootstrap — Upside & Downside Capture Profiles\n(Shaded = 95% CI)")
ax.legend()
save_fig("fig3_capture_ratios.png")

# Save CSVs
pd.DataFrame({
    "Ticker": up_capture_s.index,
    "Upside_Capture_Pct":   up_capture_s.values * 100,
    "Downside_Capture_Pct": down_capture_s.reindex(up_capture_s.index).values * 100,
    "Good_Upside":   (up_capture_s > 1).values,
    "Good_Downside": (down_capture_s.reindex(up_capture_s.index) < 1).values
}).sort_values("Upside_Capture_Pct", ascending=False).to_csv(
    os.path.join(OUTPUT_DIR, "analysis3_capture_ratios.csv"), index=False)

pd.DataFrame({
    "Metric": [
        "Full-Period Avg Upside Capture",
        "Full-Period Avg Downside Capture",
        "Upside Capture T-test vs 100%",    "Upside Capture T-test sig",
        "Upside Capture Wilcoxon p",        "Upside Capture Wilcoxon sig",
        "Downside Capture T-test vs 100%",  "Downside Capture T-test sig",
        "Downside Capture Wilcoxon p",      "Downside Capture Wilcoxon sig",
        "Bootstrap % Up Capture > 100% - Point Est",
        "Bootstrap % Up Capture > 100% - CI Lower",
        "Bootstrap % Up Capture > 100% - CI Upper",
        "Bootstrap % Up Capture > 100% - p-value",
        "Bootstrap % Dn Capture < 100% - Point Est",
        "Bootstrap % Dn Capture < 100% - CI Lower",
        "Bootstrap % Dn Capture < 100% - CI Upper",
        "Bootstrap % Dn Capture < 100% - p-value",
    ],
    "Value": [
        f"{up_capture_s.mean()*100:.1f}%",
        f"{down_capture_s.mean()*100:.1f}%",
        f"{p_ttest_uc:.4f}", sig_stars(p_ttest_uc),
        f"{p_wl_uc:.4f}",    sig_stars(p_wl_uc),
        f"{p_ttest_dc:.4f}", sig_stars(p_ttest_dc),
        f"{p_wl_dc:.4f}",    sig_stars(p_wl_dc),
        f"{up_pt*100:.1f}%",   f"{up_ci_lo*100:.1f}%",
        f"{up_ci_hi*100:.1f}%", f"{p_boot_uc:.4f}",
        f"{down_pt*100:.1f}%", f"{down_ci_lo*100:.1f}%",
        f"{down_ci_hi*100:.1f}%", f"{p_boot_dc:.4f}",
    ]
}).to_csv(os.path.join(OUTPUT_DIR, "analysis3_summary.csv"), index=False)


# =============================================================================
# ANALYSIS 4: TAIL-RISK STORY
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 4: TAIL-RISK STORY (Worst Market Months)")
print("=" * 70)

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
total_stocks_worst = stock_worst.shape[1]

# Per-stock average return in worst months
stock_worst_means = stock_worst.mean(axis=0).dropna()

# Statistical significance
excess_per_stock   = stock_worst_means.values - avg_bm_worst
ts_tail, p_tail_t  = stats.ttest_1samp(excess_per_stock, 0)
ws_tail, p_tail_wl = stats.wilcoxon(excess_per_stock)
n_whr              = len(valid_worst_hr)
whr_vals           = valid_worst_hr.values
n_beats_tail       = int((whr_vals > 0.5).sum())  # stocks with hit_rate > 50% in worst months
binom_tail         = stats.binomtest(n_beats_tail, n=n_whr, p=0.5, alternative="greater")
p_binom_tail       = binom_tail.pvalue

# Bootstraps
boot_whr     = np.array([whr_vals[np.random.randint(0, n_whr, n_whr)].mean()
                         for _ in range(N_BOOTSTRAP)])
boot_excess  = np.array([stock_worst_means.values[
                         np.random.randint(0, len(stock_worst_means),
                                           len(stock_worst_means))].mean()
                         - avg_bm_worst
                         for _ in range(N_BOOTSTRAP)])
n_wm = len(breadth)
boot_breadth = np.array([breadth.values[np.random.randint(0, n_wm, n_wm)].mean()
                         for _ in range(N_BOOTSTRAP)])

ci_whr_lo    = np.percentile(boot_whr, 2.5)
ci_whr_hi    = np.percentile(boot_whr, 97.5)
ci_excess_lo = np.percentile(boot_excess, 2.5)
ci_excess_hi = np.percentile(boot_excess, 97.5)
ci_brdth_lo  = np.percentile(boot_breadth, 2.5)
ci_brdth_hi  = np.percentile(boot_breadth, 97.5)
p_boot_tail  = bootstrap_pvalue(boot_excess, 0, "greater")

print(f"Worst months (bottom {threshold_pct}%): "
      f"{[d.strftime('%Y-%m') for d in worst_months]}")
print(f"Avg benchmark return    : {avg_bm_worst*100:.2f}%")
print(f"Avg honoree return      : {avg_stock_worst*100:.2f}%")
print(f"Avg excess              : {avg_excess*100:.2f}% "
      f"[{ci_excess_lo*100:.2f}%, {ci_excess_hi*100:.2f}%]")
print(f"Hit rate in worst months: {valid_worst_hr.mean()*100:.1f}% "
      f"[{ci_whr_lo*100:.1f}%, {ci_whr_hi*100:.1f}%]")
print(f"Breadth                 : {breadth.mean():.0f} "
      f"[{ci_brdth_lo:.0f}, {ci_brdth_hi:.0f}] / {total_stocks_worst}")
print(f"\nStatistical Significance:")
print(f"  T-test excess > 0       : p={p_tail_t:.4f}  {sig_stars(p_tail_t)}")
print(f"  Wilcoxon excess > 0     : p={p_tail_wl:.4f}  {sig_stars(p_tail_wl)}")
print(f"  Binomial hit rate > 50% : p={p_binom_tail:.4f}  {sig_stars(p_binom_tail)}")
print(f"  Bootstrap p (excess > 0): p={p_boot_tail:.4f}  {sig_stars(p_boot_tail)}")

# ── Figures ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: bar chart of worst months, benchmark vs avg honoree
ax = axes[0]
month_labels = [d.strftime("%b %Y") for d in worst_months]
x    = np.arange(len(worst_months))
w    = 0.35
bm_b = ax.bar(x - w / 2, bm_worst.values * 100, w,
              label="Benchmark", color=BRAND_GREY, alpha=0.85, edgecolor="white")
ho_b = ax.bar(x + w / 2, stock_worst.mean(axis=1).values * 100, w,
              label="Avg Honoree", color=BRAND_BLUE, alpha=0.85, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(month_labels, rotation=15)
ax.set_ylabel("Monthly Return (%)")
ax.set_title(f"Worst Market Months (Bottom {threshold_pct}%) — Benchmark vs. Honorees\n"
             f"Avg excess: {avg_excess*100:.2f}% [{ci_excess_lo*100:.2f}%, {ci_excess_hi*100:.2f}%]")
ax.axhline(0, color="black", linewidth=0.8)
ax.legend()
ax.text(0.97, 0.03,
        f"T-test excess > 0: p={p_tail_t:.3f} {sig_stars(p_tail_t)}\n"
        f"Wilcoxon: p={p_tail_wl:.3f} {sig_stars(p_tail_wl)}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BRAND_LIGHT, alpha=0.8))

# Right: breadth (how many stocks beat BM in each worst month)
ax = axes[1]
bars = ax.bar(month_labels, breadth.values, color=BRAND_BLUE, alpha=0.8, edgecolor="white")
ax.axhline(total_stocks_worst / 2, color=BRAND_RED, linewidth=1.5,
           linestyle="--", label="50% of stocks")
ax.axhline(ci_brdth_lo, color=BRAND_GOLD, linewidth=1, linestyle=":",
           label=f"95% CI [{ci_brdth_lo:.0f}–{ci_brdth_hi:.0f}]")
ax.axhline(ci_brdth_hi, color=BRAND_GOLD, linewidth=1, linestyle=":")
ax.set_ylabel("Number of Stocks Beating Benchmark")
ax.set_title(f"Breadth — Honorees Beating Benchmark in Worst Months\n"
             f"Avg: {breadth.mean():.0f}/{total_stocks_worst} stocks per month")
for bar, val in zip(bars, breadth.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            str(int(val)), ha="center", fontsize=10, fontweight="bold")
ax.legend()
save_fig("fig4_tail_risk.png")

# Save CSVs
pd.DataFrame({
    "Month": [d.strftime("%Y-%m") for d in worst_months],
    "Benchmark_Return_Pct": bm_worst.values * 100,
    "Avg_Honoree_Return_Pct": stock_worst.mean(axis=1).values * 100,
    "Excess_Return_Pct": (stock_worst.mean(axis=1) - bm_worst).values * 100,
    "Breadth_Stocks_Beating_BM": breadth.values,
    "Total_Stocks": total_stocks_worst
}).to_csv(os.path.join(OUTPUT_DIR, "analysis4_worst_months.csv"), index=False)

pd.DataFrame({
    "Metric": [
        "Threshold (bottom 5%)", "N Worst Months",
        "Avg Benchmark Return",  "Avg Honoree Return",
        "Avg Excess - Point Est", "Avg Excess - CI Lower", "Avg Excess - CI Upper",
        "T-test excess > 0 - p", "T-test sig",
        "Wilcoxon - p",          "Wilcoxon sig",
        "Binomial hit > 50% - p","Binomial sig",
        "Bootstrap excess p",    "Bootstrap sig",
        "Hit Rate - Point Est",  "Hit Rate - CI Lower", "Hit Rate - CI Upper",
        "Breadth - Point Est",   "Breadth - CI Lower", "Breadth - CI Upper",
    ],
    "Value": [
        f"{threshold_val*100:.2f}%", len(worst_months),
        f"{avg_bm_worst*100:.2f}%",  f"{avg_stock_worst*100:.2f}%",
        f"{avg_excess*100:.2f}%",    f"{ci_excess_lo*100:.2f}%",
        f"{ci_excess_hi*100:.2f}%",
        f"{p_tail_t:.4f}",  sig_stars(p_tail_t),
        f"{p_tail_wl:.4f}", sig_stars(p_tail_wl),
        f"{p_binom_tail:.4f}", sig_stars(p_binom_tail),
        f"{p_boot_tail:.4f}", sig_stars(p_boot_tail),
        f"{valid_worst_hr.mean()*100:.1f}%",
        f"{ci_whr_lo*100:.1f}%", f"{ci_whr_hi*100:.1f}%",
        f"{breadth.mean():.1f}", f"{ci_brdth_lo:.1f}", f"{ci_brdth_hi:.1f}",
    ]
}).to_csv(os.path.join(OUTPUT_DIR, "analysis4_summary.csv"), index=False)


# =============================================================================
# ANALYSIS 5: SEGMENTATION
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 5: SEGMENTATION")
print("=" * 70)

stock_summary = pd.DataFrame({
    "Ticker":             list(tickers),
    "Max_Drawdown":       stock_mdd.reindex(tickers).values,
    "Monthly_Hit_Rate":   stock_hit_rates.reindex(tickers).values,
    "Upside_Capture":     up_capture_s.reindex(tickers).values,
    "Downside_Capture":   down_capture_s.reindex(tickers).values,
    "Down_Month_Hit_Rate": stock_down_hr.reindex(tickers).values
                            if len(down_months) > 0 else np.nan,
    "Worst_Month_Hit_Rate": worst_hit_rates.reindex(tickers).values
}).merge(
    firmographics[["Symbol", "Sectors", "Industry Groups", "Domicile Country",
                   "Workforce Size", "Annual Revenue", "Structure"]],
    left_on="Ticker", right_on="Symbol", how="left"
).drop(columns=["Symbol"])

stock_summary.to_csv(os.path.join(OUTPUT_DIR, "analysis5_stock_summary.csv"),
                     index=False)

segment_dims = {
    "Sector":         "Sectors",
    "Country":        "Domicile Country",
    "Workforce_Size": "Workforce Size",
    "Revenue":        "Annual Revenue",
}

all_seg_results = {}
for seg_label, seg_col in segment_dims.items():
    if seg_col not in stock_summary.columns:
        continue
    grouped = stock_summary.dropna(subset=[seg_col]).groupby(seg_col)
    seg_result = grouped.agg(
        N_Stocks              = ("Ticker", "count"),
        Avg_Max_Drawdown      = ("Max_Drawdown", "mean"),
        Avg_Monthly_Hit_Rate  = ("Monthly_Hit_Rate", "mean"),
        Avg_Upside_Capture    = ("Upside_Capture", "mean"),
        Avg_Downside_Capture  = ("Downside_Capture", "mean"),
        Avg_Down_Month_HR     = ("Down_Month_Hit_Rate", "mean"),
        Avg_Worst_Month_HR    = ("Worst_Month_Hit_Rate", "mean"),
    ).round(4)
    seg_result = seg_result[seg_result["N_Stocks"] >= 3]
    seg_result = seg_result.sort_values("Avg_Monthly_Hit_Rate", ascending=False)
    seg_result.to_csv(os.path.join(OUTPUT_DIR,
                                   f"analysis5_segment_{seg_label.lower()}.csv"))
    all_seg_results[seg_label] = seg_result

    # Kruskal-Wallis across groups for monthly hit rate
    groups = [grp["Monthly_Hit_Rate"].dropna().values
              for _, grp in stock_summary.dropna(subset=[seg_col, "Monthly_Hit_Rate"]
                                                  ).groupby(seg_col)
              if len(grp) >= 3]
    if len(groups) >= 2:
        kw_stat, kw_p = stats.kruskal(*groups)
        print(f"\n{seg_label}: Kruskal-Wallis on Monthly Hit Rate "
              f"(H={kw_stat:.2f}, p={kw_p:.4f} {sig_stars(kw_p)})")
    print(seg_result[["N_Stocks", "Avg_Monthly_Hit_Rate",
                       "Avg_Upside_Capture", "Avg_Downside_Capture"]].to_string())

# ── Figures ────────────────────────────────────────────────────────────────────
# Fig 5a: Sector bar chart — multiple metrics
if "Sector" in all_seg_results:
    seg_df = all_seg_results["Sector"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = [
        ("Avg_Monthly_Hit_Rate", "Avg Monthly Hit Rate", 0.5),
        ("Avg_Upside_Capture",   "Avg Upside Capture",   1.0),
        ("Avg_Downside_Capture", "Avg Downside Capture", 1.0),
    ]
    labels_s = [s.replace(" ", "\n") for s in seg_df.index]
    x = np.arange(len(seg_df))
    for ax, (col, title, ref) in zip(axes, metrics):
        vals = seg_df[col].values
        bar_colors = [BRAND_BLUE if (col == "Avg_Monthly_Hit_Rate" and v > ref)
                      or (col == "Avg_Upside_Capture" and v > ref)
                      or (col == "Avg_Downside_Capture" and v < ref)
                      else BRAND_GOLD for v in vals]
        ax.barh(x, vals, color=bar_colors, alpha=0.85, edgecolor="white")
        ax.axvline(ref, color=BRAND_RED, linewidth=1.5, linestyle="--",
                   label=f"Reference ({ref:.0%})")
        ax.set_yticks(x)
        ax.set_yticklabels(labels_s, fontsize=8)
        ax.set_title(title + " by Sector", fontweight="bold")
        ax.invert_yaxis()
        for i, v in enumerate(vals):
            ax.text(v + (0.002 if col != "Avg_Downside_Capture" else 0.002),
                    i, f"{v:.2f}", va="center", fontsize=8)
    plt.suptitle("Segmentation by Sector — Key Performance Metrics",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig("fig5a_sector_metrics.png")

# Fig 5b: Sector heatmap
if "Sector" in all_seg_results:
    seg_df = all_seg_results["Sector"].copy()
    heat_cols = ["Avg_Monthly_Hit_Rate", "Avg_Max_Drawdown",
                 "Avg_Upside_Capture", "Avg_Downside_Capture",
                 "Avg_Down_Month_HR", "Avg_Worst_Month_HR"]
    heat_labels = ["Monthly\nHit Rate", "Max\nDrawdown",
                   "Upside\nCapture", "Downside\nCapture",
                   "Down-Month\nHit Rate", "Worst-Month\nHit Rate"]
    heat_data = seg_df[heat_cols].copy()
    heat_data.index = [s.replace(" ", "\n") for s in heat_data.index]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heat_data.values,
                xticklabels=heat_labels,
                yticklabels=heat_data.index,
                annot=True, fmt=".2f", cmap="RdYlGn",
                center=None, linewidths=0.5, ax=ax,
                annot_kws={"size": 9})
    ax.set_title("Sector Segmentation — Performance Metrics Heatmap",
                 fontweight="bold", pad=12)
    plt.tight_layout()
    save_fig("fig5b_sector_heatmap.png")

# Fig 5c: Country bar (top countries by hit rate)
if "Country" in all_seg_results:
    seg_df = all_seg_results["Country"].copy().head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    x   = np.arange(len(seg_df))
    hr  = seg_df["Avg_Monthly_Hit_Rate"].values
    ax.bar(x, hr * 100,
           color=[BRAND_BLUE if v > 0.5 else BRAND_GOLD for v in hr],
           alpha=0.85, edgecolor="white")
    ax.axhline(50, color=BRAND_RED, linewidth=1.5, linestyle="--",
               label="50% (random)")
    ax.set_xticks(x)
    ax.set_xticklabels(seg_df.index, rotation=20, ha="right")
    ax.set_ylabel("Avg Monthly Hit Rate (%)")
    ax.set_title("Country Segmentation — Avg Monthly Hit Rate\n"
                 "(Blue = above 50%, Gold = below 50%)", fontweight="bold")
    for i, (v, n) in enumerate(zip(hr, seg_df["N_Stocks"].values)):
        ax.text(i, v * 100 + 0.5, f"{v*100:.1f}%\n(n={n})",
                ha="center", fontsize=8)
    ax.legend()
    plt.tight_layout()
    save_fig("fig5c_country_hit_rate.png")


# =============================================================================
# MASTER SUMMARY + SIGNIFICANCE TABLE
# =============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

master = pd.DataFrame({
    "Analysis": [
        "1. Downside Resilience — % stocks beating BM MDD",
        "2. Monthly Hit Rate",
        "2. 3-Month Rolling Hit Rate",
        "2. 6-Month Rolling Hit Rate",
        "2. Down-Month Hit Rate",
        "3. Upside Capture > 100% (nested bootstrap)",
        "3. Downside Capture < 100% (nested bootstrap)",
        "4. Tail-Risk Excess Return (avg worst months)",
        "4. Tail-Risk Hit Rate",
        "4. Tail-Risk Breadth",
    ],
    "Point_Estimate": [
        f"{point_est_mdd*100:.1f}%",
        f"{point_est_hr*100:.1f}%",
        f"{results_rolling['3-Month']['point_estimate']*100:.1f}%",
        f"{results_rolling['6-Month']['point_estimate']*100:.1f}%",
        f"{pt_est_d*100:.1f}%",
        f"{up_pt*100:.1f}%",
        f"{down_pt*100:.1f}%",
        f"{avg_excess*100:.2f}%",
        f"{valid_worst_hr.mean()*100:.1f}%",
        f"{breadth.mean():.1f}/{total_stocks_worst}",
    ],
    "CI_95_Lower": [
        f"{ci_lower_mdd*100:.1f}%",
        f"{ci_lower_hr*100:.1f}%",
        f"{results_rolling['3-Month']['ci_lower']*100:.1f}%",
        f"{results_rolling['6-Month']['ci_lower']*100:.1f}%",
        f"{ci_lo_d*100:.1f}%",
        f"{up_ci_lo*100:.1f}%",
        f"{down_ci_lo*100:.1f}%",
        f"{ci_excess_lo*100:.2f}%",
        f"{ci_whr_lo*100:.1f}%",
        f"{ci_brdth_lo:.1f}",
    ],
    "CI_95_Upper": [
        f"{ci_upper_mdd*100:.1f}%",
        f"{ci_upper_hr*100:.1f}%",
        f"{results_rolling['3-Month']['ci_upper']*100:.1f}%",
        f"{results_rolling['6-Month']['ci_upper']*100:.1f}%",
        f"{ci_hi_d*100:.1f}%",
        f"{up_ci_hi*100:.1f}%",
        f"{down_ci_hi*100:.1f}%",
        f"{ci_excess_hi*100:.2f}%",
        f"{ci_whr_hi*100:.1f}%",
        f"{ci_brdth_hi:.1f}",
    ],
    "Primary_p_value": [
        f"{p_binom_mdd:.4f}",
        f"{p_ttest_hr:.4f}",
        f"{results_rolling['3-Month']['p_ttest']:.4f}",
        f"{results_rolling['6-Month']['p_ttest']:.4f}",
        f"{p_tt_d:.4f}" if not np.isnan(p_tt_d) else "N/A",
        f"{p_boot_uc:.4f}",
        f"{p_boot_dc:.4f}",
        f"{p_tail_t:.4f}",
        f"{p_binom_tail:.4f}",
        "N/A",
    ],
    "Significance": [
        sig_stars(p_binom_mdd),
        sig_stars(p_ttest_hr),
        sig_stars(results_rolling["3-Month"]["p_ttest"]),
        sig_stars(results_rolling["6-Month"]["p_ttest"]),
        sig_stars(p_tt_d) if not np.isnan(p_tt_d) else "N/A",
        sig_stars(p_boot_uc),
        sig_stars(p_boot_dc),
        sig_stars(p_tail_t),
        sig_stars(p_binom_tail),
        "N/A",
    ],
})
master.to_csv(os.path.join(OUTPUT_DIR, "master_summary.csv"), index=False)

print(master.to_string(index=False))
print(f"""
Significance key:  *** p<0.001  ** p<0.01  * p<0.05  . p<0.10  ns = not significant

Output files saved to: {OUTPUT_DIR}/
  Figures : fig0_index_performance.png, fig1_downside_resilience.png,
            fig2_consistency.png, fig3_capture_ratios.png,
            fig4_tail_risk.png, fig5a_sector_metrics.png,
            fig5b_sector_heatmap.png, fig5c_country_hit_rate.png
  CSVs    : master_summary.csv, analysis1_*, analysis2_*, analysis3_*,
            analysis4_*, analysis5_*
DONE.
""")
