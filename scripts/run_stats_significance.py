#!/usr/bin/env python3
"""
Statistical Significance Testing Framework.
Compares the LLM trading agent results against the rule-based baseline bot results.
Performs:
- Student's t-test on daily returns
- Mann-Whitney U test on daily returns
- Bootstrap confidence intervals for Sharpe and Sortino differences
- Drawdown distribution comparison
- Generates a research report saved to data-backtest/statistical_significance_report.md
"""

import os
import sys
import json
import math
import argparse
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
import numpy as np
import pandas as pd

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Standard normal CDF approximation using math.erf
def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def calculate_sharpe(returns: np.ndarray, risk_free_annual: float = 0.0) -> float:
    # Daily returns. Standard annualization is sqrt(252)
    if len(returns) == 0:
        return 0.0
    rf_daily = risk_free_annual / 252.0
    excess = returns - rf_daily
    mean_excess = np.mean(excess)
    std_excess = np.std(excess, ddof=1)
    if std_excess == 0:
        return 0.0
    return float(np.sqrt(252.0) * mean_excess / std_excess)

def calculate_sortino(returns: np.ndarray, risk_free_annual: float = 0.0) -> float:
    if len(returns) == 0:
        return 0.0
    rf_daily = risk_free_annual / 252.0
    excess = returns - rf_daily
    mean_excess = np.mean(excess)
    downside_returns = excess[excess < 0]
    if len(downside_returns) == 0:
        return 0.0
    downside_std = np.sqrt(np.sum(downside_returns**2) / len(returns))
    if downside_std == 0:
        return 0.0
    return float(np.sqrt(252.0) * mean_excess / downside_std)

def run_t_test(x: np.ndarray, y: np.ndarray):
    """Student's t-test (independent two-sample with unequal variances / Welch's t-test)."""
    # Use scipy if available, otherwise pure numpy
    try:
        from scipy import stats
        res = stats.ttest_ind(x, y, equal_var=False)
        return float(res.statistic), float(res.pvalue)
    except ImportError:
        n1 = len(x)
        n2 = len(y)
        m1, m2 = np.mean(x), np.mean(y)
        v1, v2 = np.var(x, ddof=1), np.var(y, ddof=1)
        
        # Welch's t-test
        denom = np.sqrt(v1/n1 + v2/n2)
        if denom == 0:
            return 0.0, 1.0
        t_stat = (m1 - m2) / denom
        
        # Degrees of freedom (Welch–Satterthwaite)
        df_num = (v1/n1 + v2/n2)**2
        df_den = (v1/n1)**2 / (n1 - 1) + (v2/n2)**2 / (n2 - 1)
        df = df_num / df_den
        
        # P-value approximation using normal distribution as proxy for large samples
        p_val = 2.0 * (1.0 - norm_cdf(abs(t_stat)))
        return float(t_stat), float(p_val)

def run_mann_whitney(x: np.ndarray, y: np.ndarray):
    """Mann-Whitney U test."""
    try:
        from scipy import stats
        res = stats.mannwhitneyu(x, y, alternative='two-sided')
        return float(res.statistic), float(res.pvalue)
    except ImportError:
        n1 = len(x)
        n2 = len(y)
        combined = np.concatenate([x, y])
        ranks = pd.Series(combined).rank()
        r1 = ranks[:n1].sum()
        u1 = n1 * n2 + (n1 * (n1 + 1)) / 2.0 - r1
        u2 = n1 * n2 - u1
        u_stat = min(u1, u2)
        
        # Normal approximation (large samples)
        mu_u = (n1 * n2) / 2.0
        sigma_u = np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12.0)
        if sigma_u == 0:
            return float(u_stat), 1.0
        z_stat = (u_stat - mu_u) / sigma_u
        p_val = 2.0 * (1.0 - norm_cdf(abs(z_stat)))
        return float(u_stat), float(p_val)

def run_bootstrap(x: np.ndarray, y: np.ndarray, num_bootstrap: int = 1000):
    """Bootstrap confidence intervals for difference of Sharpe and Sortino ratios."""
    np.random.seed(42) # Deterministic bootstrap seed
    sharpe_diffs = []
    sortino_diffs = []
    
    n_x, n_y = len(x), len(y)
    for _ in range(num_bootstrap):
        boot_x = np.random.choice(x, size=n_x, replace=True)
        boot_y = np.random.choice(y, size=n_y, replace=True)
        
        sh_x = calculate_sharpe(boot_x)
        sh_y = calculate_sharpe(boot_y)
        sharpe_diffs.append(sh_x - sh_y)
        
        so_x = calculate_sortino(boot_x)
        so_y = calculate_sortino(boot_y)
        sortino_diffs.append(so_x - so_y)
        
    sharpe_diffs = np.array(sharpe_diffs)
    sortino_diffs = np.array(sortino_diffs)
    
    ci_sharpe = (np.percentile(sharpe_diffs, 2.5), np.percentile(sharpe_diffs, 97.5))
    ci_sortino = (np.percentile(sortino_diffs, 2.5), np.percentile(sortino_diffs, 97.5))
    
    return ci_sharpe, ci_sortino, np.mean(sharpe_diffs), np.mean(sortino_diffs)

def calculate_drawdowns(equity: list) -> np.ndarray:
    eq = np.array(equity)
    if len(eq) == 0:
        return np.array([])
    cum_max = np.maximum.accumulate(eq)
    # Avoid zero division
    cum_max = np.where(cum_max == 0, 1.0, cum_max)
    drawdowns = (cum_max - eq) / cum_max
    return drawdowns

def find_latest_results(prefix: str) -> Optional[Path]:
    """Find the latest backtest results file matching prefix in directories."""
    data_backtest_dir = PROJECT_ROOT / "data-backtest"
    if not data_backtest_dir.exists():
        return None
    
    run_dirs = sorted(
        [d for d in data_backtest_dir.iterdir() if d.is_dir() and d.name.startswith("run-")],
        key=lambda d: d.stat().st_mtime,
        reverse=True
    )
    
    for r_dir in run_dirs:
        res_file = r_dir / "backtest_results.json"
        if res_file.exists():
            try:
                with open(res_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    model_name = data.get("llm", {}).get("model", "")
                    if prefix.lower() in model_name.lower():
                        return res_file
            except:
                continue
    return None

def main():
    parser = argparse.ArgumentParser(description="Statistical Significance Testing for LLM Trading Bot.")
    parser.add_argument("--llm", type=str, help="Path to LLM backtest_results.json")
    parser.add_argument("--baseline", type=str, help="Path to baseline backtest_results.json")
    args = parser.parse_args()

    llm_path = args.llm
    baseline_path = args.baseline

    # Auto-resolve if paths not provided
    if not llm_path:
        found = find_latest_results("llama")
        if found:
            llm_path = str(found)
            print(f"Auto-detected LLM results: {llm_path}")
        else:
            # Fallback to search any run with model not equal to rule-based
            data_backtest_dir = PROJECT_ROOT / "data-backtest"
            if data_backtest_dir.exists():
                for r_dir in sorted(data_backtest_dir.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True):
                    res_file = r_dir / "backtest_results.json"
                    if res_file.exists():
                        try:
                            with open(res_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                m = data.get("llm", {}).get("model", "")
                                if "baseline" not in m.lower():
                                    llm_path = str(res_file)
                                    print(f"Auto-detected LLM results (non-baseline): {llm_path}")
                                    break
                        except:
                            continue

    if not baseline_path:
        found = find_latest_results("baseline")
        if found:
            baseline_path = str(found)
            print(f"Auto-detected Baseline results: {baseline_path}")

    if not llm_path or not baseline_path:
        print("Error: Please provide paths to both --llm and --baseline backtest results.")
        print("Usage: python scripts/run_stats_significance.py --llm <path> --baseline <path>")
        sys.exit(1)

    # Load results
    with open(llm_path, 'r', encoding='utf-8') as f:
        llm_data = json.load(f)
    with open(baseline_path, 'r', encoding='utf-8') as f:
        base_data = json.load(f)

    llm_returns = np.array(llm_data.get("daily_returns", []))
    base_returns = np.array(base_data.get("daily_returns", []))

    if len(llm_returns) == 0 or len(base_returns) == 0:
        print("Error: One or both of the result files do not contain daily_returns history.")
        sys.exit(1)

    llm_equity = llm_data.get("equity_history", [])
    base_equity = base_data.get("equity_history", [])

    llm_dd = calculate_drawdowns(llm_equity)
    base_dd = calculate_drawdowns(base_equity)

    # 1. Student's t-test
    t_stat, p_val_t = run_t_test(llm_returns, base_returns)

    # 2. Mann-Whitney U test
    u_stat, p_val_u = run_mann_whitney(llm_returns, base_returns)

    # 3. Bootstrap Confidence Intervals
    ci_sharpe, ci_sortino, mean_diff_sharpe, mean_diff_sortino = run_bootstrap(llm_returns, base_returns, num_bootstrap=1000)

    # Calculate indicators
    llm_cap = llm_data["capital"]
    base_cap = base_data["capital"]

    report_lines = [
        "# Statistical Significance Report",
        f"Generated on: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This report evaluates the statistical significance of differences in performance between the LLM Trading Decision Agent and the rule-based baseline bot.",
        "",
        "## Performance Comparison Summary",
        "",
        "| Metric | LLM Trading Agent | Rule-Based Baseline | Difference |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Model / Agent** | {llm_data['llm']['model']} | {base_data['llm']['model']} | - |",
        f"| **Total Net Profit** | ${llm_cap['total_net_profit']:,.2f} | ${base_cap['total_net_profit']:,.2f} | ${llm_cap['total_net_profit'] - base_cap['total_net_profit']:+,.2f} |",
        f"| **Return %** | {llm_cap['total_return_pct']:+.2f}% | {base_cap['total_return_pct']:+.2f}% | {llm_cap['total_return_pct'] - base_cap['total_return_pct']:+.2f}% |",
        f"| **Max Drawdown** | {llm_cap['max_drawdown_pct']:.2f}% | {base_cap['max_drawdown_pct']:.2f}% | {llm_cap['max_drawdown_pct'] - base_cap['max_drawdown_pct']:.2f}% |",
        f"| **Sharpe Ratio** | {llm_cap['sharpe_ratio'] if llm_cap['sharpe_ratio'] is not None else 0.0:.2f} | {base_cap['sharpe_ratio'] if base_cap['sharpe_ratio'] is not None else 0.0:.2f} | {(llm_cap['sharpe_ratio'] if llm_cap['sharpe_ratio'] is not None else 0.0) - (base_cap['sharpe_ratio'] if base_cap['sharpe_ratio'] is not None else 0.0):+.2f} |",
        f"| **Sortino Ratio** | {llm_cap['sortino_ratio'] if llm_cap['sortino_ratio'] is not None else 0.0:.2f} | {base_cap['sortino_ratio'] if base_cap['sortino_ratio'] is not None else 0.0:.2f} | {(llm_cap['sortino_ratio'] if llm_cap['sortino_ratio'] is not None else 0.0) - (base_cap['sortino_ratio'] if base_cap['sortino_ratio'] is not None else 0.0):+.2f} |",
        f"| **Profit Factor** | {llm_cap['profit_factor'] if llm_cap['profit_factor'] is not None else 0.0:.2f} | {base_cap['profit_factor'] if base_cap['profit_factor'] is not None else 0.0:.2f} | {(llm_cap['profit_factor'] if llm_cap['profit_factor'] is not None else 0.0) - (base_cap['profit_factor'] if base_cap['profit_factor'] is not None else 0.0):+.2f} |",
        f"| **Recovery Factor** | {llm_cap['recovery_factor'] if llm_cap['recovery_factor'] is not None else 0.0:.2f} | {base_cap['recovery_factor'] if base_cap['recovery_factor'] is not None else 0.0:.2f} | {(llm_cap['recovery_factor'] if llm_cap['recovery_factor'] is not None else 0.0) - (base_cap['recovery_factor'] if base_cap['recovery_factor'] is not None else 0.0):+.2f} |",
        f"| **VaR (95% Daily)** | {llm_cap.get('var_95_pct', 0.0):.2f}% | {base_cap.get('var_95_pct', 0.0):.2f}% | {llm_cap.get('var_95_pct', 0.0) - base_cap.get('var_95_pct', 0.0):+.2f}% |",
        f"| **CVaR (95% Daily)** | {llm_cap.get('cvar_95_pct', 0.0):.2f}% | {base_cap.get('cvar_95_pct', 0.0):.2f}% | {llm_cap.get('cvar_95_pct', 0.0) - base_cap.get('cvar_95_pct', 0.0):+.2f}% |",
        f"| **Total Trades** | {llm_data['trading']['total_trades']} | {base_data['trading']['total_trades']} | {llm_data['trading']['total_trades'] - base_data['trading']['total_trades']:+d} |",
        f"| **Win Rate** | {llm_cap['win_rate_pct']:.1f}% | {base_cap['win_rate_pct']:.1f}% | {llm_cap['win_rate_pct'] - base_cap['win_rate_pct']:+.1f}% |",
        "",
        "## Hypothesis Testing",
        "",
        "### 1. Student's t-test (Welch's Formulation)",
        "* **Null Hypothesis ($H_0$)**: There is no difference in the mean daily return between the LLM agent and the baseline.",
        "* **Alternative Hypothesis ($H_a$)**: There is a significant difference in the mean daily return.",
        f"* **t-statistic**: `{t_stat:.4f}`",
        f"* **p-value**: `{p_val_t:.4e}`",
        f"* **Result**: {'Reject $H_0$ (Statistically Significant)' if p_val_t < 0.05 else 'Fail to reject $H_0$ (Not Statistically Significant)'} at 5% level.",
        "",
        "### 2. Mann-Whitney U Test (Non-Parametric)",
        "* **Null Hypothesis ($H_0$)**: The distributions of daily returns for both agents are identical.",
        "* **Alternative Hypothesis ($H_a$)**: The distributions are shifted (one agent stochastically dominates the other).",
        f"* **U-statistic**: `{u_stat:.2f}`",
        f"* **p-value**: `{p_val_u:.4e}`",
        f"* **Result**: {'Reject $H_0$ (Statistically Significant)' if p_val_u < 0.05 else 'Fail to reject $H_0$ (Not Statistically Significant)'} at 5% level.",
        "",
        "## Bootstrap Confidence Intervals (95% Confidence)",
        "We generated 1,000 bootstrap resamples with replacement to compute the distribution of differences in risk-adjusted performance metrics.",
        "",
        "### Sharpe Ratio Difference (LLM - Baseline)",
        f"* **Mean Sharpe Difference**: `{mean_diff_sharpe:+.4f}`",
        f"* **95% Bootstrap CI**: `[{ci_sharpe[0]:.4f}, {ci_sharpe[1]:.4f}]`",
        f"* **Significance**: {'Statistically Significant (CI does not contain 0.0)' if (ci_sharpe[0] > 0 or ci_sharpe[1] < 0) else 'Not Statistically Significant (CI contains 0.0)'}",
        "",
        "### Sortino Ratio Difference (LLM - Baseline)",
        f"* **Mean Sortino Difference**: `{mean_diff_sortino:+.4f}`",
        f"* **95% Bootstrap CI**: `[{ci_sortino[0]:.4f}, {ci_sortino[1]:.4f}]`",
        f"* **Significance**: {'Statistically Significant (CI does not contain 0.0)' if (ci_sortino[0] > 0 or ci_sortino[1] < 0) else 'Not Statistically Significant (CI contains 0.0)'}",
        "",
        "## Risk & Drawdown Distribution Analysis",
        "",
        "| Drawdown Statistic | LLM Trading Agent | Rule-Based Baseline |",
        "| :--- | :---: | :---: |",
        f"| **Maximum Drawdown** | {llm_cap['max_drawdown_pct']:.2f}% | {base_cap['max_drawdown_pct']:.2f}% |",
        f"| **Mean Intrabar Drawdown** | {np.mean(llm_dd)*100:.2f}% | {np.mean(base_dd)*100:.2f}% |",
        f"| **Drawdown Volatility (StdDev)** | {np.std(llm_dd)*100:.2f}% | {np.std(base_dd)*100:.2f}% |",
        "",
        "## Conclusion & Research Interpretation",
        f"The LLM trading decision agent achieved a final return of **{llm_cap['total_return_pct']:+.2f}%** compared to **{base_cap['total_return_pct']:+.2f}%** for the baseline.",
        f"Based on the t-test p-value of `{p_val_t:.2e}`, the performance difference {'is' if p_val_t < 0.05 else 'is NOT'} statistically significant.",
        f"Furthermore, the bootstrap 95% confidence interval for Sharpe difference is `[{ci_sharpe[0]:.4f}, {ci_sharpe[1]:.4f}]`, confirming that the difference in risk-adjusted performance {'is' if (ci_sharpe[0] > 0 or ci_sharpe[1] < 0) else 'is NOT'} statistically robust.",
        "",
        "This research suggests that LLM cognitive capabilities under financial crisis regimes " + 
        ("provide a statistically verifiable advantage" if (p_val_t < 0.05 and ci_sharpe[0] > 0) else "do not yield a statistically significant advantage over simple momentum rules") +
        " when considering transaction friction (spread/slippage)."
    ]

    report_content = "\n".join(report_lines)
    
    # Save report
    data_backtest_dir = PROJECT_ROOT / "data-backtest"
    data_backtest_dir.mkdir(parents=True, exist_ok=True)
    report_path = data_backtest_dir / "statistical_significance_report.md"
    
    with open(report_path, "w", encoding='utf-8') as fh:
        fh.write(report_content)
        
    print(f"\nReport successfully generated and saved to: {report_path}\n")
    print("=" * 60)
    print("                 HYPOTHESIS TESTING SUMMARY")
    print("=" * 60)
    print(f"t-statistic: {t_stat:.4f}  |  p-value (t-test): {p_val_t:.4e}")
    print(f"U-statistic: {u_stat:.2f}  |  p-value (MWU-test): {p_val_u:.4e}")
    print(f"Mean Sharpe Difference: {mean_diff_sharpe:+.4f} (95% CI: [{ci_sharpe[0]:.4f}, {ci_sharpe[1]:.4f}])")
    print(f"Mean Sortino Difference: {mean_diff_sortino:+.4f} (95% CI: [{ci_sortino[0]:.4f}, {ci_sortino[1]:.4f}])")
    print("=" * 60)

if __name__ == "__main__":
    main()
