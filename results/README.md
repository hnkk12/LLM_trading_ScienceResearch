# Results Directory - DSAA 2026 Evaluation Tables

This directory contains the aggregated CSV performance tables and daily return files generated for the DSAA 2026 Paper. These files compile the backtest evaluations of the four compared trading systems:

1.  **Baseline**: Deterministic rule-based SMC/Wyckoff bot.
2.  **RMDB**: Risk-Managed Deterministic Baseline (rules + risk gates).
3.  **XGBoost**: Supervised ML baseline (walk-forward classifier + risk gates).
4.  **LLM Agent**: Language model reasoning agent (Llama-3.3-70B + risk gates).

---

## 1. Directory Structure

```
results/
├── table2_combined.csv                # Consolidated Table II (Baseline vs RMDB vs LLM vs XGBoost)
└── xgboost/
    ├── aggregate_performance.csv      # Individual XGBoost run metrics (18 rows)
    ├── trade_diagnostics.csv          # Table V: Trades per run, win rates, and holding time
    ├── mdd_advantage_counts.csv       # Table III: Drawdown and return wins count vs other systems
    ├── daily_returns_AAPL_2008_2009_S0.csv
    ├── ...                            # 18 daily return & equity series files
    └── daily_returns_Gold_2022_2023_S2.csv
```

---

## 2. File Specifications & Mapping to Paper Tables

### A. Combined Performance Averages (`results/table2_combined.csv`)
*   **Target Table**: **Table II** (Consolidated performance averages across scenarios).
*   **Description**: Combines and averages the daily returns, drawdown, Sharpe, and Sortino metrics across all test windows for each system and scenario.
*   **Columns**:
    *   `system`: "Baseline", "RMDB", "XGBoost", or "LLM".
    *   `scenario`: "S0" (dynamic ATR slippage), "S1" (0.05% fixed), "S2" (0.10% fixed).
    *   `mean_return`: Mean daily portfolio return (%).
    *   `mdd`: Mean Maximum Drawdown (%).
    *   `sharpe`: Sharpe ratio (annualized, risk-free rate = 0).
    *   `sortino`: Sortino ratio (annualized, risk-free rate = 0).

### B. Trade Diagnostics (`results/xgboost/trade_diagnostics.csv`)
*   **Target Table**: **Table V** (Trade execution metrics).
*   **Description**: Summarizes XGBoost trade counts, win ratios, and average holding times per slippage scenario.
*   **Columns**:
    *   `system`: "XGBoost".
    *   `scenario`: "S0", "S1", "S2".
    *   `trades_per_run`: Average number of closed trades per run.
    *   `win_rate`: Percentage of profitable trades (%).
    *   `avg_hold_days`: Average trade holding duration (days).

### C. MDD Advantage Counts (`results/xgboost/mdd_advantage_counts.csv`)
*   **Target Table**: **Table III** (Risk governance comparative counts).
*   **Description**: Counts the number of times (out of 6 test combinations) XGBoost achieved a lower Maximum Drawdown or higher return than the comparison baselines.
*   **Columns**:
    *   `scenario`: "S0", "S1", "S2".
    *   `ref`: "vs Baseline", "vs RMDB", "vs LLM".
    *   `mdd_wins`: Number of runs where XGBoost achieved a lower MDD (max 6).
    *   `ret_wins`: Number of runs where XGBoost achieved a higher return (max 6).
    *   `mdd_adv`: Mean Maximum Drawdown reduction advantage (%).

### D. Raw Daily Returns Series (`results/xgboost/daily_returns_*.csv`)
*   **Usage**: Statistical significance tests (Welch's t-test and Mann-Whitney U test) and bootstrapping confidence intervals.
*   **Description**: Chronological daily tracking of the 18 XGBoost backtest runs.
*   **Columns**:
    *   `date`: Trading date (`YYYY-MM-DD`).
    *   `daily_return`: Portfolio percentage return of the day (float).
    *   `equity`: Mark-to-market account balance (USD).

---

## 3. How to Regenerate the Files

If you add new completed backtest directories to `data-backtest/` or modify the training features, you can fully rebuild all CSV tables in this directory by running:

```bash
python xgboost_baseline.py
```
This script will re-evaluate XGBoost, rebuild the standard runs in `data-backtest/`, parse the other systems, and update these files.
