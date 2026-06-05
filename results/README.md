# Results Directory - DSAA 2026 Evaluation Tables

This directory contains the aggregated CSV performance tables, metrics, explainability reports, and daily return files generated for the DSAA 2026 Paper. These files compile the backtest evaluations of the five compared trading systems:

1.  **Baseline**: Deterministic rule-based SMC/Wyckoff bot.
2.  **RMDB**: Risk-Managed Deterministic Baseline (rules + risk gates).
3.  **Random Forest (RF)**: Supervised ML bagging baseline (walk-forward classifier + risk gates).
4.  **XGBoost**: Supervised ML boosting baseline (walk-forward classifier + SHAP interpretability + risk gates).
5.  **LLM Agent**: Language model reasoning agent (Llama-3.3-70B + risk gates).

---

## 1. Directory Structure

```
results/
├── README.md                          # This documentation file
├── table2_combined.csv                # Consolidated Table II (Baseline vs RMDB vs RF vs XGBoost vs LLM)
├── rf/                                # Random Forest outputs
│   ├── aggregate_performance.csv      # Individual RF run metrics (18 rows)
│   └── trade_diagnostics.csv          # Table V: Average trades, win rates, and hold times for RF
└── xgboost/                           # XGBoost outputs
    ├── aggregate_performance.csv      # Individual XGBoost run metrics (18 rows)
    ├── trade_diagnostics.csv          # Table V: Average trades, win rates, and hold times for XGBoost
    ├── xgboost_feature_importance.csv # XGBoost individual feature importance (Gain-based)
    ├── shap_summary.csv               # Mean absolute SHAP values per feature
    ├── xgboost_feature_importance.png # Feature gain importance bar plot
    ├── xgboost_group_importance.png   # Grouped feature importance bar plot
    ├── shap_summary.png               # SHAP beeswarm summary plot
    ├── mdd_advantage_counts.csv       # Table III: Drawdown/return wins count vs other systems
    ├── daily_returns_AAPL_2008_2009_S0.csv
    ├── ...                            # 18 daily return & equity series files
    └── daily_returns_Gold_2022_2023_S2.csv
```

---

## 2. File Specifications & Mapping to Paper Tables

### A. Combined Performance Averages (`results/table2_combined.csv`)
*   **Target Table**: **Table II** (Consolidated performance averages across scenarios).
*   **Description**: Combines and averages the daily returns, drawdown, Sharpe, and Sortino metrics across all test windows and assets for each of the 5 systems and scenarios.
*   **Columns**:
    *   `system`: "Baseline", "RMDB", "RF", "XGBoost", or "LLM".
    *   `scenario`: "S0" (dynamic ATR slippage), "S1" (0.05% fixed), "S2" (0.10% fixed).
    *   `mean_return`: Mean daily portfolio return (%).
    *   `mdd`: Mean Maximum Drawdown (%).
    *   `sharpe`: Sharpe ratio (annualized, risk-free rate = 0).
    *   `sortino`: Sortino ratio (annualized, risk-free rate = 0).

### B. Trade Diagnostics (`results/rf/trade_diagnostics.csv` and `results/xgboost/trade_diagnostics.csv`)
*   **Target Table**: **Table V** (Trade execution metrics).
*   **Description**: Summarizes trade counts, win ratios, and average holding times per slippage scenario for the ML models.
*   **Columns**:
    *   `system`: "RF" or "XGBoost".
    *   `scenario`: "S0", "S1", "S2".
    *   `trades_per_run`: Average number of closed trades per run.
    *   `win_rate`: Percentage of profitable trades (%).
    *   `avg_hold_days`: Average trade holding duration (days).

### C. XGBoost Model Explainability Files (`results/xgboost/...`)
*   `xgboost_feature_importance.csv`: The average Information Gain score for each of the 12 features across the walk-forward models, mapped to 7 technical indicator groups.
*   `shap_summary.csv`: The mean absolute SHAP value ($mean(|\phi_j|)$) representing the average magnitude of a feature's effect on model output probability.
*   `xgboost_feature_importance.png`: Visual bar chart of individual feature importances (Gain).
*   `xgboost_group_importance.png`: Visual bar chart of grouped indicator importances (Returns, MACD, EMA, RSI, Volume Ratio, ATR, Volatility Gate).
*   `shap_summary.png`: Beeswarm plot visualizing both feature importance and the directional effect (e.g. how high vs low values of an indicator drive long signals).

### D. MDD Advantage Counts (`results/xgboost/mdd_advantage_counts.csv`)
*   **Target Table**: **Table III** (Risk governance comparative counts).
*   **Description**: Counts the number of times (out of 6 test combinations) XGBoost achieved a lower Maximum Drawdown or higher return compared to Baseline, RMDB, RF, and LLM systems.

---

## 3. How to Regenerate the Files

If you add new completed backtest directories to `data-backtest/` or modify the training features, you can fully rebuild all CSV tables in this directory using the following sequence:

1.  **Run Random Forest baseline**:
    ```bash
    python rf_baseline.py
    ```
    This generates the `RF` backtest directories under `data-backtest/` and summaries under `results/rf/`.

2.  **Run XGBoost baseline and aggregator**:
    ```bash
    python xgboost_baseline.py
    ```
    This generates the `XGBoost` backtest directories under `data-backtest/`, performs the SHAP explainability analysis and plotting, scans all completed backtests (Baseline, RMDB, RF, LLM), and recompiles the unified comparison file `table2_combined.csv` and advantage counts.
