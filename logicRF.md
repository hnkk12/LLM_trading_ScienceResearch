# Logic RF - Supervised Machine Learning Baseline (Detailed Technical Specification)

This document presents the detailed architectural design, feature engineering mathematics, walk-forward training windows, risk management overlays, model interpretability framework, and execution model of the **Random Forest Baseline Bot**. There are two identical implementations that differ only by their data paths:
1. **Original Version (`rf_baseline.py`)**: Runs on the original daily market dataset (located in the `dataset/` directory) and outputs results to the `data-backtest/` directory, with summary results in `results/rf/`.
2. **Robust/Stress-Tested Version (`rf2.py`)**: Runs on the perturbed daily market dataset (located in the `dataset_robust/` directory) and outputs results to the `data-backtest2/` directory, with summary results in `results2/rf/`.

Both versions share the exact same trading logic, feature calculation, model parameters, and risk management framework.

---

## 1. Architecture Overview & System Role

The Random Forest Baseline Bot is a **supervised machine learning control model** designed to represent the **Bagging ensemble approach**. By comparing Random Forest (Bagging) against XGBoost (Boosting) on identical data feeds, technical features, and risk overlays, we can isolate the impact of the ensemble method on high-noise financial time series.

```
Baseline (Technical Rules)
  → RMDB (Rules + Risk Gates)                  ← Isolates impact of Risk Gates
    → Random Forest (ML Bagging + Risk Gates)   ← Isolates Bagging vs Boosting (NEW)
      → XGBoost (ML Boosting + Risk Gates)      ← Isolates value of LLM Reasoning
        → LLM Agent (LLM Decisions + Risk Gates)
```

### Core Architecture:
1.  **Tabular Feature Input**: Receives 12 numeric features computed from historical price action and volume (identical to prompt inputs).
2.  **Walk-Forward Bagging Classifier**: Fits a Random Forest binary classifier on pre-stress historical windows and predicts next-day price direction.
3.  **Risk Management overlay**: Applies the exact same 1% risk-per-trade position sizing, dynamic stop-loss, and ATR-based volatility gate as the RMDB, XGBoost, and LLM bots.
4.  **Transaction Fees & Slippage**: Incorporates a taker fee of **0.05%** per transaction, a bid-ask spread of **0.02%**, and execution price slippage models (S0, S1, S2).
5.  **Interpretability Engine**: Extracts feature importance (MDI) and computes SHAP (SHapley Additive exPlanations) values on out-of-sample data to explain what technical indicators drive predictions.

---

## 2. Feature Engineering & Target Labelling

Features are calculated on the full chronological daily dataset from 2004 to 2023 to avoid cold-start problems (ensuring EMA50 and ATR100 are fully populated from day 1 of the test window).

### A. Technical Indicator Features
*   **Relative Strength Index (RSI-14)**:
    $$\text{RSI} = 100 - \frac{100}{1 + \text{RS}}, \quad \text{where } \text{RS} = \frac{\text{EMA}(\text{U}, 14)}{\text{EMA}(\text{D}, 14)}$$
*   **Moving Average Convergence Divergence (MACD)**:
    $$\text{MACD Line} = \text{EMA}(Close, 12) - \text{EMA}(Close, 26)$$
    $$\text{Signal Line} = \text{EMA}(\text{MACD Line}, 9)$$
    $$\text{Histogram} = \text{MACD Line} - \text{Signal Line}$$
*   **Exponential Moving Averages (EMA-20 & EMA-50)**:
    $$\text{EMA}_t = \alpha \times Close_t + (1 - \alpha) \times \text{EMA}_{t-1}, \quad \alpha = \frac{2}{N+1}$$
*   **Average True Range (ATR-14)**:
    $$\text{TR} = \max\left( \text{High}_t - \text{Low}_t, |\text{High}_t - Close_{t-1}|, |\text{Low}_t - Close_{t-1}| \right)$$
    $$\text{ATR}_t = \frac{\text{ATR}_{t-1} \times 13 + \text{TR}_t}{14}$$
*   **Volume Ratio**:
    $$\text{Volume Ratio} = \frac{\text{Volume}_t}{\frac{1}{20}\sum_{i=0}^{19} \text{Volume}_{t-i}}$$

### B. Price Context Features
*   **Price Returns (1-day, 5-day, 20-day)**:
    $$\text{close\_pct\_kd}_t = \frac{Close_t}{Close_{t-k}} - 1, \quad k \in \{1, 5, 20\}$$

### C. Volatility Gate Flag
*   **vol\_gate\_flag**: A binary variable indicating if the short-term ATR exceeds 2.0x the rolling mean ATR:
    $$\text{vol\_gate\_flag} = \begin{cases} 1 & \text{if } \text{ATR}_{14, t} > 2.0 \times \left( \frac{1}{20} \sum_{i=0}^{19} \text{ATR}_{14, t-i} \right) \\ 0 & \text{otherwise} \end{cases}$$

### D. Target Variable
*   **forward\_return\_sign**: The direction of price change for the next trading day:
    $$\text{forward\_return} = \frac{Close_{t+1}}{Close_t} - 1$$
    $$\text{forward\_return\_sign} = \begin{cases} 1 & \text{if } \text{forward\_return} > 0 \\ 0 & \text{otherwise} \end{cases}$$

---

## 3. Walk-Forward Training Schema

To prevent lookahead bias (data leakage), the Random Forest model is trained strictly on out-of-sample historical periods preceding the stress windows. The training weights are frozen before backtesting begins.

| Period Name | Training Window (In-Sample) | Stress Backtesting Window (Out-of-Sample) |
| :--- | :--- | :--- |
| **2008-2009** | Jan 1, 2005 - Dec 31, 2007 | Jan 1, 2008 - Dec 31, 2009 |
| **2020-2021** | Jan 1, 2017 - Dec 31, 2019 | Jan 1, 2020 - Dec 31, 2021 |
| **2022-2023** | Jan 1, 2019 - Dec 31, 2021 | Jan 1, 2022 - Dec 31, 2023 |

---

## 4. Random Forest Model Configuration

The Random Forest parameters are constrained to prevent overfitting on financial market noise, matching the depth and tree size limits of the XGBoost baseline:

*   `n_estimators`: `200` (number of decision trees in the forest)
*   `max_depth`: `4` (limits tree depth to avoid memorizing noise)
*   `class_weight`: `{0: 1.0, 1: scale_weight}` (balances class imbalance where `scale_weight = n_neg / n_pos`, matching XGBoost's `scale_pos_weight`)
*   `random_state`: `42` (deterministic random state)
*   `n_jobs`: `-1` (parallel execution across all available CPU cores)
*   `verbose`: `0`

### Signal Threshold:
A long position entry is triggered **only if** the model prediction probability exceeds $55\%$:
$$\text{Signal}_t = \begin{cases} 1 \text (Long) & \text{if } P(\text{Price rises}) > 0.55 \\ 0 \text (Flat) & \text{otherwise} \end{cases}$$

---

## 5. Risk-Managed Trading Rules

Once a signal is generated, it must satisfy the risk-management gates before execution:

1.  **Volatility Filter**: Skip entry if `vol_gate_flag == 1`.
2.  **Position Sizing (1% Risk-per-trade Rule)**:
    $$\text{Risk Amount} = \text{Current Equity}_t \times 0.01$$
    $$\text{SL Distance} = 2.0 \times \text{ATR}_{14, t}$$
    $$\text{Shares} = \text{round}\left( \frac{\text{Risk Amount}}{\text{SL Distance}}, 4 \right)$$
    *Note: Fractional sizes are allowed (up to 4 decimals).*
3.  **Capital Constraint**: Max trade value is capped at 95% of equity (leverage limits).
4.  **Stop-Loss Placement**:
    $$\text{SL Price} = \text{Entry Price} - \text{SL Distance}$$
5.  **Exits**: The position is closed if:
    *   The low price of the day touches or falls below the Stop-Loss: $\text{Low}_t \le \text{SL Price}$.
    *   The model signal turns flat: $\text{Signal}_t == 0$.

---

## 6. Slippage & Fee Specifications

*   **Taker Fee Rate**: $0.05\%$ per transaction (applied to entry and exit total values).
*   **Spread Rate**: $0.02\%$ bid-ask spread of price.
*   **Slippage Scenarios (S0, S1, S2)**:
    *   **S0 (Dynamic Slippage)**:
        $$\text{Slippage} = 0.1 \times \text{ATR}_{14, t} + 0.5 \times \text{Spread}$$
    *   **S1 (Fixed 0.05%)**:
        $$\text{Slippage} = 0.0005 \times \text{Price}_t + 0.5 \times \text{Spread}$$
    *   **S2 (Fixed 0.10%)**:
        $$\text{Slippage} = 0.0010 \times \text{Price}_t + 0.5 \times \text{Spread}$$
*   **Execution Price**:
    $$\text{Entry Price}_{\text{Long}} = Close_t + \text{Slippage}$$
    $$\text{Exit Price}_{\text{Long}} = Close_t - \text{Slippage}$$

---

## 7. Model Interpretability & Explainability (SHAP)

To identify what indicators drive the Random Forest's decision-making, we analyze feature importance and directional attribution.

### A. Feature Grouping
Features are categorized into 7 groups:
1.  **RSI**: `rsi_14`
2.  **MACD**: `macd`, `macd_signal`, `macd_hist`
3.  **EMA**: `ema_20`, `ema_50`
4.  **ATR**: `atr_14`
5.  **Volume Ratio**: `volume_ratio`
6.  **Returns**: `close_pct_1d`, `close_pct_5d`, `close_pct_20d`
7.  **Volatility Gate**: `vol_gate_flag`

### B. MDI-Based Feature Importance
Calculated using Mean Decrease in Impurity (MDI) or Gini importance, representing the fraction of total impurity reduction achieved by splitting on a feature, averaged over all 200 estimators.
$$\text{Importance}_{\text{Group}} = \sum_{f \in \text{Group}} \text{Importance}(f)$$

### C. SHAP (SHapley Additive exPlanations)
Computes Shapley values using `TreeExplainer` on the out-of-sample test splits to measure the marginal contribution of each indicator value to the output log-odds prediction.
*   **Global Impact**: Calculated using the mean absolute SHAP value across all test dates ($mean(|\phi_j|)$).
*   **Variance Dampening**: Because Random Forest relies on Bagging (averaging independent tree predictions), its SHAP values are compressed by a factor of ~10 compared to XGBoost, indicating high robustness and stability of predictions.

---

## 8. Output Result Schema

All Random Forest backtests generate matching results standard to the paper framework, located under their respective execution folders:

### A. Version 1: Original Dataset Outputs (`rf_baseline.py`)
*   **Run Directories**: `data-backtest/AAPL_RF_{period}_{scenario}/` or `data-backtest/GOLD_RF_{period}_{scenario}/`
    *   `backtest_results.json`: Complete dictionary of metadata, daily equity series, daily returns series, capital ratios, and trade performance (with non-zero calculated VaR/CVaR).
    *   `daily_returns.csv`: Daily return path.
    *   `trade_history.csv`: List of entry, exit, holding days, close types, and PnL.
    *   `backtest_summary.txt`: ASCII format metrics summary table.
*   **Averages & Interpretability**: `results/rf/`
    *   `aggregate_performance.csv`: Summary performance rows across 18 backtest combinations.
    *   `trade_diagnostics.csv`: Averages of orders count, win rate, and hold duration.
    *   `rf_feature_importance.csv`: Rank list of feature MDI values.
    *   `shap_summary.csv`: Rank list of mean absolute SHAP values.
    *   `rf_feature_importance.png` / `rf_group_importance.png` / `shap_summary.png`: Feature and SHAP beeswarm visualizations.
*   **Consolidated Report**: `results/table2_combined.csv` (Combined averages for all 5 systems).

### B. Version 2: Robust Dataset Outputs (`rf2.py`)
*   **Run Directories**: `data-backtest2/AAPL_RF_{period}_{scenario}/` or `data-backtest2/GOLD_RF_{period}_{scenario}/`
    *   *Note: Files inside are structured identically to Version 1.*
*   **Averages & Interpretability**: `results2/rf/`
    *   *Note: Diagnostic CSVs, feature importance lists, and PNG plots are saved here, representing model behavior under perturbed conditions.*
*   **Consolidated Report**: `results2/table2_combined.csv` (Combined averages for robust ML runs).
