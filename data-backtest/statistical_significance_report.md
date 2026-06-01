# Statistical Significance Report

Generated on: 2026-05-25T08:23:04.065747+00:00

This report evaluates the statistical significance of differences in performance between the LLM Trading Decision Agent and the rule-based baseline bot.

## Performance Comparison Summary

| Metric               |         LLM Trading Agent         | Rule-Based Baseline | Difference |
| :------------------- | :-------------------------------: | :-----------------: | :--------: |
| **Model / Agent**    | meta-llama/llama-3.3-70b-instruct | Rule-Based Baseline |     -      |
| **Total Net Profit** |              $10.13               |       $19.11        |   $-8.98   |
| **Return %**         |              +1.00%               |       +1.89%        |   -0.89%   |
| **Max Drawdown**     |               3.19%               |        7.89%        |   -4.70%   |
| **Sharpe Ratio**     |               0.80                |        0.59         |   +0.21    |
| **Sortino Ratio**    |               1.18                |        0.81         |   +0.37    |
| **Profit Factor**    |               1.23                |        1.18         |   +0.05    |
| **Recovery Factor**  |               0.31                |        0.24         |   +0.07    |
| **VaR (95% Daily)**  |               0.53%               |        1.90%        |   -1.38%   |
| **CVaR (95% Daily)** |               0.83%               |        2.45%        |   -1.62%   |
| **Total Trades**     |                18                 |          6          |    +12     |
| **Win Rate**         |               44.4%               |        33.3%        |   +11.1%   |

## Hypothesis Testing

### 1. Student's t-test (Welch's Formulation)

- **Null Hypothesis ($H_0$)**: There is no difference in the mean daily return between the LLM agent and the baseline.
- **Alternative Hypothesis ($H_a$)**: There is a significant difference in the mean daily return.
- **t-statistic**: `-0.1288`
- **p-value**: `8.9755e-01`
- **Result**: Fail to reject $H_0$ (Not Statistically Significant) at 5% level.

### 2. Mann-Whitney U Test (Non-Parametric)

- **Null Hypothesis ($H_0$)**: The distributions of daily returns for both agents are identical.
- **Alternative Hypothesis ($H_a$)**: The distributions are shifted (one agent stochastically dominates the other).
- **U-statistic**: `3862.00`
- **p-value**: `3.0570e-01`
- **Result**: Fail to reject $H_0$ (Not Statistically Significant) at 5% level.

## Bootstrap Confidence Intervals (95% Confidence)

We generated 1,000 bootstrap resamples with replacement to compute the distribution of differences in risk-adjusted performance metrics.

### Sharpe Ratio Difference (LLM - Baseline)

- **Mean Sharpe Difference**: `+0.0766`
- **95% Bootstrap CI**: `[-4.5372, 4.8507]`
- **Significance**: Not Statistically Significant (CI contains 0.0)

### Sortino Ratio Difference (LLM - Baseline)

- **Mean Sortino Difference**: `+0.4037`
- **95% Bootstrap CI**: `[-6.7108, 9.2816]`
- **Significance**: Not Statistically Significant (CI contains 0.0)

## Risk & Drawdown Distribution Analysis

| Drawdown Statistic               | LLM Trading Agent | Rule-Based Baseline |
| :------------------------------- | :---------------: | :-----------------: |
| **Maximum Drawdown**             |       3.19%       |        7.89%        |
| **Mean Intrabar Drawdown**       |       1.31%       |        2.85%        |
| **Drawdown Volatility (StdDev)** |       0.87%       |        2.32%        |

## Conclusion & Research Interpretation

The LLM trading decision agent achieved a final return of **+1.00%** compared to **+1.89%** for the baseline.
Based on the t-test p-value of `8.98e-01`, the performance difference is NOT statistically significant.
Furthermore, the bootstrap 95% confidence interval for Sharpe difference is `[-4.5372, 4.8507]`, confirming that the difference in risk-adjusted performance is NOT statistically robust.

This research suggests that LLM cognitive capabilities under financial crisis regimes do not yield a statistically significant advantage over simple momentum rules when considering transaction friction (spread/slippage).
