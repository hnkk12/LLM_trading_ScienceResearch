# Logic RMDB - Risk-Managed Deterministic Baseline (Detailed Technical Specification)

This document presents the detailed architectural design, risk governance model, and scalping rules of the **Risk-Managed Deterministic Baseline (RMDB) Bot** (`scripts/run_rmdb.py`).

---

## 1. Architecture Overview & System Role

The RMDB Bot is a **hybrid system** that combines the technical entry/exit indicators of the deterministic Baseline Bot with the volatility-adaptive risk governance frameworks used by the LLM Bot. It serves as a middle-ground control group to isolate whether the LLM's superior performance is due to its qualitative trading decisions or its quantitative risk management overlay.

### Core Architecture:
1.  **Technical Core**: Reuses the Smart Money Concepts (SMC), Order Block (OB), Fair Value Gap (FVG), and Break of Structure (BOS) algorithms from the Baseline Bot to identify potential entries.
2.  **Risk Management Overlay**: Implements the **Volatility Ratio (VR)**, **Market Risk Zones (Green/Yellow/Red)**, and **ATR Gate (Anti-Chasing)** logic from the LLM Active Guardian specification.
3.  **Position Sizing**: Dynamically scales entry risks (1.0%, 0.5%, or 0.0%) based on real-time market volatility.

---

## 2. Volatility-Adaptive Risk Governance Model

The RMDB Bot evaluates market volatility at the start of each daily candle to classify the environment into three distinct **Risk Zones**:

### A. Volatility Ratio ($VR$) Formula:
$$VR = \frac{\text{ATR}_{\text{Short}} \text{ (14-period)}}{\text{ATR}_{\text{Long}} \text{ (100-period)}}$$

### B. Risk Zones Classification:
1.  **Green Zone ($VR < 1.6$ - Active)**:
    *   **Market State**: Low-to-moderate, stable volatility.
    *   **Risk Allocation**: Standard **1% of Balance** per trade.
    *   **Confluence Requirement**: Standard SMC entry signals (OB mitigation, FVG mitigation, or volume breakouts) are executed without secondary trend-alignment filters.
2.  **Yellow Zone ($VR \in [1.6, 2.2]$ - Defensive)**:
    *   **Market State**: High volatility or trend transition.
    *   **Risk Allocation**: Halved to **0.5% of Balance** per trade.
    *   **Confluence Requirement**: Strict trend-alignment filters are enforced. Entry is permitted **only if** momentum supports the trend:
        *   *Long Entry*: Must have either $\text{RSI}_{14} < 50$ (value dip) OR $\text{MACD} > \text{Signal Line}$ (bullish momentum).
        *   *Short Entry*: Must have either $\text{RSI}_{14} > 50$ (value rally) OR $\text{MACD} < \text{Signal Line}$ (bearish momentum).
3.  **Red Zone ($VR > 2.2$ - Pause)**:
    *   **Market State**: Extreme volatility or market stress.
    *   **Risk Allocation**: **0% (Entries Suspended)**.
    *   **Action**: The bot pauses all new trade entries to prevent whipsaw losses. It only manages active open positions.

---

## 3. Scalping Entry & Anti-Chasing (ATR Gate) Rules

To prevent entering trades late when the price has already overextended during a breakout, the RMDB Bot utilizes an **ATR Gate (Anti-Chasing Logic)**:

*   **Bullish Breakout (Long)**:
    *   Trigger: Price closes above the recent Swing High.
    *   ATR Gate: Rejected if:
        $$\text{Current Price} - \text{Swing High Price} > 2.0 \times \text{ATR}$$
*   **Bearish Breakout (Short)**:
    *   Trigger: Price closes below the recent Swing Low.
    *   ATR Gate: Rejected if:
        $$\text{Swing Low Price} - \text{Current Price} > 2.0 \times \text{ATR}$$

---

## 4. Position, Take Profit & Stop Loss Management

### A. Position Sizing
*   The required quantity is dynamically computed using the zone risk allocation ($1.0\%$ or $0.5\%$):
    $$\text{Quantity} = \frac{\text{Balance} \times \text{Risk Percent}}{|\text{Entry Price} - \text{Stop Loss Price}|}$$

### B. Stop Loss (SL) Placement
*   **OB/FVG Entries**: Placed just beyond the boundaries of the triggering OB or FVG:
    $$\text{SL}_{\text{Long}} = \text{OB/FVG Low} - 0.05 \times \text{ATR}$$
    $$\text{SL}_{\text{Short}} = \text{OB/FVG High} + 0.05 \times \text{ATR}$$
*   **Breakout Entries**: Placed at a fixed distance of $1.2 \times \text{ATR}$ from the entry price.

### C. Take Profit (TP) Calibration
*   **Default Target**: Set at a fixed $2.0 \times \text{ATR}$ distance from entry.
*   **Liquidity Sweeps (Swing Targets)**: If the system detects a historical swing high (for Longs) or low (for Shorts) within a $1.2 \times \text{ATR}$ to $3.0 \times \text{ATR}$ range, the TP order is aligned directly with that key level to exploit liquidity targets.

### D. Immediate Reversal Rule
If an opposite signal is triggered while a position is open:
1.  The active position is closed with the reason: `"SMC Scalping Reversal opposite signal met"`.
2.  If the volatility zone is Green or Yellow, the opposing position is immediately entered.

---

## 5. Percentage-Based Slippage Robustness (S0, S1, S2)

During backtesting, the execution prices are adjusted to simulate execution costs matching the other systems:
*   **S0 (Dynamic ATR-based Slippage)**: Slippage = $0.1 \times \text{ATR} + 0.5 \times \text{Spread}$.
*   **S1 (Fixed 0.05% Slippage)**: Slippage = $0.05\% \times \text{Price} + 0.5 \times \text{Spread}$.
*   **S2 (Fixed 0.10% Slippage)**: Slippage = $0.10\% \times \text{Price} + 0.5 \times \text{Spread}$.

---

## 6. Backtest Simulation Flow

1.  **Initialization**: Loads target symbols, start/end dates, capital, and slippage settings from environment variables.
2.  **Historical Timeline Loop**:
    *   Processes OHLCV daily bars.
    *   Settles active positions using the High-Low ranges of each bar.
    *   Calculates the Volatility Ratio ($VR$) to determine the active Market Risk Zone.
    *   Evaluates entry triggers (incorporating the ATR Gate) if no positions are active.
3.  **Termination**: Force-closes open positions on the final bar, outputs backtest files (`backtest_results.json`, `trade_history.csv`, `daily_returns.csv`, `settings.json`, `prompt_template.txt`, `backtest_summary.txt`), and logs reports.
