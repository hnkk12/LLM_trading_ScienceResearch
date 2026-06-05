# Logic AI - Trading Bot (Detailed Technical Specification)

This document presents the detailed architectural design, event-driven activation mechanism, input/output data structures, and trade/risk management rules of the Large Language Model (LLM) trading bot. There are two identical implementations that differ only by their data paths:
1. **Original Version (`backtest.py` & `bot.py`)**: Runs on the original daily market dataset (located in the `dataset/` directory) and outputs results to the `data-backtest/` directory.
2. **Robust/Stress-Tested Version (`backtest2.py` & `bot.py`)**: Runs on the perturbed daily market dataset (located in the `dataset_robust/` directory) and outputs results to the `data-backtest2/` directory.

Both versions utilize the same event-driven AI reasoning core, system prompts, indicators, and risk management logic.

---

## 1. Architecture Overview & Operating Mechanism

The bot is designed as a **Swing Trader integrated with an Active Guardian Risk Management Overlay**. The AI acts as the qualitative reasoning and decision-making engine, while the Python core system is responsible for data ingestion, technical indicator calculation, position sizing, order execution, and logging.

### Basic Workflow:
1. **Data Ingestion**: The bot ingests market data (from historical CSV files during backtesting, or the exchange API during live trading) at the **1D** (daily) interval.
2. **Event-Driven AI Activation**: The Python core checks if current market conditions meet the criteria to "wake up" the AI. If met, the core packages the prompt and calls the LLM API (OpenRouter).
3. **Analysis & Response**: The LLM receives a structured prompt containing indicators, market structure data, and portfolio states. It performs chain-of-thought reasoning and returns a structured JSON decision.
4. **Order Execution**: The core processes the JSON response to enter a position (`entry`), close a position (`close`), or modify parameters (`hold`).
5. **Logging & Reporting**: Automatically records logs into `ai_decisions.csv`, `trade_history.csv`, and sends real-time status alerts via Telegram.

---

## 2. Event-Driven AI Activation Mechanism

To optimize API latency and reduce token costs during consolidations (sideways markets), the bot employs an event-driven activation mechanism. The LLM is **only invoked (woken up)** when at least one of the following conditions is satisfied:

*   **Condition A (Active Position Management)**: If the portfolio has any open positions, the AI is called on every new bar to monitor the position, adjust trailing stops, or execute early closures.
*   **Condition B (Significant Price Volatility)**: The closing price of the current daily candle changes by $\ge 0.8\%$ compared to the previous candle's close for any tracked asset.
*   **Condition C (RSI Extremes / Momentum Reversal)**: The 14-period RSI enters extreme zones ($\text{RSI} < 35$ or $\text{RSI} > 65$), signaling potential trend exhaustion or reversal.
*   **Condition D (Boundary Bars)**: The first bar (to establish initial strategy) and the final bar (to liquidate all remaining holdings) of the simulation run always trigger the AI.

> [!NOTE]
> If none of these conditions are met, the core system skips the LLM API call for that candle and automatically maintains the active stop-loss (SL) and take-profit (TP) orders. This mechanism reduces API calls by approximately 80%.

---

## 3. Input Data Structure (LLM Context)

Upon activation, the core compiles the market metrics and account states into a detailed text prompt. The context elements include:

### A. Strategic Guidelines (System Prompt)
Defines the trading persona (e.g., Active Guardian), core position sizing rules (Strict 1% Risk), leverage parameters, volatility ratio definitions, and risk-zone containment protocols.

### B. Account & Portfolio State
*   **Available Cash**: The cash balance available to meet margin requirements.
*   **Total Equity**: Calculated as $\text{Available Cash} + \text{Initial Margin} + \text{Unrealized PnL}$.
*   **Open Positions**:
    *   Asset symbol, side (`long`/`short`), quantity, and leverage.
    *   Entry price, current stop loss (SL), and take profit (TP) levels.
    *   Allocated margin, fees paid, unrealized PnL ($), and current ROI (%).
    *   Trailing stop adjustment history and reasoning.

### C. Market Data & Technical Indicators
*   **OHLCV**: Open, High, Low, Close prices, and Volume of the current candle.
*   **EMAs (20, 50, 200)**: Used to determine the primary market trend and dynamic support/resistance zones.
*   **RSI (14)**: Identifies market momentum and overbought/oversold states.
*   **MACD & Signal Line**: Determines momentum crossovers and money flow direction.
*   **ATR (14)**: Measures market volatility to calibrate dynamic SL/TP margins.
*   **Volume Ratio**: The current volume divided by the 20-period average volume.

---

## 4. Money Management & Risk Control

The bot enforces strict capital preservation rules to prevent catastrophic drawdowns and account liquidation:

### A. Strict 1% Risk Rule
The maximum capital at risk (maximum loss if the stop-loss is hit) for any new trade is strictly capped at **1% of the Available Balance**.

### B. Dynamic Position Sizing Formula
The order quantity and required margin are computed dynamically by the Python core based on the stop-loss distance proposed by the AI:

1.  **Stop Distance**:
    $$\text{Stop Distance} = |\text{Entry Price} - \text{Stop Loss Price}|$$
2.  **Order Quantity**:
    $$\text{Quantity} = \frac{\text{Available Balance} \times 0.01}{\text{Stop Distance}}$$
3.  **Position Value**:
    $$\text{Position Value} = \text{Quantity} \times \text{Entry Price}$$
4.  **Margin Required**:
    $$\text{Margin Required} = \frac{\text{Position Value}}{\text{Leverage}}$$

> [!WARNING]
> If $\text{Margin Required} + \text{Execution Fees}$ exceeds the available account cash, the trade order is **rejected** by the execution engine to preserve solvency.

### C. Market Risk Zones
Using the **Volatility Ratio (VR = Short-term ATR / Long-term ATR)**, the system divides the market into three operational risk zones:
*   **Green Zone ($VR < 1.6$ - Active)**: Stable market conditions; standard entry confluences (2 factors) applied. Risks **1% of Balance**.
*   **Yellow Zone ($VR \in [1.6, 2.2]$ - Defensive)**: Volatile market; requires stricter confluences (3+ factors) and halves position sizes (capping trade risk to **0.5% of Balance**).
*   **Red Zone ($VR > 2.2$ - Pause)**: Extreme market stress; **suspends all new trade entries** and instructs the AI to focus entirely on managing and trailing active positions.

### D. Drawdown Protection
If the account net balance drops by $3\%$ within a rolling week, all entry gates are automatically tightened to "Red Zone" (Pause) mode, stopping any new trade entries.

### E. Anti-Chasing Logic
The AI is instructed to never enter a trade if the price has already moved more than **2 ATRs** away from the original breakout/signal trigger level.

---

## 5. Position & In-Trade Management

Once a position is active, the bot maintains dual-layered monitoring:

### A. Intrabar TP/SL Settlement (Zero API Cost)
At each new price bar, before activating the AI, the core checks the High/Low range of the previous candle:
*   If the price touches or breaches the **Stop Loss**: The core immediately liquidates the position with the label `"Stop loss hit"`.
*   If the price touches or breaches the **Take Profit**: The core liquidates the position with the label `"Take profit hit"`.

### B. Trailing Stop Management
When the AI outputs a `hold` decision, it can propose an adjusted Stop Loss level:
*   **Safety Restriction**: The core only accepts the new Stop Loss if it reduces the overall risk of the position (i.e., increasing SL for Long positions, decreasing SL for Short positions). If the AI attempts to widen the stop loss, the core rejects the adjustment and retains the previous level.
*   **The 20% Rule**: If the price comes within **20%** of the stop loss, the AI is forbidden from executing a manual close. It must let the stop-loss order do its job to prevent emotional panic exits.

### C. Fast Early Exit
If the AI detects a structural trend reversal or determines that a trade is failing to gain momentum after **3 bars**, it can output a `close` signal to execute an early exit and limit capital impairment.

---

## 6. Percentage-Based Slippage Robustness (S0, S1, S2)

During backtesting, execution prices are adjusted to account for slippage and spread based on the `BACKTEST_SLIPPAGE_MODE` environment variable:

*   **S0 (Dynamic ATR-based Slippage)**: Default mode. Slippage is derived dynamically from market volatility.
    *   $\text{Slippage} = 0.1 \times \text{ATR}$
    *   $\text{Spread} = \text{Current Price} \times \text{Spread Percentage}$ (default 0.02%)
    *   $\text{Entry Price}_{\text{Long}} = \text{Current Price} + \text{Slippage} + 0.5 \times \text{Spread}$
    *   $\text{Exit Price}_{\text{Long}} = \text{Current Price} - \text{Slippage} - 0.5 \times \text{Spread}$
*   **S1 (Fixed 0.05% Slippage)**: Simulates standard market slippage.
    *   $\text{Slippage} = \text{Current Price} \times 0.0005$
    *   $\text{Entry Price}_{\text{Long}} = \text{Current Price} + \text{Slippage} + 0.5 \times \text{Spread}$
    *   $\text{Exit Price}_{\text{Long}} = \text{Current Price} - \text{Slippage} - 0.5 \times \text{Spread}$
*   **S2 (Fixed 0.10% Slippage)**: Simulates high-slippage market stress.
    *   $\text{Slippage} = \text{Current Price} \times 0.0010$
    *   $\text{Entry Price}_{\text{Long}} = \text{Current Price} + \text{Slippage} + 0.5 \times \text{Spread}$
    *   $\text{Exit Price}_{\text{Long}} = \text{Current Price} - \text{Slippage} - 0.5 \times \text{Spread}$

---

## 7. Output Data Structure (LLM Response Schema)

The LLM must respond with a single, structured JSON document containing the trade decision:

```json
{
  "AAPL": {
    "signal": "entry|hold|close|reject",
    "side": "long|short",
    "quantity": 0.0,
    "profit_target": 0.0,
    "stop_loss": 0.0,
    "leverage": 10,
    "confidence": 0.85,
    "risk_usd": 10.0,
    "market_mode": "Active|Defensive|Pause",
    "justification": "Detailed natural language reasoning analyzing technical indicators and market structure."
  }
}
```

### JSON Fields Explanation:
*   **`signal`**: The requested action.
    *   `entry`: Initiate a new position (valid only if no open position exists).
    *   `close`: Liquidate the current position.
    *   `hold`: Keep the position open (allows stop loss updates).
    *   `reject`: Take no action (bypass opportunity due to high risk).
*   **`side`**: Trade direction (`long` or `short`).
*   **`quantity`**: Recommended order size (processed and verified by core formulas).
*   **`profit_target`**: The target Take Profit price.
*   **`stop_loss`**: The protection Stop Loss price.
*   **`leverage`**: Leverage multiplier (e.g., `5`, `10`).
*   **`risk_usd`**: Dollar value at risk (must align with the 1% parameter).
*   **`justification`**: Chain-of-thought analysis explaining the trade logic.

---

## 8. Output Result Schema

All AI backtests generate matching results standard to the paper framework, saved under their respective run directories:

### A. Version 1: Original Dataset Outputs (`backtest.py`)
*   **Run Directories**: `data-backtest/AAPL_BACKTEST_{period}_{scenario}/` or `data-backtest/GOLD_BACKTEST_{period}_{scenario}/`
    *   `backtest_results.json`: Complete dictionary of metadata, daily equity series, daily returns series, capital ratios, and trade performance (featuring non-zero VaR/CVaR).
    *   `daily_returns.csv`: Daily return path.
    *   `trade_history.csv`: List of entry, exit, holding days, close types, and net PnL.
    *   `ai_messages.csv`: Saved transcripts of all LLM inputs and JSON outputs.
    *   `backtest_summary.txt`: ASCII format metrics summary table.

### B. Version 2: Robust Dataset Outputs (`backtest2.py`)
*   **Run Directories**: `data-backtest2/AAPL_BACKTEST_{period}_{scenario}/` or `data-backtest2/GOLD_BACKTEST_{period}_{scenario}/`
    *   *Note: Files inside are structured identically to Version 1, representing Llama's performance and decisions on perturbed stress-testing datasets.*
