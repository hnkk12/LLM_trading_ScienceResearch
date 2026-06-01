# Logic AI - Trading Bot (Detailed Technical Specification)

This document presents the detailed architectural design, event-driven activation mechanism, input/output data structures, and trade/risk management rules of the Large Language Model (LLM) trading bot.

---

## 1. Architecture Overview & Operating Mechanism

The bot is designed as a **Swing Trader integrated with an Active Guardian Risk Management Overlay**. The AI acts as the reasoning and decision-making engine, while the Python core system is responsible for data ingestion, technical indicator calculation, position sizing, order execution, and logging.

### Basic Workflow:
1. **Data Collection**: The bot ingests market data (from historical CSV files during backtesting, or the exchange API during live trading) at the **1D** interval.
2. **Event-Driven AI Activation**: The Python core checks if current market conditions meet the criteria to "wake up" the AI. If met, the core packages the prompt and calls the LLM API.
3. **Analysis & Response**: The LLM receives a structured prompt containing indicators, market structure data, and portfolio states. It performs chain-of-thought reasoning and returns a structured JSON decision.
4. **Order Execution**: The core processes the JSON response to enter a position (`entry`), close a position (`close`), or modify stop-loss/take-profit parameters (`hold`/`trail`).
5. **Logging & Reporting**: Automatically records logs into `ai_decisions.csv`, `trade_history.csv`, and sends real-time status alerts via Telegram.

---

## 2. Event-Driven AI Activation Mechanism

To optimize API latency and reduce token costs during consolidations (sideways markets), the bot employs an event-driven activation mechanism. The LLM is **only invoked (woken up)** when at least one of the following conditions is satisfied:

*   **Condition A (Active Position Management)**: If the portfolio has any open positions, the AI is called on every new bar to monitor the position, adjust trailing stops, or execute early closures.
*   **Condition B (Significant Price Volatility)**: The closing price of the current candle changes by $\ge 0.8\%$ compared to the previous candle's close for any tracked asset.
*   **Condition C (RSI Extremes / Momentum Reversal)**: The 14-period RSI enters extreme zones ($\text{RSI} < 35$ or $\text{RSI} > 65$), signaling potential trend exhaustion or reversal.
*   **Condition D (Boundary Bars)**: The first bar (to establish initial strategy) and the final bar (to liquidate all remaining holdings) of the simulation run always trigger the AI.

> [!NOTE]
> If none of these conditions are met, the core system skips the LLM API call for that candle and automatically maintains the active stop-loss (SL) and take-profit (TP) orders. This mechanism reduces API calls by approximately 80%.

---

## 3. Input Data Structure (LLM Context)

Upon activation, the core compiles the market metrics and account states into a detailed text prompt. The context elements include:

### A. Strategic Guidelines (System Prompt)
Defines the trading persona (e.g., Active Guardian, Sniper, or Aggressive), core position sizing rules (Strict 1% Risk), leverage parameters, volatility ratio definitions, and risk-zone containment protocols.

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
*   **Green Zone ($VR < 1.6$ - Stable)**: Low-to-moderate volatility. The AI is permitted to execute entry orders using standard **2-factor confluences**.
*   **Yellow Zone ($VR \in [1.6, 2.2]$ - Volatile)**: High volatility. The AI requires at least **3-factor confluences** to enter trades, and position size is automatically halved (capping trade risk to 0.5%).
*   **Red Zone ($VR > 2.2$ - Extreme)**: Severe volatility or drawdowns. The core **suspends all new trade entries** and instructs the AI to focus entirely on managing and trailing active positions.

---

## 5. Position & In-Trade Management

Once a position is active, the bot maintains dual-layered monitoring via the core script and the LLM:

### A. Intrabar TP/SL Settlement (Zero API Cost)
At each new price bar, before activating the AI, the core checks the High/Low range of the previous candle:
*   If the price touches or breaches the **Stop Loss**: The core immediately liquidates the position with the label `"Stop loss hit"`.
*   If the price touches or breaches the **Take Profit**: The core liquidates the position with the label `"Take profit hit"`.

### B. Trailing Stop Management
When the AI outputs a `hold` decision, it can propose an adjusted Stop Loss level:
*   **Safety Restriction**: The core only accepts the new Stop Loss if it reduces the overall risk of the position (i.e., increasing SL for Long positions, decreasing SL for Short positions). If the AI attempts to widen the stop loss, the core rejects the adjustment and retains the previous level.

### C. Fast Early Exit
If the AI detects a structural trend reversal or determines that a trade is failing to gain momentum after 3 candles, it can output a `close` signal to execute an early exit and limit capital impairment.

---

## 6. Output Data Structure (LLM Response Schema)

The LLM must respond with a single, structured JSON document containing the trade decision for the evaluated assets:

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
    "justification": "Detailed natural language reasoning analyzing technical indicators and market structure.",
    "confluence_tags": ["EMA Crossover", "RSI Support"],
    "trigger_tags": ["RSI Bounce"],
    "reasoning_categories": ["Trend Following"]
  }
}
```

### JSON Fields Explanation:
*   **`signal`**: The requested action.
    *   `entry`: Initiate a new position (valid only if no open position exists for the asset).
    *   `close`: Liquidate the current position.
    *   `hold`: Keep the position open (allows stop loss updates).
    *   `reject`: Take no action (or bypass potential opportunities due to risk).
*   **`side`**: Trade direction (`long` or `short`).
*   **`quantity`**: Recommended order size (processed by core formulas).
*   **`profit_target`**: The target Take Profit price.
*   **`stop_loss`**: The protection Stop Loss price.
*   **`leverage`**: Leverage multiplier (e.g., `5`, `10`).
*   **`risk_usd`**: Cash value at risk (matches the 1% parameter).
*   **`justification`**: Chain-of-thought analysis explaining the trade logic.
