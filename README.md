# LLM Multi-Asset Trading & Risk Governance Framework

An event-driven backtesting and live execution framework built to evaluate Large Language Models (LLMs) as adaptive trading agents and risk governors, compared against a deterministic Smart Money Concepts (SMC) rule-based baseline.

---

## 🚀 Key Features

*   **Dual Execution Systems**:
    *   **AI Agent Mode ([bot.py](bot.py))**: Integrates state-of-the-art LLMs (DeepSeek, Llama, Gemini) with a structured reasoning loop to dynamically analyze charts, adjust stop losses, and manage positions.
    *   **Deterministic SMC Baseline ([scripts/run_baseline.py](scripts/run_baseline.py))**: A rule-based trading program that scans market structures to identify Order Blocks (OB), Fair Value Gaps (FVG), Breaks of Structure (BOS/CHoCH), and Liquidity Sweeps.
*   **Event-Driven AI Wakeup**: Minimizes API invocation costs by ~80% by only querying the LLM when critical events occur: intraday price fluctuations $\ge 0.8\%$, RSI entering extreme zones ($<35$ or $>65$), or during active position management.
*   **Volatility-Adaptive Risk Governance**: Categorizes market risk based on short-term/long-term Volatility Ratios ($VR$):
    *   **Green Zone ($VR < 1.6$)**: Stable market conditions; standard entry confluences applied.
    *   **Yellow Zone ($VR \in [1.6, 2.2]$)**: Volatile market; requires stricter confluences and halves position sizes (0.5% risk).
    *   **Red Zone ($VR > 2.2$)**: Extreme market stress; suspends all new entry trades to preserve capital.
*   **High-Fidelity Backtesting Harness**: Simulates historical price actions with intrabar high-low resolution for stop-loss (SL) and take-profit (TP) execution, incorporating exchange transaction fees (maker/taker) and slippage.
*   **Statistical Significance Suite**: Features a validation tool that performs Student's t-test, Mann-Whitney U, and bootstrap resampling to mathematically evaluate the performance spread between the AI agent and the baseline.
*   **Interactive HTML Replay Engine**: Generates dynamic web interfaces from backtest logs to visualize candlesticks, technical indicators, trade entry/exit markers, and time-series portfolio equity curves.
*   **Live/Paper Dashboard**: Built-in Streamlit dashboard to monitor live portfolio balances and AI decisions in real-time.

---

## 📁 Repository Structure

```text
LLM_trading_ScienceResearch/
├── dataset/                  # Historical daily OHLCV CSV files (e.g. AAPL, gold)
├── prompts/                  # System prompts and instructions for AI agents
│   ├── system_prompt.txt     # Default "Active Guardian" risk overlay strategy
│   ├── system_prompt_aggressive.txt
│   └── system_prompt_sniper.txt
├── replay/                   # Web-based timeline replay visualizer
│   ├── build_replay_site.py  # Script to compile backtest CSVs into HTML
│   └── index.html            # Visual replay interface template
├── scripts/                  # Auxiliary operational scripts
│   ├── run_baseline.py       # Rule-Based Baseline execution engine
│   ├── run_stats_significance.py  # Statistical significance and bootstrap suite
│   └── recalculate_portfolio.py   # Portfolio balance recalculator from raw logs
├── data-backtest/            # Backtest results (official run logs and local outputs)
├── bot.py                    # Core live-trading execution engine & LLM connector
├── backtest.py               # Historical timeline simulation harness
├── dashboard.py              # Streamlit monitoring dashboard
├── LogicAI.md                # Technical document: AI decision framework details
├── logicbaseline.md          # Technical document: SMC rule-based logic details
├── requirements.txt          # Python library dependencies
└── .env.example              # Environment configuration template
```

---

## 🛠 Getting Started

### 1. Installation
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy the environment template to create your `.env` file:
```bash
cp .env.example .env
```
Open `.env` and configure your API keys (e.g., `OPENROUTER_API_KEY` for Llama/DeepSeek or `GEMINI_API_KEY` for Gemini), backtest parameters, and initial capital settings.

---

## 📈 Running Simulations

### 1. Run LLM Backtest
Simulate the AI Agent on historical datasets using the configurations defined in your `.env`:
```bash
python backtest.py
```
Outputs (including decision transcripts and trade logs) are saved in `data-backtest/run-YYYYMMDD-HHMMSS/`.

### 2. Run Deterministic Baseline
Simulate the rule-based SMC Baseline on the same period:
```bash
python scripts/run_baseline.py
```

### 3. Generate Interactive Trade Replay
To visually inspect price action, indicator crossovers, and trade executions, build the interactive HTML replay:
```bash
python replay/build_replay_site.py --data data-backtest/run-YOUR_RUN_ID
```
Open `replay/index.html` in any web browser to interact with the chart.

### 4. Evaluate Statistical Significance
Compare the AI agent's performance directly against the baseline to run Welch's t-test, Mann-Whitney U, and bootstrap intervals:
```bash
python scripts/run_stats_significance.py --agent data-backtest/run-AGENT_ID --baseline data-backtest/run-BASELINE_ID
```
This generates a detailed comparison report: `data-backtest/statistical_significance_report.md`.

### 5. Launch Live Dashboard
Launch the dashboard to monitor active paper or live trades:
```bash
streamlit run dashboard.py
```

---

## 📘 Technical Documentation

*   For an in-depth look at prompt design, volatility boundaries, and risk governance logic: See [LogicAI.md](LogicAI.md).
*   For technical implementation details of the technical indicators (EMA, MACD, RSI, ATR) and SMC logic (OB, FVG, BOS, CHoCH): See [logicbaseline.md](logicbaseline.md).

---

## 🎓 Research Paper Context

This codebase serves as the replication package for the manuscript:
> **"Can Large Language Models Trade under Market Stress? Evidence from AAPL and XAUUSD Backtesting across Crisis Regimes"**
> *Prepared for submission to IEEE DSAA 2026 — Application, Data and Benchmark Track*

For peer reviewers, the raw execution logs, exact prompt logs (`ai_messages.csv`), trade history, and portfolio states evaluated in the paper are preserved under `data-backtest/`.
