# LLM Multi-Asset Trading & Risk Governance Framework

An event-driven backtesting and live execution framework built to evaluate Large Language Models (LLMs) as adaptive trading agents and risk governors, compared against a deterministic Smart Money Concepts (SMC) rule-based baseline and a hybrid Risk-Managed Deterministic Baseline (RMDB).

---

## 🚀 Key Features

*   **Three Execution Systems**:
    *   **AI Agent Mode ([bot.py](bot.py) & [backtest.py](backtest.py))**: Integrates state-of-the-art LLMs (Llama, Gemini) with a structured reasoning loop to dynamically analyze charts, adjust stop losses, and manage positions under strict risk governance.
    *   **Deterministic SMC Baseline ([scripts/run_baseline.py](scripts/run_baseline.py))**: A pure rule-based trading program that scans market structures to identify Order Blocks (OB), Fair Value Gaps (FVG), Breaks of Structure (BOS/CHoCH), and Liquidity Sweeps.
    *   **Risk-Managed Deterministic Baseline (RMDB) ([scripts/run_rmdb.py](scripts/run_rmdb.py))**: A hybrid control model combining the Baseline's deterministic SMC entry rules with the LLM's quantitative risk governance overlay (Volatility Zones, ATR-Gate).
*   **Event-Driven AI Wakeup**: Minimizes API invocation costs by ~80% by only querying the LLM when critical events occur: daily price fluctuations $\ge 0.8\%$, RSI entering extreme zones ($<35$ or $>65$), or during active position management.
*   **Volatility-Adaptive Risk Governance**: Categorizes market risk based on short-term/long-term Volatility Ratios ($VR$):
    *   **Green Zone ($VR < 1.6$)**: Stable market conditions; standard entry confluences applied. Risks **1.0% of Balance**.
    *   **Yellow Zone ($VR \in [1.6, 2.2]$)**: Volatile market; requires stricter confluences and halves position sizes (**0.5% risk**).
    *   **Red Zone ($VR > 2.2$)**: Extreme market stress; **suspends all new entry trades** to preserve capital.
*   **Slippage Robustness (S0, S1, S2)**:
    *   **S0**: Dynamic volatility-based slippage ($0.1 \times \text{ATR}$ + spread).
    *   **S1**: Fixed $0.05\%$ slippage per side (+ spread).
    *   **S2**: Fixed $0.10\%$ slippage per side (+ spread).
*   **High-Fidelity Backtesting Harness**: Simulates historical price actions with intrabar high-low resolution for stop-loss (SL) and take-profit (TP) execution, incorporating exchange transaction fees and slippage.
*   **Statistical Significance Suite**: Features a validation tool that performs Student's t-test, Mann-Whitney U, and bootstrap resampling to mathematically evaluate the performance spread between the AI agent and the baselines.
*   **Interactive HTML Replay Engine**: Generates dynamic web interfaces from backtest logs to visualize candlesticks, technical indicators, trade entry/exit markers, and time-series portfolio equity curves.

---

## 📁 Repository Structure

```text
LLM_trading_ScienceResearch/
├── dataset/                  # Historical daily OHLCV CSV files (e.g. AAPL, gold)
├── prompts/                  # System prompts and instructions for AI agents
│   └── system_prompt.txt     # Default "Active Guardian" risk overlay strategy
├── scripts/                  # Auxiliary operational scripts
│   ├── run_baseline.py       # Rule-Based Baseline execution engine
│   ├── run_rmdb.py           # Risk-Managed Deterministic Baseline (RMDB) engine
│   ├── run_stats_significance.py  # Statistical significance and bootstrap suite
│   └── recalculate_portfolio.py   # Portfolio balance recalculator from raw logs
├── data-backtest/            # Backtest results (official run logs and local outputs)
├── bot.py                    # Core live-trading execution engine & LLM connector
├── backtest.py               # Historical timeline simulation harness for LLM
├── dashboard.py              # Streamlit monitoring dashboard
├── LogicAI.md                # Technical document: AI decision framework details
├── logicbaseline.md          # Technical document: SMC rule-based logic details
├── logicrmdb.md              # Technical document: Hybrid RMDB logic details
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
Open `.env` and configure your API keys, backtest parameters, initial capital settings, and execution options:
*   `BACKTEST_SLIPPAGE_MODE`: Set to `S0`, `S1`, or `S2` to select the transaction slippage scenario.
*   `BACKTEST_RUN_ID`: Set to your run ID (e.g., `AAPL_Baseline_2008_2009_S0` or `AAPL_RMDB_2008_2009_S0`) to save results in distinct folders.

---

## 📈 Running Simulations

### 1. Run LLM Backtest
Simulate the AI Agent (Llama) on historical datasets using the configurations defined in your `.env`:
```bash
python backtest.py
```
Outputs (including decision transcripts and trade logs) are saved in `data-backtest/run-YOUR_RUN_ID/`.

### 2. Run Deterministic Baseline
Simulate the rule-based SMC Baseline on the same period:
```bash
python scripts/run_baseline.py
```

### 3. Run RMDB (Risk-Managed Deterministic Baseline)
Simulate the hybrid RMDB model combining rule-based entry with volatility-based risk adjustments:
```bash
python scripts/run_rmdb.py
```

### 4. Evaluate Statistical Significance
Compare the AI agent's performance directly against the baselines to run Welch's t-test, Mann-Whitney U, and bootstrap intervals:
```bash
python scripts/run_stats_significance.py --agent data-backtest/YOUR_AGENT_RUN_ID --baseline data-backtest/YOUR_BASELINE_RUN_ID
```
This generates a detailed comparison report: `data-backtest/statistical_significance_report.md`.

### 5. Launch Live Dashboard
Launch the dashboard to monitor active paper or live trades:
```bash
streamlit run dashboard.py
```

---

## 📘 Technical Documentation

*   For an in-depth look at prompt design, volatility boundaries, and risk governance logic of the LLM Bot: See [LogicAI.md](LogicAI.md).
*   For technical implementation details of the technical indicators and SMC logic of the Baseline Bot: See [logicbaseline.md](logicbaseline.md).
*   For details of the hybrid Volatility Zones and SMC criteria combination of the RMDB Bot: See [logicrmdb.md](logicrmdb.md).

---

## 🎓 Research Paper Context

This codebase serves as the replication package for the manuscript:
**"Can Large Language Models Trade under Market Stress? Evidence from AAPL and XAUUSD Backtesting across Crisis Regimes"**
*Prepared for submission to IEEE DSAA 2026 — Application, Data and Benchmark Track*

For peer reviewers, the raw execution logs, exact prompt logs (`ai_messages.csv`), trade history, and portfolio states evaluated in the paper are preserved under `data-backtest/`.
