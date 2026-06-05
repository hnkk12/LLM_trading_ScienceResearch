"""
XGBoost Baseline for Robust Dataset Testing
Runs 18 full backtests: 2 assets (AAPL, Gold) x 3 periods x 3 scenarios
Using local robust daily datasets (dataset_robust) and identical fee/slippage models.
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import ta
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS = {
    "AAPL": PROJECT_ROOT / "dataset_robust" / "AAPL_D_2004_2023_ROBUST.csv",
    "Gold": PROJECT_ROOT / "dataset_robust" / "GOLD_D_2004_2023_ROBUST.csv",
}

STRESS_WINDOWS = {
    "2008-2009": {
        "train_start": "2005-01-01",
        "train_end":   "2007-12-31",
        "test_start":  "2008-01-01",
        "test_end":    "2009-12-31",
    },
    "2020-2021": {
        "train_start": "2017-01-01",
        "train_end":   "2019-12-31",
        "test_start":  "2020-01-01",
        "test_end":    "2021-12-31",
    },
    "2022-2023": {
        "train_start": "2019-01-01",
        "train_end":   "2021-12-31",
        "test_start":  "2022-01-01",
        "test_end":    "2023-12-31",
    },
}

SCENARIOS = {
    "S0": {"type": "atr",   "value": 0.1},    # 0.1 * ATR slippage
    "S1": {"type": "fixed", "value": 0.0005}, # 0.05% slippage
    "S2": {"type": "fixed", "value": 0.001},  # 0.10% slippage
}

RISK_CONFIG = {
    "risk_per_trade":      0.01,   # 1% risk per trade
    "stop_loss_atr_mult":  2.0,    # SL = 2 * ATR
    "volatility_gate_mult": 2.0,   # skip if ATR > 2x rolling mean ATR
    "initial_capital":     1000.0, # 1,000 USD to match Baseline/RMDB/Llama/XGBoost
    "fee_rate_taker":      0.0005, # 0.05% taker fee
}

OUTPUT_RESULTS_DIR = PROJECT_ROOT / "results2" / "xgboost"
os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)

DATA_BACKTEST_DIR = PROJECT_ROOT / "data-backtest2"
os.makedirs(DATA_BACKTEST_DIR, exist_ok=True)

# ─── DATA LOADING & PARSING ───────────────────────────────────────────────────

def parse_dataset_volume(vol_str: str) -> float:
    """Parse volume strings like '1.08B', '842.07M', etc."""
    if not isinstance(vol_str, str):
        return 0.0
    vol_str = vol_str.strip().upper()
    if vol_str == "-" or not vol_str:
        return 0.0
    
    multiplier = 1.0
    if vol_str.endswith("B"):
        multiplier = 1e9
        vol_str = vol_str[:-1]
    elif vol_str.endswith("M"):
        multiplier = 1e6
        vol_str = vol_str[:-1]
    elif vol_str.endswith("K"):
        multiplier = 1e3
        vol_str = vol_str[:-1]
    
    try:
        return float(vol_str.replace(",", "")) * multiplier
    except ValueError:
        return 0.0

def to_clean_float(series: pd.Series) -> pd.Series:
    """Strip commas and convert series to numeric floats."""
    if series.dtype == object:
        return pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce")
    return pd.to_numeric(series, errors="coerce")

# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features identical to LLM prompt inputs."""
    df = df.copy()
    
    # Sort chronological
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Standardize column headers
    df = df.rename(columns={
        "Price": "Close",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Vol.": "Volume"
    })

    # Clean data types
    df["Close"] = to_clean_float(df["Close"])
    df["Open"] = to_clean_float(df["Open"])
    df["High"] = to_clean_float(df["High"])
    df["Low"] = to_clean_float(df["Low"])
    df["Volume"] = df["Volume"].apply(parse_dataset_volume)

    # Price returns
    df["close_pct_1d"]  = df["Close"].pct_change(1)
    df["close_pct_5d"]  = df["Close"].pct_change(5)
    df["close_pct_20d"] = df["Close"].pct_change(20)

    # Technical indicators
    df["rsi_14"]      = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    macd_ind          = ta.trend.MACD(df["Close"])
    df["macd"]        = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_hist"]   = macd_ind.macd_diff()
    df["ema_20"]      = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
    df["ema_50"]      = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()
    df["atr_14"]      = ta.volatility.AverageTrueRange(
                            df["High"], df["Low"], df["Close"], window=14
                        ).average_true_range()
    
    df["volume_sma"] = df["Volume"].rolling(window=20).mean().fillna(1.0)
    df["volume_ratio"] = (df["Volume"] / df["volume_sma"].replace(0, np.nan)).fillna(1.0)

    # Volatility gate feature
    df["atr_rolling_mean"] = df["atr_14"].rolling(20).mean()
    df["vol_gate_flag"]    = (df["atr_14"] > RISK_CONFIG["volatility_gate_mult"]
                               * df["atr_rolling_mean"]).astype(int)

    # Forward return label (next-day close vs today close direction)
    df["forward_return"]      = df["Close"].shift(-1) / df["Close"] - 1
    df["forward_return_sign"] = (df["forward_return"] > 0).astype(int)

    df.dropna(subset=FEATURE_COLS + ["forward_return_sign"], inplace=True)
    return df.reset_index(drop=True)

FEATURE_COLS = [
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "ema_20", "ema_50", "atr_14", "volume_ratio",
    "close_pct_1d", "close_pct_5d", "close_pct_20d",
    "vol_gate_flag",
]

# ─── RISK-MANAGED BACKTEST ────────────────────────────────────────────────────

def run_backtest(
    df_test: pd.DataFrame,
    probas: np.ndarray,
    scenario_cfg: dict,
    asset_name: str,
) -> dict:
    """
    Simulates a daily backtest using the same execution, slippage, and fee models.
    """
    capital = RISK_CONFIG["initial_capital"]
    position_shares = 0
    entry_price = 0.0
    stop_loss = 0.0
    entry_fee = 0.0
    entry_date = None
    
    trades = []
    equity_history = []
    daily_returns = []
    
    dates = df_test["Date"].tolist()
    closes = df_test["Close"].tolist()
    highs = df_test["High"].tolist()
    lows = df_test["Low"].tolist()
    atrs = df_test["atr_14"].tolist()
    vol_flags = df_test["vol_gate_flag"].tolist()
    
    current_equity = capital
    prev_equity = capital
    
    for i in range(len(df_test)):
        date = dates[i]
        price = closes[i]
        high = highs[i]
        low = lows[i]
        atr = atrs[i]
        vol_gate = vol_flags[i] == 1
        
        prob = probas[i]
        sig = 1 if prob > 0.55 else 0
        
        # ── EXIT LOGIC ──
        if position_shares > 0:
            hit_stop = low <= stop_loss
            exit_signal = sig == 0
            
            if hit_stop:
                # Sell at Stop Loss price
                spread = stop_loss * 0.0002
                if scenario_cfg["type"] == "atr":
                    slip = scenario_cfg["value"] * atr
                else:
                    slip = stop_loss * scenario_cfg["value"]
                
                exec_price = stop_loss - slip - 0.5 * spread
                pnl = position_shares * (exec_price - entry_price)
                fee = position_shares * exec_price * RISK_CONFIG["fee_rate_taker"]
                
                capital += position_shares * exec_price - fee
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry": entry_price,
                    "exit": exec_price,
                    "pnl": pnl - fee - entry_fee,
                    "hold_days": (date - entry_date).days,
                    "type": "Stop Loss"
                })
                position_shares = 0
                
            elif exit_signal:
                # Sell at Close price
                spread = price * 0.0002
                if scenario_cfg["type"] == "atr":
                    slip = scenario_cfg["value"] * atr
                else:
                    slip = price * scenario_cfg["value"]
                
                exec_price = price - slip - 0.5 * spread
                pnl = position_shares * (exec_price - entry_price)
                fee = position_shares * exec_price * RISK_CONFIG["fee_rate_taker"]
                
                capital += position_shares * exec_price - fee
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry": entry_price,
                    "exit": exec_price,
                    "pnl": pnl - fee - entry_fee,
                    "hold_days": (date - entry_date).days,
                    "type": "Signal Exit"
                })
                position_shares = 0

        # ── ENTRY LOGIC ──
        if position_shares == 0 and sig == 1 and not vol_gate:
            # 1% Account Risk Sizing
            risk_usd = current_equity * RISK_CONFIG["risk_per_trade"]
            sl_distance = RISK_CONFIG["stop_loss_atr_mult"] * atr
            
            if sl_distance > 0:
                shares = risk_usd / sl_distance
                
                # Check maximum capital constraint (95% of equity)
                max_pos_val = current_equity * 0.95
                if shares * price > max_pos_val:
                    shares = max_pos_val / price
                
                shares = round(shares, 4)
                if shares > 0:
                    # Execute entry at Close price
                    spread = price * 0.0002
                    if scenario_cfg["type"] == "atr":
                        slip = scenario_cfg["value"] * atr
                    else:
                        slip = price * scenario_cfg["value"]
                    
                    exec_price = price + slip + 0.5 * spread
                    entry_fee = shares * exec_price * RISK_CONFIG["fee_rate_taker"]
                    
                    capital -= (shares * exec_price + entry_fee)
                    position_shares = shares
                    entry_price = exec_price
                    stop_loss = exec_price - sl_distance
                    entry_date = date

        # ── EQUITY SNAPSHOT ──
        if position_shares > 0:
            current_equity = capital + (position_shares * price)
        else:
            current_equity = capital
            
        equity_history.append(current_equity)
        daily_return = (current_equity - prev_equity) / prev_equity
        daily_returns.append(daily_return)
        prev_equity = current_equity

    # Close any remaining position at the end
    if position_shares > 0:
        last_price = closes[-1]
        last_date = dates[-1]
        spread = last_price * 0.0002
        if scenario_cfg["type"] == "atr":
            slip = scenario_cfg["value"] * atrs[-1]
        else:
            slip = last_price * scenario_cfg["value"]
        
        exec_price = last_price - slip - 0.5 * spread
        pnl = position_shares * (exec_price - entry_price)
        fee = position_shares * exec_price * RISK_CONFIG["fee_rate_taker"]
        
        capital += position_shares * exec_price - fee
        trades.append({
            "entry_date": entry_date,
            "exit_date": last_date,
            "entry": entry_price,
            "exit": exec_price,
            "pnl": pnl - fee - entry_fee,
            "hold_days": (last_date - entry_date).days,
            "type": "Force Close"
        })
        current_equity = capital
        equity_history[-1] = current_equity
        daily_returns[-1] = (current_equity - equity_history[-2]) / equity_history[-2] if len(equity_history) > 1 else 0.0

    return {
        "equity": equity_history,
        "daily_returns": daily_returns,
        "trades": trades,
        "final_equity": current_equity
    }

# ─── METRIC COMPUTATIONS ──────────────────────────────────────────────────────

def calculate_metrics(result: dict) -> dict:
    returns = np.array(result["daily_returns"])
    equity = np.array(result["equity"])
    trades = result["trades"]
    
    total_ret_pct = (result["final_equity"] / RISK_CONFIG["initial_capital"] - 1) * 100
    mean_ret_pct = np.mean(returns) * 100
    
    # MDD
    eq_series = pd.Series(equity)
    roll_max = eq_series.cummax()
    drawdown = (eq_series - roll_max) / roll_max
    mdd = abs(drawdown.min()) * 100
    
    # Sharpe
    std_ret = np.std(returns, ddof=1)
    sharpe = (np.mean(returns) / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
    
    # Sortino
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else std_ret
    sortino = (np.mean(returns) / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0
    
    # Trade diagnostics
    n_trades = len(trades)
    winning_trades = sum(1 for t in trades if t["pnl"] > 0)
    win_rate = (winning_trades / n_trades * 100) if n_trades > 0 else 0.0
    avg_hold = np.mean([t["hold_days"] for t in trades]) if trades else 0.0
    
    return {
        "mean_return": round(mean_ret_pct, 4),
        "total_return": round(total_ret_pct, 4),
        "mdd": round(mdd, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "n_trades": n_trades,
        "win_rate": round(win_rate, 2),
        "avg_hold_days": round(avg_hold, 2),
    }

# ─── MAIN EXECUTION ───────────────────────────────────────────────────────────

def run_all():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Starting XGBoost Robust Baseline Backtest Engine...")

    aggregate_rows = []
    trade_diag_rows = []
    
    all_feature_importances = []
    all_X_test = []
    all_shap_values = []

    for asset_name, data_path in ASSETS.items():
        if not data_path.exists():
            logging.error(f"Dataset not found for {asset_name} at {data_path}. Skipping.")
            continue
            
        logging.info(f"Processing asset: {asset_name} from {data_path}")
        raw_df = pd.read_csv(data_path)
        df_full = compute_features(raw_df)

        for period_name, window in STRESS_WINDOWS.items():
            logging.info(f"Running period: {period_name}")

            # Walk-forward splits
            train_mask = (df_full["Date"] >= window["train_start"]) & (df_full["Date"] <= window["train_end"])
            test_mask = (df_full["Date"] >= window["test_start"]) & (df_full["Date"] <= window["test_end"])
            
            df_train = df_full.loc[train_mask]
            df_test = df_full.loc[test_mask]

            if len(df_train) < 100 or len(df_test) < 50:
                logging.warning(f"Insufficient data for {asset_name} {period_name}. Train len: {len(df_train)}, Test len: {len(df_test)}. Skipping.")
                continue

            X_train = df_train[FEATURE_COLS]
            y_train = df_train["forward_return_sign"]
            X_test = df_test[FEATURE_COLS]

            # Model fit
            n_pos = y_train.sum()
            n_neg = len(y_train) - n_pos
            scale_weight = n_neg / n_pos if n_pos > 0 else 1.0

            model = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_weight,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
            model.fit(X_train, y_train)

            # Predictions
            probas = model.predict_proba(X_test)[:, 1]

            # Collect feature importances
            all_feature_importances.append(model.feature_importances_)

            # Compute and collect SHAP values
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_test)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
            elif len(shap_vals.shape) == 3:
                shap_vals = shap_vals[:, :, 1]
            all_shap_values.append(shap_vals)
            all_X_test.append(X_test)

            for scen_name, scen_cfg in SCENARIOS.items():
                logging.info(f"  Scenario: {scen_name}")
                
                # Execute Backtest
                res = run_backtest(df_test, probas, scen_cfg, asset_name)
                m = calculate_metrics(res)
                
                # Calculate VaR and CVaR (95% Daily)
                daily_ret_array = np.array(res["daily_returns"])
                if len(daily_ret_array) > 0:
                    var_95_raw = np.percentile(daily_ret_array, 5)
                    var_95_val = -var_95_raw if var_95_raw < 0 else 0.0
                    losses_beyond = daily_ret_array[daily_ret_array <= var_95_raw]
                    cvar_95_val = -losses_beyond.mean() if len(losses_beyond) > 0 and losses_beyond.mean() < 0 else 0.0
                else:
                    var_95_val = 0.0
                    cvar_95_val = 0.0
                
                logging.info(f"    Total Return={m['total_return']:+.2f}% | MDD={m['mdd']:.2f}% | Sharpe={m['sharpe']:.2f} | Trades={m['n_trades']}")

                # ─── SAVE TO STANDARDIZED DATA-BACKTEST DIR ───
                run_id = f"{asset_name.upper()}_XGBOOST_{period_name.replace('-', '_')}_{scen_name}"
                run_dir = DATA_BACKTEST_DIR / run_id
                os.makedirs(run_dir, exist_ok=True)
                
                # Build json results
                results_json = {
                    "run_id": run_id,
                    "run_directory": str(run_dir),
                    "cache_directory": str(DATA_BACKTEST_DIR / "cache"),
                    "timeframe": {
                        "start": pd.to_datetime(df_test["Date"].iloc[0]).isoformat(),
                        "end": pd.to_datetime(df_test["Date"].iloc[-1]).isoformat(),
                        "interval": "1d",
                        "bars": len(df_test)
                    },
                    "symbols": [asset_name],
                    "capital": {
                        "start": RISK_CONFIG["initial_capital"],
                        "final_balance": res["final_equity"],
                        "final_equity": res["final_equity"],
                        "total_net_profit": res["final_equity"] - RISK_CONFIG["initial_capital"],
                        "total_return_pct": m["total_return"],
                        "max_drawdown_pct": m["mdd"],
                        "recovery_factor": round((res["final_equity"] - RISK_CONFIG["initial_capital"]) / (RISK_CONFIG["initial_capital"] * (m["mdd"]/100)) if m["mdd"] > 0 else 0, 4),
                        "profit_factor": round(sum(t["pnl"] for t in res["trades"] if t["pnl"] > 0) / abs(sum(t["pnl"] for t in res["trades"] if t["pnl"] < 0)) if sum(t["pnl"] for t in res["trades"] if t["pnl"] < 0) < 0 else 0, 4),
                        "win_rate_pct": m["win_rate"],
                        "sharpe_ratio": m["sharpe"],
                        "sortino_ratio": m["sortino"],
                        "var_95_pct": var_95_val * 100,
                        "cvar_95_pct": cvar_95_val * 100,
                        "gross_profit": round(sum(t["pnl"] for t in res["trades"] if t["pnl"] > 0), 4),
                        "gross_loss": round(abs(sum(t["pnl"] for t in res["trades"] if t["pnl"] < 0)), 4)
                    },
                    "daily_returns": res["daily_returns"],
                    "equity_history": res["equity"],
                    "llm": {
                        "model": "XGBoost Baseline",
                        "temperature": 0.0,
                        "max_tokens": None,
                        "thinking": None,
                        "system_prompt": {
                            "source": "xgboost_classifier",
                            "file": None,
                            "override": False,
                            "preview": "XGBoost Classifier + 1% Risk Gating",
                            "full": "XGBoost classifier trained on walk-forward historical features. Order execution guided by volatility gating and stop loss gates."
                        }
                    },
                    "trading": {
                        "total_trades": m["n_trades"],
                        "closed_trades": m["n_trades"],
                        "partial_closes": 0,
                        "close_events": m["n_trades"],
                        "winning_trades": sum(1 for t in res["trades"] if t["pnl"] > 0),
                        "losing_trades": sum(1 for t in res["trades"] if t["pnl"] < 0),
                        "win_rate_pct": m["win_rate"],
                        "net_realized_pnl": res["final_equity"] - RISK_CONFIG["initial_capital"],
                        "gross_win": round(sum(t["pnl"] for t in res["trades"] if t["pnl"] > 0), 4),
                        "gross_loss": round(abs(sum(t["pnl"] for t in res["trades"] if t["pnl"] < 0)), 4),
                        "profit_factor": round(sum(t["pnl"] for t in res["trades"] if t["pnl"] > 0) / abs(sum(t["pnl"] for t in res["trades"] if t["pnl"] < 0)) if sum(t["pnl"] for t in res["trades"] if t["pnl"] < 0) < 0 else 0, 4),
                        "avg_holding_time_seconds": round(m["avg_hold_days"] * 86400, 2)
                    },
                    "generated_at": pd.Timestamp.now().isoformat()
                }

                with open(run_dir / "backtest_results.json", "w", encoding="utf-8") as f:
                    json.dump(results_json, f, indent=2)

                # Save daily returns to data-backtest folder
                daily_ret_df = pd.DataFrame({"daily_return": res["daily_returns"]}, index=df_test["Date"])
                daily_ret_df.to_csv(run_dir / "daily_returns.csv")

                # Save trade history to data-backtest folder
                if res["trades"]:
                    trade_history_df = pd.DataFrame(res["trades"])
                    trade_history_df.to_csv(run_dir / "trade_history.csv", index=False)

                # Generate summary text table and save to backtest_summary.txt
                try:
                    total_net_profit = res["final_equity"] - RISK_CONFIG["initial_capital"]
                    win_pct_total = m["win_rate"]
                    
                    gross_win = sum(t["pnl"] for t in res["trades"] if t["pnl"] > 0)
                    gross_loss = abs(sum(t["pnl"] for t in res["trades"] if t["pnl"] < 0))
                    pf_val = gross_win / gross_loss if gross_loss > 0 else 0.0
                    
                    rf_val = (res["final_equity"] - RISK_CONFIG["initial_capital"]) / (RISK_CONFIG["initial_capital"] * (m["mdd"]/100)) if m["mdd"] > 0 else 0.0
                    avg_holding = f"{m['avg_hold_days']:.1f}d"
                    
                    crisis_period = f"{period_name} ({scen_name})"
                    
                    rows = [
                        ("Model", "XGBoost Baseline"),
                        ("Asset", asset_name),
                        ("Crisis Period", crisis_period),
                        ("Initial Capital", f"${RISK_CONFIG['initial_capital']:,.2f}"),
                        ("Final Capital", f"${res['final_equity']:,.2f}"),
                        ("Net Profit", f"{'+' if total_net_profit >= 0 else '-'}${abs(total_net_profit):,.2f}"),
                        ("Return %", f"{m['total_return']:+.2f}%"),
                        ("Total Trades", str(m["n_trades"])),
                        ("Win Rate", f"{win_pct_total:.1f}%"),
                        ("Profit Factor", f"{pf_val:.2f}"),
                        ("Sharpe Ratio", f"{m['sharpe']:.2f}"),
                        ("Sortino Ratio", f"{m['sortino']:.2f}"),
                        ("Maximum Drawdown", f"{m['mdd']:.2f}%"),
                        ("Recovery Factor", f"{rf_val:.2f}"),
                        ("VaR/CVaR (95%)", f"{var_95_val*100:.2f}% / {cvar_95_val*100:.2f}%"),
                        ("Avg Holding Time", avg_holding),
                    ]
                    
                    col1_w = max(len(r[0]) for r in rows) + 2
                    col2_w = max(len(str(r[1])) for r in rows) + 2
                    
                    border = f"+{'-' * col1_w}+{'-' * col2_w}+"
                    header = f"| {'Metric':<{col1_w-2}} | {'Value':<{col2_w-2}} |"
                    
                    table_lines = [border, header, border]
                    for met, val in rows:
                        table_lines.append(f"| {met:<{col1_w-2}} | {str(val):<{col2_w-2}} |")
                    table_lines.append(border)
                    table_str = "\n".join(table_lines)
                    
                    with open(run_dir / "backtest_summary.txt", "w", encoding="utf-8") as sf:
                        sf.write(table_str)
                except Exception as exc:
                    logging.warning(f"Failed to generate backtest_summary.txt: {exc}")

                # ─── SAVE TO RESULTS/XGBOOST DIR ───
                fname_daily = f"daily_returns_{asset_name}_{period_name.replace('-', '_')}_{scen_name}.csv"
                daily_out_path = OUTPUT_RESULTS_DIR / fname_daily
                
                daily_paper_df = pd.DataFrame({
                    "date": df_test["Date"],
                    "daily_return": res["daily_returns"],
                    "equity": res["equity"]
                })
                daily_paper_df.to_csv(daily_out_path, index=False)

                # Collect row for performance aggregate
                aggregate_rows.append({
                    "system": "XGBoost",
                    "asset": asset_name,
                    "period": period_name,
                    "scenario": scen_name,
                    "mean_return": m["mean_return"],
                    "mdd": m["mdd"],
                    "sharpe": m["sharpe"],
                    "sortino": m["sortino"],
                })

                # Collect row for trade diagnostics
                trade_diag_rows.append({
                    "system": "XGBoost",
                    "asset": asset_name,
                    "period": period_name,
                    "scenario": scen_name,
                    "trades_per_run": m["n_trades"],
                    "win_rate": m["win_rate"],
                    "avg_hold_days": m["avg_hold_days"],
                })

    # Save aggregated performance CSV
    agg_df = pd.DataFrame(aggregate_rows)
    agg_df.to_csv(OUTPUT_RESULTS_DIR / "aggregate_performance.csv", index=False)
    logging.info(f"Saved aggregate performance -> {OUTPUT_RESULTS_DIR}/aggregate_performance.csv")

    # Save trade diagnostics CSV
    diag_df = pd.DataFrame(trade_diag_rows)
    diag_summary = diag_df.groupby(["system", "scenario"]).mean(numeric_only=True).round(2).reset_index()
    diag_summary.to_csv(OUTPUT_RESULTS_DIR / "trade_diagnostics.csv", index=False)
    logging.info(f"Saved trade diagnostics -> {OUTPUT_RESULTS_DIR}/trade_diagnostics.csv")

    # ─── FEATURE IMPORTANCE & SHAP AGGREGATION ───
    if all_feature_importances:
        # 1. Feature Importance aggregation
        importances_df = pd.DataFrame(all_feature_importances, columns=FEATURE_COLS)
        mean_importances = importances_df.mean()
        
        # Map to groups
        feature_to_group = {
            "rsi_14": "RSI",
            "macd": "MACD",
            "macd_signal": "MACD",
            "macd_hist": "MACD",
            "ema_20": "EMA",
            "ema_50": "EMA",
            "atr_14": "ATR",
            "volume_ratio": "volume ratio",
            "close_pct_1d": "returns",
            "close_pct_5d": "returns",
            "close_pct_20d": "returns",
            "vol_gate_flag": "volatility gate"
        }
        
        fi_rows = []
        for feat in FEATURE_COLS:
            fi_rows.append({
                "Feature": feat,
                "Group": feature_to_group[feat],
                "Importance": mean_importances[feat]
            })
        fi_df = pd.DataFrame(fi_rows)
        fi_df = fi_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)
        fi_df.to_csv(OUTPUT_RESULTS_DIR / "xgboost_feature_importance.csv", index=False)
        logging.info(f"Saved feature importance -> {OUTPUT_RESULTS_DIR}/xgboost_feature_importance.csv")
        
        # 2. SHAP aggregation
        X_test_all = pd.concat(all_X_test, axis=0).reset_index(drop=True)
        shap_vals_all = np.concatenate(all_shap_values, axis=0)
        
        mean_abs_shap = np.mean(np.abs(shap_vals_all), axis=0)
        shap_rows = []
        for i, feat in enumerate(FEATURE_COLS):
            shap_rows.append({
                "Feature": feat,
                "Group": feature_to_group[feat],
                "Mean_Abs_SHAP": mean_abs_shap[i]
            })
        shap_df = pd.DataFrame(shap_rows)
        shap_df = shap_df.sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
        shap_df.to_csv(OUTPUT_RESULTS_DIR / "shap_summary.csv", index=False)
        logging.info(f"Saved SHAP summary -> {OUTPUT_RESULTS_DIR}/shap_summary.csv")
        
        # 3. Plot Feature Importance and SHAP
        try:
            # Individual Feature Importance Plot
            plt.figure(figsize=(10, 6))
            sorted_fi = fi_df.sort_values(by="Importance", ascending=True)
            plt.barh(sorted_fi["Feature"], sorted_fi["Importance"], color="skyblue")
            plt.title("XGBoost Average Feature Importance (Gain)", fontsize=14, fontweight="bold")
            plt.xlabel("Relative Importance (Gain)")
            plt.tight_layout()
            plt.savefig(OUTPUT_RESULTS_DIR / "xgboost_feature_importance.png", dpi=300)
            plt.close()
            
            # Group Importance Plot
            group_fi = fi_df.groupby("Group")["Importance"].sum().reset_index()
            group_fi = group_fi.sort_values(by="Importance", ascending=True)
            plt.figure(figsize=(10, 6))
            plt.barh(group_fi["Group"], group_fi["Importance"], color="coral")
            plt.title("XGBoost Average Feature Importance by Group", fontsize=14, fontweight="bold")
            plt.xlabel("Total Group Importance")
            plt.tight_layout()
            plt.savefig(OUTPUT_RESULTS_DIR / "xgboost_group_importance.png", dpi=300)
            plt.close()
            
            # SHAP Summary Beeswarm Plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_vals_all, X_test_all, show=False)
            plt.title("SHAP Summary Beeswarm Plot (All Assets & Periods)", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(OUTPUT_RESULTS_DIR / "shap_summary.png", dpi=300)
            plt.close()
            logging.info("Saved feature importance and SHAP plots.")
        except Exception as e:
            logging.warning(f"Failed to generate plots: {e}")

    # ─── MDD ADVANTAGE COUNTS GENERATION ───
    # Parse existing runs from data-backtest directory to generate comparison counts
    generate_mdd_advantage_counts(agg_df)


def generate_mdd_advantage_counts(xgb_agg_df: pd.DataFrame):
    """
    Scans data-backtest for existing Baseline, RMDB, and LLM runs, 
    then computes returns and MDD wins comparison tables.
    """
    logging.info("Scanning completed backtests in data-backtest/ for table comparisons...")
    
    parsed_runs = []
    
    # Scan all directories in data-backtest
    for d in DATA_BACKTEST_DIR.iterdir():
        if not d.is_dir() or d.name == "cache":
            continue
        res_file = d / "backtest_results.json"
        if not res_file.exists():
            continue
            
        try:
            with open(res_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            model_name = data.get("llm", {}).get("model", "")
            
            # Map model string to paper system name
            if "xgboost" in model_name.lower():
                # We already have XGBoost from current run
                continue
            elif "random forest" in model_name.lower() or "rf" in model_name.lower():
                system = "RF"
            elif "risk-managed" in model_name.lower():
                system = "RMDB"
            elif "baseline" in model_name.lower():
                system = "Baseline"
            elif "llama" in model_name.lower() or "llm" in model_name.lower():
                system = "LLM"
            else:
                continue
                
            # Parse scenario and period from run_id or timeframe dates
            run_id = data.get("run_id", "")
            
            # Extract scenario
            scen = "S0"
            for s_opt in ["S0", "S1", "S2"]:
                if s_opt in run_id:
                    scen = s_opt
                    break
                    
            # Extract period based on start year
            start_year = pd.to_datetime(data["timeframe"]["start"]).year
            if start_year == 2008:
                period = "2008-2009"
            elif start_year == 2020:
                period = "2020-2021"
            elif start_year == 2022:
                period = "2022-2023"
            else:
                continue
                
            asset_raw = data["symbols"][0].upper()
            asset = "AAPL" if "AAPL" in asset_raw else "Gold"
            
            returns = np.array(data["daily_returns"])
            mean_ret = np.mean(returns) * 100
            mdd = data["capital"]["max_drawdown_pct"]
            sharpe = data["capital"].get("sharpe_ratio", 0.0)
            sortino = data["capital"].get("sortino_ratio", 0.0)
            
            parsed_runs.append({
                "system": system,
                "asset": asset,
                "period": period,
                "scenario": scen,
                "mean_return": mean_ret,
                "mdd": mdd,
                "sharpe": sharpe,
                "sortino": sortino
            })
        except Exception as e:
            logging.warning(f"Could not parse results from {res_file}: {e}")
            continue

    if not parsed_runs:
        logging.warning("No other system runs (Baseline, RMDB, LLM) were found in data-backtest/. MDD advantage counts will be blank.")
        # Create empty template to fulfill the required filename
        empty_adv = pd.DataFrame(columns=["scenario", "ref", "mdd_wins", "ret_wins", "mdd_adv"])
        empty_adv.to_csv(OUTPUT_RESULTS_DIR / "mdd_advantage_counts.csv", index=False)
        return

    others_df = pd.DataFrame(parsed_runs)
    xgb_subset = xgb_agg_df[["asset", "period", "scenario", "mean_return", "mdd"]].copy()

    adv_rows = []
    
    for ref_system in ["Baseline", "RMDB", "LLM", "RF"]:
        ref_df = others_df[others_df["system"] == ref_system]
        if ref_df.empty:
            logging.warning(f"No completed runs found for system: {ref_system}. Skipping comparison.")
            continue
            
        for scen in ["S0", "S1", "S2"]:
            # Slice both to compare under same scenario
            xgb_scen = xgb_subset[xgb_subset["scenario"] == scen]
            ref_scen = ref_df[ref_df["scenario"] == scen]
            
            # Merge on asset and period
            merged = pd.merge(xgb_scen, ref_scen, on=["asset", "period"], suffixes=("_xgb", "_ref"))
            if merged.empty:
                continue
                
            mdd_wins = int((merged["mdd_xgb"] < merged["mdd_ref"]).sum())
            ret_wins = int((merged["mean_return_xgb"] > merged["mean_return_ref"]).sum())
            mdd_adv = float((merged["mdd_ref"] - merged["mdd_xgb"]).mean())
            
            adv_rows.append({
                "scenario": scen,
                "ref": f"vs {ref_system}",
                "mdd_wins": mdd_wins,
                "ret_wins": ret_wins,
                "mdd_adv": round(mdd_adv, 4)
            })

    adv_df = pd.DataFrame(adv_rows)
    adv_df.to_csv(OUTPUT_RESULTS_DIR / "mdd_advantage_counts.csv", index=False)
    logging.info(f"Saved MDD advantage counts -> {OUTPUT_RESULTS_DIR}/mdd_advantage_counts.csv")

    # Combine everything to results2/table2_combined.csv if other runs are available
    all_perf = pd.concat([others_df, xgb_agg_df], ignore_index=True)
    combined_table_path = OUTPUT_RESULTS_DIR.parent / "table2_combined.csv"
    
    summary_combined = all_perf.groupby(["system", "scenario"])[["mean_return", "mdd", "sharpe", "sortino"]].mean().round(4)
    summary_combined.to_csv(combined_table_path)
    logging.info(f"Saved combined Table II -> {combined_table_path}")


if __name__ == "__main__":
    run_all()
