#!/usr/bin/env python3
"""
Rule-Based Baseline Trading Bot for backtesting evaluation.
Reuses the exact same dataset loader, timeline simulation, slippage, and fee models
as backtest.py, but makes decisions using deterministic rules:
- Long Entry: Price > EMA20 and RSI > 50 and MACD > MACD Signal
- Short Entry: Price < EMA20 and RSI < 50 and MACD < MACD Signal
- Stop Loss: 2x ATR from entry
- Profit Target: 3x ATR from entry
- Exits: TP/SL hits, or opposite signals generating reversal entries
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Add project root to python path to import bot and backtest
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import backtest
from backtest import BacktestConfig, HistoricalBinanceClient, KLINE_COLUMNS, load_from_dataset, interval_to_timedelta

def configure_environment(cfg: BacktestConfig) -> None:
    os.environ["TRADEBOT_DATA_DIR"] = str(cfg.run_dir)
    os.environ["HYPERLIQUID_LIVE_TRADING"] = "false"
    if cfg.start_capital is not None:
        os.environ["PAPER_START_CAPITAL"] = str(cfg.start_capital)
    if cfg.disable_telegram:
        os.environ["TELEGRAM_BOT_TOKEN"] = ""
        os.environ["TELEGRAM_CHAT_ID"] = ""

def main() -> None:
    backtest.configure_logging()
    dotenv_path = PROJECT_ROOT / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path, override=False)
    else:
        load_dotenv(override=False)

    cfg = BacktestConfig.from_environment()
    configure_environment(cfg)

    import bot
    bot.LLM_MODEL_NAME = "Rule-Based Baseline"
    bot.LLM_TEMPERATURE = 0.0

    # Override symbols if provided in environment
    env_symbols = os.getenv("BACKTEST_SYMBOLS")
    if env_symbols:
        requested_symbols = [s.strip().upper() for s in env_symbols.split(",") if s.strip()]
        if requested_symbols:
            bot.SYMBOLS = requested_symbols
            bot.SYMBOL_TO_COIN = {s: s for s in requested_symbols}
            bot.COIN_TO_SYMBOL = {s: s for s in requested_symbols}

    if getattr(bot, "INTERVAL", None) != cfg.interval:
        bot.INTERVAL = cfg.interval
        if hasattr(bot, "_INTERVAL_TO_SECONDS"):
            bot.CHECK_INTERVAL = bot._INTERVAL_TO_SECONDS[cfg.interval]

    if bot.hyperliquid_trader.is_live:
        bot.hyperliquid_trader._requested_live = False

    # Load datasets
    intervals_needed = {cfg.interval, "1h", "4h"}
    symbol_frames = {}
    for symbol in bot.SYMBOLS:
        symbol_frames[symbol] = {}
        local_frame = load_from_dataset(symbol, cfg)
        for interval in intervals_needed:
            symbol_frames[symbol][interval] = local_frame

    historical_client = HistoricalBinanceClient(symbol_frames)
    bot.client = historical_client

    primary_symbol = bot.SYMBOLS[0]
    if primary_symbol not in symbol_frames or symbol_frames[primary_symbol][cfg.interval].empty:
        logging.error("No data found for primary symbol %s in dataset folder", primary_symbol)
        return

    primary_interval_frame = symbol_frames[primary_symbol][cfg.interval]
    timeline_mask = (primary_interval_frame["timestamp"] >= cfg.start_ms) & (
        primary_interval_frame["timestamp"] <= cfg.end_ms
    )
    timeline = primary_interval_frame.loc[timeline_mask, "timestamp"].astype(np.int64).tolist()
    if not timeline:
        logging.error("No data available for %s between %s and %s in local dataset", cfg.interval, cfg.start, cfg.end)
        return

    time_holder = {"value": int(timeline[0])}
    def simulated_time() -> datetime:
        return datetime.fromtimestamp(time_holder["value"] / 1000, tz=timezone.utc)

    if cfg.start_capital is not None:
        bot.START_CAPITAL = cfg.start_capital

    bot.set_time_provider(simulated_time)
    bot.reset_state(bot.START_CAPITAL)
    bot.init_csv_files()
    bot.register_equity_snapshot(bot.START_CAPITAL)

    interval_seconds = int(interval_to_timedelta(cfg.interval).total_seconds())

    logging.info("Starting Baseline Backtest on %s | Capital: $%.2f", ", ".join(bot.SYMBOLS), bot.START_CAPITAL)

    for idx, timestamp_ms in enumerate(timeline, start=1):
        time_holder["value"] = int(timestamp_ms)
        historical_client.set_current_timestamp(int(timestamp_ms))
        bot.iteration_counter += 1
        bot.current_iteration_messages = []

        # 1. Check auto exits (TP/SL)
        bot.check_stop_loss_take_profit()

        # 2. Rule evaluation
        for symbol in bot.SYMBOLS:
            coin = bot.SYMBOL_TO_COIN.get(symbol, symbol)
            data = bot.fetch_market_data(symbol)
            if not data:
                continue

            price = float(data["price"])
            ema20 = float(data.get("ema20", price))
            rsi = float(data.get("rsi", 50.0))
            macd = float(data.get("macd", 0.0))
            macd_signal = float(data.get("macd_signal", 0.0))
            atr = float(data.get("atr", 0.0))

            long_entry_condition = (price > ema20) and (rsi > 50) and (macd > macd_signal)
            short_entry_condition = (price < ema20) and (rsi < 50) and (macd < macd_signal)

            # Retrieve active position
            pos = bot.positions.get(coin)

            if pos is None:
                # No active position: Evaluate entries
                if long_entry_condition:
                    decision = {
                        "coin": coin,
                        "action": "entry",
                        "side": "long",
                        "leverage": 10.0,
                        "risk_usd": bot.balance * 0.05,
                        "stop_loss": price - 2 * atr if atr > 0 else price * 0.98,
                        "profit_target": price + 3 * atr if atr > 0 else price * 1.03,
                        "justification": f"Long entry rule met. Price {price:.4f} > EMA20 {ema20:.4f}, RSI {rsi:.2f}, MACD {macd:.4f} > Signal {macd_signal:.4f}",
                        "confluence_tags": ["EMA_crossover", "RSI_over_50", "MACD_bullish"],
                        "trigger_tags": ["EMA_crossover"],
                        "reasoning_categories": ["trend_following", "momentum"]
                    }
                    bot.execute_entry(coin, decision, price)
                elif short_entry_condition:
                    decision = {
                        "coin": coin,
                        "action": "entry",
                        "side": "short",
                        "leverage": 10.0,
                        "risk_usd": bot.balance * 0.05,
                        "stop_loss": price + 2 * atr if atr > 0 else price * 1.02,
                        "profit_target": price - 3 * atr if atr > 0 else price * 0.97,
                        "justification": f"Short entry rule met. Price {price:.4f} < EMA20 {ema20:.4f}, RSI {rsi:.2f}, MACD {macd:.4f} < Signal {macd_signal:.4f}",
                        "confluence_tags": ["EMA_crossover", "RSI_under_50", "MACD_bearish"],
                        "trigger_tags": ["EMA_crossover"],
                        "reasoning_categories": ["trend_following", "momentum"]
                    }
                    bot.execute_entry(coin, decision, price)
            else:
                # Active position: Check for reversals
                pos_side = pos["side"].lower()
                if pos_side == "long" and short_entry_condition:
                    # Close Long
                    close_dec = {
                        "coin": coin,
                        "action": "close",
                        "justification": "Opposite signal (Short entry condition met)"
                    }
                    bot.execute_close(coin, close_dec, price)
                    # Immediately open Short
                    decision = {
                        "coin": coin,
                        "action": "entry",
                        "side": "short",
                        "leverage": 10.0,
                        "risk_usd": bot.balance * 0.05,
                        "stop_loss": price + 2 * atr if atr > 0 else price * 1.02,
                        "profit_target": price - 3 * atr if atr > 0 else price * 0.97,
                        "justification": f"Reversal short entry rule met. Price {price:.4f} < EMA20 {ema20:.4f}, RSI {rsi:.2f}",
                        "confluence_tags": ["EMA_crossover", "RSI_under_50", "MACD_bearish"],
                        "trigger_tags": ["EMA_crossover"],
                        "reasoning_categories": ["trend_following", "momentum"]
                    }
                    bot.execute_entry(coin, decision, price)
                elif pos_side == "short" and long_entry_condition:
                    # Close Short
                    close_dec = {
                        "coin": coin,
                        "action": "close",
                        "justification": "Opposite signal (Long entry condition met)"
                    }
                    bot.execute_close(coin, close_dec, price)
                    # Immediately open Long
                    decision = {
                        "coin": coin,
                        "action": "entry",
                        "side": "long",
                        "leverage": 10.0,
                        "risk_usd": bot.balance * 0.05,
                        "stop_loss": price - 2 * atr if atr > 0 else price * 0.98,
                        "profit_target": price + 3 * atr if atr > 0 else price * 1.03,
                        "justification": f"Reversal long entry rule met. Price {price:.4f} > EMA20 {ema20:.4f}, RSI {rsi:.2f}",
                        "confluence_tags": ["EMA_crossover", "RSI_over_50", "MACD_bullish"],
                        "trigger_tags": ["EMA_crossover"],
                        "reasoning_categories": ["trend_following", "momentum"]
                    }
                    bot.execute_entry(coin, decision, price)

        total_equity = bot.calculate_total_equity()
        bot.register_equity_snapshot(total_equity)
        bot.save_state()

        if idx % 50 == 0 or idx == len(timeline):
            logging.info(
                "Processed bar %d/%d at %s | Equity: %.2f | Positions: %d",
                idx,
                len(timeline),
                simulated_time().isoformat(),
                total_equity,
                len(bot.positions)
            )

    # Force close remaining positions at end
    if bot.positions:
        logging.info("Force-closing remaining positions at end of backtest...")
        open_coins = list(bot.positions.keys())
        for coin in open_coins:
            symbol = bot.COIN_TO_SYMBOL.get(coin)
            if not symbol:
                continue
            data = bot.fetch_market_data(symbol)
            if not data:
                continue
            bot.execute_close(coin, {"action": "close", "justification": "End of backtest"}, data["price"])

    final_equity = bot.calculate_total_equity()
    total_net_profit = final_equity - bot.START_CAPITAL
    total_return_pct = (total_net_profit / bot.START_CAPITAL) * 100 if bot.START_CAPITAL else 0.0
    sortino = bot.calculate_sortino_ratio(bot.equity_history, interval_seconds, bot.RISK_FREE_RATE)
    sharpe = bot.calculate_sharpe_ratio(bot.equity_history, interval_seconds, bot.RISK_FREE_RATE)
    max_drawdown = bot.calculate_max_drawdown(bot.equity_history)
    trade_stats = bot.summarize_trades(bot.TRADES_CSV)

    recovery_factor = None
    if max_drawdown is not None and max_drawdown > 0:
        max_dd_amount = bot.START_CAPITAL * max_drawdown
        recovery_factor = total_net_profit / max_dd_amount if max_dd_amount > 0 else None

    # Calculate daily returns for VaR/CVaR and exports
    var_95, cvar_95, daily_returns = 0.0, 0.0, []
    if len(bot.equity_history) >= 2 and timeline:
        start_ts = timeline[0] - int(interval_seconds * 1000)
        all_ts = [start_ts] + list(timeline)
        equity_vals = bot.equity_history[:len(all_ts)]
        if len(equity_vals) < len(all_ts):
            equity_vals = equity_vals + [equity_vals[-1]] * (len(all_ts) - len(equity_vals))

        dates = pd.to_datetime(all_ts, unit='ms', utc=True)
        ts_series = pd.Series(equity_vals, index=dates)
        daily_equity = ts_series.resample('1D').last().ffill()
        if len(daily_equity) >= 2:
            daily_returns_series = daily_equity.pct_change().dropna()
        else:
            daily_returns_series = ts_series.pct_change().dropna()

        daily_returns = daily_returns_series.tolist()
        if not daily_returns_series.empty:
            var_95_raw = np.percentile(daily_returns_series, 5)
            var_95 = -var_95_raw if var_95_raw < 0 else 0.0
            losses_beyond = daily_returns_series[daily_returns_series <= var_95_raw]
            if not losses_beyond.empty:
                cvar_95 = -losses_beyond.mean() if losses_beyond.mean() < 0 else 0.0
            else:
                cvar_95 = var_95

    def format_seconds(seconds: Optional[float]) -> str:
        if seconds is None: return "N/A"
        if seconds < 60: return f"{seconds:.1f}s"
        if seconds < 3600: return f"{seconds/60:.1f}m"
        if seconds < 86400: return f"{seconds/3600:.1f}h"
        return f"{seconds/86400:.1f}d"

    total_closed = trade_stats['close_events']
    win_pct_total = (trade_stats['winning_trades'] / total_closed * 100) if total_closed > 0 else 0.0

    results = {
        "run_id": cfg.run_id,
        "run_directory": str(cfg.run_dir),
        "cache_directory": str(cfg.cache_dir),
        "timeframe": {
            "start": cfg.start.isoformat(),
            "end": cfg.end.isoformat(),
            "interval": cfg.interval,
            "bars": len(timeline),
        },
        "symbols": list(bot.SYMBOL_TO_COIN.values()),
        "capital": {
            "start": bot.START_CAPITAL,
            "final_balance": bot.balance,
            "final_equity": final_equity,
            "total_net_profit": total_net_profit,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": (max_drawdown * 100) if max_drawdown is not None else None,
            "recovery_factor": recovery_factor,
            "profit_factor": trade_stats['profit_factor'],
            "win_rate_pct": win_pct_total,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "var_95_pct": var_95 * 100,
            "cvar_95_pct": cvar_95 * 100,
            "gross_profit": trade_stats['gross_win'],
            "gross_loss": trade_stats['gross_loss'],
        },
        "daily_returns": daily_returns,
        "equity_history": bot.equity_history,
        "llm": {
            "model": "Rule-Based Baseline",
            "temperature": 0.0,
            "max_tokens": None,
            "thinking": None,
            "system_prompt": {
                "source": "deterministic_rules",
                "file": None,
                "override": False,
                "preview": "EMA Cross + RSI filter + MACD trend + ATR SL/TP",
                "full": "Long if Price > EMA20 and RSI > 50 and MACD > MACD Signal. SL 2xATR, TP 3xATR.",
            },
        },
        "trading": trade_stats,
        "generated_at": simulated_time().isoformat(),
    }

    results_path = cfg.run_dir / "backtest_results.json"
    with open(results_path, "w", encoding='utf-8') as fh:
        json.dump(results, fh, indent=2)

    logging.info("Baseline backtest complete. Results written to %s", results_path)

    # Send Telegram notification if enabled
    if not cfg.disable_telegram and bot.TELEGRAM_BOT_TOKEN:
        try:
            rf_val = recovery_factor if recovery_factor is not None else 0.0
            pf_val = trade_stats['profit_factor'] if trade_stats['profit_factor'] is not None else 0.0
            sharpe_val = sharpe if sharpe is not None else 0.0
            sortino_val = sortino if sortino is not None else 0.0
            max_dd_pct = (max_drawdown * 100) if max_drawdown is not None else 0.0
            avg_holding = format_seconds(trade_stats['avg_holding_time_seconds'])
            
            crisis_period = f"{cfg.start.strftime('%Y-%m-%d')} to {cfg.end.strftime('%Y-%m-%d')}"
            
            rows = [
                ("Model", "Rule-Based Baseline"),
                ("Asset", ", ".join(bot.SYMBOLS)),
                ("Crisis Period", crisis_period),
                ("Initial Capital", f"${bot.START_CAPITAL:,.2f}"),
                ("Final Capital", f"${final_equity:,.2f}"),
                ("Net Profit", f"{'+' if total_net_profit >= 0 else '-'}${abs(total_net_profit):,.2f}"),
                ("Return %", f"{total_return_pct:+.2f}%"),
                ("Total Trades", str(trade_stats['total_trades'])),
                ("Win Rate", f"{win_pct_total:.1f}%"),
                ("Profit Factor", f"{pf_val:.2f}"),
                ("Sharpe Ratio", f"{sharpe_val:.2f}"),
                ("Sortino Ratio", f"{sortino_val:.2f}"),
                ("Maximum Drawdown", f"{max_dd_pct:.2f}%"),
                ("Recovery Factor", f"{rf_val:.2f}"),
                ("VaR/CVaR (95%)", f"{var_95*100:.2f}% / {cvar_95*100:.2f}%"),
                ("Avg Holding Time", avg_holding),
            ]
            
            col1_w = max(len(r[0]) for r in rows) + 2
            col2_w = max(len(str(r[1])) for r in rows) + 2
            
            border = f"+{'-' * col1_w}+{'-' * col2_w}+"
            header = f"| {'Metric':<{col1_w-2}} | {'Value':<{col2_w-2}} |"
            
            table_lines = [border, header, border]
            for m, v in rows:
                table_lines.append(f"| {m:<{col1_w-2}} | {str(v):<{col2_w-2}} |")
            table_lines.append(border)
            table_str = "\n".join(table_lines)
            
            msg = f"📊 *Baseline Backtest Research Summary*\n```\n{table_str}\n```"
            bot.send_telegram_message(msg)
            logging.info("Sent baseline summary to Telegram.")
        except Exception as exc:
            logging.warning("Failed to send Telegram summary: %s", exc)

if __name__ == "__main__":
    main()
