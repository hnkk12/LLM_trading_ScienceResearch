#!/usr/bin/env python3
"""
Risk-Managed Deterministic Baseline (RMDB) Trading Bot for backtesting evaluation.
Reuses the exact same dataset loader, timeline simulation, slippage, and fee models
as run_baseline.py and backtest.py, but incorporates the exact same risk settings
as Llama:
- Risk 1% (halved to 0.5% in Yellow Zone, 0% in Red Zone)
- ATR gate / Volatility-Adaptive Risk Zones (Green/Yellow/Red)
- Cùng fee, slippage, asset, period và timeframe.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Add project root to python path to import bot and backtest
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import backtest
from backtest import BacktestConfig, HistoricalBinanceClient, KLINE_COLUMNS, load_from_dataset, interval_to_timedelta

def detect_candlestick_patterns(df: pd.DataFrame, num_candles: int = 2) -> List[Dict[str, Any]]:
    """Detect Pinbar, Engulfing, and Inside Bar patterns for the last num_candles."""
    patterns = []
    if len(df) < 5:
        return patterns

    for offset in range(num_candles, 0, -1):
        idx = len(df) - offset
        if idx <= 0:
            continue
        
        row = df.iloc[idx]
        prev_row = df.iloc[idx - 1]
        
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        po, ph, pl, pc = float(prev_row["open"]), float(prev_row["high"]), float(prev_row["low"]), float(prev_row["close"])
        
        candle_range = h - l
        if candle_range <= 0:
            continue
            
        body = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        
        pattern_name = None
        details = ""
        
        # 1. Pinbar
        if body <= candle_range * 0.35:
            if lower_shadow >= candle_range * 0.6:
                pattern_name = "Bullish Pinbar"
                details = f"Long lower shadow at {l:.2f}"
            elif upper_shadow >= candle_range * 0.6:
                pattern_name = "Bearish Pinbar"
                details = f"Long upper shadow at {h:.2f}"
                
        # 2. Engulfing
        if not pattern_name:
            if c > o and pc < po:
                if o <= pc and c >= po and (c - o) > (po - pc):
                    pattern_name = "Bullish Engulfing"
                    details = "Body engulfs previous bearish body"
            elif c < o and pc > po:
                if o >= pc and c <= po and (o - c) > (pc - po):
                    pattern_name = "Bearish Engulfing"
                    details = "Body engulfs previous bullish body"
                    
        # 3. Inside Bar
        if not pattern_name:
            if h <= ph and l >= pl:
                pattern_name = "Inside Bar"
                details = f"Range completely within previous bar ({pl:.2f} - {ph:.2f})"
                
        if pattern_name:
            patterns.append({
                "bar_offset": -offset,
                "pattern": pattern_name,
                "details": details,
                "price": c
            })
            
    return patterns


def detect_fvgs(df: pd.DataFrame, limit: int = 15) -> Dict[str, List[Dict[str, Any]]]:
    """Detect unmitigated Bullish and Bearish Fair Value Gaps in the last limit candles."""
    bullish_fvgs = []
    bearish_fvgs = []
    
    if len(df) < 5:
        return {"bullish": [], "bearish": []}
        
    start_idx = max(2, len(df) - limit)
    
    for i in range(start_idx, len(df)):
        h_prev2 = float(df.iloc[i - 2]["high"])
        l_curr = float(df.iloc[i]["low"])
        
        if l_curr > h_prev2:
            fvg_low = h_prev2
            fvg_high = l_curr
            mitigated = False
            for j in range(i + 1, len(df)):
                low_j = float(df.iloc[j]["low"])
                if low_j <= fvg_low:
                    mitigated = True
                    break
                elif low_j < fvg_high:
                    fvg_high = low_j
            
            if not mitigated and (fvg_high - fvg_low) > 0.05 * float(df.iloc[i]["close"]) / 100:
                bullish_fvgs.append({
                    "low": fvg_low,
                    "high": fvg_high,
                    "bar_index": i,
                    "bar_offset": i - len(df),
                    "gap_size_pct": (fvg_high - fvg_low) / fvg_low * 100
                })
                
        l_prev2 = float(df.iloc[i - 2]["low"])
        h_curr = float(df.iloc[i]["high"])
        
        if l_prev2 > h_curr:
            fvg_low = h_curr
            fvg_high = l_prev2
            mitigated = False
            for j in range(i + 1, len(df)):
                high_j = float(df.iloc[j]["high"])
                if high_j >= fvg_high:
                    mitigated = True
                    break
                elif high_j > fvg_low:
                    fvg_low = high_j
                    
            if not mitigated and (fvg_high - fvg_low) > 0.05 * float(df.iloc[i]["close"]) / 100:
                bearish_fvgs.append({
                    "low": fvg_low,
                    "high": fvg_high,
                    "bar_index": i,
                    "bar_offset": i - len(df),
                    "gap_size_pct": (fvg_high - fvg_low) / fvg_low * 100
                })
                
    return {"bullish": bullish_fvgs, "bearish": bearish_fvgs}


def detect_order_blocks(df: pd.DataFrame, limit: int = 20) -> Dict[str, List[Dict[str, Any]]]:
    """Detect recent unmitigated Bullish and Bearish Order Blocks."""
    bullish_obs = []
    bearish_obs = []
    
    if len(df) < 15:
        return {"bullish": [], "bearish": []}
        
    body_sizes = (df["close"] - df["open"]).abs()
    avg_body = body_sizes.rolling(window=15).mean().fillna(0.0)
    
    start_idx = max(5, len(df) - limit)
    
    for i in range(start_idx, len(df)):
        avg_sz = float(avg_body.iloc[i])
        curr_body = float(body_sizes.iloc[i])
        
        if curr_body >= 1.5 * avg_sz and avg_sz > 0:
            c = float(df.iloc[i]["close"])
            o = float(df.iloc[i]["open"])
            
            if c > o:
                for k in range(i - 1, i - 5, -1):
                    if k < 0:
                        break
                    prev_c = float(df.iloc[k]["close"])
                    prev_o = float(df.iloc[k]["open"])
                    prev_l = float(df.iloc[k]["low"])
                    prev_h = float(df.iloc[k]["high"])
                    
                    if prev_c < prev_o:
                        ob_low = prev_l
                        ob_high = prev_h
                        
                        mitigated = False
                        for j in range(i, len(df)):
                            low_j = float(df.iloc[j]["low"])
                            if low_j < ob_low:
                                mitigated = True
                                break
                        
                        if not mitigated:
                            bullish_obs.append({
                                "low": ob_low,
                                "high": ob_high,
                                "bar_index": k,
                                "bar_offset": k - len(df),
                                "type": "Bullish OB"
                            })
                        break
                        
            elif c < o:
                for k in range(i - 1, i - 5, -1):
                    if k < 0:
                        break
                    prev_c = float(df.iloc[k]["close"])
                    prev_o = float(df.iloc[k]["open"])
                    prev_l = float(df.iloc[k]["low"])
                    prev_h = float(df.iloc[k]["high"])
                    
                    if prev_c > prev_o:
                        ob_low = prev_l
                        ob_high = prev_h
                        
                        mitigated = False
                        for j in range(i, len(df)):
                            high_j = float(df.iloc[j]["high"])
                            if high_j > ob_high:
                                mitigated = True
                                break
                                
                        if not mitigated:
                            bearish_obs.append({
                                "low": ob_low,
                                "high": ob_high,
                                "bar_index": k,
                                "bar_offset": k - len(df),
                                "type": "Bearish OB"
                            })
                        break
                        
    return {"bullish": bullish_obs, "bearish": bearish_obs}


def detect_market_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect the most recent BOS/CHoCH by scanning historical bars backwards."""
    result = {
        "structure_status": "Range",
        "last_break_type": None,
        "last_break_direction": None,
        "break_price": None,
        "swing_high": None,
        "swing_low": None
    }
    
    if len(df) < 20:
        return result
        
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    
    swing_highs = []
    swing_lows = []
    
    for i in range(2, len(df) - 3):
        if highs[i] == max(highs[i-2:i+3]):
            swing_highs.append(highs[i])
        if lows[i] == min(lows[i-2:i+3]):
            swing_lows.append(lows[i])
            
    if swing_highs:
        result["swing_high"] = swing_highs[-1]
    if swing_lows:
        result["swing_low"] = swing_lows[-1]
        
    for i in range(len(df) - 1, 10, -1):
        sh_before = None
        for k in range(i - 3, 2, -1):
            if highs[k] == max(highs[k-2:k+3]):
                sh_before = highs[k]
                break
        sl_before = None
        for k in range(i - 3, 2, -1):
            if lows[k] == min(lows[k-2:k+3]):
                sl_before = lows[k]
                break
                
        if sh_before and closes[i] > sh_before:
            result["last_break_direction"] = "Bullish"
            result["break_price"] = sh_before
            ema20_i = float(df["ema20"].iloc[i]) if "ema20" in df else closes[i]
            ema50_i = float(df["ema50"].iloc[i]) if "ema50" in df else closes[i]
            if ema20_i > ema50_i:
                result["structure_status"] = "Bullish Continuation"
                result["last_break_type"] = "BOS"
            else:
                result["structure_status"] = "Bullish Reversal"
                result["last_break_type"] = "CHoCH"
            break
        elif sl_before and closes[i] < sl_before:
            result["last_break_direction"] = "Bearish"
            result["break_price"] = sl_before
            ema20_i = float(df["ema20"].iloc[i]) if "ema20" in df else closes[i]
            ema50_i = float(df["ema50"].iloc[i]) if "ema50" in df else closes[i]
            if ema20_i < ema50_i:
                result["structure_status"] = "Bearish Continuation"
                result["last_break_type"] = "BOS"
            else:
                result["structure_status"] = "Bearish Reversal"
                result["last_break_type"] = "CHoCH"
            break
            
    return result

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
    bot.LLM_MODEL_NAME = "Risk-Managed Deterministic Baseline"
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

    logging.info("Starting RMDB Backtest on %s | Capital: $%.2f", ", ".join(bot.SYMBOLS), bot.START_CAPITAL)

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
            
            # Fetch execution klines directly from client to get full DataFrame
            klines_exec = bot.client.get_klines(symbol, cfg.interval, limit=120)
            if not klines_exec or len(klines_exec) < 20:
                continue
                
            df_exec = pd.DataFrame(klines_exec, columns=KLINE_COLUMNS)
            df_exec[KLINE_COLUMNS[1:6]] = df_exec[KLINE_COLUMNS[1:6]].astype(float)
            
            # Calculate indicators using bot functions
            df_exec = bot.add_indicator_columns(df_exec, ema_lengths=(20, 50))
            df_exec["atr"] = bot.calculate_atr_series(df_exec, 14)
            df_exec["atr_long"] = bot.calculate_atr_series(df_exec, 100)
            
            latest_row = df_exec.iloc[-1]
            price = float(latest_row["close"])
            ema20 = float(latest_row["ema20"])
            ema50 = float(latest_row["ema50"])
            atr = float(latest_row["atr"]) if not pd.isna(latest_row["atr"]) else 0.0
            atr_long = float(latest_row["atr_long"]) if not pd.isna(latest_row["atr_long"]) else 0.0
            
            # 3. Volatility Ratio & Zone Classification
            vr = atr / atr_long if atr_long > 0 else 1.0
            if vr < 1.6:
                vr_zone = "GREEN"
                risk_pct = 0.01  # Strict 1% risk
            elif vr <= 2.2:
                vr_zone = "YELLOW"
                risk_pct = 0.005  # 50% reduced position size/risk (0.5%)
            else:
                vr_zone = "RED"
                risk_pct = 0.0  # Suspend entries (0%)
            
            # Detect patterns and structure
            patterns = detect_candlestick_patterns(df_exec, num_candles=2)
            fvgs = detect_fvgs(df_exec, limit=15)
            obs = detect_order_blocks(df_exec, limit=20)
            ms = detect_market_structure(df_exec)
            
            # Trend Check (Wyckoff markup/markdown phase proxy)
            trend_bullish = (ema20 > ema50) and (ms["structure_status"] in ["Bullish Continuation", "Bullish Reversal"])
            trend_bearish = (ema20 < ema50) and (ms["structure_status"] in ["Bearish Continuation", "Bearish Reversal"])
            
            # SMC Mitigation Check
            mitigates_bullish_ob = False
            bullish_ob_target = None
            for ob in obs["bullish"]:
                if ob["low"] <= price <= ob["high"]:
                    mitigates_bullish_ob = True
                    bullish_ob_target = ob
                    break
                    
            mitigates_bullish_fvg = False
            bullish_fvg_target = None
            for fvg in fvgs["bullish"]:
                if fvg["low"] <= price <= fvg["high"]:
                    mitigates_bullish_fvg = True
                    bullish_fvg_target = fvg
                    break
                    
            mitigates_bearish_ob = False
            bearish_ob_target = None
            for ob in obs["bearish"]:
                if ob["low"] <= price <= ob["high"]:
                    mitigates_bearish_ob = True
                    bearish_ob_target = ob
                    break
                    
            mitigates_bearish_fvg = False
            bearish_fvg_target = None
            for fvg in fvgs["bearish"]:
                if fvg["low"] <= price <= fvg["high"]:
                    mitigates_bearish_fvg = True
                    bearish_fvg_target = fvg
                    break
            
            # Calculate volume ratio for breakout trigger
            df_exec["volume_sma"] = df_exec["volume"].rolling(window=20).mean().fillna(1.0)
            df_exec["volume_ratio"] = (df_exec["volume"] / df_exec["volume_sma"].replace(0, np.nan)).fillna(1.0)
            volume_ratio = float(df_exec["volume_ratio"].iloc[-1])
            
            # Check Breakout trigger
            is_bullish_breakout = (ms["last_break_type"] == "BOS") and (ms["last_break_direction"] == "Bullish") and (volume_ratio > 1.2)
            is_bearish_breakout = (ms["last_break_type"] == "BOS") and (ms["last_break_direction"] == "Bearish") and (volume_ratio > 1.2)
            
            # Stricter Confluences for Yellow Zone (3+ factors: trend + trigger + RSI/MACD confirm)
            yellow_long_confluence = True
            yellow_short_confluence = True
            if vr_zone == "YELLOW":
                rsi14 = float(latest_row["rsi14"]) if "rsi14" in latest_row and not pd.isna(latest_row["rsi14"]) else 50.0
                macd = float(latest_row["macd"]) if "macd" in latest_row and not pd.isna(latest_row["macd"]) else 0.0
                macd_signal = float(latest_row["macd_signal"]) if "macd_signal" in latest_row and not pd.isna(latest_row["macd_signal"]) else 0.0
                yellow_long_confluence = (rsi14 > 50) or (macd > macd_signal)
                yellow_short_confluence = (rsi14 < 50) or (macd < macd_signal)
            
            # ATR Gate (Anti-Chasing: no entry if price moved > 2 ATRs from trigger breakout level)
            is_long_chasing = False
            is_short_chasing = False
            if is_bullish_breakout:
                breakout_ref = ms["swing_high"]
                if breakout_ref and price - breakout_ref > 2 * atr:
                    is_long_chasing = True
            if is_bearish_breakout:
                breakout_ref = ms["swing_low"]
                if breakout_ref and breakout_ref - price > 2 * atr:
                    is_short_chasing = True
            
            # Final Entry Signals
            if vr_zone == "RED":
                long_entry_condition = False
                short_entry_condition = False
            else:
                long_entry_condition = trend_bullish and (mitigates_bullish_ob or mitigates_bullish_fvg or is_bullish_breakout) and yellow_long_confluence and not is_long_chasing
                short_entry_condition = trend_bearish and (mitigates_bearish_ob or mitigates_bearish_fvg or is_bearish_breakout) and yellow_short_confluence and not is_short_chasing
            
            # Retrieve active position
            pos = bot.positions.get(coin)
            risk_usd = bot.balance * risk_pct
            
            if pos is None:
                # No active position: Evaluate entries
                if long_entry_condition and risk_usd > 0:
                    # SL placement: tight SL (0.05x ATR) below OB/FVG, or default to 1.2x ATR
                    sl_ref = price - 1.2 * atr if atr > 0 else price * 0.99
                    trigger_source = "1.2x ATR"
                    if mitigates_bullish_ob and bullish_ob_target:
                        sl_ref = bullish_ob_target["low"] - 0.05 * atr if atr > 0 else bullish_ob_target["low"]
                        trigger_source = f"OB Low ${bullish_ob_target['low']:.2f}"
                    elif mitigates_bullish_fvg and bullish_fvg_target:
                        sl_ref = bullish_fvg_target["low"] - 0.05 * atr if atr > 0 else bullish_fvg_target["low"]
                        trigger_source = f"FVG Low ${bullish_fvg_target['low']:.2f}"
                    elif is_bullish_breakout:
                        trigger_source = f"BOS Bullish Breakout (Vol Ratio {volume_ratio:.1f}x)"
                    
                    stop_loss = sl_ref if sl_ref < price else (price - 1.2 * atr if atr > 0 else price * 0.99)
                    
                    # TP placement: quick TP (2.0x ATR) or near swing high
                    tp_ref = price + 2.0 * atr if atr > 0 else price * 1.02
                    if ms["swing_high"] and (price + 1.2 * atr < ms["swing_high"] < price + 3.0 * atr):
                        tp_ref = ms["swing_high"]
                    profit_target = tp_ref
                    
                    decision = {
                        "coin": coin,
                        "action": "entry",
                        "side": "long",
                        "leverage": 10.0,
                        "risk_usd": risk_usd,
                        "stop_loss": stop_loss,
                        "profit_target": profit_target,
                        "justification": f"RMDB SMC Long ({vr_zone} Zone, VR: {vr:.2f}). Trigger: {trigger_source}, Target: ${profit_target:.2f}, Stop: ${stop_loss:.2f}",
                        "confluence_tags": ["SMC_scalping", "OB_FVG_limit" if not is_bullish_breakout else "BOS_breakout", ms["structure_status"], f"Zone_{vr_zone}"],
                        "trigger_tags": ["OB_mitigation"] if mitigates_bullish_ob else (["FVG_mitigation"] if mitigates_bullish_fvg else ["BOS_breakout"]),
                        "reasoning_categories": ["SMC", "Scalping", "Risk_Managed"]
                    }
                    bot.execute_entry(coin, decision, price)
                    
                elif short_entry_condition and risk_usd > 0:
                    # SL placement: tight SL (0.05x ATR) above OB/FVG, or default to 1.2x ATR
                    sl_ref = price + 1.2 * atr if atr > 0 else price * 1.01
                    trigger_source = "1.2x ATR"
                    if mitigates_bearish_ob and bearish_ob_target:
                        sl_ref = bearish_ob_target["high"] + 0.05 * atr if atr > 0 else bearish_ob_target["high"]
                        trigger_source = f"OB High ${bearish_ob_target['high']:.2f}"
                    elif mitigates_bearish_fvg and bearish_fvg_target:
                        sl_ref = bearish_fvg_target["high"] + 0.05 * atr if atr > 0 else bearish_fvg_target["high"]
                        trigger_source = f"FVG High ${bearish_fvg_target['high']:.2f}"
                    elif is_bearish_breakout:
                        trigger_source = f"BOS Bearish Breakout (Vol Ratio {volume_ratio:.1f}x)"
                        
                    stop_loss = sl_ref if sl_ref > price else (price + 1.2 * atr if atr > 0 else price * 1.01)
                    
                    # TP placement: quick TP (2.0x ATR) or near swing low
                    tp_ref = price - 2.0 * atr if atr > 0 else price * 0.98
                    if ms["swing_low"] and (price - 3.0 * atr < ms["swing_low"] < price - 1.2 * atr):
                        tp_ref = ms["swing_low"]
                    profit_target = tp_ref
                    
                    decision = {
                        "coin": coin,
                        "action": "entry",
                        "side": "short",
                        "leverage": 10.0,
                        "risk_usd": risk_usd,
                        "stop_loss": stop_loss,
                        "profit_target": profit_target,
                        "justification": f"RMDB SMC Short ({vr_zone} Zone, VR: {vr:.2f}). Trigger: {trigger_source}, Target: ${profit_target:.2f}, Stop: ${stop_loss:.2f}",
                        "confluence_tags": ["SMC_scalping", "OB_FVG_limit" if not is_bearish_breakout else "BOS_breakout", ms["structure_status"], f"Zone_{vr_zone}"],
                        "trigger_tags": ["OB_mitigation"] if mitigates_bearish_ob else (["FVG_mitigation"] if mitigates_bearish_fvg else ["BOS_breakout"]),
                        "reasoning_categories": ["SMC", "Scalping", "Risk_Managed"]
                    }
                    bot.execute_entry(coin, decision, price)
            else:
                # Active position: Check for reversals (Only reverse if we are not in RED zone)
                pos_side = pos["side"].lower()
                if pos_side == "long" and short_entry_condition and risk_usd > 0:
                    close_dec = {
                        "coin": coin,
                        "action": "close",
                        "justification": "SMC Scalping Reversal opposite signal met (Short Entry Triggered)"
                    }
                    bot.execute_close(coin, close_dec, price)
                    
                    # Immediately open Short
                    sl_ref = price + 1.2 * atr if atr > 0 else price * 1.01
                    trigger_source = "1.2x ATR"
                    if mitigates_bearish_ob and bearish_ob_target:
                        sl_ref = bearish_ob_target["high"] + 0.05 * atr if atr > 0 else bearish_ob_target["high"]
                        trigger_source = f"OB High ${bearish_ob_target['high']:.2f}"
                    elif mitigates_bearish_fvg and bearish_fvg_target:
                        sl_ref = bearish_fvg_target["high"] + 0.05 * atr if atr > 0 else bearish_fvg_target["high"]
                        trigger_source = f"FVG High ${bearish_fvg_target['high']:.2f}"
                    elif is_bearish_breakout:
                        trigger_source = f"BOS Bearish Breakout (Vol Ratio {volume_ratio:.1f}x)"
                    stop_loss = sl_ref if sl_ref > price else (price + 1.2 * atr if atr > 0 else price * 1.01)
                    
                    tp_ref = price - 2.0 * atr if atr > 0 else price * 0.98
                    if ms["swing_low"] and (price - 3.0 * atr < ms["swing_low"] < price - 1.2 * atr):
                        tp_ref = ms["swing_low"]
                    profit_target = tp_ref
                    
                    decision = {
                        "coin": coin,
                        "action": "entry",
                        "side": "short",
                        "leverage": 10.0,
                        "risk_usd": risk_usd,
                        "stop_loss": stop_loss,
                        "profit_target": profit_target,
                        "justification": f"Reversal short entry rule met. Trigger: {trigger_source}, Target: ${profit_target:.2f}, Stop: ${stop_loss:.2f}",
                        "confluence_tags": ["SMC_scalping", "OB_FVG_limit" if not is_bearish_breakout else "BOS_breakout", ms["structure_status"], f"Zone_{vr_zone}"],
                        "trigger_tags": ["OB_mitigation"] if mitigates_bearish_ob else (["FVG_mitigation"] if mitigates_bearish_fvg else ["BOS_breakout"]),
                        "reasoning_categories": ["SMC", "Scalping", "Risk_Managed"]
                    }
                    bot.execute_entry(coin, decision, price)
                    
                elif pos_side == "short" and long_entry_condition and risk_usd > 0:
                    close_dec = {
                        "coin": coin,
                        "action": "close",
                        "justification": "SMC Scalping Reversal opposite signal met (Long Entry Triggered)"
                    }
                    bot.execute_close(coin, close_dec, price)
                    
                    # Immediately open Long
                    sl_ref = price - 1.2 * atr if atr > 0 else price * 0.99
                    trigger_source = "1.2x ATR"
                    if mitigates_bullish_ob and bullish_ob_target:
                        sl_ref = bullish_ob_target["low"] - 0.05 * atr if atr > 0 else bullish_ob_target["low"]
                        trigger_source = f"OB Low ${bullish_ob_target['low']:.2f}"
                    elif mitigates_bullish_fvg and bullish_fvg_target:
                        sl_ref = bullish_fvg_target["low"] - 0.05 * atr if atr > 0 else bullish_fvg_target["low"]
                        trigger_source = f"FVG Low ${bullish_fvg_target['low']:.2f}"
                    elif is_bullish_breakout:
                        trigger_source = f"BOS Bullish Breakout (Vol Ratio {volume_ratio:.1f}x)"
                    stop_loss = sl_ref if sl_ref < price else (price - 1.2 * atr if atr > 0 else price * 0.99)
                    
                    tp_ref = price + 2.0 * atr if atr > 0 else price * 1.02
                    if ms["swing_high"] and (price + 1.2 * atr < ms["swing_high"] < price + 3.0 * atr):
                        tp_ref = ms["swing_high"]
                    profit_target = tp_ref
                    
                    decision = {
                        "coin": coin,
                        "action": "entry",
                        "side": "long",
                        "leverage": 10.0,
                        "risk_usd": risk_usd,
                        "stop_loss": stop_loss,
                        "profit_target": profit_target,
                        "justification": f"Reversal long entry rule met. Trigger: {trigger_source}, Target: ${profit_target:.2f}, Stop: ${stop_loss:.2f}",
                        "confluence_tags": ["SMC_scalping", "OB_FVG_limit" if not is_bullish_breakout else "BOS_breakout", ms["structure_status"], f"Zone_{vr_zone}"],
                        "trigger_tags": ["OB_mitigation"] if mitigates_bullish_ob else (["FVG_mitigation"] if mitigates_bullish_fvg else ["BOS_breakout"]),
                        "reasoning_categories": ["SMC", "Scalping", "Risk_Managed"]
                    }
                    bot.execute_entry(coin, decision, price)

        total_equity = bot.calculate_total_equity()
        bot.register_equity_snapshot(total_equity)
        bot.save_state()

        if idx % 50 == 0 or idx == len(timeline):
            logging.info(
                "Processed bar %d/%d at %s | Equity: %.2f | Positions: %d | VR Zone: %s (VR: %.2f)",
                idx,
                len(timeline),
                simulated_time().isoformat(),
                total_equity,
                len(bot.positions),
                vr_zone,
                vr
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
            "model": "Risk-Managed Deterministic Baseline",
            "temperature": 0.0,
            "max_tokens": None,
            "thinking": None,
            "system_prompt": {
                "source": "deterministic_rmdb_rules",
                "file": None,
                "override": False,
                "preview": "SMC + Wyckoff + Volatility Zones RMDB Baseline",
                "full": "RMDB Baseline: 1% risk in Green Zone, 0.5% risk in Yellow Zone with stricter confluences, 0% risk (suspend entries) in Red Zone. Also applies ATR gate / Anti-Chasing to ignore breakouts closing too far.",
            },
        },
        "trading": trade_stats,
        "generated_at": simulated_time().isoformat(),
    }

    results_path = cfg.run_dir / "backtest_results.json"
    with open(results_path, "w", encoding='utf-8') as fh:
        json.dump(results, fh, indent=2)

    logging.info("RMDB backtest complete. Results written to %s", results_path)

    # Save daily returns to CSV
    if 'daily_returns_series' in locals() and not daily_returns_series.empty:
        daily_returns_series.to_csv(cfg.run_dir / "daily_returns.csv", header=["daily_return"])

    # Save settings to settings.json
    settings_data = {
        "start": cfg.start.isoformat(),
        "end": cfg.end.isoformat(),
        "interval": cfg.interval,
        "symbols": bot.SYMBOLS,
        "start_capital": bot.START_CAPITAL,
        "model": "Risk-Managed Deterministic Baseline",
        "temperature": 0.0,
        "max_tokens": 0,
        "slippage_mode": os.getenv("BACKTEST_SLIPPAGE_MODE", "S0"),
        "spread_pct": float(os.getenv("BACKTEST_SPREAD_PCT", "0.0002")),
        "fee_rate_taker": bot.TAKER_FEE_RATE,
        "fee_rate_maker": bot.MAKER_FEE_RATE,
    }
    with open(cfg.run_dir / "settings.json", "w", encoding="utf-8") as sf:
        json.dump(settings_data, sf, indent=2)

    # Save explanation to prompt_template.txt
    try:
        with open(cfg.run_dir / "prompt_template.txt", "w", encoding="utf-8") as pf:
            pf.write("Risk-Managed Deterministic Baseline (RMDB). No LLM used. 1% Risk, Green/Yellow/Red Zones, ATR Gate.")
    except Exception as e:
        logging.warning("Failed to write prompt_template.txt: %s", e)

    # Generate summary text table and save to backtest_summary.txt
    try:
        rf_val = recovery_factor if recovery_factor is not None else 0.0
        pf_val = trade_stats['profit_factor'] if trade_stats['profit_factor'] is not None else 0.0
        sharpe_val = sharpe if sharpe is not None else 0.0
        sortino_val = sortino if sortino is not None else 0.0
        max_dd_pct = (max_drawdown * 100) if max_drawdown is not None else 0.0
        avg_holding = format_seconds(trade_stats['avg_holding_time_seconds'])
        
        crisis_period = f"{cfg.start.strftime('%Y-%m-%d')} to {cfg.end.strftime('%Y-%m-%d')}"
        
        rows = [
            ("Model", "RMDB"),
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
        
        # Save to backtest_summary.txt
        with open(cfg.run_dir / "backtest_summary.txt", "w", encoding="utf-8") as sf:
            sf.write(table_str)
            
        # Send Telegram notification if enabled
        if not cfg.disable_telegram and bot.TELEGRAM_BOT_TOKEN:
            msg = f"📊 *RMDB Backtest Research Summary*\n```\n{table_str}\n```"
            bot.send_telegram_message(msg)
            logging.info("Sent RMDB summary to Telegram.")
    except Exception as exc:
        logging.warning("Failed to generate or send RMDB summary: %s", exc)

if __name__ == "__main__":
    main()
