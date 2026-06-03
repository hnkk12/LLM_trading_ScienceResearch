#!/usr/bin/env python3
"""
Backtesting harness for the DeepSeek Multi-Asset Trading Bot.

It replays historical Binance data, calls the LLM for decisions on each bar,
and reuses the live trading execution and logging pipeline while writing output
to an isolated data directory per run.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from binance.client import Client
from dotenv import load_dotenv

# Columns returned by Binance kline endpoints
KLINE_COLUMNS: List[str] = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_base",
    "taker_quote",
    "ignore",
]

DEFAULT_INTERVAL = "15m"
LONG_CONTEXT_INTERVAL = "4h"
STRUCTURE_INTERVAL = "1h"
INTERVAL_TO_DELTA = {
    "1m": timedelta(minutes=1),
    "3m": timedelta(minutes=3),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "2h": timedelta(hours=2),
    "4h": timedelta(hours=4),
    "6h": timedelta(hours=6),
    "8h": timedelta(hours=8),
    "12h": timedelta(hours=12),
    "1d": timedelta(days=1),
}
SUPPORTED_INTERVALS = tuple(INTERVAL_TO_DELTA.keys())
WARMUP_BARS = {
    "1m": 500,
    "3m": 300,
    "5m": 240,
    "15m": 200,
    "30m": 180,
    "1h": 150,
    "1d": 100,
    LONG_CONTEXT_INTERVAL: 120,
}

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BACKTEST_DIR = PROJECT_ROOT / "data-backtest"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_datetime(value: Optional[str], fallback: datetime) -> datetime:
    if not value:
        return fallback
    try:
        parsed = pd.to_datetime(value, utc=True)
    except Exception as exc:  # pragma: no cover - parsing guard
        logging.warning("Failed to parse datetime '%s': %s; using fallback %s", value, exc, fallback)
        return fallback
    if isinstance(parsed, pd.Series):
        parsed = parsed.iloc[0]
    if isinstance(parsed, pd.Timestamp):
        parsed = parsed.to_pydatetime()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def interval_to_timedelta(interval: str) -> timedelta:
    try:
        return INTERVAL_TO_DELTA[interval]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported interval '{interval}'") from exc


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize_kline_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    normalized = df.copy()
    for col in ("timestamp", "close_time", "trades"):
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    float_columns = ["open", "high", "low", "close", "volume", "quote_volume", "taker_base", "taker_quote"]
    for col in float_columns:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    normalized.dropna(subset=["timestamp"], inplace=True)
    normalized["timestamp"] = normalized["timestamp"].astype(np.int64)
    normalized["close_time"] = normalized["close_time"].fillna(normalized["timestamp"]).astype(np.int64)
    normalized["trades"] = normalized["trades"].fillna(0).astype(int)
    normalized.sort_values("timestamp", inplace=True)
    normalized.reset_index(drop=True, inplace=True)
    return normalized


@dataclass
class BacktestConfig:
    start: datetime
    end: datetime
    interval: str
    base_dir: Path
    run_dir: Path
    cache_dir: Path
    run_id: str
    model: Optional[str]
    temperature: Optional[float]
    max_tokens: Optional[int]
    thinking: Optional[str]
    system_prompt: Optional[str]
    system_prompt_file: Optional[str]
    start_capital: Optional[float]
    disable_telegram: bool

    @property
    def start_ms(self) -> int:
        return int(self.start.timestamp() * 1000)

    @property
    def end_ms(self) -> int:
        return int(self.end.timestamp() * 1000)

    @staticmethod
    def from_environment() -> "BacktestConfig":
        now_utc = datetime.now(timezone.utc)
        default_end = now_utc
        default_start = default_end - timedelta(days=7)

        start = ensure_utc(parse_datetime(os.getenv("BACKTEST_START"), default_start))
        end = ensure_utc(parse_datetime(os.getenv("BACKTEST_END"), default_end))
        if start >= end:
            raise ValueError("BACKTEST_START must be earlier than BACKTEST_END")

        interval = os.getenv("BACKTEST_INTERVAL", DEFAULT_INTERVAL).lower()
        if interval not in SUPPORTED_INTERVALS:
            logging.warning(
                "Interval %s not explicitly supported; defaulting to %s",
                interval,
                DEFAULT_INTERVAL,
            )
            interval = DEFAULT_INTERVAL

        base_dir_raw = os.getenv("BACKTEST_DATA_DIR")
        if base_dir_raw:
            base_dir = Path(base_dir_raw).expanduser()
            if not base_dir.is_absolute():
                base_dir = (PROJECT_ROOT / base_dir).resolve()
        else:
            base_dir = DEFAULT_BACKTEST_DIR
        cache_dir = base_dir / "cache"

        run_id = os.getenv("BACKTEST_RUN_ID")
        if not run_id:
            run_id = f"run-{now_utc.strftime('%Y%m%d-%H%M%S')}"
        run_dir = base_dir / run_id

        model = os.getenv("BACKTEST_LLM_MODEL")
        if model is None:
            model = os.getenv("BACKTEST_MODEL")

        temperature_raw = os.getenv("BACKTEST_TEMPERATURE")
        temperature = 0.0  # Force 0.0 by default for deterministic backtests
        if temperature_raw:
            try:
                temp_val = float(temperature_raw)
                if temp_val in (0.0, 0.1):
                    temperature = temp_val
                else:
                    logging.warning("BACKTEST_TEMPERATURE '%s' overridden to 0.0 for determinism.", temperature_raw)
                    temperature = 0.0
            except ValueError:
                logging.warning("Invalid BACKTEST_TEMPERATURE '%s'; defaulting to 0.0.", temperature_raw)

        max_tokens_raw = os.getenv("BACKTEST_MAX_TOKENS")
        max_tokens = None
        if max_tokens_raw:
            try:
                max_tokens = int(max_tokens_raw)
            except ValueError:
                logging.warning("Invalid BACKTEST_MAX_TOKENS '%s'; ignoring.", max_tokens_raw)

        thinking_raw = os.getenv("BACKTEST_LLM_THINKING")
        if thinking_raw is None:
            thinking_raw = os.getenv("BACKTEST_THINKING")
        thinking = None
        if thinking_raw is not None:
            thinking_raw = thinking_raw.strip()
            if thinking_raw:
                thinking = thinking_raw

        system_prompt_file_raw = os.getenv("BACKTEST_SYSTEM_PROMPT_FILE")
        system_prompt_file = None
        if system_prompt_file_raw:
            prompt_path = Path(system_prompt_file_raw).expanduser()
            if not prompt_path.is_absolute():
                prompt_path = (PROJECT_ROOT / prompt_path).resolve()
            system_prompt_file = str(prompt_path)

        system_prompt = os.getenv("BACKTEST_SYSTEM_PROMPT")

        start_capital_raw = os.getenv("BACKTEST_START_CAPITAL")
        start_capital = None
        if start_capital_raw:
            try:
                start_capital = float(start_capital_raw)
            except ValueError:
                logging.warning("Invalid BACKTEST_START_CAPITAL '%s'; ignoring.", start_capital_raw)

        disable_telegram = os.getenv("BACKTEST_DISABLE_TELEGRAM", "false").strip().lower() in {"1", "true", "yes", "on"}

        base_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        return BacktestConfig(
            start=start,
            end=end,
            interval=interval,
            base_dir=base_dir,
            run_dir=run_dir,
            cache_dir=cache_dir,
            run_id=run_id,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
            system_prompt=system_prompt,
            system_prompt_file=system_prompt_file,
            start_capital=start_capital,
            disable_telegram=disable_telegram,
        )


def ensure_cached_klines(
    client: Client,
    cfg: BacktestConfig,
    symbol: str,
    interval: str,
) -> pd.DataFrame:
    warmup = WARMUP_BARS.get(interval, 0)
    interval_delta = interval_to_timedelta(interval)
    start_with_buffer = cfg.start - interval_delta * warmup
    start_with_buffer = ensure_utc(start_with_buffer)
    end_with_buffer = ensure_utc(cfg.end)

    cache_path = cfg.cache_dir / f"{symbol}_{interval}.csv"
    if cache_path.exists():
        cached = normalize_kline_dataframe(pd.read_csv(cache_path))
    else:
        cached = pd.DataFrame(columns=KLINE_COLUMNS)

    start_ms_required = int(start_with_buffer.timestamp() * 1000)
    end_ms_required = int(end_with_buffer.timestamp() * 1000)

    have_coverage = False
    if not cached.empty:
        cached_start = int(cached["timestamp"].min())
        cached_end = int(cached["timestamp"].max())
        have_coverage = cached_start <= start_ms_required and cached_end >= end_ms_required

    if not have_coverage:
        start_str = start_with_buffer.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_with_buffer.strftime("%Y-%m-%d %H:%M:%S")
        logging.info("Downloading %s %s klines from Binance (%s → %s)...", symbol, interval, start_str, end_str)
        # Pass millisecond timestamps to avoid ambiguous string date parsing in python-binance.
        klines = client.get_historical_klines(symbol, interval, start_ms_required, end_ms_required)
        fetched = normalize_kline_dataframe(pd.DataFrame(klines, columns=KLINE_COLUMNS))
        if cached.empty:
            cached = fetched
        else:
            cached = pd.concat([cached, fetched], ignore_index=True)
            cached.drop_duplicates(subset="timestamp", keep="last", inplace=True)
            cached.sort_values("timestamp", inplace=True)
            cached.reset_index(drop=True, inplace=True)
        cached.to_csv(cache_path, index=False)

    trimmed = cached[cached["timestamp"] <= end_ms_required].copy()
    trimmed.reset_index(drop=True, inplace=True)
    return trimmed


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


def load_from_dataset(symbol: str, cfg: BacktestConfig) -> pd.DataFrame:
    """Load and normalize data from the local 'dataset' folder."""
    dataset_dir = PROJECT_ROOT / "dataset"
    # Find files starting with symbol (e.g., AAPL_D_2008_2009.csv)
    files = list(dataset_dir.glob(f"{symbol}*.csv"))
    if not files:
        logging.warning("No local dataset found for %s in %s", symbol, dataset_dir)
        return pd.DataFrame(columns=KLINE_COLUMNS)
    
    # Try to find a file matching the start year of the backtest
    start_year = str(cfg.start.year)
    matching_files = [f for f in files if start_year in f.name]
    if matching_files:
        file_path = matching_files[0]
    else:
        file_path = files[0]
    logging.info("Loading local dataset for %s: %s", symbol, file_path)
    
    try:
        df = pd.read_csv(file_path)
        # Expected columns: "Date","Price","Open","High","Low","Vol.","Change %"
        # Map to Binance format
        normalized = pd.DataFrame()
        # Some CSVs might use different case for Date
        date_col = next((c for c in df.columns if c.lower() == "date"), "Date")
        normalized["timestamp"] = pd.to_datetime(df[date_col]).astype(np.int64) // 10**6
        
        price_col = next((c for c in df.columns if c.lower() in ["price", "close"]), "Price")
        open_col = next((c for c in df.columns if c.lower() == "open"), "Open")
        high_col = next((c for c in df.columns if c.lower() == "high"), "High")
        low_col = next((c for c in df.columns if c.lower() == "low"), "Low")
        vol_col = next((c for c in df.columns if c.lower() in ["vol.", "volume"]), "Vol.")
        
        def to_clean_float(series: pd.Series) -> pd.Series:
            if series.dtype == object:
                return pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce")
            return pd.to_numeric(series, errors="coerce")

        normalized["open"] = to_clean_float(df[open_col])
        normalized["high"] = to_clean_float(df[high_col])
        normalized["low"] = to_clean_float(df[low_col])
        normalized["close"] = to_clean_float(df[price_col])
        normalized["volume"] = df[vol_col].apply(parse_dataset_volume)
        
        # Add required kline columns
        normalized["close_time"] = normalized["timestamp"] + 86399999 # Default to 1 day
        normalized["quote_volume"] = 0.0
        normalized["trades"] = 0
        normalized["taker_base"] = 0.0
        normalized["taker_quote"] = 0.0
        normalized["ignore"] = 0
        
        return normalize_kline_dataframe(normalized)
    except Exception as exc:
        logging.error("Failed to load dataset %s: %s", file_path, exc)
        return pd.DataFrame(columns=KLINE_COLUMNS)


class HistoricalBinanceClient:
    """Minimal Binance client shim that replays cached klines."""

    def __init__(self, frames: Dict[str, Dict[str, pd.DataFrame]]) -> None:
        self._frames = frames
        self._current_timestamp_ms: Optional[int] = None
        self._indices: Dict[str, Dict[str, Optional[int]]] = {
            symbol: {interval: None for interval in intervals}
            for symbol, intervals in frames.items()
        }

    def set_current_timestamp(self, timestamp_ms: int) -> None:
        self._current_timestamp_ms = timestamp_ms
        for symbol, interval_frames in self._frames.items():
            for interval, df in interval_frames.items():
                timestamps = df["timestamp"].to_numpy(dtype=np.int64)
                idx = np.searchsorted(timestamps, timestamp_ms, side="right") - 1
                if 0 <= idx < len(timestamps):
                    self._indices[symbol][interval] = int(idx)
                else:
                    self._indices[symbol][interval] = None

    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[List[float]]:
        if symbol not in self._frames or interval not in self._frames[symbol]:
            return []
        idx = self._indices[symbol][interval]
        if idx is None:
            return []
        df = self._frames[symbol][interval]
        start_idx = max(0, idx - max(0, limit - 1))
        subset = df.iloc[start_idx : idx + 1]
        return subset[KLINE_COLUMNS].values.tolist()

    def futures_open_interest_hist(self, symbol: str, period: str, limit: int = 30) -> List[Dict[str, float]]:
        return []

    def futures_funding_rate(self, symbol: str, limit: int = 30) -> List[Dict[str, float]]:
        return []

    @property
    def current_timestamp_ms(self) -> Optional[int]:
        return self._current_timestamp_ms

    @property
    def current_datetime(self) -> Optional[datetime]:
        if self._current_timestamp_ms is None:
            return None
        return datetime.fromtimestamp(self._current_timestamp_ms / 1000, tz=timezone.utc)


def configure_environment(cfg: BacktestConfig) -> None:
    os.environ["TRADEBOT_DATA_DIR"] = str(cfg.run_dir)
    os.environ["HYPERLIQUID_LIVE_TRADING"] = "false"
    if cfg.start_capital is not None:
        os.environ["PAPER_START_CAPITAL"] = str(cfg.start_capital)
    if cfg.model:
        os.environ["TRADEBOT_LLM_MODEL"] = cfg.model
    if cfg.temperature is not None:
        os.environ["TRADEBOT_LLM_TEMPERATURE"] = str(cfg.temperature)
    if cfg.max_tokens is not None:
        os.environ["TRADEBOT_LLM_MAX_TOKENS"] = str(cfg.max_tokens)
    if cfg.thinking is not None:
        os.environ["TRADEBOT_LLM_THINKING"] = cfg.thinking
    if cfg.system_prompt_file:
        os.environ["TRADEBOT_SYSTEM_PROMPT_FILE"] = cfg.system_prompt_file
        os.environ.pop("TRADEBOT_SYSTEM_PROMPT", None)
    elif cfg.system_prompt is not None:
        os.environ["TRADEBOT_SYSTEM_PROMPT"] = cfg.system_prompt
        os.environ.pop("TRADEBOT_SYSTEM_PROMPT_FILE", None)
    if cfg.disable_telegram:
        os.environ["TELEGRAM_BOT_TOKEN"] = ""
        os.environ["TELEGRAM_CHAT_ID"] = ""


def main() -> None:
    configure_logging()
    dotenv_path = PROJECT_ROOT / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path, override=False)
    else:
        load_dotenv(override=False)

    cfg = BacktestConfig.from_environment()
    logging.info("Backtest LLM override from env: %s", cfg.model)
    print(f"Backtest LLM override from env: {cfg.model}")
    configure_environment(cfg)

    import bot  # pylint: disable=import-error
    
    # Override symbols if provided in environment
    env_symbols = os.getenv("BACKTEST_SYMBOLS")
    if env_symbols:
        requested_symbols = [s.strip().upper() for s in env_symbols.split(",") if s.strip()]
        if requested_symbols:
            bot.SYMBOLS = requested_symbols
            # Also update SYMBOL_TO_COIN to avoid KeyErrors
            bot.SYMBOL_TO_COIN = {s: s for s in requested_symbols}
            bot.COIN_TO_SYMBOL = {s: s for s in requested_symbols}
            logging.info("Overriding bot symbols with: %s", bot.SYMBOLS)
    if hasattr(bot, "refresh_llm_configuration_from_env"):
        bot.refresh_llm_configuration_from_env()
    if hasattr(bot, "log_system_prompt_info"):
        bot.log_system_prompt_info("Backtest system prompt")
        print(f"System prompt for this backtest: {bot.describe_system_prompt_source()}")

    if getattr(bot, "INTERVAL", None) != cfg.interval:
        logging.info("Aligning bot interval with backtest interval: %s → %s", getattr(bot, "INTERVAL", None), cfg.interval)
        bot.INTERVAL = cfg.interval
        if hasattr(bot, "_INTERVAL_TO_SECONDS"):
            bot.CHECK_INTERVAL = bot._INTERVAL_TO_SECONDS[cfg.interval]  # type: ignore[attr-defined]

    logging.info("Backtest configured with LLM model: %s", bot.LLM_MODEL_NAME)
    print(f"LLM model for this backtest: {bot.LLM_MODEL_NAME}")

    if bot.hyperliquid_trader.is_live:
        logging.warning("Hyperliquid trader reports live mode; forcing paper mode for backtest.")
        bot.hyperliquid_trader._requested_live = False  # type: ignore[attr-defined]

    # Use local dataset instead of Binance API
    intervals_needed = {cfg.interval, STRUCTURE_INTERVAL, LONG_CONTEXT_INTERVAL}
    symbol_frames: Dict[str, Dict[str, pd.DataFrame]] = {}
    for symbol in bot.SYMBOLS:
        symbol_frames[symbol] = {}
        # Load local data once per symbol and reuse for all intervals
        local_frame = load_from_dataset(symbol, cfg)
        for interval in intervals_needed:
            symbol_frames[symbol][interval] = local_frame

    historical_client = HistoricalBinanceClient(symbol_frames)
    bot.client = historical_client  # type: ignore[assignment]

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
        logging.info("Starting capital set to BACKTEST_START_CAPITAL: $%.2f", bot.START_CAPITAL)

    bot.set_time_provider(simulated_time)
    # Ensure the bot's internal state matches our backtest starting capital
    bot.reset_state(bot.START_CAPITAL)
    bot.init_csv_files()
    bot.register_equity_snapshot(bot.START_CAPITAL)

    interval_seconds = int(interval_to_timedelta(cfg.interval).total_seconds())

    logging.info("LLM model used for this backtest: %s", bot.LLM_MODEL_NAME)
    print(f"LLM model used for this backtest: {bot.LLM_MODEL_NAME}")
    print(f"Starting capital: ${bot.START_CAPITAL:.2f}")

    ai_calls_count = 0
    for idx, timestamp_ms in enumerate(timeline, start=1):
        time_holder["value"] = int(timestamp_ms)
        historical_client.set_current_timestamp(int(timestamp_ms))
        bot.iteration_counter += 1
        bot.current_iteration_messages = []

        # 1. Check Auto-Execution (TP/SL) - No API cost
        bot.check_stop_loss_take_profit()
        
        # 2. Event-Driven Logic: Decide if we should wake up the AI
        should_call_ai = False
        event_reason = ""
        
        # Reason A: We have active positions (Check every bar for management)
        has_positions = len(bot.positions) > 0
        if has_positions:
            should_call_ai = True
            event_reason = "Active position management"
            
        # Reason B: Moderate Price Volatility (> 0.8% move)
        for symbol in bot.SYMBOLS:
            klines = historical_client.get_klines(symbol, cfg.interval, limit=2)
            if len(klines) >= 2:
                # klines is List[List[float]], index 4 is 'close'
                prev_close = float(klines[-2][4])
                curr_close = float(klines[-1][4])
                change = abs((curr_close - prev_close) / prev_close)
                if change >= 0.008:
                    should_call_ai = True
                    event_reason = f"Volatility detected in {symbol} ({change*100:.1f}%)"
                    break
                    should_call_ai = True
                    event_reason = f"Volatility detected in {symbol} ({change*100:.1f}%)"
                    break
        
        # Reason C: RSI Extremes (Potential reversals)
        if not should_call_ai:
            for symbol in bot.SYMBOLS:
                data = bot.fetch_market_data(symbol)
                if data and (data["rsi"] < 35 or data["rsi"] > 65):
                    should_call_ai = True
                    event_reason = f"RSI extreme in {symbol} ({data['rsi']:.1f})"
                    break
        
        # Reason D: First and Last days (Always call for initial setup and final wrap)
        if idx == 1 or idx == len(timeline):
            should_call_ai = True
            event_reason = "Initial/Final iteration"

        if should_call_ai:
            logging.info("Waking up AI for: %s", event_reason)
            prompt = bot.format_trading_prompt()
            decisions = bot.call_llm_api(prompt)
            ai_calls_count += 1

            if not decisions:
                logging.warning("Iteration %d: no decisions returned by LLM.", idx)
            else:
                bot.process_ai_decisions(decisions)
        else:
            # Skip AI call - just log progress
            if idx % 10 == 0:
                logging.info("Skipping AI call for bar %d/%d (No major events)", idx, len(timeline))

        total_equity = bot.calculate_total_equity()
        bot.register_equity_snapshot(total_equity)
        bot.log_portfolio_state() # Reduced logging for speed
        bot.save_state()

        current_dt = simulated_time()
        if should_call_ai or idx % 50 == 0:
            logging.info(
                "Processed bar %d/%d at %s | Equity: %.2f | Positions: %d | AI Calls: %d",
                idx,
                len(timeline),
                current_dt.isoformat(),
                total_equity,
                len(bot.positions),
                ai_calls_count
            )

    # Force close all remaining positions at the end of backtest to realize PnL
    if bot.positions:
        logging.info("End of backtest reached. Force-closing %d remaining positions...", len(bot.positions))
        final_timestamp_ms = int(timeline[-1])
        time_holder["value"] = final_timestamp_ms
        historical_client.set_current_timestamp(final_timestamp_ms)
        
        # Create a list of coins to avoid 'dictionary size changed during iteration'
        open_coins = list(bot.positions.keys())
        for coin in open_coins:
            symbol = bot.COIN_TO_SYMBOL.get(coin)
            if not symbol: continue
            data = bot.fetch_market_data(symbol)
            if not data: continue
            
            # Use a dummy 'close' decision
            bot.execute_close(coin, {"signal": "close", "justification": "End of backtest force-close"}, data["price"])

    final_equity = bot.calculate_total_equity()
    total_net_profit = final_equity - bot.START_CAPITAL
    total_return_pct = (total_net_profit / bot.START_CAPITAL) * 100 if bot.START_CAPITAL else 0.0
    sortino = bot.calculate_sortino_ratio(bot.equity_history, interval_seconds, bot.RISK_FREE_RATE)
    sharpe = bot.calculate_sharpe_ratio(bot.equity_history, interval_seconds, bot.RISK_FREE_RATE)
    max_drawdown = bot.calculate_max_drawdown(bot.equity_history)
    trade_stats = bot.summarize_trades(bot.TRADES_CSV)

    recovery_factor = None
    if max_drawdown is not None and max_drawdown > 0:
        # max_drawdown is decimal pct, e.g. 0.1 for 10%
        # recovery_factor = Total Net Profit / (Start Capital * Max Drawdown)
        max_dd_amount = bot.START_CAPITAL * max_drawdown
        recovery_factor = total_net_profit / max_dd_amount if max_dd_amount > 0 else None

    # Calculate daily returns for VaR/CVaR and exports
    var_95, cvar_95, daily_returns = 0.0, 0.0, []
    if len(bot.equity_history) >= 2 and timeline:
        try:
            delta = interval_to_timedelta(cfg.interval)
        except Exception:
            delta = timedelta(minutes=15)
        start_ts = timeline[0] - int(delta.total_seconds() * 1000)
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

    # Helper for duration formatting
    def format_seconds(seconds: Optional[float]) -> str:
        if seconds is None: return "N/A"
        if seconds < 60: return f"{seconds:.1f}s"
        if seconds < 3600: return f"{seconds/60:.1f}m"
        if seconds < 86400: return f"{seconds/3600:.1f}h"
        return f"{seconds/86400:.1f}d"

    # Calculate additional percentages for display
    total_closed = trade_stats['close_events']
    win_pct_total = (trade_stats['winning_trades'] / total_closed * 100) if total_closed > 0 else 0
    loss_pct_total = (trade_stats['losing_trades'] / total_closed * 100) if total_closed > 0 else 0

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
            "model": bot.LLM_MODEL_NAME,
            "temperature": bot.LLM_TEMPERATURE,
            "max_tokens": bot.LLM_MAX_TOKENS,
            "thinking": bot.LLM_THINKING_PARAM,
            "system_prompt": {
                "source": (
                    "file"
                    if cfg.system_prompt_file
                    else ("env" if cfg.system_prompt is not None else "default")
                ),
                "file": cfg.system_prompt_file,
                "override": bool(cfg.system_prompt_file or cfg.system_prompt),
                "preview": bot.TRADING_RULES_PROMPT[:200],
                "full": bot.TRADING_RULES_PROMPT,
            },
        },
        "trading": {
            **trade_stats,
            "win_trades_pct_total": win_pct_total,
            "loss_trades_pct_total": loss_pct_total,
        },
        "generated_at": simulated_time().isoformat(),
    }

    results_path = cfg.run_dir / "backtest_results.json"
    with open(results_path, "w", encoding='utf-8') as fh:
        json.dump(results, fh, indent=2)

    logging.info("Backtest complete. Results written to %s", results_path)

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
        "model": bot.LLM_MODEL_NAME,
        "temperature": bot.LLM_TEMPERATURE,
        "max_tokens": bot.LLM_MAX_TOKENS,
        "slippage_mode": os.getenv("BACKTEST_SLIPPAGE_MODE", "S0"),
        "spread_pct": float(os.getenv("BACKTEST_SPREAD_PCT", "0.0002")),
        "fee_rate_taker": bot.TAKER_FEE_RATE,
        "fee_rate_maker": bot.MAKER_FEE_RATE,
    }
    with open(cfg.run_dir / "settings.json", "w", encoding="utf-8") as sf:
        json.dump(settings_data, sf, indent=2)

    # Copy system prompt to prompt_template.txt
    if cfg.system_prompt_file and Path(cfg.system_prompt_file).exists():
        try:
            import shutil
            shutil.copy(cfg.system_prompt_file, cfg.run_dir / "prompt_template.txt")
        except Exception as e:
            logging.warning("Failed to copy system prompt file: %s", e)
    else:
        try:
            with open(cfg.run_dir / "prompt_template.txt", "w", encoding="utf-8") as pf:
                pf.write(bot.TRADING_RULES_PROMPT)
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
            ("Model", bot.LLM_MODEL_NAME),
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
            msg = f"📊 *Backtest Research Summary*\n```\n{table_str}\n```"
            bot.send_telegram_message(msg)
            logging.info("Sent backtest summary to Telegram.")
    except Exception as exc:
        logging.warning("Failed to generate or send backtest summary: %s", exc)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # pragma: no cover - interactive guard
        logging.info("Backtest interrupted by user.")
