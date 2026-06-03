#!/usr/bin/env python3
"""
DeepSeek Multi-Asset Paper Trading Bot
Uses Binance API for market data and OpenRouter API for DeepSeek Chat V3.1 trading decisions
"""
from __future__ import annotations

import os
import re
import time
import json
import logging
import csv
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.exceptions import RequestException, Timeout
from binance.client import Client
from dotenv import load_dotenv
from colorama import Fore, Style, init as colorama_init

from hyperliquid_client import HyperliquidTradingClient

colorama_init(autoreset=True)

BASE_DIR = Path(__file__).resolve().parent
DOTENV_PATH = BASE_DIR / ".env"

if DOTENV_PATH.exists():
    dotenv_loaded = load_dotenv(dotenv_path=DOTENV_PATH, override=True)
else:
    dotenv_loaded = load_dotenv(override=True)

DEFAULT_DATA_DIR = BASE_DIR / "data"
DATA_DIR = Path(os.getenv("TRADEBOT_DATA_DIR", str(DEFAULT_DATA_DIR))).expanduser()
DATA_DIR.mkdir(parents=True, exist_ok=True)

EARLY_ENV_WARNINGS: List[str] = []

def _parse_bool_env(value: Optional[str], *, default: bool = False) -> bool:
    """Convert environment string to bool with sensible defaults."""
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_float_env(value: Optional[str], *, default: float) -> float:
    """Convert environment string to float with fallback and logging."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        EARLY_ENV_WARNINGS.append(
            f"Invalid float environment value '{value}'; using default {default:.2f}"
        )
        return default


def _parse_int_env(value: Optional[str], *, default: int) -> int:
    """Convert environment string to int with fallback and logging."""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        EARLY_ENV_WARNINGS.append(
            f"Invalid int environment value '{value}'; using default {default}"
        )
        return default


def _parse_thinking_env(value: Optional[str]) -> Optional[Any]:
    """Parse LLM thinking budget/configuration from environment."""
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        return int(raw)
    except (TypeError, ValueError):
        pass
    try:
        return float(raw)
    except (TypeError, ValueError):
        pass
    return raw


# ───────────────────────── CONFIG ─────────────────────────
API_KEY = os.getenv("BN_API_KEY", "")
API_SECRET = os.getenv("BN_SECRET", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_SIGNALS_CHAT_ID = os.getenv("TELEGRAM_SIGNALS_CHAT_ID", "")

# Proxy configuration
HTTP_PROXY = os.getenv("HTTP_PROXY")
HTTPS_PROXY = os.getenv("HTTPS_PROXY")
PROXIES = {
    "http": HTTP_PROXY,
    "https": HTTPS_PROXY,
} if HTTP_PROXY or HTTPS_PROXY else None

HYPERLIQUID_LIVE_TRADING = _parse_bool_env(
    os.getenv("HYPERLIQUID_LIVE_TRADING"),
    default=False,
)
HYPERLIQUID_WALLET_ADDRESS = os.getenv("HYPERLIQUID_WALLET_ADDRESS", "")
HYPERLIQUID_PRIVATE_KEY = os.getenv("HYPERLIQUID_PRIVATE_KEY", "")

PAPER_START_CAPITAL = _parse_float_env(
    os.getenv("PAPER_START_CAPITAL"),
    default=10000.0,
)
HYPERLIQUID_CAPITAL = _parse_float_env(
    os.getenv("HYPERLIQUID_CAPITAL"),
    default=500.0,
)

START_CAPITAL = HYPERLIQUID_CAPITAL if HYPERLIQUID_LIVE_TRADING else PAPER_START_CAPITAL

# Trading symbols to monitor
SYMBOLS = ["ETHUSDT", "SOLUSDT", "XRPUSDT", "BTCUSDT", "DOGEUSDT", "BNBUSDT"]
SYMBOL_TO_COIN = {
    "ETHUSDT": "ETH",
    "SOLUSDT": "SOL", 
    "XRPUSDT": "XRP",
    "BTCUSDT": "BTC",
    "DOGEUSDT": "DOGE",
    "BNBUSDT": "BNB"
}
COIN_TO_SYMBOL = {coin: symbol for symbol, coin in SYMBOL_TO_COIN.items()}

DEFAULT_TRADING_RULES_PROMPT = """
You are a high-performance aggressive crypto trader. Your goal is to maximize account growth by decisively entering high-probability setups.

CORE DIRECTIVES:
- Be Decisive: One clear signal is enough to act.
- Lower Barriers: 3/6 secondary conditions are sufficient.
- Risk Profile: Accept R/R as low as 1.2:1.
- Trend Following: Ride strong trends with conviction.

EXECUTION RULES:
- Risk: Default to 5% of capital per trade.
- Leverage: Default to 10x.
- Surival is priority, but growth is the mission.
""".strip()

SYSTEM_PROMPT_SOURCE: Dict[str, Any] = {"type": "default"}


def _load_system_prompt() -> str:
    """Load system prompt from env variables or fall back to default."""
    global SYSTEM_PROMPT_SOURCE
    prompt_file = os.getenv("TRADEBOT_SYSTEM_PROMPT_FILE")
    if prompt_file:
        path = Path(prompt_file).expanduser()
        if not path.is_absolute():
            path = (BASE_DIR / path).resolve()
        try:
            if path.exists():
                SYSTEM_PROMPT_SOURCE = {"type": "file", "path": str(path)}
                return path.read_text(encoding="utf-8").strip()
            EARLY_ENV_WARNINGS.append(
                f"System prompt file '{path}' not found; using default prompt."
            )
        except Exception as exc:
            EARLY_ENV_WARNINGS.append(
                f"Failed to read system prompt file '{path}': {exc}; using default prompt."
            )

    prompt_env = os.getenv("TRADEBOT_SYSTEM_PROMPT")
    if prompt_env:
        SYSTEM_PROMPT_SOURCE = {"type": "env"}
        return prompt_env.strip()

    SYSTEM_PROMPT_SOURCE = {"type": "default"}
    return DEFAULT_TRADING_RULES_PROMPT


def describe_system_prompt_source() -> str:
    """Return human-readable description of the active system prompt."""
    source_type = SYSTEM_PROMPT_SOURCE.get("type", "default")
    if source_type == "file":
        return f"file:{SYSTEM_PROMPT_SOURCE.get('path', '?')}"
    if source_type == "env":
        return "env:TRADEBOT_SYSTEM_PROMPT"
    return "default prompt"


TRADING_RULES_PROMPT = _load_system_prompt()

DEFAULT_INTERVAL = "15m"
_INTERVAL_TO_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
}


def _load_trade_interval(default: str = DEFAULT_INTERVAL) -> str:
    """Resolve trade interval from environment."""
    raw = os.getenv("TRADEBOT_INTERVAL")
    if raw:
        candidate = raw.strip().lower()
        if candidate in _INTERVAL_TO_SECONDS:
            return candidate
        EARLY_ENV_WARNINGS.append(
            f"Unsupported TRADEBOT_INTERVAL '{raw}'; using default {default}."
        )
    return default


INTERVAL = _load_trade_interval()
CHECK_INTERVAL = _INTERVAL_TO_SECONDS[INTERVAL]
DEFAULT_RISK_FREE_RATE = 0.0  # Annualized baseline for Sortino ratio calculations
DEFAULT_LLM_MODEL = "llama-3.1-8b-instant"


def _load_llm_model_name() -> str:
    raw = os.getenv("TRADEBOT_LLM_MODEL", DEFAULT_LLM_MODEL)
    if not raw:
        return DEFAULT_LLM_MODEL
    value = raw.strip()
    return value or DEFAULT_LLM_MODEL


def _load_llm_temperature() -> float:
    return _parse_float_env(
        os.getenv("TRADEBOT_LLM_TEMPERATURE"),
        default=0.7,
    )


def _load_llm_max_tokens() -> int:
    return _parse_int_env(
        os.getenv("TRADEBOT_LLM_MAX_TOKENS"),
        default=4000,
    )


def refresh_llm_configuration_from_env() -> None:
    """Reload LLM-related runtime settings from environment variables."""
    global LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_THINKING_PARAM, TRADING_RULES_PROMPT
    LLM_MODEL_NAME = _load_llm_model_name()
    LLM_TEMPERATURE = _load_llm_temperature()
    LLM_MAX_TOKENS = _load_llm_max_tokens()
    LLM_THINKING_PARAM = _parse_thinking_env(os.getenv("TRADEBOT_LLM_THINKING"))
    TRADING_RULES_PROMPT = _load_system_prompt()


def log_system_prompt_info(prefix: str = "System prompt in use") -> None:
    """Log the current system prompt configuration."""
    description = describe_system_prompt_source()
    logging.info("%s: %s", prefix, description)


LLM_MODEL_NAME = _load_llm_model_name()
LLM_TEMPERATURE = _load_llm_temperature()
LLM_MAX_TOKENS = _load_llm_max_tokens()
LLM_THINKING_PARAM = _parse_thinking_env(os.getenv("TRADEBOT_LLM_THINKING"))

# Indicator settings
EMA_LEN = 20
RSI_LEN = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Binance fee structure (as decimals)
MAKER_FEE_RATE = 0.0         # 0.0000%
TAKER_FEE_RATE = 0.000275    # 0.0275%

# ───────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)

for warning_msg in EARLY_ENV_WARNINGS:
    logging.warning(warning_msg)
EARLY_ENV_WARNINGS.clear()

def _resolve_risk_free_rate() -> float:
    """Determine the annualized risk-free rate used in Sortino calculations."""
    env_value = os.getenv("SORTINO_RISK_FREE_RATE")
    if env_value is None:
        env_value = os.getenv("RISK_FREE_RATE")
    if env_value is None:
        return DEFAULT_RISK_FREE_RATE
    try:
        return float(env_value)
    except (TypeError, ValueError):
        logging.warning(
            "Invalid SORTINO_RISK_FREE_RATE/RISK_FREE_RATE value '%s'; using default %.4f",
            env_value,
            DEFAULT_RISK_FREE_RATE,
        )
        return DEFAULT_RISK_FREE_RATE

RISK_FREE_RATE = _resolve_risk_free_rate()

if not dotenv_loaded:
    logging.warning(f"No .env file found at {DOTENV_PATH}; falling back to system environment variables.")

if OPENROUTER_API_KEY:
    masked_key = (
        OPENROUTER_API_KEY
        if len(OPENROUTER_API_KEY) <= 12
        else f"{OPENROUTER_API_KEY[:6]}...{OPENROUTER_API_KEY[-4:]}"
    )
    logging.info(
        "OpenRouter API key detected: %s (length %d)",
        masked_key,
        len(OPENROUTER_API_KEY),
    )
else:
    logging.error("OPENROUTER_API_KEY not found; please check your .env file.")

client: Optional[Client] = None

try:
    hyperliquid_trader = HyperliquidTradingClient(
        live_mode=HYPERLIQUID_LIVE_TRADING,
        wallet_address=HYPERLIQUID_WALLET_ADDRESS,
        secret_key=HYPERLIQUID_PRIVATE_KEY,
    )
except Exception as exc:
    logging.critical("Hyperliquid live trading initialization failed: %s", exc)
    raise SystemExit(1) from exc

def get_binance_client() -> Optional[Client]:
    """Return a connected Binance client or None if initialization failed."""
    global client

    if client is not None:
        return client

    if not API_KEY or not API_SECRET:
        logging.error("BN_API_KEY and/or BN_SECRET missing; unable to initialize Binance client.")
        return None

    try:
        logging.info("Attempting to initialize Binance client...")
        client = Client(API_KEY, API_SECRET, testnet=False)
        logging.info("Binance client initialized successfully.")
    except Timeout as exc:
        logging.warning(
            "Timed out while connecting to Binance API: %s. Will retry automatically without exiting.",
            exc,
        )
        client = None
    except RequestException as exc:
        logging.error(
            "Network error while connecting to Binance API: %s. Will retry automatically.",
            exc,
        )
        client = None
    except Exception as exc:
        logging.error(
            "Unexpected error while initializing Binance client: %s",
            exc,
            exc_info=True,
        )
        client = None

    return client

# ──────────────────────── GLOBAL STATE ─────────────────────
balance: float = START_CAPITAL
positions: Dict[str, Dict[str, Any]] = {}  # coin -> position info
trade_history: List[Dict[str, Any]] = []
def _default_time_provider() -> datetime:
    """Return current UTC time; overridable for testing/backtests."""
    return datetime.now(timezone.utc)


_current_time_provider: Callable[[], datetime] = _default_time_provider


def get_current_time() -> datetime:
    """Return the current time from the active provider."""
    return _current_time_provider()


def set_time_provider(provider: Optional[Callable[[], datetime]]) -> None:
    """Override the time provider; pass None to restore wall-clock time."""
    global _current_time_provider
    _current_time_provider = provider or _default_time_provider


BOT_START_TIME = get_current_time()
invocation_count: int = 0
iteration_counter: int = 0
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
current_iteration_messages: List[str] = []
equity_history: List[float] = []

# CSV files
STATE_CSV = DATA_DIR / "portfolio_state.csv"
STATE_JSON = DATA_DIR / "portfolio_state.json"
TRADES_CSV = DATA_DIR / "trade_history.csv"
DECISIONS_CSV = DATA_DIR / "ai_decisions.csv"
MESSAGES_CSV = DATA_DIR / "ai_messages.csv"
STATE_COLUMNS = [
    'timestamp',
    'total_balance',
    'total_equity',
    'total_return_pct',
    'num_positions',
    'position_details',
    'total_margin',
    'net_unrealized_pnl',
    'btc_price',
]
last_btc_price: Optional[float] = None

# ───────────────────────── CSV LOGGING ──────────────────────

def init_csv_files() -> None:
    """Initialize CSV files with headers."""
    if not STATE_CSV.exists():
        with open(STATE_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(STATE_COLUMNS)
    else:
        try:
            df = pd.read_csv(STATE_CSV, encoding='utf-8')
        except Exception as exc:
            logging.warning("Unable to load %s for schema check: %s", STATE_CSV, exc)
        else:
            if list(df.columns) != STATE_COLUMNS:
                for column in STATE_COLUMNS:
                    if column not in df.columns:
                        df[column] = np.nan
                try:
                    df = df[STATE_COLUMNS]
                except KeyError:
                    # Fall back to writing header only if severe mismatch
                    df = pd.DataFrame(columns=STATE_COLUMNS)
                df.to_csv(STATE_CSV, index=False, encoding='utf-8')
    
    TRADES_COLUMNS = [
        'timestamp', 'coin', 'action', 'side', 'quantity', 'price',
        'profit_target', 'stop_loss', 'leverage', 'confidence',
        'pnl', 'balance_after', 'reason', 'confluence_tags', 'trigger_tags', 'reasoning_categories'
    ]
    if not TRADES_CSV.exists():
        with open(TRADES_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(TRADES_COLUMNS)
    else:
        try:
            df = pd.read_csv(TRADES_CSV, encoding='utf-8')
            if list(df.columns) != TRADES_COLUMNS:
                for col in TRADES_COLUMNS:
                    if col not in df.columns:
                        df[col] = ""
                df = df[TRADES_COLUMNS]
                df.to_csv(TRADES_CSV, index=False, encoding='utf-8')
        except Exception as exc:
            logging.warning("Unable to migrate %s schema: %s", TRADES_CSV, exc)
    
    DECISIONS_COLUMNS = [
        'timestamp', 'coin', 'signal', 'reasoning', 'confidence',
        'confluence_tags', 'trigger_tags', 'reasoning_categories'
    ]
    if not DECISIONS_CSV.exists():
        with open(DECISIONS_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(DECISIONS_COLUMNS)
    else:
        try:
            df = pd.read_csv(DECISIONS_CSV, encoding='utf-8')
            if list(df.columns) != DECISIONS_COLUMNS:
                for col in DECISIONS_COLUMNS:
                    if col not in df.columns:
                        df[col] = ""
                df = df[DECISIONS_COLUMNS]
                df.to_csv(DECISIONS_CSV, index=False, encoding='utf-8')
        except Exception as exc:
            logging.warning("Unable to migrate %s schema: %s", DECISIONS_CSV, exc)

    if not MESSAGES_CSV.exists():
        with open(MESSAGES_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'direction', 'role', 'content', 'metadata'
            ])

def get_btc_benchmark_price() -> Optional[float]:
    """Fetch the current BTC/USDT price for benchmarking."""
    global last_btc_price
    # Skip benchmark if not in crypto mode or BTCUSDT data is missing
    if "BTCUSDT" not in SYMBOLS and not any("BTC" in s for s in SYMBOLS):
        return None
        
    data = fetch_market_data("BTCUSDT")
    if data and "price" in data:
        try:
            last_btc_price = float(data["price"])
        except (TypeError, ValueError):
            logging.debug("Received non-numeric BTC price: %s", data["price"])
    return last_btc_price

def log_portfolio_state() -> None:
    """Log current portfolio state."""
    total_equity = calculate_total_equity()
    total_return = ((total_equity - START_CAPITAL) / START_CAPITAL) * 100
    total_margin = calculate_total_margin()
    net_unrealized = total_equity - balance - total_margin
    
    position_details = "; ".join([
        f"{coin}:{pos['side']}:{pos['quantity']:.4f}@{pos['entry_price']:.4f}"
        for coin, pos in positions.items()
    ]) if positions else "No positions"
    
    btc_price = get_btc_benchmark_price()
    btc_price_str = f"{btc_price:.2f}" if btc_price is not None else ""

    with open(STATE_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            get_current_time().isoformat(),
            f"{balance:.2f}",
            f"{total_equity:.2f}",
            f"{total_return:.2f}",
            len(positions),
            position_details,
            f"{total_margin:.2f}",
            f"{net_unrealized:.2f}",
            btc_price_str,
        ])

def log_trade(coin: str, action: str, details: Dict[str, Any]) -> None:
    """Log trade execution."""
    with open(TRADES_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            get_current_time().isoformat(),
            coin,
            action,
            details.get('side', ''),
            details.get('quantity', 0),
            details.get('price', 0),
            details.get('profit_target', 0),
            details.get('stop_loss', 0),
            details.get('leverage', 1),
            details.get('confidence', 0),
            details.get('pnl', 0),
            balance,
            details.get('reason', ''),
            details.get('confluence_tags', ''),
            details.get('trigger_tags', ''),
            details.get('reasoning_categories', '')
        ])

def log_ai_decision(coin: str, signal: str, reasoning: str, confidence: float, confluence_tags: str = '', trigger_tags: str = '', reasoning_categories: str = '') -> None:
    """Log AI decision."""
    with open(DECISIONS_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            get_current_time().isoformat(),
            coin,
            signal,
            reasoning,
            confidence,
            confluence_tags,
            trigger_tags,
            reasoning_categories
        ])


def log_ai_message(direction: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Log raw messages exchanged with the AI provider."""
    with open(MESSAGES_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            get_current_time().isoformat(),
            direction,
            role,
            content,
            json.dumps(metadata) if metadata else ""
        ])

def strip_ansi_codes(text: str) -> str:
    """Remove ANSI color codes so Telegram receives plain text."""
    return ANSI_ESCAPE_RE.sub("", text)

def escape_markdown(text: str) -> str:
    """Escape characters that have special meaning in Telegram Markdown."""
    if not text:
        return text
    specials = r"_*[]()~`>#+-=|{}.!\\"
    return "".join(f"\\{char}" if char in specials else char for char in text)

def record_iteration_message(text: str) -> None:
    """Record console output for this iteration to share via Telegram."""
    if current_iteration_messages is not None:
        current_iteration_messages.append(strip_ansi_codes(text).rstrip())

def send_telegram_message(text: str, chat_id: Optional[str] = None, parse_mode: Optional[str] = None) -> None:
    """Send a notification message to Telegram if credentials are configured.

    If `chat_id` is provided it will be used; otherwise `TELEGRAM_CHAT_ID` is used.
    This allows sending different message types to a dedicated signals group (`TELEGRAM_SIGNALS_CHAT_ID`).
    """
    effective_chat = (chat_id or TELEGRAM_CHAT_ID or "").strip()
    if not TELEGRAM_BOT_TOKEN or not effective_chat:
        return

    try:
        payload = {
            "chat_id": effective_chat,
            "text": text,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        
        response = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json=payload,
            timeout=10,
        )
        if response.status_code == 200:
            return

        response_text_lower = response.text.lower()
        logging.warning(
            "Telegram notification failed (%s): %s",
            response.status_code,
            response.text,
        )
        if (
            response.status_code == 400
            and "can't parse entities" in response_text_lower
            and parse_mode
        ):
            fallback_payload = {
                "chat_id": effective_chat,
                "text": strip_ansi_codes(text),
            }
            try:
                fallback_response = requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                    json=fallback_payload,
                    timeout=10,
                )
                if fallback_response.status_code != 200:
                    logging.warning(
                        "Telegram fallback notification failed (%s): %s",
                        fallback_response.status_code,
                        fallback_response.text,
                    )
            except Exception as fallback_exc:
                logging.error("Fallback Telegram message failed: %s", fallback_exc)
    except Exception as exc:
        logging.error("Error sending Telegram message: %s", exc)
        
def notify_error(
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    log_error: bool = True,
) -> None:
    """Log an error and forward a brief description to Telegram."""
    if log_error:
        logging.error(message)
    log_ai_message(
        direction="error",
        role="system",
        content=message,
        metadata=metadata,
    )
    send_telegram_message(message, parse_mode=None)

# ───────────────────────── STATE MGMT ───────────────────────

def load_state() -> None:
    """Load persisted balance and positions if available."""
    global balance, positions, iteration_counter

    if not STATE_JSON.exists():
        logging.info("No existing state file found; starting fresh.")
        return

    try:
        with open(STATE_JSON, "r") as f:
            data = json.load(f)

        balance = float(data.get("balance", START_CAPITAL))
        try:
            iteration_counter = int(data.get("iteration", 0))
        except (TypeError, ValueError):
            iteration_counter = 0
        loaded_positions = data.get("positions", {})
        if isinstance(loaded_positions, dict):
            restored_positions: Dict[str, Dict[str, Any]] = {}
            for coin, pos in loaded_positions.items():
                if not isinstance(pos, dict):
                    continue
                fees_paid_raw = pos.get("fees_paid", pos.get("entry_fee", 0.0))
                if fees_paid_raw is None:
                    fees_paid_value = 0.0
                else:
                    try:
                        fees_paid_value = float(fees_paid_raw)
                    except (TypeError, ValueError):
                        fees_paid_value = 0.0
                risk_usd_raw = pos.get("risk_usd", 0.0)
                try:
                    risk_usd_value = float(risk_usd_raw)
                except (TypeError, ValueError):
                    risk_usd_value = 0.0
                initial_stop_raw = pos.get("initial_stop", pos.get("stop_loss", 0.0))
                try:
                    initial_stop_value = float(initial_stop_raw)
                except (TypeError, ValueError):
                    initial_stop_value = float(pos.get("stop_loss", 0.0))
                initial_risk_per_unit_raw = pos.get("initial_risk_per_unit", 0.0)
                try:
                    initial_risk_per_unit_value = float(initial_risk_per_unit_raw)
                except (TypeError, ValueError):
                    initial_risk_per_unit_value = 0.0
                initial_risk_usd_raw = pos.get("initial_risk_usd", risk_usd_value)
                try:
                    initial_risk_usd_value = float(initial_risk_usd_raw)
                except (TypeError, ValueError):
                    initial_risk_usd_value = risk_usd_value
                trail_history_raw = pos.get("trail_history", [])
                trail_history_value = [
                    hist for hist in trail_history_raw
                    if isinstance(hist, dict)
                ]

                fee_rate_raw = pos.get("fee_rate", TAKER_FEE_RATE)
                try:
                    fee_rate_value = float(fee_rate_raw)
                except (TypeError, ValueError):
                    fee_rate_value = TAKER_FEE_RATE

                restored_positions[coin] = {
                    "side": pos.get("side", "long"),
                    "quantity": float(pos.get("quantity", 0.0)),
                    "entry_price": float(pos.get("entry_price", 0.0)),
                    "profit_target": float(pos.get("profit_target", 0.0)),
                    "stop_loss": float(pos.get("stop_loss", 0.0)),
                    "leverage": float(pos.get("leverage", 1)),
                    "confidence": float(pos.get("confidence", 0.0)),
                    "invalidation_condition": pos.get("invalidation_condition", ""),
                    "margin": float(pos.get("margin", 0.0)),
                    "fees_paid": fees_paid_value,
                    "fee_rate": fee_rate_value,
                    "liquidity": pos.get("liquidity", "taker"),
                    "entry_justification": pos.get("entry_justification", ""),
                    "last_justification": pos.get("last_justification", pos.get("entry_justification", "")),
                    "risk_usd": risk_usd_value,
                    "trade_type": pos.get("trade_type"),
                    "trail_phase": pos.get("trail_phase", "Phase 1"),
                    "initial_stop": initial_stop_value,
                    "initial_risk_per_unit": initial_risk_per_unit_value,
                    "initial_risk_usd": initial_risk_usd_value,
                    "trail_history": trail_history_value,
                }
            positions = restored_positions
        logging.info(
            "Loaded state from %s (balance: %.2f, positions: %d)",
            STATE_JSON,
            balance,
            len(positions),
        )
    except Exception as e:
        logging.error("Failed to load state from %s: %s", STATE_JSON, e, exc_info=True)
        balance = START_CAPITAL
        positions = {}

def save_state() -> None:
    """Persist current balance, open positions, and iteration counter."""
    try:
        # Calculate performance metrics for the state file
        trade_stats = summarize_trades(TRADES_CSV)
        total_equity = calculate_total_equity()
        total_net_profit = total_equity - START_CAPITAL
        mdd = calculate_max_drawdown(equity_history)
        
        recovery_factor = 0.0
        if mdd is not None and mdd > 0:
            max_dd_amount = START_CAPITAL * mdd
            recovery_factor = total_net_profit / max_dd_amount if max_dd_amount > 0 else 0.0

        performance = {
            "total_net_profit": total_net_profit,
            "total_return_pct": (total_net_profit / START_CAPITAL * 100) if START_CAPITAL else 0.0,
            "max_drawdown_pct": (mdd * 100) if mdd is not None else 0.0,
            "recovery_factor": recovery_factor,
            "profit_factor": trade_stats.get("profit_factor"),
            "win_rate_pct": trade_stats.get("win_rate_pct"),
            "total_trades": trade_stats.get("total_trades"),
            "winning_trades": trade_stats.get("winning_trades"),
            "losing_trades": trade_stats.get("losing_trades"),
        }

        with open(STATE_JSON, "w", encoding='utf-8') as f:
            json.dump(
                {
                    "balance": balance,
                    "positions": positions,
                    "performance": performance,
                    "iteration": iteration_counter,
                    "updated_at": get_current_time().isoformat(),
                },
                f,
                indent=2,
            )
    except Exception as e:
        logging.error("Failed to save state to %s: %s", STATE_JSON, e, exc_info=True)


def reset_state(initial_balance: Optional[float] = None) -> None:
    """Reset in-memory trading state to start a fresh run."""
    global balance, positions, trade_history, iteration_counter, equity_history, invocation_count, current_iteration_messages, BOT_START_TIME
    balance = float(initial_balance) if initial_balance is not None else START_CAPITAL
    positions = {}
    trade_history = []
    iteration_counter = 0
    invocation_count = 0
    equity_history.clear()
    current_iteration_messages = []
    BOT_START_TIME = get_current_time()


def load_equity_history() -> None:
    """Populate the in-memory equity history for performance calculations."""
    equity_history.clear()
    if not STATE_CSV.exists():
        return
    try:
        df = pd.read_csv(STATE_CSV, usecols=["total_equity"])
    except ValueError:
        logging.warning(
            "%s missing 'total_equity' column; Sortino ratio unavailable until new data is logged.",
            STATE_CSV,
        )
        return
    except Exception as exc:
        logging.warning("Unable to load historical equity data: %s", exc)
        return

    values = pd.to_numeric(df["total_equity"], errors="coerce").dropna()
    if not values.empty:
        equity_history.extend(float(v) for v in values.tolist())

def register_equity_snapshot(total_equity: float) -> None:
    """Append the latest equity to the history if it is a finite value."""
    if total_equity is None:
        return
    if isinstance(total_equity, (int, float, np.floating)) and np.isfinite(total_equity):
        equity_history.append(float(total_equity))

# ───────────────────────── INDICATORS ───────────────────────

def calculate_rsi_series(close: pd.Series, period: int) -> pd.Series:
    """Return RSI series for specified period using Wilder's smoothing."""
    delta = close.astype(float).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    alpha = 1 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def add_indicator_columns(
    df: pd.DataFrame,
    ema_lengths: Iterable[int] = (EMA_LEN,),
    rsi_periods: Iterable[int] = (RSI_LEN,),
    macd_params: Iterable[int] = (MACD_FAST, MACD_SLOW, MACD_SIGNAL),
) -> pd.DataFrame:
    """Return copy of df with EMA, RSI, and MACD columns added."""
    ema_lengths = tuple(dict.fromkeys(ema_lengths))  # remove duplicates, preserve order
    rsi_periods = tuple(dict.fromkeys(rsi_periods))
    fast, slow, signal = macd_params

    result = df.copy()
    close = result["close"]

    for span in ema_lengths:
        result[f"ema{span}"] = close.ewm(span=span, adjust=False).mean()

    for period in rsi_periods:
        result[f"rsi{period}"] = calculate_rsi_series(close, period)

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    result["macd"] = macd_line
    result["macd_signal"] = macd_line.ewm(span=signal, adjust=False).mean()

    return result


def calculate_atr_series(df: pd.DataFrame, period: int) -> pd.Series:
    """Return Average True Range series for the provided period."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr_components = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    alpha = 1 / period
    return true_range.ewm(alpha=alpha, adjust=False).mean()


def calculate_adx_series(df: pd.DataFrame, period: int) -> pd.Series:
    """Return Average Directional Index (ADX) series."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_high = high.shift(1)
    prev_low = low.shift(1)

    plus_dm = (high - prev_high).where((high - prev_high) > (prev_low - low), 0.0)
    plus_dm = plus_dm.where(plus_dm > 0, 0.0)
    minus_dm = (prev_low - low).where((prev_low - low) > (high - prev_high), 0.0)
    minus_dm = minus_dm.where(minus_dm > 0, 0.0)

    tr_components = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)

    alpha = 1 / period
    smoothed_tr = true_range.ewm(alpha=alpha, adjust=False).mean().replace(0, np.nan)
    smoothed_plus_dm = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
    minus_di = 100 * (smoothed_minus_dm / smoothed_tr)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    return dx.ewm(alpha=alpha, adjust=False).mean()


def calculate_indicators(df: pd.DataFrame) -> pd.Series:
    """Calculate technical indicators and return the latest row."""
    enriched = add_indicator_columns(
        df,
        ema_lengths=(EMA_LEN,),
        rsi_periods=(RSI_LEN,),
        macd_params=(MACD_FAST, MACD_SLOW, MACD_SIGNAL),
    )
    enriched["rsi"] = enriched[f"rsi{RSI_LEN}"]
    enriched["atr"] = calculate_atr_series(enriched, 14)
    return enriched.iloc[-1]

def fetch_market_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch current market data for a symbol."""
    binance_client = get_binance_client()
    if not binance_client:
        logging.warning("Skipping market data fetch for %s: Binance client unavailable.", symbol)
        return None

    try:
        # Get recent klines
        klines = binance_client.get_klines(symbol=symbol, interval=INTERVAL, limit=50)

        df = pd.DataFrame(
            klines,
            columns=[
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
            ],
        )

        if df.empty:
            logging.warning("Market data for %s is empty; skipping.", symbol)
            return None

        df["close"] = df["close"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["open"] = df["open"].astype(float)

        last = calculate_indicators(df)
        latest_bar = df.iloc[-1]
        last_high = float(latest_bar["high"])
        last_low = float(latest_bar["low"])
        last_close = float(latest_bar["close"])

        # Get funding rate for perpetual futures
        try:
            funding_info = binance_client.futures_funding_rate(symbol=symbol, limit=1)
            funding_rate = float(funding_info[0]["fundingRate"]) if funding_info else 0
        except:
            funding_rate = 0

        return {
            "symbol": symbol,
            "price": last_close,
            "high": last_high,
            "low": last_low,
            "ema20": last["ema20"],
            "rsi": last["rsi"],
            "macd": last["macd"],
            "macd_signal": last["macd_signal"],
            "funding_rate": funding_rate,
            "atr": last["atr"] if "atr" in last else 0.0,
        }
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return None


def round_series(values: Iterable[Any], precision: int) -> List[float]:
    """Round numeric iterable to the given precision, skipping NaNs."""
    rounded: List[float] = []
    for value in values:
        try:
            if pd.isna(value):
                continue
        except TypeError:
            # Non-numeric/NA sentinel types fall back to ValueError later
            pass
        try:
            rounded.append(round(float(value), precision))
        except (TypeError, ValueError):
            continue
    return rounded


def collect_prompt_market_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Return rich market snapshot for prompt composition."""
    binance_client = get_binance_client()
    if not binance_client:
        return None

    try:
        execution_klines = binance_client.get_klines(symbol=symbol, interval=INTERVAL, limit=200)
        df_execution = pd.DataFrame(
            execution_klines,
            columns=[
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
            ],
        )
        if df_execution.empty:
            logging.warning("Skipping market snapshot for %s: execution klines unavailable.", symbol)
            return None

        numeric_cols = ["open", "high", "low", "close", "volume"]
        df_execution[numeric_cols] = df_execution[numeric_cols].astype(float)
        df_execution["mid_price"] = (df_execution["high"] + df_execution["low"]) / 2
        df_execution = add_indicator_columns(
            df_execution,
            ema_lengths=(EMA_LEN,),
            rsi_periods=(RSI_LEN,),
            macd_params=(MACD_FAST, MACD_SLOW, MACD_SIGNAL),
        )
        df_execution["atr"] = calculate_atr_series(df_execution, 14)
        df_execution["atr_long"] = calculate_atr_series(df_execution, 100)
        latest_atr = float(df_execution["atr"].iloc[-1])
        latest_atr_long = float(df_execution["atr_long"].iloc[-1]) if not pd.isna(df_execution["atr_long"].iloc[-1]) else 0.0
        volatility_ratio = latest_atr / latest_atr_long if latest_atr_long > 0 else 1.0

        structure_klines = binance_client.get_klines(symbol=symbol, interval="1h", limit=100)
        df_structure = pd.DataFrame(
            structure_klines,
            columns=[
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
            ],
        )
        if df_structure.empty:
            logging.warning("Skipping market snapshot for %s: structure klines unavailable.", symbol)
            return None
        df_structure[numeric_cols] = df_structure[numeric_cols].astype(float)
        df_structure = add_indicator_columns(
            df_structure,
            ema_lengths=(20, 50),
            rsi_periods=(14,),
            macd_params=(MACD_FAST, MACD_SLOW, MACD_SIGNAL),
        )
        df_structure["atr"] = calculate_atr_series(df_structure, 14)
        df_structure["swing_high"] = df_structure["high"].rolling(window=5, center=True).max()
        df_structure["swing_low"] = df_structure["low"].rolling(window=5, center=True).min()
        df_structure["volume_sma"] = df_structure["volume"].rolling(window=20).mean()
        df_structure["volume_ratio"] = df_structure["volume"] / df_structure["volume_sma"].replace(0, np.nan)

        trend_klines = binance_client.get_klines(symbol=symbol, interval="4h", limit=100)
        df_trend = pd.DataFrame(
            trend_klines,
            columns=[
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
            ],
        )
        if df_trend.empty:
            logging.warning("Skipping market snapshot for %s: trend klines unavailable.", symbol)
            return None
        df_trend[numeric_cols] = df_trend[numeric_cols].astype(float)
        df_trend = add_indicator_columns(
            df_trend,
            ema_lengths=(20, 50, 200),
            rsi_periods=(14,),
            macd_params=(MACD_FAST, MACD_SLOW, MACD_SIGNAL),
        )
        df_trend["macd_histogram"] = df_trend["macd"] - df_trend["macd_signal"]
        df_trend["atr"] = calculate_atr_series(df_trend, 14)
        df_trend["macd_histogram_avg"] = df_trend["macd_histogram"].abs().rolling(window=20, min_periods=5).mean()
        df_trend["adx"] = calculate_adx_series(df_trend, 14)

        latest_trend = df_trend.iloc[-1]
        macd_hist_value = float(latest_trend["macd_histogram"])
        macd_hist_avg = float(df_trend["macd_histogram_avg"].iloc[-1]) if not pd.isna(
            df_trend["macd_histogram_avg"].iloc[-1]
        ) else 0.0
        macd_ratio = macd_hist_value / macd_hist_avg if macd_hist_avg else 0.0
        rsi_component = (float(latest_trend["rsi14"]) - 50.0) / 50.0
        adx_value = float(latest_trend.get("adx", 0.0))
        adx_component = adx_value / 25.0 if adx_value else 0.0
        ema_component = 1.0 if latest_trend["ema20"] > latest_trend["ema50"] else 0.0
        trend_strength = (
            ema_component * 0.30
            + macd_ratio * 0.30
            + rsi_component * 0.20
            + adx_component * 0.20
        )

        try:
            oi_hist = binance_client.futures_open_interest_hist(symbol=symbol, period="5m", limit=30)
            open_interest_values = [float(entry["sumOpenInterest"]) for entry in oi_hist]
        except Exception as exc:
            logging.debug("Open interest history unavailable for %s: %s", symbol, exc)
            open_interest_values = []

        try:
            funding_hist = binance_client.futures_funding_rate(symbol=symbol, limit=30)
            funding_rates = [float(entry["fundingRate"]) for entry in funding_hist]
        except Exception as exc:
            logging.debug("Funding rate history unavailable for %s: %s", symbol, exc)
            funding_rates = []

        funding_latest = funding_rates[-1] if funding_rates else 0.0
        price = float(df_execution["close"].iloc[-1])

        exec_tail = df_execution.tail(10)
        struct_tail = df_structure.tail(10)
        trend_tail = df_trend.tail(10)

        open_interest_latest = open_interest_values[-1] if open_interest_values else None
        open_interest_average = float(np.mean(open_interest_values)) if open_interest_values else None

        return {
            "symbol": symbol,
            "coin": SYMBOL_TO_COIN[symbol],
            "price": price,
            "execution": {
                "ema20": float(df_execution["ema20"].iloc[-1]),
                "rsi14": float(df_execution["rsi14"].iloc[-1]),
                "macd": float(df_execution["macd"].iloc[-1]),
                "macd_signal": float(df_execution["macd_signal"].iloc[-1]),
                "atr": float(df_execution["atr"].iloc[-1]),
                "series": {
                    "mid_prices": round_series(exec_tail["mid_price"], 3),
                    "ema20": round_series(exec_tail["ema20"], 3),
                    "macd": round_series(exec_tail["macd"], 3),
                    "rsi14": round_series(exec_tail["rsi14"], 3),
                    "atr": round_series(exec_tail["atr"], 3),
                },
            },
            "structure": {
                "ema20": float(df_structure["ema20"].iloc[-1]),
                "ema50": float(df_structure["ema50"].iloc[-1]),
                "rsi14": float(df_structure["rsi14"].iloc[-1]),
                "macd": float(df_structure["macd"].iloc[-1]),
                "macd_signal": float(df_structure["macd_signal"].iloc[-1]),
                "swing_high": float(df_structure["swing_high"].iloc[-1]),
                "swing_low": float(df_structure["swing_low"].iloc[-1]),
                "volume_ratio": float(df_structure["volume_ratio"].iloc[-1]),
                "atr": float(df_structure["atr"].iloc[-1]),
                "series": {
                    "close": round_series(struct_tail["close"], 3),
                    "ema20": round_series(struct_tail["ema20"], 3),
                    "ema50": round_series(struct_tail["ema50"], 3),
                    "rsi14": round_series(struct_tail["rsi14"], 3),
                    "macd": round_series(struct_tail["macd"], 3),
                    "swing_high": round_series(struct_tail["swing_high"], 3),
                    "swing_low": round_series(struct_tail["swing_low"], 3),
                    "atr": round_series(struct_tail["atr"], 3),
                },
            },
            "trend": {
                "ema20": float(df_trend["ema20"].iloc[-1]),
                "ema50": float(df_trend["ema50"].iloc[-1]),
                "ema200": float(df_trend["ema200"].iloc[-1]),
                "rsi14": float(df_trend["rsi14"].iloc[-1]),
                "macd": float(df_trend["macd"].iloc[-1]),
                "macd_signal": float(df_trend["macd_signal"].iloc[-1]),
                "macd_histogram": float(df_trend["macd_histogram"].iloc[-1]),
                "atr": float(df_trend["atr"].iloc[-1]),
                "macd_histogram_avg": float(df_trend["macd_histogram_avg"].iloc[-1]) if not pd.isna(
                    df_trend["macd_histogram_avg"].iloc[-1]
                ) else 0.0,
                "adx": float(df_trend["adx"].iloc[-1]) if not pd.isna(df_trend["adx"].iloc[-1]) else 0.0,
                "trend_strength": float(trend_strength),
                "current_volume": float(df_trend["volume"].iloc[-1]),
                "average_volume": float(df_trend["volume"].mean()),
                "series": {
                    "close": round_series(trend_tail["close"], 3),
                    "ema20": round_series(trend_tail["ema20"], 3),
                    "ema50": round_series(trend_tail["ema50"], 3),
                    "macd": round_series(trend_tail["macd"], 3),
                    "rsi14": round_series(trend_tail["rsi14"], 3),
                    "macd_histogram": round_series(trend_tail["macd_histogram"], 3),
                    "adx": round_series(trend_tail["adx"], 3),
                },
            },
            "funding_rate": funding_latest,
            "funding_rates": funding_rates,
            "open_interest": {
                "latest": open_interest_latest,
                "average": open_interest_average,
            },
            "trend_strength": float(trend_strength),
            "trend_components": {
                "ema_component": float(ema_component),
                "macd_ratio": float(macd_ratio),
                "rsi_component": float(rsi_component),
                "adx_component": float(adx_component),
            },
            "volatility_ratio": float(volatility_ratio),
        }
    except Exception as exc:
        logging.error("Failed to build market snapshot for %s: %s", symbol, exc, exc_info=True)
        return None

# ───────────────────── AI DECISION MAKING ───────────────────

def format_trading_prompt() -> str:
    """Compose a rich prompt for the trading agent."""
    global invocation_count
    invocation_count += 1

    now = get_current_time()
    minutes_running = int((now - BOT_START_TIME).total_seconds() // 60)

    market_snapshots: Dict[str, Dict[str, Any]] = {}
    for symbol in SYMBOLS:
        snapshot = collect_prompt_market_data(symbol)
        if snapshot:
            market_snapshots[snapshot["coin"]] = snapshot

    total_margin = calculate_total_margin()
    total_equity = balance + total_margin
    for coin, pos in positions.items():
        current_price = market_snapshots.get(coin, {}).get("price", pos["entry_price"])
        total_equity += calculate_unrealized_pnl(coin, current_price)

    total_return = ((total_equity - START_CAPITAL) / START_CAPITAL) * 100 if START_CAPITAL else 0.0
    net_unrealized_total = total_equity - balance - total_margin

    def fmt(value: Optional[float], digits: int = 3) -> str:
        if value is None:
            return "N/A"
        try:
            if pd.isna(value):
                return "N/A"
        except TypeError:
            pass
        return f"{value:.{digits}f}"

    def fmt_rate(value: Optional[float]) -> str:
        if value is None:
            return "N/A"
        try:
            if pd.isna(value):
                return "N/A"
        except TypeError:
            pass
        return f"{value:.6g}"

    weekly_dd = calculate_weekly_drawdown()
    drawdown_protection_active = weekly_dd >= 0.03

    prompt_lines: List[str] = []
    prompt_lines.append(
        f"It has been {minutes_running} minutes since you started trading. "
        f"The current time is {now.isoformat()} and you've been invoked {invocation_count} times. "
        "Below, we are providing you with a variety of state data, price data, and predictive signals so you can discover alpha. "
        "Below that is your current account information, value, performance, positions, etc."
    )
    prompt_lines.append("ALL PRICE OR SIGNAL SERIES BELOW ARE ORDERED OLDEST → NEWEST.")
    prompt_lines.append(
        f"Timeframe note: Execution uses {INTERVAL} candles, Structure uses 1h candles, Trend uses 4h candles."
    )
    prompt_lines.append("-" * 80)
    prompt_lines.append("GLOBAL RISK & VOLATILITY CONTROLS")
    prompt_lines.append(f"- Current Weekly Drawdown: {weekly_dd * 100:.2f}%")
    prompt_lines.append(f"- Drawdown Protection Status: {'ACTIVE (Pause Entries - Red Zone Mode)' if drawdown_protection_active else 'INACTIVE'}")
    prompt_lines.append("-" * 80)
    prompt_lines.append("CURRENT MARKET STATE FOR ALL COINS (Multi-Timeframe Analysis)")

    # Build market snapshot section
    is_daily_only = (INTERVAL == "1d")
    
    for symbol in SYMBOLS:
        coin = SYMBOL_TO_COIN.get(symbol, symbol)
        data = market_snapshots.get(coin)
        if not data:
            continue

        execution = data["execution"]
        structure = data["structure"]
        trend = data["trend"]
        trend_components = data.get("trend_components", {})
        open_interest = data.get("open_interest", {})
        funding_rates = data.get("funding_rates", [])
        funding_avg_str = fmt_rate(float(np.mean(funding_rates))) if funding_rates else "N/A"
        
        vr = data.get("volatility_ratio", 1.0)
        if vr < 1.6:
            vr_zone = "ACTIVE (Green Zone - High Frequency, Normal Risk)"
        elif vr <= 2.2:
            vr_zone = "DEFENSIVE (Yellow Zone - Moderate Frequency, 50% Reduced Position Size/Risk)"
        else:
            vr_zone = "PAUSE (Red Zone - Zero Frequency, Pause Entries)"

        prompt_lines.append(f"\n{coin} MARKET SNAPSHOT")
        prompt_lines.append(f"Current Price: {fmt(data['price'], 3)}")
        prompt_lines.append(f"  Volatility Ratio (VR): {fmt(vr, 2)}")
        prompt_lines.append(f"  Volatility Zone: {vr_zone}")
        
        if is_daily_only:
            # Optimized 1D prompt: Skip redundant timeframe labels
            prompt_lines.append(f"  ANALYSIS (1D):")
            prompt_lines.append(f"    EMA Alignment: EMA20={fmt(trend['ema20'], 3)}, EMA50={fmt(trend['ema50'], 3)}, EMA200={fmt(trend['ema200'], 3)}")
            prompt_lines.append(f"    Indicators: RSI14={fmt(trend['rsi14'], 2)}, MACD={fmt(trend['macd'], 3)}, ATR14={fmt(trend['atr'], 3)}, ADX14={fmt(trend.get('adx'), 2)}")
            
            if trend["macd"] > trend["macd_signal"]:
                macd_direction = "bullish"
            elif trend["macd"] < trend["macd_signal"]:
                macd_direction = "bearish"
            else:
                macd_direction = "neutral"
            prompt_lines.append(f"    MACD Crossover: {macd_direction}")

            prompt_lines.append(
                f"    Trend Strength Score: {fmt(data.get('trend_strength'), 2)} "
                f"(EMA {fmt(trend_components.get('ema_component'), 2)}, "
                f"MACD ratio {fmt(trend_components.get('macd_ratio'), 2)}, "
                f"RSI {fmt(trend_components.get('rsi_component'), 2)}, "
                f"ADX {fmt(trend_components.get('adx_component'), 2)})"
            )
            prompt_lines.append(f"    History (last 10): {json.dumps(trend['series']['close'])}")
        else:
            # Original Multi-timeframe prompt
            prompt_lines.append(f"\n  4H TREND TIMEFRAME:")
            prompt_lines.append(f"    EMA Alignment: EMA20={fmt(trend['ema20'], 3)}, EMA50={fmt(trend['ema50'], 3)}, EMA200={fmt(trend['ema200'], 3)}")
            ema_trend = (
                "BULLISH"
                if trend["ema20"] > trend["ema50"]
                else "BEARISH"
                if trend["ema20"] < trend["ema50"]
                else "NEUTRAL"
            )
            prompt_lines.append(f"    Trend Classification: {ema_trend}")
            prompt_lines.append(
                f"    MACD: {fmt(trend['macd'], 3)}, Signal: {fmt(trend['macd_signal'], 3)}, Histogram: {fmt(trend['macd_histogram'], 3)}"
            )
            prompt_lines.append(
                f"    MACD Histogram Avg (|20|): {fmt(trend.get('macd_histogram_avg'), 3)}"
            )
            prompt_lines.append(f"    RSI14: {fmt(trend['rsi14'], 2)}")
            prompt_lines.append(f"    ATR14: {fmt(trend['atr'], 3)}")
            prompt_lines.append(f"    ADX14: {fmt(trend.get('adx'), 2)}")
            prompt_lines.append(
                f"    Trend Strength Score: {fmt(trend.get('trend_strength'), 2)} "
                f"(EMA {fmt(trend_components.get('ema_component'), 2)}, "
                f"MACD ratio {fmt(trend_components.get('macd_ratio'), 2)}, "
                f"RSI {fmt(trend_components.get('rsi_component'), 2)}, "
                f"ADX {fmt(trend_components.get('adx_component'), 2)})"
            )
            prompt_lines.append(
                f"    Volume: Current {fmt(trend['current_volume'], 2)}, Average {fmt(trend['average_volume'], 2)}"
            )
            prompt_lines.append(
                f"    4H Series (last 10): Close={json.dumps(trend['series']['close'])}"
            )
            prompt_lines.append(
                f"                         EMA20={json.dumps(trend['series']['ema20'])}, EMA50={json.dumps(trend['series']['ema50'])}"
            )
            prompt_lines.append(
                f"                         MACD={json.dumps(trend['series']['macd'])}, RSI14={json.dumps(trend['series']['rsi14'])}"
            )

            prompt_lines.append(f"\n  1H STRUCTURE TIMEFRAME:")
            prompt_lines.append(
                f"    EMA20: {fmt(structure['ema20'], 3)}, EMA50: {fmt(structure['ema50'], 3)}"
            )
            struct_position = "above" if data["price"] > structure["ema20"] else "below"
            prompt_lines.append(f"    Price relative to 1H EMA20: {struct_position}")
            prompt_lines.append(
                f"    Swing High: {fmt(structure['swing_high'], 3)}, Swing Low: {fmt(structure['swing_low'], 3)}"
            )
            prompt_lines.append(f"    RSI14: {fmt(structure['rsi14'], 2)}")
            prompt_lines.append(
                f"    MACD: {fmt(structure['macd'], 3)}, Signal: {fmt(structure['macd_signal'], 3)}"
            )
            prompt_lines.append(f"    ATR14: {fmt(structure['atr'], 3)}")
            prompt_lines.append(f"    Volume Ratio: {fmt(structure['volume_ratio'], 2)}x (>1.5 = volume spike)")
            prompt_lines.append(
                f"    1H Series (last 10): Close={json.dumps(structure['series']['close'])}"
            )
            prompt_lines.append(
                f"                         EMA20={json.dumps(structure['series']['ema20'])}, EMA50={json.dumps(structure['series']['ema50'])}"
            )
            prompt_lines.append(
                f"                         Swing High={json.dumps(structure['series']['swing_high'])}, Swing Low={json.dumps(structure['series']['swing_low'])}"
            )
            prompt_lines.append(
                f"                         RSI14={json.dumps(structure['series']['rsi14'])}"
            )
            prompt_lines.append(
                f"                         ATR14={json.dumps(structure['series']['atr'])}"
            )

            prompt_lines.append(f"\n  {INTERVAL.upper()} EXECUTION TIMEFRAME:")
            prompt_lines.append(
                f"    EMA20: {fmt(execution['ema20'], 3)} (Price {'above' if data['price'] > execution['ema20'] else 'below'} EMA20)"
            )
            prompt_lines.append(
                f"    MACD: {fmt(execution['macd'], 3)}, Signal: {fmt(execution['macd_signal'], 3)}"
            )
            if execution["macd"] > execution["macd_signal"]:
                macd_direction = "bullish"
            elif execution["macd"] < execution["macd_signal"]:
                macd_direction = "bearish"
            else:
                macd_direction = "neutral"

            prompt_lines.append(f"    MACD Crossover: {macd_direction}")
            prompt_lines.append(f"    RSI14: {fmt(execution['rsi14'], 2)}")
            rsi_zone = (
                "oversold (<35)"
                if execution["rsi14"] < 35
                else "overbought (>65)"
                if execution["rsi14"] > 65
                else "neutral"
            )
            prompt_lines.append(f"    RSI Zone: {rsi_zone}")
            prompt_lines.append(f"    ATR14: {fmt(execution['atr'], 3)}")
            prompt_lines.append(
                f"    {INTERVAL.upper()} Series (last 10): Mid-Price={json.dumps(execution['series']['mid_prices'])}"
            )
            prompt_lines.append(
                f"                          EMA20={json.dumps(execution['series']['ema20'])}"
            )
            prompt_lines.append(
                f"                          MACD={json.dumps(execution['series']['macd'])}"
            )
            prompt_lines.append(
                f"                          RSI14={json.dumps(execution['series']['rsi14'])}"
            )
            prompt_lines.append(
                f"                          ATR14={json.dumps(execution['series']['atr'])}"
            )


        prompt_lines.append(f"\n  MARKET SENTIMENT:")
        prompt_lines.append(
            f"    Open Interest: Latest={fmt(open_interest.get('latest'), 2)}, Average={fmt(open_interest.get('average'), 2)}"
        )
        prompt_lines.append(
            f"    Funding Rate: Latest={fmt_rate(data['funding_rate'])}, Average={funding_avg_str}"
        )
        # Build trade history section (Feedback Loop)
        if TRADES_CSV.exists():
            try:
                trades_df = pd.read_csv(TRADES_CSV)
                if not trades_df.empty:
                    # Get last 10 close/partial close events
                    recent_outcomes = trades_df[trades_df["action"].isin(["CLOSE", "CLOSE_PARTIAL"])].tail(10)
                    if not recent_outcomes.empty:
                        prompt_lines.append("-" * 80)
                        prompt_lines.append("RECENT TRADE OUTCOMES (Feedback Loop - Learn from these!)")
                        for _, row in recent_outcomes.iterrows():
                            pnl_val = float(row.get("pnl", 0.0))
                            outcome = "[WIN]" if pnl_val > 0 else "[LOSS]" if pnl_val < 0 else "[BREAKEVEN]"
                            coin_val = row.get("coin", "N/A")
                            side_val = row.get("side", "N/A")
                            reason_val = row.get("reason", "N/A")
                            prompt_lines.append(f"{outcome} {coin_val} {side_val} | PnL: ${pnl_val:.2f} | Reason: {reason_val}")

                        # Calculate recent win rate
                        wins = (recent_outcomes["pnl"] > 0).sum()
                        total_recent = len(recent_outcomes)
                        recent_wr = (wins / total_recent) * 100
                        prompt_lines.append(f"Recent Win Rate (last {total_recent}): {recent_wr:.1f}%")
                        if recent_wr < 40:
                            prompt_lines.append("NOTICE: You are currently in a losing streak. The market may be volatile or your current strategy is being exploited. Consider being more selective (REJECT more) or widening stops.")
            except Exception as exc:
                logging.debug("Failed to read trade history for prompt: %s", exc)

        prompt_lines.append("-" * 80)
        prompt_lines.append("ACCOUNT INFORMATION AND PERFORMANCE")

    prompt_lines.append(f"- Total Return (%): {fmt(total_return, 2)}")
    prompt_lines.append(f"- Available Cash: {fmt(balance, 2)}")
    prompt_lines.append(f"- Margin Allocated: {fmt(total_margin, 2)}")
    prompt_lines.append(f"- Unrealized PnL: {fmt(net_unrealized_total, 2)}")
    prompt_lines.append(f"- Current Account Value: {fmt(total_equity, 2)}")
    prompt_lines.append("Open positions and performance details:")

    for coin, pos in positions.items():
        current_price = market_snapshots.get(coin, {}).get("price", pos["entry_price"])
        quantity = pos["quantity"]
        gross_unrealized = calculate_unrealized_pnl(coin, current_price)
        leverage = pos.get("leverage", 1) or 1
        if pos["side"] == "long":
            liquidation_price = pos["entry_price"] * max(0.0, 1 - 1 / leverage)
        else:
            liquidation_price = pos["entry_price"] * (1 + 1 / leverage)
        notional_value = quantity * current_price
        initial_stop = pos.get("initial_stop", pos.get("stop_loss"))
        initial_risk_per_unit = float(pos.get("initial_risk_per_unit", 0.0) or 0.0)
        if initial_risk_per_unit:
            if pos["side"] == "long":
                r_multiple = (current_price - pos["entry_price"]) / initial_risk_per_unit
            else:
                r_multiple = (pos["entry_price"] - current_price) / initial_risk_per_unit
        else:
            r_multiple = 0.0
        snapshot = market_snapshots.get(coin, {})
        execution = snapshot.get("execution", {})
        structure = snapshot.get("structure", {})
        trend = snapshot.get("trend", {})
        position_payload = {
            "symbol": coin,
            "side": pos["side"],
            "quantity": quantity,
            "entry_price": pos["entry_price"],
            "current_price": current_price,
            "liquidation_price": liquidation_price,
            "unrealized_pnl": gross_unrealized,
            "leverage": pos.get("leverage", 1),
            "trade_type": pos.get("trade_type"),
            "trail_phase": pos.get("trail_phase", "Phase 1"),
            "initial_stop": initial_stop,
            "initial_risk_per_unit": initial_risk_per_unit,
            "initial_risk_usd": pos.get("initial_risk_usd"),
            "current_r_multiple": r_multiple,
            "exit_plan": {
                "profit_target": pos.get("profit_target"),
                "stop_loss": pos.get("stop_loss"),
                "invalidation_condition": pos.get("invalidation_condition"),
            },
            "confidence": pos.get("confidence", 0.0),
            "risk_usd": pos.get("risk_usd"),
            "sl_oid": pos.get("sl_oid", -1),
            "tp_oid": pos.get("tp_oid", -1),
            "wait_for_fill": pos.get("wait_for_fill", False),
            "entry_oid": pos.get("entry_oid", -1),
            "notional_usd": notional_value,
            "prompt_metrics": {
                "atr_15m": execution.get("atr"),
                "atr_1h": structure.get("atr"),
                "atr_4h": trend.get("atr"),
                "ema20_15m": execution.get("ema20"),
                "ema20_1h": structure.get("ema20"),
                "ema20_4h": trend.get("ema20"),
                "trend_strength": snapshot.get("trend_strength"),
            },
            "trail_history_tail": pos.get("trail_history", [])[-5:],
        }
        prompt_lines.append(f"{coin} position data: {json.dumps(position_payload)}")

    prompt_lines.append(
        """
INSTRUCTIONS:
Return ONLY valid JSON (no extra text). For each coin supply:
{
  "BNB": {
    "signal": "entry|hold|close",
    "side": "long|short",              // required for entry
    "quantity": 0.0,
    "profit_target": 0.0,
    "stop_loss": 0.0,
    "leverage": 3,
    "confidence": 0.75,
    "risk_usd": 150.0,
    "invalidation_condition": "1H close below 1080",
    "trade_type": "TYPE A|TYPE B|TYPE C",
    "phase": "Phase 1|Phase 2|Phase 3|Phase 4",
    "justification": "Concise multi-timeframe reasoning with rule references.",
    "confluence_tags": ["EMA_crossover", "RSI_oversold", "Volatility_spike"], // List of key confluences matched
    "trigger_tags": ["indicator_crossover", "support_bounce", "breakout"], // Trigger types
    "reasoning_categories": ["trend_following", "mean_reversion", "breakout"] // Analytical categories
  }
}
Optional for partial closes: include "close_fraction" (0-1), "close_percent", or "close_quantity" for the amount to exit.
Do not include commentary outside the JSON response.
""".strip()
    )

    return "\n".join(prompt_lines)

def format_prompt_for_deepseek() -> str:
    """Deprecated: use format_trading_prompt instead."""
    return format_trading_prompt()

def _recover_partial_decisions(json_str: str) -> Optional[Tuple[Dict[str, Any], List[str]]]:
    """Attempt to salvage individual coin decisions from truncated JSON."""
    coins = list(SYMBOL_TO_COIN.values())
    recovered: Dict[str, Any] = {}
    missing: List[str] = []

    for coin in coins:
        marker = f'"{coin}"'
        marker_idx = json_str.find(marker)
        if marker_idx == -1:
            missing.append(coin)
            continue

        obj_start = json_str.find('{', marker_idx)
        if obj_start == -1:
            missing.append(coin)
            continue

        depth = 0
        in_string = False
        escaped = False
        end_idx: Optional[int] = None

        for idx in range(obj_start, len(json_str)):
            char = json_str[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    end_idx = idx
                    break

        if end_idx is None:
            missing.append(coin)
            continue

        block = json_str[obj_start:end_idx + 1]
        try:
            recovered[coin] = json.loads(block)
        except json.JSONDecodeError:
            missing.append(coin)

    if not recovered:
        return None

    missing = list(dict.fromkeys(missing))

    fallback_message = "Missing data from truncated AI response; defaulting to hold."
    for coin in coins:
        if coin not in recovered:
            recovered[coin] = {
                "signal": "hold",
                "justification": fallback_message,
                "confidence": 0.0,
            }

    return recovered, missing

def call_llm_api(prompt: str) -> Optional[Dict[str, Any]]:
    """Call the configured LLM API (Google Gemini direct or OpenRouter)."""
    # Detect provider and call appropriate helper
    is_gemini_model = "gemini" in LLM_MODEL_NAME.lower()
    
    if is_gemini_model and GEMINI_API_KEY:
        return _call_google_gemini_api(prompt)
    else:
        return _call_openrouter_api(prompt)

def _call_google_gemini_api(prompt: str) -> Optional[Dict[str, Any]]:
    """Call Google AI Studio API directly for Gemini models."""
    try:
        # Standardize model name for Google API
        model_name = LLM_MODEL_NAME
        if "/" in model_name:
            model_name = model_name.split("/")[-1]
            if not model_name.startswith("gemini-"):
                model_name = f"gemini-{model_name}"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
        
        request_metadata = {
            "model": model_name,
            "temperature": LLM_TEMPERATURE,
            "max_output_tokens": LLM_MAX_TOKENS,
        }

        log_ai_message(
            direction="sent",
            role="system",
            content=TRADING_RULES_PROMPT,
            metadata=request_metadata,
        )
        log_ai_message(
            direction="sent",
            role="user",
            content=prompt,
            metadata=request_metadata,
        )

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"SYSTEM RULES:\n{TRADING_RULES_PROMPT}\n\nUSER PROMPT:\n{prompt}"}]
                }
            ],
            "generationConfig": {
                "temperature": LLM_TEMPERATURE,
                "maxOutputTokens": LLM_MAX_TOKENS,
            }
        }

        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code != 200:
            notify_error(
                f"Google Gemini API error: {response.status_code}",
                metadata={
                    "status_code": response.status_code,
                    "response_text": response.text,
                },
            )
            return None

        result = response.json()
        candidates = result.get("candidates")
        if not candidates:
            notify_error("Google Gemini API returned no candidates", metadata=result)
            return None

        content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        return _extract_json_from_llm_response(content, response.status_code, result.get("usageMetadata"))

    except Exception as e:
        logging.exception("Error calling Google Gemini API")
        notify_error(f"Error calling Google Gemini API: {e}")
        return None

def _call_openrouter_api(prompt: str) -> Optional[Dict[str, Any]]:
    """Call OpenRouter API with robust parameter handling."""
    try:
        request_payload: Dict[str, Any] = {
            "model": LLM_MODEL_NAME,
            "messages": [
                {"role": "system", "content": TRADING_RULES_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS
        }

        # Only add 'thinking' if explicitly configured and model supports it (usually DeepSeek)
        if LLM_THINKING_PARAM is not None:
            request_payload["thinking"] = LLM_THINKING_PARAM

        log_ai_message(
            direction="sent",
            role="system",
            content=TRADING_RULES_PROMPT,
            metadata=request_payload,
        )
        log_ai_message(
            direction="sent",
            role="user",
            content=prompt,
            metadata=request_payload,
        )

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/crypto-trading-bot",
                "X-Title": "AI Trading Bot",
            },
            json=request_payload,
            timeout=30
        )

        if response.status_code != 200:
            notify_error(
                f"OpenRouter API error: {response.status_code}",
                metadata={
                    "status_code": response.status_code,
                    "response_text": response.text,
                    "model": LLM_MODEL_NAME,
                },
            )
            return None

        result = response.json()
        choices = result.get("choices")
        if not choices:
            notify_error("OpenRouter API returned no choices", metadata=result)
            return None

        content = choices[0].get("message", {}).get("content", "")
        return _extract_json_from_llm_response(content, response.status_code, result.get("usage"), result.get("id"))

    except Exception as e:
        logging.exception("Error calling OpenRouter API")
        notify_error(f"Error calling OpenRouter API: {e}")
        return None

def _extract_json_from_llm_response(content: str, status_code: int, usage: Any = None, response_id: str = "") -> Optional[Dict[str, Any]]:
    """Sanitize and extract JSON from LLM response text."""
    log_ai_message(
        direction="received",
        role="assistant",
        content=content,
        metadata={
            "status_code": status_code,
            "response_id": response_id,
            "usage": usage,
        }
    )

    # Extract JSON from response (handle markdown blocks or extra text)
    start = content.find('{')
    end = content.rfind('}') + 1
    if start != -1 and end > start:
        json_str = content[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as decode_err:
            recovery = _recover_partial_decisions(json_str)
            if recovery:
                decisions, missing_coins = recovery
                logging.warning("Recovered partial JSON (missing: %s)", ", ".join(missing_coins))
                return decisions
            
            notify_error(f"JSON decode failed: {decode_err}", metadata={"raw_excerpt": json_str[:500]})
            return None
    else:
        notify_error("No JSON found in LLM response", metadata={"content_preview": content[:500]})
        return None

def call_deepseek_api(prompt: str) -> Optional[Dict[str, Any]]:
    """Deprecated: use call_llm_api instead."""
    return call_llm_api(prompt)

# ───────────────────── POSITION MANAGEMENT ──────────────────

def calculate_unrealized_pnl(coin: str, current_price: float) -> float:
    """Calculate unrealized PnL for a position."""
    if coin not in positions:
        return 0.0
    
    pos = positions[coin]
    if pos['side'] == 'long':
        pnl = (current_price - pos['entry_price']) * pos['quantity']
    else:  # short
        pnl = (pos['entry_price'] - current_price) * pos['quantity']
    
    return pnl

def calculate_net_unrealized_pnl(coin: str, current_price: float) -> float:
    """Calculate unrealized PnL after subtracting fees already paid."""
    gross_pnl = calculate_unrealized_pnl(coin, current_price)
    fees_paid = positions.get(coin, {}).get('fees_paid', 0.0)
    return gross_pnl - fees_paid

def calculate_pnl_for_price(pos: Dict[str, Any], target_price: float) -> float:
    """Return gross PnL for a hypothetical exit price."""
    try:
        quantity = float(pos.get('quantity', 0.0))
        entry_price = float(pos.get('entry_price', 0.0))
    except (TypeError, ValueError):
        return 0.0
    side = str(pos.get('side', 'long')).lower()
    if side == 'short':
        return (entry_price - target_price) * quantity
    return (target_price - entry_price) * quantity

def estimate_exit_fee(pos: Dict[str, Any], exit_price: float) -> float:
    """Estimate taker/maker fee required to exit the position at the given price."""
    try:
        quantity = float(pos.get('quantity', 0.0))
    except (TypeError, ValueError):
        quantity = 0.0
    fee_rate = pos.get('fee_rate', TAKER_FEE_RATE)
    try:
        fee_rate_value = float(fee_rate)
    except (TypeError, ValueError):
        fee_rate_value = TAKER_FEE_RATE
    estimated_fee = quantity * exit_price * fee_rate_value
    return max(estimated_fee, 0.0)

def format_leverage_display(leverage: Any) -> str:
    """Return leverage formatted as '<value>x' while handling strings gracefully."""
    if leverage is None:
        return "n/a"
    if isinstance(leverage, str):
        cleaned = leverage.strip()
        if not cleaned:
            return "n/a"
        if cleaned.lower().endswith('x'):
            return cleaned.lower()
        try:
            value = float(cleaned)
        except (TypeError, ValueError):
            return cleaned
    else:
        try:
            value = float(leverage)
        except (TypeError, ValueError):
            return str(leverage)
    if value.is_integer():
        return f"{int(value)}x"
    return f"{value:g}x"

def calculate_total_margin() -> float:
    """Return sum of margin allocated across all open positions."""
    return sum(float(pos.get('margin', 0.0)) for pos in positions.values())

def calculate_total_equity() -> float:
    """Calculate total equity (balance + unrealized PnL)."""
    total = balance + calculate_total_margin()
    
    for coin in positions:
        symbol = next((s for s, c in SYMBOL_TO_COIN.items() if c == coin), None)
        if not symbol:
            continue
        data = fetch_market_data(symbol)
        if data:
            total += calculate_unrealized_pnl(coin, data['price'])
    
    return total

def calculate_sortino_ratio(
    equity_values: Iterable[float],
    period_seconds: float,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> Optional[float]:
    """
    Compute the annualized Sortino ratio from equity snapshots.

    Args:
        equity_values: Sequence of equity values in chronological order.
        period_seconds: Average period between snapshots (used to annualize).
        risk_free_rate: Annualized risk-free rate (decimal form).
    """
    values = [float(v) for v in equity_values if isinstance(v, (int, float, np.floating)) and np.isfinite(v)]
    if len(values) < 2:
        return None

    returns = np.diff(values) / np.array(values[:-1], dtype=float)
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        return None

    period_seconds = float(period_seconds) if period_seconds and period_seconds > 0 else CHECK_INTERVAL
    periods_per_year = (365 * 24 * 60 * 60) / period_seconds
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        return None

    per_period_rf = risk_free_rate / periods_per_year
    excess_return = returns.mean() - per_period_rf
    if not np.isfinite(excess_return):
        return None

    downside_diff = np.minimum(returns - per_period_rf, 0.0)
    downside_squared = downside_diff ** 2
    downside_deviation = np.sqrt(np.mean(downside_squared))
    if downside_deviation <= 0 or not np.isfinite(downside_deviation):
        return None

    sortino = (excess_return / downside_deviation) * np.sqrt(periods_per_year)
    if not np.isfinite(sortino):
        return None
    return float(sortino)

def calculate_sharpe_ratio(
    equity_values: Iterable[float],
    period_seconds: float,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> Optional[float]:
    """
    Compute the annualized Sharpe ratio from equity snapshots.
    """
    values = [float(v) for v in equity_values if isinstance(v, (int, float, np.floating)) and np.isfinite(v)]
    if len(values) < 2:
        return None

    returns = np.diff(values) / np.array(values[:-1], dtype=float)
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        return None

    period_seconds = float(period_seconds) if period_seconds and period_seconds > 0 else CHECK_INTERVAL
    periods_per_year = (365 * 24 * 60 * 60) / period_seconds
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        return None

    per_period_rf = risk_free_rate / periods_per_year
    excess_returns = returns - per_period_rf
    mean_excess_return = excess_returns.mean()
    std_return = returns.std()
    
    if std_return <= 0 or not np.isfinite(std_return):
        return None

    sharpe = (mean_excess_return / std_return) * np.sqrt(periods_per_year)
    if not np.isfinite(sharpe):
        return None
    return float(sharpe)

def calculate_max_drawdown(equity_values: Iterable[float]) -> Optional[float]:
    """Compute the maximum drawdown as a decimal percentage (e.g., 0.1 for 10%)."""
    values = np.array([float(v) for v in equity_values if np.isfinite(v)], dtype=float)
    if values.size < 2:
        return None
    peaks = np.maximum.accumulate(values)
    # Avoid division by zero
    valid_peaks = np.where(peaks > 0, peaks, np.nan)
    drawdowns = (peaks - values) / valid_peaks
    mdd = float(np.nanmax(drawdowns)) if not np.all(np.isnan(drawdowns)) else 0.0
    return mdd

def calculate_weekly_drawdown() -> float:
    """Compute rolling drawdown over the last 7 items in equity_history (representing a week)."""
    if len(equity_history) < 2:
        return 0.0
    lookback = min(7, len(equity_history))
    recent_equity = equity_history[-lookback:]
    max_recent = max(recent_equity)
    current = recent_equity[-1]
    if max_recent <= 0:
        return 0.0
    dd = (max_recent - current) / max_recent
    return max(dd, 0.0)

def summarize_trades(trades_path: Path) -> Dict[str, Any]:
    """Calculate aggregate trade statistics from a trade history CSV."""
    empty_stats = {
        "total_trades": 0,
        "closed_trades": 0,
        "partial_closes": 0,
        "close_events": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "breakeven_trades": 0,
        "win_rate_pct": 0.0,
        "net_realized_pnl": 0.0,
        "gross_win": 0.0,
        "gross_loss": 0.0,
        "profit_factor": None,
        "avg_trade_pnl": None,
        "avg_holding_time_seconds": None,
        "max_consecutive_wins": 0,
        "max_consecutive_losses": 0,
    }

    if not trades_path.exists():
        return dict(empty_stats)

    try:
        df = pd.read_csv(trades_path)
    except Exception as exc:
        logging.warning("Unable to load trade history from %s: %s", trades_path, exc)
        return dict(empty_stats)

    if df.empty or "action" not in df:
        return dict(empty_stats)

    # Calculate holding time and consecutive streaks
    holding_times = []
    open_positions_track: Dict[str, List[Dict[str, Any]]] = {}
    
    # Sort by timestamp to ensure chronological processing
    try:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
    except Exception:
        return dict(empty_stats)
        
    df = df.sort_values("timestamp_dt")
    
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_streak_type = None # 'win' or 'loss'
    current_streak_count = 0

    actions = df["action"].astype(str).str.upper().str.strip()
    
    for _, row in df.iterrows():
        coin = row["coin"]
        action = str(row["action"]).upper()
        ts = row["timestamp_dt"]
        
        if action == "ENTRY":
            if coin not in open_positions_track:
                open_positions_track[coin] = []
            open_positions_track[coin].append({"ts": ts, "qty": float(row["quantity"])})
        elif action in ["CLOSE", "CLOSE_PARTIAL"]:
            close_qty = float(row["quantity"])
            pnl = float(row.get("pnl", 0))
            
            # Match with entries (FIFO) to calculate holding time
            if coin in open_positions_track:
                while close_qty > 0 and open_positions_track[coin]:
                    entry = open_positions_track[coin][0]
                    if entry["qty"] <= close_qty + 1e-8:
                        # Full entry closed
                        duration = (ts - entry["ts"]).total_seconds()
                        holding_times.append(duration)
                        close_qty -= entry["qty"]
                        open_positions_track[coin].pop(0)
                    else:
                        # Partial entry closed
                        duration = (ts - entry["ts"]).total_seconds()
                        holding_times.append(duration)
                        entry["qty"] -= close_qty
                        close_qty = 0
            
            # Streak calculation
            if pnl > 0:
                if current_streak_type == 'win':
                    current_streak_count += 1
                else:
                    current_streak_type = 'win'
                    current_streak_count = 1
                max_consecutive_wins = max(max_consecutive_wins, current_streak_count)
            elif pnl < 0:
                if current_streak_type == 'loss':
                    current_streak_count += 1
                else:
                    current_streak_type = 'loss'
                    current_streak_count = 1
                max_consecutive_losses = max(max_consecutive_losses, current_streak_count)
            elif pnl == 0:
                current_streak_type = None
                current_streak_count = 0

    entries_mask = actions == "ENTRY"
    closes_mask = actions == "CLOSE"
    partial_mask = actions == "CLOSE_PARTIAL"
    close_events_mask = closes_mask | partial_mask

    total_trades_count = int(entries_mask.sum())
    full_closes = int(closes_mask.sum())
    partial_closes = int(partial_mask.sum())

    close_trades = df.loc[close_events_mask].copy()
    if close_trades.empty:
        return {
            **empty_stats,
            "total_trades": total_trades_count,
            "closed_trades": full_closes,
            "partial_closes": partial_closes,
        }

    close_trades["pnl"] = pd.to_numeric(close_trades.get("pnl", 0), errors="coerce")
    close_trades = close_trades[np.isfinite(close_trades["pnl"])]

    close_events = int(len(close_trades))
    winning = int((close_trades["pnl"] > 0).sum())
    losing = int((close_trades["pnl"] < 0).sum())
    breakeven = int((close_trades["pnl"] == 0).sum())
    win_rate = (winning / close_events) * 100 if close_events else 0.0
    net_realized = float(close_trades["pnl"].sum()) if close_events else 0.0
    avg_trade = net_realized / close_events if close_events else None

    wins = close_trades[close_trades["pnl"] > 0]["pnl"]
    losses = close_trades[close_trades["pnl"] < 0]["pnl"]
    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(-losses.sum()) if not losses.empty else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None
    
    avg_holding_time = float(np.mean(holding_times)) if holding_times else None

    return {
        "total_trades": total_trades_count,
        "closed_trades": full_closes,
        "partial_closes": partial_closes,
        "close_events": close_events,
        "winning_trades": winning,
        "losing_trades": losing,
        "breakeven_trades": breakeven,
        "win_rate_pct": float(win_rate),
        "net_realized_pnl": net_realized,
        "gross_win": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "avg_trade_pnl": avg_trade,
        "avg_holding_time_seconds": avg_holding_time,
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
    }

def execute_entry(coin: str, decision: Dict[str, Any], current_price: float) -> None:
    """Execute entry trade."""
    global balance
    
    if coin in positions:
        logging.warning(f"{coin}: Already have position, skipping entry")
        return
    
    side = str(decision.get('side', 'long')).lower()
    raw_reason = str(decision.get('justification', '')).strip()
    reason_text_compact = " ".join(raw_reason.split()) if raw_reason else ""
    if reason_text_compact:
        contradictory_phrases = (
            "no entry",
            "no long entry",
            "no short entry",
            "do not enter",
            "avoid entry",
            "skip entry",
        )
        reason_lower = reason_text_compact.lower()
        if any(phrase in reason_lower for phrase in contradictory_phrases):
            logging.warning(
                "%s: Skipping entry because AI justification contradicts signal (%s)",
                coin,
                reason_text_compact,
            )
            return

    # Calculate slippage/spread for backtesting
    is_backtest = os.getenv("BACKTEST_RUN_ID") is not None
    symbol = COIN_TO_SYMBOL.get(coin)
    entry_price = current_price
    if is_backtest and symbol:
        slippage_mode = os.getenv("BACKTEST_SLIPPAGE_MODE", "S0").upper()
        spread_pct = float(os.getenv("BACKTEST_SPREAD_PCT", "0.0002"))
        spread = current_price * spread_pct
        if slippage_mode == "S1":
            slippage = current_price * 0.0005
        elif slippage_mode == "S2":
            slippage = current_price * 0.0010
        else:
            slippage_factor = float(os.getenv("BACKTEST_SLIPPAGE_FACTOR", "0.1"))
            data = fetch_market_data(symbol)
            atr = data.get("atr", 0.0) if data else 0.0
            slippage = slippage_factor * atr
        if side == 'long':
            entry_price = current_price + slippage + 0.5 * spread
        else:
            entry_price = current_price - slippage - 0.5 * spread
        logging.info(
            f"{coin}: Slippage/spread applied. Raw: {current_price:.4f} -> Slipped Entry: {entry_price:.4f} "
            f"(Slippage: {slippage:.4f}, Spread: {spread:.4f})"
        )

    leverage_raw = decision.get('leverage', 10)
    try:
        leverage = float(leverage_raw)
        if leverage <= 0:
            leverage = 1.0
    except (TypeError, ValueError):
        logging.warning(f"{coin}: Invalid leverage '%s'; defaulting to 1x", leverage_raw)
        leverage = 1.0
    leverage_display = format_leverage_display(leverage)

    risk_usd_raw = decision.get('risk_usd', balance * 0.05)
    try:
        risk_usd = float(risk_usd_raw)
    except (TypeError, ValueError):
        logging.warning(f"{coin}: Invalid risk_usd '%s'; defaulting to 1%% of balance.", risk_usd_raw)
        risk_usd = balance * 0.01

    if not np.isfinite(risk_usd) or risk_usd <= 0:
        logging.warning(
            "%s: Received non-positive risk (%s); skipping entry to avoid zero-sized trade.",
            coin,
            risk_usd_raw,
        )
        return

    try:
        stop_loss_price = float(decision['stop_loss'])
        profit_target_price = float(decision['profit_target'])
    except (KeyError, TypeError, ValueError):
        logging.warning(f"{coin}: Invalid stop loss or profit target in decision; skipping entry.")
        return
    if stop_loss_price <= 0 or profit_target_price <= 0:
        logging.warning(
            "%s: Non-positive stop loss (%s) or profit target (%s); skipping entry.",
            coin,
            stop_loss_price,
            profit_target_price,
        )
        return
    
    if side == 'long':
        if stop_loss_price >= entry_price:
            logging.warning(
                "%s: Stop loss %s not below entry price %s for long; skipping entry.",
                coin,
                stop_loss_price,
                entry_price,
            )
            return
        if profit_target_price <= entry_price:
            logging.warning(
                "%s: Profit target %s not above entry price %s for long; skipping entry.",
                coin,
                profit_target_price,
                entry_price,
            )
            return
    elif side == 'short':
        if stop_loss_price <= entry_price:
            logging.warning(
                "%s: Stop loss %s not above entry price %s for short; skipping entry.",
                coin,
                stop_loss_price,
                entry_price,
            )
            return
        if profit_target_price >= entry_price:
            logging.warning(
                "%s: Profit target %s not below entry price %s for short; skipping entry.",
                coin,
                profit_target_price,
                entry_price,
            )
            return
    
    # Calculate position size based on risk
    stop_distance = abs(entry_price - stop_loss_price)
    if not np.isfinite(stop_distance) or stop_distance <= 0:
        logging.warning(f"{coin}: Invalid stop loss distance; skipping entry.")
        return
    
    quantity = risk_usd / stop_distance
    if not np.isfinite(quantity) or quantity <= 0:
        logging.warning(
            "%s: Computed invalid position size (risk=%s, distance=%s); skipping entry.",
            coin,
            risk_usd,
            stop_distance,
        )
        return

    position_value = quantity * entry_price
    margin_required = position_value / leverage if leverage else position_value
    
    liquidity = str(decision.get('liquidity', 'taker')).lower()
    fee_rate = decision.get('fee_rate')
    if fee_rate is not None:
        try:
            fee_rate = float(fee_rate)
        except (TypeError, ValueError):
            logging.warning(f"{coin}: Invalid fee_rate provided ({fee_rate}); defaulting to Binance schedule.")
            fee_rate = None
    if fee_rate is None:
        fee_rate = MAKER_FEE_RATE if liquidity == 'maker' else TAKER_FEE_RATE
    entry_fee = position_value * fee_rate
    
    total_cost = margin_required + entry_fee
    if total_cost > balance:
        logging.warning(
            f"{coin}: Insufficient balance ${balance:.2f} for margin ${margin_required:.2f} "
            f"and fees ${entry_fee:.2f}"
        )
        return

    live_entry_receipt = None
    if hyperliquid_trader.is_live:
        live_entry_receipt = hyperliquid_trader.place_entry_with_sl_tp(
            coin=coin,
            side=side,
            size=quantity,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=profit_target_price,
            leverage=leverage,
            liquidity=liquidity,
        )
        if not live_entry_receipt.get("success"):
            logging.error(
                "%s: Live Hyperliquid entry rejected; aborting simulated entry. Response: %s",
                coin,
                live_entry_receipt.get("entry_result"),
            )
            return

    # Open position
    trade_type_raw = str(decision.get("trade_type", "TYPE A")).strip().upper()
    trade_type = trade_type_raw or "TYPE A"
    phase_raw = str(decision.get("phase", "Phase 1")).strip()
    trail_phase = phase_raw if phase_raw else "Phase 1"
    now_iso = get_current_time().isoformat()

    positions[coin] = {
        'side': side,
        'quantity': quantity,
        'entry_price': entry_price,
        'profit_target': profit_target_price,
        'stop_loss': stop_loss_price,
        'leverage': leverage,
        'confidence': decision.get('confidence', 0),
        'invalidation_condition': decision.get('invalidation_condition', ''),
        'margin': margin_required,
        'fees_paid': entry_fee,
        'fee_rate': fee_rate,
        'liquidity': liquidity,
        'risk_usd': risk_usd,
        'trade_type': trade_type,
        'trail_phase': trail_phase,
        'initial_stop': stop_loss_price,
        'initial_risk_per_unit': stop_distance,
        'initial_risk_usd': risk_usd,
        'trail_history': [
            {
                "timestamp": now_iso,
                "phase": trail_phase,
                "stop_loss": stop_loss_price,
                "reason": "Initial stop",
            }
        ],
        'wait_for_fill': decision.get('wait_for_fill', False),
        'entry_oid': decision.get('entry_oid', -1),
        'tp_oid': decision.get('tp_oid', -1),
        'sl_oid': decision.get('sl_oid', -1),
        'entry_justification': raw_reason,
        'last_justification': raw_reason,
    }
    if hyperliquid_trader.is_live and live_entry_receipt:
        positions[coin]['entry_oid'] = live_entry_receipt.get('entry_oid', positions[coin]['entry_oid'])
        positions[coin]['tp_oid'] = live_entry_receipt.get('take_profit_oid', positions[coin]['tp_oid'])
        positions[coin]['sl_oid'] = live_entry_receipt.get('stop_loss_oid', positions[coin]['sl_oid'])
        positions[coin]['live_trading'] = True
    
    balance -= total_cost
    
    target_price = profit_target_price
    stop_price = stop_loss_price

    gross_at_target = calculate_pnl_for_price(positions[coin], target_price)
    gross_at_stop = calculate_pnl_for_price(positions[coin], stop_price)
    exit_fee_target = estimate_exit_fee(positions[coin], target_price)
    exit_fee_stop = estimate_exit_fee(positions[coin], stop_price)
    net_at_target = gross_at_target - (entry_fee + exit_fee_target)
    net_at_stop = gross_at_stop - (entry_fee + exit_fee_stop)

    expected_reward = max(gross_at_target, 0.0)
    expected_risk = max(-gross_at_stop, 0.0)
    if expected_risk > 0:
        rr_value = expected_reward / expected_risk if expected_reward > 0 else 0.0
        rr_display = f"{rr_value:.2f}:1"
    else:
        rr_display = "n/a"

    line = f"{Fore.GREEN}[ENTRY] {coin} {side.upper()} {leverage_display} @ ${entry_price:.4f}"
    print(line)
    record_iteration_message(line)
    line = f"  ├─ Size: {quantity:.4f} {coin} | Margin: ${margin_required:.2f}"
    print(line)
    record_iteration_message(line)
    line = f"  ├─ Risk: ${risk_usd:.2f} | Liquidity: {liquidity}"
    print(line)
    record_iteration_message(line)
    line = f"  ├─ Target: ${target_price:.4f} | Stop: ${stop_price:.4f} | {trade_type} {trail_phase}"
    print(line)
    record_iteration_message(line)
    reason_text = raw_reason or "No justification provided."
    reason_text = " ".join(reason_text.split())
    reason_text_for_signal = escape_markdown(reason_text)

    line = (
        f"  ├─ PnL @ Target: ${gross_at_target:+.2f} "
        f"(Net: ${net_at_target:+.2f})"
    )
    print(line)
    record_iteration_message(line)
    line = (
        f"  ├─ PnL @ Stop: ${gross_at_stop:+.2f} "
        f"(Net: ${net_at_stop:+.2f})"
    )
    print(line)
    record_iteration_message(line)
    if entry_fee > 0:
        line = f"  ├─ Estimated Fee: ${entry_fee:.2f} ({liquidity} @ {fee_rate*100:.4f}%)"
        print(line)
        record_iteration_message(line)
    if hyperliquid_trader.is_live and live_entry_receipt:
        entry_oid = live_entry_receipt.get("entry_oid")
        if entry_oid is not None:
            line = f"  ├─ Hyperliquid Entry OID: {entry_oid}"
            print(line)
            record_iteration_message(line)
        sl_oid = live_entry_receipt.get("stop_loss_oid")
        if sl_oid is not None:
            line = f"  ├─ Hyperliquid SL OID: {sl_oid}"
            print(line)
            record_iteration_message(line)
        tp_oid = live_entry_receipt.get("take_profit_oid")
        if tp_oid is not None:
            line = f"  ├─ Hyperliquid TP OID: {tp_oid}"
            print(line)
            record_iteration_message(line)
    line = f"  ├─ Confidence: {decision.get('confidence', 0)*100:.0f}%"
    print(line)
    record_iteration_message(line)
    line = f"  ├─ Reward/Risk: {rr_display}"
    print(line)
    record_iteration_message(line)
    line = f"  └─ Reason: {reason_text}"
    print(line)
    record_iteration_message(line)
    
    # Send rich ENTRY signal to the dedicated signals group (if configured).
    try:
        # Format percentage confidence
        confidence_pct = decision.get('confidence', 0) * 100
        
        # Determine emoji based on side
        side_emoji = "🟢" if side.lower() == "long" else "🔴"
        
        signal_text = (
            f"{side_emoji} *ENTRY SIGNAL* {side_emoji}\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"*Asset:* `{coin}`\n"
            f"*Direction:* {side.upper()} {leverage_display}\n"
            f"*Entry Price:* `${entry_price:.4f}`\n"
            f"\n"
            f"📊 *Position Details*\n"
            f"• Size: `{quantity:.4f} {coin}`\n"
            f"• Margin: `${margin_required:.2f}`\n"
            f"• Risk: `${risk_usd:.2f}`\n"
            f"\n"
            f"🎯 *Targets & Stops*\n"
            f"• Target: `${profit_target_price:.4f}` ({'+' if gross_at_target >= 0 else ''}`${gross_at_target:.2f}`)\n"
            f"• Stop Loss: `${stop_loss_price:.4f}` (`${gross_at_stop:.2f}`)\n"
            f"• R/R Ratio: `{rr_display}`\n"
            f"\n"
            f"⚙️ *Execution*\n"
            f"• Liquidity: `{liquidity}`\n"
            f"• Confidence: `{confidence_pct:.0f}%`\n"
            f"• Entry Fee: `${entry_fee:.2f}`\n"
            f"\n"
            f"💭 *Reasoning*\n"
            f"_{reason_text_for_signal}_\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"🕐 {get_current_time().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        # If TELEGRAM_SIGNALS_CHAT_ID is set, prefer it; otherwise fall back to TELEGRAM_CHAT_ID
        send_telegram_message(signal_text, chat_id=TELEGRAM_SIGNALS_CHAT_ID, parse_mode="Markdown")
    except Exception as exc:
        # Keep trading even if notifications fail
        logging.debug("Failed to send ENTRY signal to Telegram (non-fatal): %s", exc)
    
    log_trade(coin, 'ENTRY', {
        'side': side,
        'quantity': quantity,
        'price': entry_price,
        'profit_target': decision['profit_target'],
        'stop_loss': decision['stop_loss'],
        'leverage': leverage,
        'confidence': decision.get('confidence', 0),
        'trade_type': trade_type,
        'phase': trail_phase,
        'pnl': 0,
        'reason': f"{reason_text or 'AI entry signal'} | {trade_type} {trail_phase} | Fees: ${entry_fee:.2f}",
        'confluence_tags': ';'.join(decision.get('confluence_tags', [])) if isinstance(decision.get('confluence_tags'), list) else str(decision.get('confluence_tags', '')),
        'trigger_tags': ';'.join(decision.get('trigger_tags', [])) if isinstance(decision.get('trigger_tags'), list) else str(decision.get('trigger_tags', '')),
        'reasoning_categories': ';'.join(decision.get('reasoning_categories', [])) if isinstance(decision.get('reasoning_categories'), list) else str(decision.get('reasoning_categories', ''))
    })
    save_state()

def execute_close(coin: str, decision: Dict[str, Any], current_price: float) -> None:
    """Execute full or partial close."""
    global balance

    if coin not in positions:
        logging.warning(f"{coin}: No position to close")
        return

    pos = positions[coin]
    original_quantity = float(pos.get("quantity", 0.0))
    if original_quantity <= 0:
        logging.warning(f"{coin}: Position quantity is non-positive; skipping close.")
        return

    # Calculate slippage/spread for backtesting exit price
    is_backtest = os.getenv("BACKTEST_RUN_ID") is not None
    symbol = COIN_TO_SYMBOL.get(coin)
    exit_price = current_price
    if is_backtest and symbol:
        slippage_mode = os.getenv("BACKTEST_SLIPPAGE_MODE", "S0").upper()
        spread_pct = float(os.getenv("BACKTEST_SPREAD_PCT", "0.0002"))
        spread = current_price * spread_pct
        if slippage_mode == "S1":
            slippage = current_price * 0.0005
        elif slippage_mode == "S2":
            slippage = current_price * 0.0010
        else:
            slippage_factor = float(os.getenv("BACKTEST_SLIPPAGE_FACTOR", "0.1"))
            data = fetch_market_data(symbol)
            atr = data.get("atr", 0.0) if data else 0.0
            slippage = slippage_factor * atr
        if pos["side"].lower() == "long":
            exit_price = current_price - slippage - 0.5 * spread
        else:
            exit_price = current_price + slippage + 0.5 * spread
        logging.info(
            f"{coin}: Close slippage/spread applied. Raw: {current_price:.4f} -> Slipped Exit: {exit_price:.4f} "
            f"(Slippage: {slippage:.4f}, Spread: {spread:.4f})"
        )

    def _to_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # Determine requested close size
    close_quantity = None
    quantity_override = decision.get("close_quantity")
    if quantity_override is None:
        quantity_override = decision.get("quantity")
    qty_value = _to_float(quantity_override)
    if qty_value is not None and qty_value > 0:
        close_quantity = qty_value

    if close_quantity is None:
        fraction = _to_float(decision.get("close_fraction"))
        if fraction is None:
            percent = _to_float(decision.get("close_percent"))
            if percent is not None:
                fraction = percent / 100.0
        if fraction is not None:
            if fraction <= 0:
                close_quantity = 0.0
            else:
                close_quantity = original_quantity * min(fraction, 1.0)

    if close_quantity is None:
        close_quantity = original_quantity

    EPS = 1e-8
    if close_quantity <= EPS:
        logging.warning("%s: Requested close size %.8f too small; skipping.", coin, close_quantity)
        return

    if close_quantity > original_quantity:
        close_quantity = original_quantity

    remaining_quantity = original_quantity - close_quantity
    if remaining_quantity <= EPS:
        close_quantity = original_quantity
        remaining_quantity = 0.0
        is_partial = False
    else:
        is_partial = True

    raw_reason = str(decision.get("justification", "")).strip()
    reason_text = raw_reason or pos.get("last_justification") or "AI close signal"
    reason_text = " ".join(reason_text.split())
    reason_text_for_signal = escape_markdown(reason_text)

    side = str(pos.get("side", "long")).lower()
    entry_price = float(pos.get("entry_price", 0.0))
    leverage_value = pos.get("leverage", 1)

    if side == "long":
        gross_pnl = (exit_price - entry_price) * close_quantity
    else:
        gross_pnl = (entry_price - exit_price) * close_quantity

    fee_rate_raw = pos.get("fee_rate", TAKER_FEE_RATE)
    fee_rate = _to_float(fee_rate_raw)
    if fee_rate is None or fee_rate < 0:
        fee_rate = TAKER_FEE_RATE

    exit_fee = close_quantity * exit_price * fee_rate

    fees_paid_total = float(pos.get("fees_paid", 0.0))
    entry_fee_share = fees_paid_total * (close_quantity / original_quantity)
    total_fees = entry_fee_share + exit_fee
    net_pnl = gross_pnl - total_fees

    margin_total = float(pos.get("margin", 0.0))
    margin_released = margin_total * (close_quantity / original_quantity)
    remaining_margin = max(margin_total - margin_released, 0.0)

    live_close_receipt = None
    if hyperliquid_trader.is_live:
        live_close_receipt = hyperliquid_trader.close_position(
            coin=coin,
            side=pos["side"],
            size=close_quantity,
            fallback_price=current_price,
        )
        if not live_close_receipt.get("success"):
            logging.error(
                "%s: Live Hyperliquid close rejected; position remains open. Response: %s",
                coin,
                live_close_receipt.get("close_result"),
            )
            return

    # Return proportional margin and realized PnL
    balance += margin_released + net_pnl

    label = "[PARTIAL CLOSE]" if is_partial else "[CLOSE]"
    color = Fore.GREEN if net_pnl >= 0 else Fore.RED
    line = f"{color}{label} {coin} {pos['side'].upper()} {close_quantity:.4f} @ ${exit_price:.4f}"
    print(line)
    record_iteration_message(line)

    line = f"  ├─ Entry: ${entry_price:.4f} | Gross PnL: ${gross_pnl:.2f}"
    print(line)
    record_iteration_message(line)

    if total_fees > 0:
        line = (
            f"  ├─ Fees This Exit: ${total_fees:.2f} "
            f"(entry share ${entry_fee_share:.2f}, exit ${exit_fee:.2f})"
        )
        print(line)
        record_iteration_message(line)

    if is_partial:
        line = f"  ├─ Remaining: {remaining_quantity:.4f} {coin} | Margin: ${remaining_margin:.2f}"
        print(line)
        record_iteration_message(line)

    if hyperliquid_trader.is_live and live_close_receipt:
        close_oid = live_close_receipt.get("close_oid")
        if close_oid is not None:
            line = f"  ├─ Hyperliquid Close OID: {close_oid}"
            print(line)
            record_iteration_message(line)

    line = f"  ├─ Net PnL: ${net_pnl:.2f}"
    print(line)
    record_iteration_message(line)
    line = f"  ├─ Reason: {reason_text}"
    print(line)
    record_iteration_message(line)
    line = f"  └─ Balance: ${balance:.2f}"
    print(line)
    record_iteration_message(line)

    # Build Telegram message
    result_emoji = "➖"
    result_label = "BREAKEVEN"
    if net_pnl > 0:
        result_emoji = "✅"
        result_label = "PROFIT"
    elif net_pnl < 0:
        result_emoji = "❌"
        result_label = "LOSS"

    header = (
        f"{result_emoji} *PARTIAL CLOSE SIGNAL - {result_label}* {result_emoji}"
        if is_partial
        else f"{result_emoji} *CLOSE SIGNAL - {result_label}* {result_emoji}"
    )

    price_change_pct = ((exit_price - entry_price) / entry_price) * 100 if entry_price else 0.0
    price_change_sign = "+" if price_change_pct >= 0 else ""

    roi_pct = (net_pnl / margin_released) * 100 if margin_released else 0.0
    roi_sign = "+" if roi_pct >= 0 else ""

    telegram_message = (
        f"{header}\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"*Asset:* `{coin}`\n"
        f"*Direction:* {pos['side'].upper()}\n"
        f"*Closed Size:* `{close_quantity:.4f} {coin}`\n"
        f"*Entry:* `${entry_price:.4f}`\n"
        f"*Exit:* `${exit_price:.4f}` ({price_change_sign}{price_change_pct:.2f}%)\n"
        f"\n"
        f"💰 *P&L Summary*\n"
        f"• Gross: `${gross_pnl:.2f}`\n"
        f"• Fees (entry share + exit): `${total_fees:.2f}`\n"
        f"• *Net:* `{net_pnl:+.2f}`\n"
        f"• ROI on released margin: `{roi_sign}{roi_pct:.1f}%`\n"
    )
    if is_partial:
        telegram_message += (
            f"\n"
            f"📉 *Position Remainder*\n"
            f"• Remaining Size: `{remaining_quantity:.4f} {coin}`\n"
            f"• Remaining Margin: `${remaining_margin:.2f}`\n"
        )
    telegram_message += (
        f"\n"
        f"💭 *Reasoning*\n"
        f"_{reason_text_for_signal}_\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {get_current_time().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )

    try:
        send_telegram_message(
            telegram_message,
            chat_id=TELEGRAM_SIGNALS_CHAT_ID,
            parse_mode="Markdown",
        )
    except Exception as exc:
        logging.debug("Failed to send CLOSE signal to Telegram (non-fatal): %s", exc)

    action_name = "CLOSE_PARTIAL" if is_partial else "CLOSE"
    log_reason = (
        f"{reason_text} | {pos.get('trade_type', 'TYPE A')} {pos.get('trail_phase', '')} | "
        f"Closed {close_quantity:.4f}, Remaining {remaining_quantity:.4f} | "
        f"Gross: ${gross_pnl:.2f} | Fees: ${total_fees:.2f}"
    )

    log_trade(
        coin,
        action_name,
        {
            "side": pos["side"],
            "quantity": close_quantity,
            "price": exit_price,
            "profit_target": 0,
            "stop_loss": 0,
            "leverage": leverage_value,
            "confidence": 0,
            "pnl": net_pnl,
            "reason": log_reason,
            "confluence_tags": ';'.join(decision.get('confluence_tags', [])) if isinstance(decision.get('confluence_tags'), list) else str(decision.get('confluence_tags', '')),
            "trigger_tags": ';'.join(decision.get('trigger_tags', [])) if isinstance(decision.get('trigger_tags'), list) else str(decision.get('trigger_tags', '')),
            "reasoning_categories": ';'.join(decision.get('reasoning_categories', [])) if isinstance(decision.get('reasoning_categories'), list) else str(decision.get('reasoning_categories', ''))
        },
    )

    if is_partial:
        pos["quantity"] = remaining_quantity
        pos["margin"] = remaining_margin
        pos["fees_paid"] = max(fees_paid_total - entry_fee_share, 0.0)
        initial_risk_per_unit = _to_float(pos.get("initial_risk_per_unit")) or 0.0
        pos["initial_risk_usd"] = initial_risk_per_unit * remaining_quantity

        current_stop = _to_float(pos.get("stop_loss"))
        if current_stop is None:
            current_stop = entry_price
        if side == "long":
            per_unit_risk = max(entry_price - current_stop, 0.0)
        else:
            per_unit_risk = max(current_stop - entry_price, 0.0)
        pos["risk_usd"] = per_unit_risk * remaining_quantity

        pos["last_justification"] = reason_text
        save_state()
    else:
        del positions[coin]
        save_state()

def process_ai_decisions(decisions: Dict[str, Any]) -> None:
    """Handle AI decisions for each tracked coin."""
    for coin in SYMBOL_TO_COIN.values():
        if coin not in decisions:
            continue

        decision = decisions[coin]
        signal = decision.get("signal", "hold")

        conf_tags = ';'.join(decision.get('confluence_tags', [])) if isinstance(decision.get('confluence_tags'), list) else str(decision.get('confluence_tags', ''))
        trig_tags = ';'.join(decision.get('trigger_tags', [])) if isinstance(decision.get('trigger_tags'), list) else str(decision.get('trigger_tags', ''))
        rc_tags = ';'.join(decision.get('reasoning_categories', [])) if isinstance(decision.get('reasoning_categories'), list) else str(decision.get('reasoning_categories', ''))

        log_ai_decision(
            coin,
            signal,
            decision.get("justification", ""),
            decision.get("confidence", 0),
            confluence_tags=conf_tags,
            trigger_tags=trig_tags,
            reasoning_categories=rc_tags
        )

        symbol = COIN_TO_SYMBOL.get(coin)
        if not symbol:
            logging.debug("No symbol mapping found for coin %s", coin)
            continue

        data = fetch_market_data(symbol)
        if not data:
            continue

        current_price = data["price"]

        if signal == "entry":
            execute_entry(coin, decision, current_price)
        elif signal == "close":
            execute_close(coin, decision, current_price)
        elif signal == "hold":
            if coin not in positions:
                continue
            pos = positions[coin]
            state_changed = False
            hold_updates: List[str] = []
            raw_reason = str(decision.get("justification", "")).strip()
            if raw_reason:
                reason_text = " ".join(raw_reason.split())
                pos["last_justification"] = reason_text
            else:
                existing_reason = str(pos.get("last_justification", "")).strip()
                reason_text = existing_reason or "No justification provided."
                if not existing_reason:
                    pos["last_justification"] = reason_text
            try:
                quantity = float(pos.get("quantity", 0.0))
            except (TypeError, ValueError):
                quantity = 0.0
            try:
                fees_paid = float(pos.get("fees_paid", 0.0))
            except (TypeError, ValueError):
                fees_paid = 0.0
            try:
                entry_price = float(pos.get("entry_price", 0.0))
            except (TypeError, ValueError):
                entry_price = 0.0
            try:
                target_price = float(pos.get("profit_target", entry_price))
            except (TypeError, ValueError):
                target_price = entry_price
            try:
                stop_price = float(pos.get("stop_loss", entry_price))
            except (TypeError, ValueError):
                stop_price = entry_price
            leverage_display = format_leverage_display(pos.get("leverage", 1.0))
            try:
                margin_value = float(pos.get("margin", 0.0))
            except (TypeError, ValueError):
                margin_value = 0.0
            try:
                risk_value = float(pos.get("risk_usd", 0.0))
            except (TypeError, ValueError):
                risk_value = 0.0

            # Optional updates from AI for trailing / targets / trade type
            def _coerce_float(value: Any) -> Optional[float]:
                if value is None:
                    return None
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            new_stop_candidate = _coerce_float(decision.get("stop_loss"))
            new_target_candidate = _coerce_float(decision.get("profit_target"))
            new_phase_raw = str(decision.get("phase", "")).strip()
            new_trade_type_raw = str(decision.get("trade_type", "")).strip().upper()
            timestamp_iso = get_current_time().isoformat()

            initial_stop = pos.get("initial_stop", stop_price)
            try:
                initial_stop = float(initial_stop)
            except (TypeError, ValueError):
                initial_stop = stop_price
            pos["initial_stop"] = initial_stop

            initial_risk_per_unit = pos.get("initial_risk_per_unit")
            try:
                initial_risk_per_unit = float(initial_risk_per_unit)
            except (TypeError, ValueError):
                initial_risk_per_unit = None
            if not initial_risk_per_unit or initial_risk_per_unit == 0:
                initial_risk_per_unit = abs(entry_price - initial_stop)
                pos["initial_risk_per_unit"] = initial_risk_per_unit

            if new_stop_candidate is not None and new_stop_candidate > 0:
                can_update = False
                if pos["side"] == "long":
                    if new_stop_candidate >= stop_price - 1e-6 and new_stop_candidate <= current_price:
                        if new_stop_candidate > stop_price + 1e-6:
                            can_update = True
                    else:
                        logging.debug(
                            "%s: Ignoring stop update %.6f (long) because it would reduce protection or exceed price.",
                            coin,
                            new_stop_candidate,
                        )
                else:
                    if new_stop_candidate <= stop_price + 1e-6 and new_stop_candidate >= current_price:
                        if new_stop_candidate < stop_price - 1e-6:
                            can_update = True
                    else:
                        logging.debug(
                            "%s: Ignoring stop update %.6f (short) because it would reduce protection or exceed price.",
                            coin,
                            new_stop_candidate,
                        )
                if can_update:
                    pos["stop_loss"] = new_stop_candidate
                    stop_price = new_stop_candidate
                    state_changed = True
                    phase_value = new_phase_raw or pos.get("trail_phase", "Phase 1")
                    pos["trail_phase"] = phase_value
                    pos.setdefault("trail_history", []).append(
                        {
                            "timestamp": timestamp_iso,
                            "phase": phase_value,
                            "stop_loss": new_stop_candidate,
                            "reason": decision.get("trail_reason", "") or reason_text,
                        }
                    )
                    hold_updates.append(
                        f"  ├─ Stop adjusted to ${new_stop_candidate:.4f} ({phase_value})"
                    )
            if new_target_candidate is not None and new_target_candidate > 0:
                if pos["side"] == "long" and new_target_candidate >= current_price:
                    if abs(new_target_candidate - target_price) > 1e-6:
                        pos["profit_target"] = new_target_candidate
                        target_price = new_target_candidate
                        state_changed = True
                        hold_updates.append(
                            f"  ├─ Target updated to ${new_target_candidate:.4f}"
                        )
                elif pos["side"] == "short" and new_target_candidate <= current_price:
                    if abs(new_target_candidate - target_price) > 1e-6:
                        pos["profit_target"] = new_target_candidate
                        target_price = new_target_candidate
                        state_changed = True
                        hold_updates.append(
                            f"  ├─ Target updated to ${new_target_candidate:.4f}"
                        )
            if new_trade_type_raw:
                pos["trade_type"] = new_trade_type_raw
            if new_phase_raw:
                pos["trail_phase"] = new_phase_raw

            gross_unrealized = calculate_unrealized_pnl(coin, current_price)
            estimated_exit_fee_now = estimate_exit_fee(pos, current_price)
            total_fees_now = fees_paid + estimated_exit_fee_now
            net_unrealized = gross_unrealized - total_fees_now

            gross_at_target = calculate_pnl_for_price(pos, target_price)
            exit_fee_target = estimate_exit_fee(pos, target_price)
            net_at_target = gross_at_target - (fees_paid + exit_fee_target)

            gross_at_stop = calculate_pnl_for_price(pos, stop_price)
            exit_fee_stop = estimate_exit_fee(pos, stop_price)
            net_at_stop = gross_at_stop - (fees_paid + exit_fee_stop)

            expected_reward = max(gross_at_target, 0.0)
            expected_risk = max(-gross_at_stop, 0.0)
            if expected_risk > 0:
                rr_value = expected_reward / expected_risk if expected_reward > 0 else 0.0
                rr_display = f"{rr_value:.2f}:1"
            else:
                rr_display = "n/a"

            initial_risk_per_unit = float(pos.get("initial_risk_per_unit") or 0.0)
            if initial_risk_per_unit:
                if pos["side"] == "long":
                    r_multiple = (current_price - entry_price) / initial_risk_per_unit
                else:
                    r_multiple = (entry_price - current_price) / initial_risk_per_unit
            else:
                r_multiple = 0.0

            pnl_color = Fore.GREEN if net_unrealized >= 0 else Fore.RED
            gross_color = Fore.GREEN if gross_unrealized >= 0 else Fore.RED
            net_display = f"{net_unrealized:+.2f}"
            gross_display = f"{gross_unrealized:+.2f}"
            gross_target_display = f"{gross_at_target:+.2f}"
            gross_stop_display = f"{gross_at_stop:+.2f}"
            net_target_display = f"{net_at_target:+.2f}"
            net_stop_display = f"{net_at_stop:+.2f}"

            line = f"{Fore.BLUE}[HOLD] {coin} {pos['side'].upper()} {leverage_display}"
            print(line)
            record_iteration_message(line)
            line = (
                f"  ├─ Size: {quantity:.4f} {coin} | Margin: ${margin_value:.2f} | "
                f"{pos.get('trade_type', 'TYPE A')} {pos.get('trail_phase', 'Phase 1')}"
            )
            print(line)
            record_iteration_message(line)
            line = f"  ├─ TP: ${target_price:.4f} | SL: ${stop_price:.4f} | Risk ${risk_value:.2f}"
            print(line)
            record_iteration_message(line)
            line = (
                f"  ├─ PnL: {pnl_color}${net_display}{Style.RESET_ALL} "
                f"(Gross: {gross_color}${gross_display}{Style.RESET_ALL}, Fees: ${total_fees_now:.2f})"
            )
            print(line)
            record_iteration_message(line)
            line = (
                f"  ├─ PnL @ Target: ${gross_target_display} "
                f"(Net: ${net_target_display})"
            )
            print(line)
            record_iteration_message(line)
            line = (
                f"  ├─ PnL @ Stop: ${gross_stop_display} "
                f"(Net: ${net_stop_display})"
            )
            print(line)
            record_iteration_message(line)
            line = f"  ├─ Reward/Risk: {rr_display}"
            print(line)
            record_iteration_message(line)
            line = f"  ├─ R Multiple: {r_multiple:.2f}R"
            print(line)
            record_iteration_message(line)
            if hold_updates:
                for update_line in hold_updates:
                    print(update_line)
                    record_iteration_message(update_line)
            line = f"  └─ Reason: {reason_text}"
            print(line)
            record_iteration_message(line)
            if state_changed:
                save_state()

def check_stop_loss_take_profit() -> None:
    """Check and execute stop loss / take profit for all positions using intrabar extremes."""
    if hyperliquid_trader.is_live:
        return
    for coin in list(positions.keys()):
        symbol = [s for s, c in SYMBOL_TO_COIN.items() if c == coin][0]
        data = fetch_market_data(symbol)
        if not data:
            continue

        pos = positions[coin]
        current_price = float(data.get("price", pos["entry_price"]))
        candle_high = data.get("high")
        candle_low = data.get("low")

        exit_reason = None
        exit_price = current_price

        if pos["side"] == "long":
            if candle_low is not None and candle_low <= pos["stop_loss"]:
                exit_reason = "Stop loss hit"
                exit_price = pos["stop_loss"]
            elif candle_high is not None and candle_high >= pos["profit_target"]:
                exit_reason = "Take profit hit"
                exit_price = pos["profit_target"]
        else:  # short
            if candle_high is not None and candle_high >= pos["stop_loss"]:
                exit_reason = "Stop loss hit"
                exit_price = pos["stop_loss"]
            elif candle_low is not None and candle_low <= pos["profit_target"]:
                exit_reason = "Take profit hit"
                exit_price = pos["profit_target"]

        if exit_reason:
            execute_close(coin, {"justification": exit_reason}, exit_price)

# ─────────────────────────── MAIN ──────────────────────────

def main() -> None:
    """Main trading loop."""
    global current_iteration_messages, iteration_counter
    logging.info("Initializing AI Multi-Asset Paper Trading Bot...")
    init_csv_files()
    load_equity_history()
    load_state()
    
    if not OPENROUTER_API_KEY and not GEMINI_API_KEY:
        logging.error("No LLM API key found (OPENROUTER_API_KEY or GEMINI_API_KEY). Please check your .env file.")
        return
    
    logging.info(f"Starting capital: ${START_CAPITAL:.2f}")
    logging.info(f"Monitoring: {', '.join(SYMBOL_TO_COIN.values())}")
    if hyperliquid_trader.is_live:
        logging.warning(
            "Hyperliquid LIVE trading enabled. Orders will be sent to mainnet using wallet %s.",
            hyperliquid_trader.masked_wallet,
        )
    else:
        logging.info("Hyperliquid live trading disabled; running in paper mode only.")
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        logging.info("Telegram notifications enabled (chat: %s).", TELEGRAM_CHAT_ID)
    else:
        logging.info("Telegram notifications disabled; missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID.")
    log_system_prompt_info("System prompt selected")
    logging.info("LLM model configured: %s", LLM_MODEL_NAME)
    
    while True:
        try:
            iteration_counter += 1
            current_iteration_messages = []

            if not get_binance_client():
                retry_delay = min(CHECK_INTERVAL, 60)
                logging.warning(
                    "Binance client unavailable; retrying in %d seconds without exiting.",
                    retry_delay,
                )
                time.sleep(retry_delay)
                continue

            line = f"\n{Fore.CYAN}{'='*20}"
            print(line)
            record_iteration_message(line)
            current_dt = get_current_time()
            line = f"{Fore.CYAN}Iteration {iteration_counter} - {current_dt.strftime('%Y-%m-%d %H:%M:%S')}"
            print(line)
            record_iteration_message(line)
            line = f"{Fore.CYAN}{'='*20}\n"
            print(line)
            record_iteration_message(line)
            
            # Check stop loss / take profit first
            check_stop_loss_take_profit()
            
            # Get AI decisions
            logging.info("Requesting trading decisions from AI (%s)...", LLM_MODEL_NAME)
            prompt = format_trading_prompt()
            decisions = call_llm_api(prompt)
            
            if not decisions:
                logging.warning("No decisions received from AI")
            else:
                process_ai_decisions(decisions)
            
            # Display portfolio summary
            total_equity = calculate_total_equity()
            total_net_profit = total_equity - START_CAPITAL
            total_return = (total_net_profit / START_CAPITAL) * 100 if START_CAPITAL else 0.0
            equity_color = Fore.GREEN if total_return >= 0 else Fore.RED
            total_margin = calculate_total_margin()
            net_unrealized_total = total_equity - balance - total_margin
            net_color = Fore.GREEN if net_unrealized_total >= 0 else Fore.RED
            register_equity_snapshot(total_equity)
            
            # Performance metrics
            sortino_ratio = calculate_sortino_ratio(
                equity_history,
                CHECK_INTERVAL,
                RISK_FREE_RATE,
            )
            mdd = calculate_max_drawdown(equity_history)
            recovery_factor = 0.0
            if mdd is not None and mdd > 0:
                max_dd_amount = START_CAPITAL * mdd
                recovery_factor = total_net_profit / max_dd_amount if max_dd_amount > 0 else 0.0
            
            trade_stats = summarize_trades(TRADES_CSV)
            profit_factor = trade_stats.get("profit_factor")
            win_rate = trade_stats.get("win_rate_pct", 0.0)

            line = f"\n{Fore.YELLOW}{'─'*20}"
            print(line)
            record_iteration_message(line)
            line = f"{Fore.YELLOW}PORTFOLIO SUMMARY"
            print(line)
            record_iteration_message(line)
            line = f"{Fore.YELLOW}{'─'*20}"
            print(line)
            record_iteration_message(line)
            line = f"Available Balance: ${balance:.2f}"
            print(line)
            record_iteration_message(line)
            if total_margin > 0:
                line = f"Margin Allocated: ${total_margin:.2f}"
                print(line)
                record_iteration_message(line)
            
            line = f"Total Equity: {equity_color}${total_equity:.2f} ({total_return:+.2f}% | ${total_net_profit:+.2f}){Style.RESET_ALL}"
            print(line)
            record_iteration_message(line)
            
            line = f"Unrealized PnL: {net_color}${net_unrealized_total:.2f}{Style.RESET_ALL}"
            print(line)
            record_iteration_message(line)
            
            total_closed = trade_stats.get("close_events", 0)
            winning_trades = trade_stats.get("winning_trades", 0)
            losing_trades = trade_stats.get("losing_trades", 0)
            loss_rate = (losing_trades / total_closed * 100) if total_closed > 0 else 0.0

            perf_line = f"Profit Trades: {winning_trades} wins ({win_rate:.1f}%)"
            print(perf_line)
            record_iteration_message(perf_line)

            loss_line = f"Loss Trades: {losing_trades} losses ({loss_rate:.1f}%)"
            if profit_factor is not None:
                loss_line += f" | Profit Factor: {profit_factor:.2f}"
            print(loss_line)
            record_iteration_message(loss_line)
            
            rf_line = f"Recovery Factor: {recovery_factor:.2f}"
            if mdd is not None:
                rf_line += f" | MaxDD: {mdd*100:.2f}%"
            print(rf_line)
            record_iteration_message(rf_line)

            if sortino_ratio is not None:
                sortino_color = Fore.GREEN if sortino_ratio >= 0 else Fore.RED
                line = f"Sortino Ratio: {sortino_color}{sortino_ratio:+.2f}{Style.RESET_ALL}"
            else:
                line = "Sortino Ratio: N/A (need more data)"
            print(line)
            record_iteration_message(line)
            
            line = f"Open Positions: {len(positions)}"
            print(line)
            record_iteration_message(line)
            line = f"{Fore.YELLOW}{'─'*20}\n"
            print(line)
            record_iteration_message(line)

            if current_iteration_messages:
                send_telegram_message("\n".join(current_iteration_messages), parse_mode=None)
            
            # Log state
            log_portfolio_state()
            save_state()
            
            # Wait for next check
            logging.info(f"Waiting {CHECK_INTERVAL} seconds until next check...")
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n\nShutting down bot...")
            save_state()
            break
        except Exception as e:
            logging.error(f"Error in main loop: {e}", exc_info=True)
            save_state()
            time.sleep(60)

if __name__ == "__main__":
    main()
