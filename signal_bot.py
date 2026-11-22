"""
Trading Signal Bot (শুধু সিগন্যাল দেয়, নিজে ট্রেড করে না)

কৌশল: EMA ক্রসওভার + RSI ফিল্টার
  BUY  = দ্রুত EMA ধীর EMA-কে ওপরে কেটে গেলে এবং RSI < RSI_OVERBOUGHT
  SELL = দ্রুত EMA ধীর EMA-কে নিচে কেটে গেলে এবং RSI > RSI_OVERSOLD

ব্যবহার:
  pip install ccxt pandas numpy requests
  python signal_bot.py demo       # নকল ডেটায় টেস্ট (ইন্টারনেট লাগে না)
  python signal_bot.py backtest   # পুরনো রিয়েল ডেটায় ব্যাকটেস্ট
  python signal_bot.py live       # লাইভ সিগন্যাল (কনসোল + ঐচ্ছিক Telegram)
  python signal_bot.py once       # একবার চেক করে বের হয় (GitHub Actions-এর জন্য)
  python signal_bot.py test       # Telegram-এ টেস্ট মেসেজ পাঠায়

Telegram চালু করতে (ঐচ্ছিক):
  export TELEGRAM_BOT_TOKEN="আপনার_টোকেন"
  export TELEGRAM_CHAT_ID="আপনার_চ্যাট_আইডি"

সতর্কতা: এটা শিক্ষামূলক টুল। লাভের কোনো নিশ্চয়তা নেই।
"""

import os
import sys
import time

import numpy as np
import pandas as pd

# ====================== সেটিংস (এখানে বদলান) ======================
# এনভায়রনমেন্ট ভেরিয়েবল দিলে সেটা ব্যবহার হবে (GitHub Actions-এর জন্য সুবিধা)
EXCHANGE = os.getenv("EXCHANGE", "binance")   # ccxt-সমর্থিত যেকোনো এক্সচেঞ্জ (API key লাগে না)
SYMBOL = os.getenv("SYMBOL", "BTC/USDT")      # যেমন "ETH/USDT", "SOL/USDT"
TIMEFRAME = os.getenv("TIMEFRAME", "15m")     # 5m, 15m, 1h, 4h, 1d
CANDLES = 1000             # কতগুলো ক্যান্ডেল আনবে

EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

FEE_PCT = 0.1              # ব্যাকটেস্টে প্রতি ট্রেডে ফি (%)
POLL_SECONDS = 20          # লাইভ মোডে কত সেকেন্ড পরপর চেক করবে
# ==================================================================


# ---------------------- ডেটা ----------------------
def fetch_ohlcv(symbol=SYMBOL, timeframe=TIMEFRAME, limit=CANDLES) -> pd.DataFrame:
    import ccxt

    ex = getattr(ccxt, EXCHANGE)({"enableRateLimit": True})
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df.set_index("time")


def synthetic_ohlcv(n=1500, seed=7) -> pd.DataFrame:
    """নকল প্রাইস ডেটা, শুধু কোড টেস্টের জন্য।"""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.002, n)
    close = 100 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame({"close": close}, index=idx)


# ---------------------- ইন্ডিকেটর ----------------------
def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["rsi"] = rsi(df["close"], RSI_PERIOD)

    above = df["ema_fast"] > df["ema_slow"]
    cross_up = above & ~above.shift(1, fill_value=False)
    cross_down = ~above & above.shift(1, fill_value=True)

    df["signal"] = 0
    df.loc[cross_up & (df["rsi"] < RSI_OVERBOUGHT), "signal"] = 1    # BUY
    df.loc[cross_down & (df["rsi"] > RSI_OVERSOLD), "signal"] = -1   # SELL
    # প্রথম কয়েকটা ক্যান্ডেলে ইন্ডিকেটর স্থির হয়নি, তাই বাদ
    df.iloc[: EMA_SLOW * 2, df.columns.get_loc("signal")] = 0
    return df


# ---------------------- ব্যাকটেস্ট ----------------------
def backtest(df: pd.DataFrame) -> dict:
    """
    সিগন্যালের পরের ক্যান্ডেলের ক্লোজে এন্ট্রি ধরা হয় (lookahead এড়াতে)।
    পজিশন: +1 লং, -1 শর্ট; উল্টো সিগন্যাল এলে ফ্লিপ।
    """
    df = add_signals(df)
    position = df["signal"].replace(0, np.nan).ffill().fillna(0)
    position = position.shift(1).fillna(0)  # এক ক্যান্ডেল দেরিতে কার্যকর

    ret = df["close"].pct_change().fillna(0)
    trades = position.diff().abs().fillna(0)            # ফ্লিপে 2, নতুন এন্ট্রিতে 1
    strat_ret = position * ret - trades * (FEE_PCT / 100)

    equity = (1 + strat_ret).cumprod()
    buy_hold = (1 + ret).cumprod()
    drawdown = (equity / equity.cummax() - 1).min()

    # প্রতিটি ট্রেডের ফলাফল
    change_points = position.ne(position.shift()).cumsum()
    trade_returns = strat_ret.groupby(change_points).sum()
    trade_returns = trade_returns[position.groupby(change_points).first() != 0]
    wins = (trade_returns > 0).sum()

    return {
        "ক্যান্ডেল সংখ্যা": len(df),
        "মোট ট্রেড": int(len(trade_returns)),
        "জয়ের হার (%)": round(100 * wins / max(len(trade_returns), 1), 1),
        "স্ট্র্যাটেজি রিটার্ন (%)": round((equity.iloc[-1] - 1) * 100, 2),
        "বাই-অ্যান্ড-হোল্ড রিটার্ন (%)": round((buy_hold.iloc[-1] - 1) * 100, 2),
        "সর্বোচ্চ ড্রডাউন (%)": round(drawdown * 100, 2),
    }


def print_report(title: str, stats: dict):
    print(f"\n=== {title} ===")
    for k, v in stats.items():
        print(f"{k:32s}: {v}")
    print("\nমনে রাখবেন: ব্যাকটেস্টের ফল ভবিষ্যতের গ্যারান্টি নয়।\n")


# ---------------------- Telegram ----------------------
def send_telegram(text: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        import requests

        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
    except Exception as e:
        print(f"[Telegram error] {e}")


# ---------------------- লাইভ ----------------------
def live():
    print(f"লাইভ সিগন্যাল চালু: {SYMBOL} | {TIMEFRAME} | বন্ধ করতে Ctrl+C")
    last_sent_candle = None
    while True:
        try:
            df = add_signals(fetch_ohlcv(limit=200))
            # শেষ ক্যান্ডেল এখনো চলমান, তাই শেষ বন্ধ হওয়া ক্যান্ডেল দেখি
            row = df.iloc[-2]
            candle_time = df.index[-2]

            if row["signal"] != 0 and candle_time != last_sent_candle:
                side = "BUY 🟢" if row["signal"] == 1 else "SELL 🔴"
                msg = (
                    f"{side} সিগন্যাল\n"
                    f"{SYMBOL} ({TIMEFRAME})\n"
                    f"দাম: {row['close']:.4f}\n"
                    f"RSI: {row['rsi']:.1f}\n"
                    f"ক্যান্ডেল: {candle_time}"
                )
                print("\n" + msg)
                send_telegram(msg)
                last_sent_candle = candle_time
            else:
                print(f"{time.strftime('%H:%M:%S')} | দাম {df['close'].iloc[-1]:.4f} | নতুন সিগন্যাল নেই", end="\r")
        except KeyboardInterrupt:
            print("\nবন্ধ করা হলো।")
            break
        except Exception as e:
            print(f"\n[ত্রুটি] {e} — আবার চেষ্টা করছি")
        time.sleep(POLL_SECONDS)


# ---------------------- একবার চেক (GitHub Actions) ----------------------
STATE_FILE = "last_signal.txt"   # শেষ যে ক্যান্ডেলের সিগন্যাল পাঠানো হয়েছে, সেটা এখানে থাকে


def once():
    df = add_signals(fetch_ohlcv(limit=200))
    row = df.iloc[-2]                 # শেষ বন্ধ হওয়া ক্যান্ডেল
    candle_id = str(df.index[-2])

    last = ""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as f:
            last = f.read().strip()

    if row["signal"] != 0 and candle_id != last:
        side = "BUY 🟢" if row["signal"] == 1 else "SELL 🔴"
        msg = (
            f"{side} সিগন্যাল\n"
            f"{SYMBOL} ({TIMEFRAME})\n"
            f"দাম: {row['close']:.4f}\n"
            f"RSI: {row['rsi']:.1f}\n"
            f"ক্যান্ডেল: {candle_id}"
        )
        print(msg)
        send_telegram(msg)
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            f.write(candle_id)
    else:
        print(f"নতুন সিগন্যাল নেই | {SYMBOL} {TIMEFRAME} | দাম {df['close'].iloc[-1]:.4f}")


def test_telegram():
    if not os.getenv("TELEGRAM_BOT_TOKEN") or not os.getenv("TELEGRAM_CHAT_ID"):
        print("TELEGRAM_BOT_TOKEN বা TELEGRAM_CHAT_ID সেট করা নেই।")
        return
    send_telegram("✅ টেস্ট মেসেজ: সিগন্যাল বট ঠিকমতো যুক্ত হয়েছে।")
    print("টেস্ট মেসেজ পাঠানো হয়েছে, Telegram দেখুন।")


# ---------------------- মেইন ----------------------
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if mode == "demo":
        print_report("ডেমো (নকল ডেটা)", backtest(synthetic_ohlcv()))
    elif mode == "backtest":
        print_report(f"ব্যাকটেস্ট {SYMBOL} {TIMEFRAME}", backtest(fetch_ohlcv()))
    elif mode == "live":
        live()
    elif mode == "once":
        once()
    elif mode == "test":
        test_telegram()
    else:
        print("ব্যবহার: python signal_bot.py [demo|backtest|live|once|test]")
