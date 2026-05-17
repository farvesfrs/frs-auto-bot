#!/usr/bin/env python3
"""
╔══════════════════════════════════════════╗
║         FRS AUTO BOT v1.2               ║
║   by Farvees - FRS UNIQUE SPARE PARTS   ║
╠══════════════════════════════════════════╣
║ Strategy : FRS SMC v5 + VWAP Zeiierman  ║
║          + CCI (25,EMA14) + Fibonacci   ║
║ Timeframe: 4H                           ║
║ Exchange : Binance → MEXC (Phase 2)     ║
║ Mode     : SPOT → FUTURES               ║
║ v1.2     : Paper Trading / Sim Mode     ║
╚══════════════════════════════════════════╝
"""

import os
import ccxt
import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime
from flask import Flask, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════
# PORTABLE KEY STORAGE
# Uses Replit DB when running on Replit,
# falls back to env vars → local JSON file
# on any other host (Render, VPS, etc.)
# ═══════════════════════════════════════════
import json as _json

_KEYS_FILE = os.path.join(BASE_DIR, "saved_keys.json")

def _replit_db_available():
    return bool(os.environ.get("REPLIT_DB_URL"))

def _kv_set(key, value):
    if _replit_db_available():
        from replit import db
        db[key] = value
        return
    try:
        data = _json.load(open(_KEYS_FILE)) if os.path.exists(_KEYS_FILE) else {}
    except Exception:
        data = {}
    data[key] = value
    with open(_KEYS_FILE, "w") as f:
        _json.dump(data, f)

def _kv_get(key, default=""):
    if _replit_db_available():
        from replit import db
        val = db.get(key, default)
        return str(val) if val else default
    # check environment variable override first (useful for Render dashboard)
    env_map = {"frs_api_key": "API_KEY", "frs_secret_key": "SECRET_KEY"}
    env_val = os.environ.get(env_map.get(key, ""), "")
    if env_val:
        return env_val
    try:
        data = _json.load(open(_KEYS_FILE)) if os.path.exists(_KEYS_FILE) else {}
        return data.get(key, default)
    except Exception:
        return default

app = Flask(__name__)

# ═══════════════════════════════════════════
# GLOBAL BOT STATE
# ═══════════════════════════════════════════
bot_state = {
    "running"       : False,
    "paper_mode"    : False,
    "config"        : {},
    "logs"          : [],
    "last_signal"   : "WAIT",
    "in_trade"      : False,
    "entry_price"   : None,
    "trade_side"    : None,
    "pnl"           : 0.0,
    "total_trades"  : 0,
    "wins"          : 0,
    "losses"        : 0,
    "paper_balance" : 1000.0,
    "paper_equity"  : 1000.0,
    "paper_trades"  : [],
    "confluence"    : {
        "buy_checks" : {},
        "sell_checks": {},
        "buy_score"  : 0,
        "sell_score" : 0,
        "cci"        : 0,
        "vwap"       : 0,
        "trend"      : "—",
        "ema_dir"    : "—",
        "price"      : 0,
    },
}

# ═══════════════════════════════════════════
# HELPER: LOG
# ═══════════════════════════════════════════
def log(msg):
    ts    = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    bot_state["logs"].insert(0, entry)
    bot_state["logs"] = bot_state["logs"][:60]
    print(entry)

# ═══════════════════════════════════════════
# INDICATOR 1 — EMA
# ═══════════════════════════════════════════
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

# ═══════════════════════════════════════════
# INDICATOR 2 — FRS SMC v5 (exact Pine logic)
# ═══════════════════════════════════════════
def detect_frs_smc(df):
    c  = df["close"]
    o  = df["open"]
    h  = df["high"]
    l  = df["low"]

    ema20  = ema(c, 20)
    ema50  = ema(c, 50)
    ema200 = ema(c, 200)

    bc = c < o
    gc = c > o

    bOB = bc.shift(1) & gc & ((c - o) > (h.shift(1) - l.shift(1)) * 0.5)
    sOB = gc.shift(1) & bc & ((o - c) > (h.shift(1) - l.shift(1)) * 0.5)

    buy_sig  = bOB & (c > ema20) & (c > ema50)
    sell_sig = sOB & (c < ema20) & (c < ema50)

    trend_bull = c.iloc[-1] > ema200.iloc[-1]
    ema_up     = ema20.iloc[-1] > ema50.iloc[-1]

    return {
        "bull_ob"    : bool(bOB.iloc[-1]),
        "bear_ob"    : bool(sOB.iloc[-1]),
        "buy_signal" : bool(buy_sig.iloc[-1]),
        "sell_signal": bool(sell_sig.iloc[-1]),
        "trend"      : "BULLISH" if trend_bull else "BEARISH",
        "ema_dir"    : "UP" if ema_up else "DOWN",
        "ema20"      : round(ema20.iloc[-1], 4),
        "ema50"      : round(ema50.iloc[-1], 4),
    }

# ═══════════════════════════════════════════
# INDICATOR 3 — CCI (Length:25, EMA:14)
# ═══════════════════════════════════════════
def calc_cci(df, length=25, ema_len=14):
    src = (df["high"] + df["low"]) / 2
    ma  = src.rolling(length).mean()
    mad = src.rolling(length).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci     = (src - ma) / (0.015 * mad)
    cci_ema = cci.ewm(span=ema_len, adjust=False).mean()
    return cci, cci_ema

# ═══════════════════════════════════════════
# INDICATOR 4 — Harmonic Rolling VWAP Zeiierman
# ═══════════════════════════════════════════
def calc_vwap_zeiierman(df, window=100):
    src   = (df["high"] + df["low"] + df["close"]) / 3
    pv    = src * df["volume"]
    rvwap = pv.rolling(window).sum() / df["volume"].rolling(window).sum()

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"]  - df["close"].shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    dev = tr.rolling(window).std()

    ub2 = rvwap + dev * 2.5
    lb2 = rvwap - dev * 2.5
    ub3 = rvwap + dev * 3.0
    lb3 = rvwap - dev * 3.0

    return rvwap, ub2, lb2, ub3, lb3

# ═══════════════════════════════════════════
# INDICATOR 5 — Fibonacci (0.5 - 0.62 zone)
# ═══════════════════════════════════════════
def check_fib_zone(df, lookback=50):
    high  = df["high"].iloc[-lookback:].max()
    low   = df["low"].iloc[-lookback:].min()
    price = df["close"].iloc[-1]
    rng   = high - low

    fib50  = high - rng * 0.500
    fib62  = high - rng * 0.618

    bull_fib = fib62 <= price <= fib50
    bear_fib = (high - rng * 0.236) <= price <= high

    return bull_fib, bear_fib, fib50, fib62

# ═══════════════════════════════════════════
# CONFLUENCE ENGINE
# ═══════════════════════════════════════════
def run_confluence(df):
    price = df["close"].iloc[-1]

    smc                          = detect_frs_smc(df)
    cci, cci_ema                 = calc_cci(df, 25, 14)
    rvwap, ub2, lb2, ub3, lb3   = calc_vwap_zeiierman(df, 100)
    bull_fib, bear_fib, f50, f62 = check_fib_zone(df)

    cci_val  = cci.iloc[-1]
    vwap_val = rvwap.iloc[-1]
    lb2_val  = lb2.iloc[-1]
    ub2_val  = ub2.iloc[-1]

    buy_checks = {
        "FRS SMC BUY"    : smc["buy_signal"],
        "CCI < -100"     : cci_val < -100,
        "VWAP LowerBand" : price <= lb2_val,
        "Fib 0.5-0.62"   : bull_fib,
        "EMA Bullish"    : smc["ema_dir"] == "UP",
    }

    sell_checks = {
        "FRS SMC SELL"   : smc["sell_signal"],
        "CCI > +100"     : cci_val > 100,
        "VWAP UpperBand" : price >= ub2_val,
        "Fib Bear Zone"  : bear_fib,
        "EMA Bearish"    : smc["ema_dir"] == "DOWN",
    }

    buy_score  = sum(buy_checks.values())
    sell_score = sum(sell_checks.values())

    if buy_score == 5:
        signal = "BUY"
    elif sell_score == 5:
        signal = "SELL"
    else:
        signal = "WAIT"

    return {
        "price"      : round(price, 6),
        "signal"     : signal,
        "buy_score"  : buy_score,
        "sell_score" : sell_score,
        "buy_checks" : buy_checks,
        "sell_checks": sell_checks,
        "cci"        : round(cci_val, 2),
        "vwap"       : round(vwap_val, 6),
        "trend"      : smc["trend"],
        "ema_dir"    : smc["ema_dir"],
    }

# ═══════════════════════════════════════════
# EXCHANGE CONNECT
# Paper mode uses public API only (no keys needed)
# ═══════════════════════════════════════════
def get_exchange(config, paper=False):
    name   = config.get("exchange", "binance").lower()
    mode   = config.get("mode", "spot").lower()
    params = {
        "enableRateLimit": True,
        "options"        : {
            "defaultType"            : mode,
            "adjustForTimeDifference": True,
            "recvWindow"             : 10000,
        },
    }
    if not paper:
        params["apiKey"] = config.get("api_key", "")
        params["secret"] = config.get("secret_key", "")

    if name == "binance":
        ex = ccxt.binance(params)
        if not paper:
            ex.load_time_difference()
        return ex
    elif name == "mexc":
        return ccxt.mexc(params)
    raise ValueError(f"Unknown exchange: {name}")

# ═══════════════════════════════════════════
# FETCH 4H CANDLES
# ═══════════════════════════════════════════
def fetch_candles(exchange, symbol, limit=300):
    ohlcv = exchange.fetch_ohlcv(symbol, "4h", limit=limit)
    df = pd.DataFrame(ohlcv,
        columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

# ═══════════════════════════════════════════
# TRADE AMOUNT (real mode)
# ═══════════════════════════════════════════
def get_trade_amount(exchange, config):
    mode = config.get("amount_mode", "fixed")
    if mode == "percent":
        pct  = float(config.get("capital_percent", 50)) / 100
        bal  = exchange.fetch_balance()
        usdt = bal.get("USDT", {}).get("free", 0)
        return max(10.0, usdt * pct)
    return max(10.0, float(config.get("trade_amount", 10)))

# ═══════════════════════════════════════════
# PAPER TRADE AMOUNT
# ═══════════════════════════════════════════
def get_paper_amount(config):
    mode = config.get("amount_mode", "fixed")
    if mode == "percent":
        pct = float(config.get("capital_percent", 50)) / 100
        return max(10.0, bot_state["paper_balance"] * pct)
    return max(10.0, float(config.get("trade_amount", 10)))

# ═══════════════════════════════════════════
# PLACE REAL ORDER
# ═══════════════════════════════════════════
def place_order(exchange, side, symbol, amount_usdt):
    try:
        ticker = exchange.fetch_ticker(symbol)
        price  = ticker["last"]
        qty    = exchange.amount_to_precision(
                    symbol, amount_usdt / price)
        order  = exchange.create_order(
                    symbol, "market", side, float(qty))
        log(f"✅ {side.upper()} {qty} {symbol} @ ${price:.4f}")
        return order, price
    except Exception as e:
        log(f"❌ Order error: {e}")
        return None, None

# ═══════════════════════════════════════════
# PAPER ORDER — simulated, no real money
# ═══════════════════════════════════════════
def paper_order(side, symbol, amount_usdt, price):
    qty = round(amount_usdt / price, 6)
    log(f"📝 [PAPER] {side.upper()} {qty} {symbol} @ ${price:.4f} (sim ${amount_usdt:.2f})")
    record = {
        "side"  : side,
        "symbol": symbol,
        "qty"   : qty,
        "price" : price,
        "amount": amount_usdt,
        "time"  : datetime.now().strftime("%H:%M:%S"),
    }
    bot_state["paper_trades"].insert(0, record)
    bot_state["paper_trades"] = bot_state["paper_trades"][:50]
    return {"id": "paper", "status": "closed"}, price

# ═══════════════════════════════════════════
# BOT MAIN LOOP
# ═══════════════════════════════════════════
def bot_loop():
    config     = bot_state["config"]
    symbol     = config.get("coin", "BTC/USDT")
    tp_pct     = float(config.get("tp_percent", 4.0))
    sl_pct     = float(config.get("sl_percent", 2.0))
    ex_name    = config.get("exchange", "Binance").upper()
    paper      = bot_state["paper_mode"]

    in_trade    = False
    entry_price = None
    trade_side  = None

    mode_label = "📝 PAPER" if paper else "💰 LIVE"
    log(f"🤖 FRS AUTO BOT v1.2 Started! {mode_label}")
    log(f"📡 Exchange : {ex_name}")
    log(f"🪙 Coin     : {symbol}")
    log(f"⏰ Timeframe: 4H")
    log(f"🎯 TP: {tp_pct}% | SL: {sl_pct}%")

    if paper:
        log(f"💵 Paper Balance: ${bot_state['paper_balance']:.2f} USDT (virtual)")
        log("🔒 No real orders will be placed")

    try:
        exchange = get_exchange(config, paper=paper)
        if not paper:
            balance  = exchange.fetch_balance()
            usdt_bal = balance.get("USDT", {}).get("free", 0)
            log(f"✅ {ex_name} Connected! Balance: ${usdt_bal:.2f}")
        else:
            exchange.load_markets()
            log(f"✅ {ex_name} Public Data Connected!")
    except Exception as e:
        log(f"❌ Connect failed: {e}")
        bot_state["running"] = False
        return

    while bot_state["running"]:
        try:
            df     = fetch_candles(exchange, symbol)
            result = run_confluence(df)

            price  = result["price"]
            signal = result["signal"]
            bot_state["last_signal"] = signal

            bot_state["confluence"] = {
                "buy_checks" : result["buy_checks"],
                "sell_checks": result["sell_checks"],
                "buy_score"  : result["buy_score"],
                "sell_score" : result["sell_score"],
                "cci"        : result["cci"],
                "vwap"       : result["vwap"],
                "trend"      : result["trend"],
                "ema_dir"    : result["ema_dir"],
                "price"      : price,
            }

            log(f"📊 ${price} | CCI:{result['cci']} | "
                f"Trend:{result['trend']} | "
                f"BUY:{result['buy_score']}/5 "
                f"SELL:{result['sell_score']}/5 → {signal}")

            # ── TP / SL check ──
            if in_trade and entry_price:
                if trade_side == "buy":
                    pnl = (price - entry_price) / entry_price * 100
                else:
                    pnl = (entry_price - price) / entry_price * 100

                bot_state["pnl"]         = round(pnl, 2)
                bot_state["in_trade"]    = True
                bot_state["entry_price"] = entry_price

                # Update paper equity live
                if paper:
                    amt = get_paper_amount(config)
                    bot_state["paper_equity"] = round(
                        bot_state["paper_balance"] + amt * pnl / 100, 2)

                if pnl >= tp_pct:
                    log(f"🎯 TP HIT! +{pnl:.2f}%")
                    amt = get_paper_amount(config) if paper else get_trade_amount(exchange, config)
                    if paper:
                        paper_order("sell", symbol, amt, price)
                        profit = round(amt * pnl / 100, 2)
                        bot_state["paper_balance"] = round(
                            bot_state["paper_balance"] + profit, 2)
                        bot_state["paper_equity"] = bot_state["paper_balance"]
                        log(f"💵 Paper Balance: ${bot_state['paper_balance']:.2f} (+${profit:.2f})")
                    else:
                        place_order(exchange, "sell", symbol, amt)
                    bot_state["wins"] += 1
                    bot_state["total_trades"] += 1
                    in_trade = entry_price = trade_side = None
                    bot_state["in_trade"]    = False
                    bot_state["entry_price"] = None

                elif pnl <= -sl_pct:
                    log(f"🛑 SL HIT! {pnl:.2f}%")
                    amt = get_paper_amount(config) if paper else get_trade_amount(exchange, config)
                    if paper:
                        paper_order("sell", symbol, amt, price)
                        loss = round(amt * abs(pnl) / 100, 2)
                        bot_state["paper_balance"] = round(
                            bot_state["paper_balance"] - loss, 2)
                        bot_state["paper_equity"] = bot_state["paper_balance"]
                        log(f"💵 Paper Balance: ${bot_state['paper_balance']:.2f} (-${loss:.2f})")
                    else:
                        place_order(exchange, "sell", symbol, amt)
                    bot_state["losses"] += 1
                    bot_state["total_trades"] += 1
                    in_trade = entry_price = trade_side = None
                    bot_state["in_trade"]    = False
                    bot_state["entry_price"] = None

            # ── New entry ──
            if not in_trade:
                amt = get_paper_amount(config) if paper else get_trade_amount(exchange, config)

                if signal == "BUY":
                    log(f"🟢 5/5 Confluence! BUY ${amt:.2f}")
                    if paper:
                        order, ep = paper_order("buy", symbol, amt, price)
                    else:
                        order, ep = place_order(exchange, "buy", symbol, amt)
                    if order:
                        in_trade    = True
                        entry_price = ep
                        trade_side  = "buy"
                        bot_state["in_trade"]    = True
                        bot_state["entry_price"] = ep
                        bot_state["trade_side"]  = "buy"

                elif signal == "SELL":
                    if config.get("mode") == "futures":
                        log(f"🔴 5/5 Confluence! SHORT ${amt:.2f}")
                        if paper:
                            order, ep = paper_order("sell", symbol, amt, price)
                        else:
                            order, ep = place_order(exchange, "sell", symbol, amt)
                        if order:
                            in_trade    = True
                            entry_price = ep
                            trade_side  = "sell"
                            bot_state["in_trade"]    = True
                            bot_state["entry_price"] = ep
                            bot_state["trade_side"]  = "sell"
                    else:
                        log("🔴 SELL signal — SPOT mode: waiting for BUY first")

            for _ in range(180):
                if not bot_state["running"]:
                    break
                time.sleep(10)

        except Exception as e:
            log(f"❌ Loop error: {e}")
            time.sleep(60)

    log("⏹ Bot stopped.")
    bot_state["in_trade"] = False

# ═══════════════════════════════════════════
# FLASK ROUTES
# ═══════════════════════════════════════════
@app.route("/")
def index():
    ui_path = os.path.join(BASE_DIR, "ui.html")
    return open(ui_path, encoding="utf-8").read()

@app.route("/start", methods=["POST"])
def start():
    if bot_state["running"]:
        return jsonify({"ok": False, "message": "⚠️ Already running!"})
    cfg   = request.get_json()
    paper = cfg.get("paper_mode", False)

    if not paper and (not cfg.get("api_key") or not cfg.get("secret_key")):
        return jsonify({"ok": False, "message": "❌ API Key & Secret required for Live mode!"})

    paper_start_bal = float(cfg.get("paper_balance", 1000.0))

    bot_state.update({
        "config"        : cfg,
        "running"       : True,
        "paper_mode"    : paper,
        "logs"          : [],
        "pnl"           : 0,
        "total_trades"  : 0,
        "wins"          : 0,
        "losses"        : 0,
        "in_trade"      : False,
        "entry_price"   : None,
        "last_signal"   : "WAIT",
        "paper_balance" : paper_start_bal,
        "paper_equity"  : paper_start_bal,
        "paper_trades"  : [],
        "confluence"    : {
            "buy_checks" : {}, "sell_checks": {},
            "buy_score"  : 0,  "sell_score" : 0,
            "cci"        : 0,  "vwap"       : 0,
            "trend"      : "—","ema_dir"    : "—",
            "price"      : 0,
        },
    })
    threading.Thread(target=bot_loop, daemon=True).start()
    label = "📝 Paper Trading" if paper else "💰 Live Trading"
    return jsonify({"ok": True, "message": f"🚀 FRS Bot Started! ({label})"})

@app.route("/stop", methods=["POST"])
def stop():
    bot_state["running"] = False
    return jsonify({"ok": True, "message": "⏹ Bot Stopped!"})

# ═══════════════════════════════════════════
# KEY STORAGE — Replit DB (server-side only)
# ═══════════════════════════════════════════
def mask_key(key):
    """Show first 4 and last 4 chars, mask the middle."""
    if not key or len(key) < 9:
        return "****"
    return key[:4] + "*" * (len(key) - 8) + key[-4:]

@app.route("/save-keys", methods=["POST"])
def save_keys():
    data = request.get_json()
    api_key    = data.get("api_key", "").strip()
    secret_key = data.get("secret_key", "").strip()
    if not api_key or not secret_key:
        return jsonify({"ok": False, "message": "❌ Both keys are required."})
    _kv_set("frs_api_key",    api_key)
    _kv_set("frs_secret_key", secret_key)
    return jsonify({
        "ok"        : True,
        "message"   : "Keys Saved Permanently ✅",
        "masked_api": mask_key(api_key),
        "masked_sec": mask_key(secret_key),
    })

@app.route("/load-keys")
def load_keys():
    api_key    = _kv_get("frs_api_key",    "")
    secret_key = _kv_get("frs_secret_key", "")
    return jsonify({
        "has_keys"  : bool(api_key and secret_key),
        "api_key"   : str(api_key)    if api_key    else "",
        "secret_key": str(secret_key) if secret_key else "",
        "masked_api": mask_key(str(api_key))    if api_key    else "",
        "masked_sec": mask_key(str(secret_key)) if secret_key else "",
    })

@app.route("/status")
def status():
    return jsonify({
        "running"       : bot_state["running"],
        "paper_mode"    : bot_state["paper_mode"],
        "last_signal"   : bot_state["last_signal"],
        "in_trade"      : bot_state["in_trade"],
        "entry_price"   : bot_state["entry_price"],
        "trade_side"    : bot_state["trade_side"],
        "pnl"           : bot_state["pnl"],
        "total_trades"  : bot_state["total_trades"],
        "wins"          : bot_state["wins"],
        "losses"        : bot_state["losses"],
        "paper_balance" : bot_state["paper_balance"],
        "paper_equity"  : bot_state["paper_equity"],
        "paper_trades"  : bot_state["paper_trades"][:10],
        "logs"          : bot_state["logs"][:25],
        "confluence"    : bot_state["confluence"],
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 45)
    print("  🤖 FRS AUTO BOT v1.2 by Farvees")
    print(f"  📱 Open: http://localhost:{port}")
    print("  ✅ Paper Trading Mode Added")
    print("=" * 45)
    app.run(host="0.0.0.0", port=port, debug=False)
