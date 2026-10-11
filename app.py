from flask import Flask, render_template, request, jsonify
import threading
import time
import os
import ccxt
import pandas as pd
import numpy as np
import datetime
import traceback
import sys
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ==========================================
# ESTADO GLOBAL DEL BOT Y UI
# ==========================================
class BotState:
    def __init__(self):
        self.is_running = False
        self.api_key = ''
        self.api_secret = ''
        self.max_margin_usd = 200.0
        self.symbols = ['BTC/USD:USD']
        self.base_currency = 'USD'
        self.timeframe = '15m'
        self.sl_pct = 2.5
        self.tp_pct = 50.0
        self.ts_activation_pct = 1.0
        self.ts_distance_pct = 0.5

        self.logs = []
        self.balance = 0.0
        self.total_balance = 0.0

        # Estados por símbolo: { symbol: data }
        self.open_positions = {}      # { 'BTC/USD:USD': {...} }
        self.last_updates = {}        # { 'BTC/USD:USD': 'HH:MM:SS' }
        self.indicators = {}          # { 'BTC/USD:USD': { rsi, adx, macd, ... } }
        self.trade_histories = {}     # { 'BTC/USD:USD': [ trades ] }
        self.strategy_metrics = {}    # { 'BTC/USD:USD': { trend, score, ... } }

        self.engine = None
        self.last_candle_timestamp = {}
        self.chart_data = {}          # { 'BTC/USD:USD': [ { time, open, high, low, close, ema21, ema55, atr } ] }

bot_state = BotState()

def log_msg(msg):
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    formatted = f"[{timestamp}] {msg}"
    bot_state.logs.insert(0, formatted)
    if len(bot_state.logs) > 100:
        bot_state.logs.pop()
    print(formatted)

# ==========================================
# MOTOR CUANTITATIVO DTM-V TRADING ENGINE
# ==========================================
class QuantitativeTradingEngine:
    def __init__(self, api_key, api_secret, symbols, timeframe, base_currency, sl_pct, tp_pct, ts_act_pct, ts_dist_pct):
        self.active_trades = {}  # { symbol: trade_dict }
        self.sl_pct = sl_pct / 100.0
        self.tp_pct = tp_pct / 100.0
        self.ts_act_pct = ts_act_pct / 100.0
        self.ts_dist_pct = ts_dist_pct / 100.0
        self.max_leverage = 5
        self.kraken_fee = 0.0005

        self.symbols = symbols
        self.timeframe = timeframe
        self.base_currency = base_currency

        self.exchange = ccxt.krakenfutures({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })

    def load_kraken_history(self, symbol):
        """Carga el historial de operaciones cerradas desde Kraken para un símbolo."""
        log_msg(f"🔄 Sincronizando historial {symbol}...")
        history = []

        try:
            trades = self.exchange.fetch_my_trades(symbol, limit=20)
            for t in trades:
                side = 'LONG' if t.get('side') == 'buy' else 'SHORT'
                price = t.get('price', 0)
                qty   = t.get('amount', 0)
                ts    = t.get('timestamp', time.time() * 1000)
                date_str = datetime.datetime.fromtimestamp(ts / 1000).strftime('%d/%m %H:%M')
                history.append({
                    'id': str(t.get('id', 'N/A'))[:8],
                    'side': side, 'entry': price, 'qty': qty, 'lev': '--',
                    'status': 'CLOSED', 'timestamp': date_str, 'pnl_pct': 'Kraken',
                    'invested_usd': '--', 'pnl_usd': '--'
                })
        except Exception as e:
            log_msg(f"⚠️ fetch_my_trades {symbol} falló: {e}")

        if not history:
            try:
                orders = self.exchange.fetch_orders(symbol, limit=20, params={'status': 'closed'})
                for o in orders:
                    if o.get('status') not in ('closed', 'filled'): continue
                    side = 'LONG' if o.get('side') == 'buy' else 'SHORT'
                    price = float(o.get('average') or o.get('price') or 0)
                    qty   = float(o.get('filled') or o.get('amount') or 0)
                    ts    = o.get('timestamp', time.time() * 1000)
                    date_str = datetime.datetime.fromtimestamp(ts / 1000).strftime('%d/%m %H:%M')
                    if price == 0 or qty == 0: continue
                    history.append({
                        'id': str(o.get('id', 'N/A'))[:8],
                        'side': side, 'entry': price, 'qty': qty, 'lev': '--',
                        'status': 'CLOSED', 'timestamp': date_str, 'pnl_pct': 'Kraken',
                        'invested_usd': '--', 'pnl_usd': '--'
                    })
            except Exception as e:
                log_msg(f"⚠️ fetch_orders fallback {symbol} falló: {e}")

        history.sort(key=lambda x: x['timestamp'], reverse=True)
        bot_state.trade_histories[symbol] = history[:20]
        log_msg(f"✅ Historial {symbol}: {len(history)} operaciones.")

    @staticmethod
    def calculate_indicators(df):
        """Calcula indicadores técnicos de alta precisión de forma vectorizada."""
        # Precios
        close = df['c']
        high = df['h']
        low = df['l']

        # 1. EMAs de Tendencia Institucional
        df['ema9'] = close.ewm(span=9, adjust=False).mean()
        df['ema21'] = close.ewm(span=21, adjust=False).mean()
        df['ema55'] = close.ewm(span=55, adjust=False).mean()
        df['ema200'] = close.ewm(span=200, adjust=False).mean()

        # 2. ATR (Average True Range - 14)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean().bfill()

        # 3. RSI Wilder (14)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
        loss = ((-delta).where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / (loss.replace(0, 1e-8))
        df['rsi'] = 100.0 - (100.0 / (1.0 + rs))

        # 4. ADX (Average Directional Index - 14) para filtro de rango/tendencia
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
        
        tr_smooth = tr.ewm(alpha=1/14, adjust=False).mean().replace(0, 1e-8)
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / tr_smooth)
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / tr_smooth)
        
        dx = 100 * ((plus_di - minus_di).abs() / ((plus_di + minus_di).replace(0, 1e-8)))
        df['adx'] = dx.ewm(alpha=1/14, adjust=False).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di

        # 5. MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # 6. Bandas de Bollinger y Keltner (Squeeze Detection)
        df['bb_mid'] = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + (2.0 * bb_std)
        df['bb_lower'] = df['bb_mid'] - (2.0 * bb_std)

        # Retornos logarítmicos
        df['ret'] = np.log(close / close.shift(1)).fillna(0)
        return df

    def get_real_data(self, symbol):
        """Descarga OHLCV y calcula indicadores técnicos."""
        ohlcv = []
        for attempt in range(3):
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=500)
                break
            except Exception as e:
                log_msg(f"⚠️ Aviso Kraken API ({symbol}): Reintentando en 2s... Error: {e}")
                time.sleep(2)

        if not ohlcv:
            raise ValueError(f"No se pudieron descargar datos de {symbol} tras 3 intentos.")

        df = pd.DataFrame(ohlcv, columns=['t', 'o', 'h', 'l', 'c', 'v'])
        for col in ['o', 'h', 'l', 'c', 'v']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = self.calculate_indicators(df)
        df = df.dropna().reset_index(drop=True)

        ticker = self.exchange.fetch_ticker(symbol)
        current_price = float(ticker['last'])
        return df, current_price

    def analyze_market_regime(self, df, symbol):
        """
        Estrategia Institucional Dynamic Trend-Momentum & Volatility Breakout (DTM-V).
        Evalúa tendencia macro, compresión/expansión, fuerza de momentum y gatillos de entrada.
        """
        last = df.iloc[-1]
        prev = df.iloc[-2]

        close = float(last['c'])
        ema9 = float(last['ema9'])
        ema21 = float(last['ema21'])
        ema55 = float(last['ema55'])
        ema200 = float(last['ema200']) if 'ema200' in last else ema55

        rsi = float(last['rsi'])
        adx = float(last['adx'])
        plus_di = float(last['plus_di'])
        minus_di = float(last['minus_di'])
        macd = float(last['macd'])
        macd_sig = float(last['macd_signal'])
        macd_hist = float(last['macd_hist'])
        prev_hist = float(prev['macd_hist'])
        atr = float(last['atr'])

        # --- EVALUACIÓN DE TENDENCIA ---
        is_uptrend = (ema21 > ema55) and (close > ema55)
        is_downtrend = (ema21 < ema55) and (close < ema55)

        # Regla de oro macro: EMA 200 como árbitro final
        above_macro = close > ema200
        below_macro = close < ema200

        # --- FILTRO DE VOLATILIDAD / RANGO (ADX) ---
        has_trend_strength = adx >= 20.0
        is_strong_trend = adx >= 28.0

        # --- SCORE INSTITUCIONAL DE DIRECCIÓN (-100 a +100) ---
        bull_score = 0
        bear_score = 0

        # Factores Tendencia (40 pts)
        if is_uptrend: bull_score += 25
        if is_downtrend: bear_score += 25
        if above_macro: bull_score += 15
        if below_macro: bear_score += 15

        # Factores Momentum MACD & DIs (35 pts)
        if macd > macd_sig: bull_score += 15
        else: bear_score += 15

        if macd_hist > 0 and macd_hist > prev_hist: bull_score += 10 # Momentum acelerando
        elif macd_hist < 0 and macd_hist < prev_hist: bear_score += 10

        if plus_di > minus_di: bull_score += 10
        elif minus_di > plus_di: bear_score += 10

        # Factores RSI (25 pts)
        if 50.0 < rsi < 70.0: bull_score += 20 # Zona óptima de impulso alcista
        elif rsi >= 70.0: bull_score += 5     # Alerta sobrecompra
        elif 30.0 < rsi < 50.0: bear_score += 20 # Zona óptima de impulso bajista
        elif rsi <= 30.0: bear_score += 5     # Alerta sobreventa

        total_pts = bull_score + bear_score
        bull_pct = round((bull_score / total_pts) * 100, 1) if total_pts > 0 else 50.0
        bear_pct = round((bear_score / total_pts) * 100, 1) if total_pts > 0 else 50.0

        # --- GATILLO DE ENTRADA CONSERVADOR & RENTABLE ---
        signal = "NEUTRAL"
        regime_desc = "RANGO LATERAL"

        if has_trend_strength:
            if bull_pct >= 65 and is_uptrend and above_macro and (macd_hist > 0):
                # Gatillo alcista: retroceso respetando EMA 21 o ruptura de momentum
                if (close > ema21) and (prev['c'] <= prev['ema21'] or macd_hist > prev_hist) and (rsi < 72):
                    signal = "LONG"
                    regime_desc = "TENDENCIA ALCISTA FUERTE"
                else:
                    regime_desc = "ALCISTA (ESPERANDO PULLBACK)"
            elif bear_pct >= 65 and is_downtrend and below_macro and (macd_hist < 0):
                # Gatillo bajista
                if (close < ema21) and (prev['c'] >= prev['ema21'] or macd_hist < prev_hist) and (rsi > 28):
                    signal = "SHORT"
                    regime_desc = "TENDENCIA BAJISTA FUERTE"
                else:
                    regime_desc = "BAJISTA (ESPERANDO PULLBACK)"
            else:
                regime_desc = "TRANSICIÓN / CONSOLIDACIÓN"
        else:
            regime_desc = "RANGO CON BAJA VOLATILIDAD (FILTRO ADX)"

        confidence = max(bull_pct, bear_pct) / 100.0

        metrics = {
            "regime": regime_desc,
            "signal": signal,
            "confidence": confidence,
            "bull_score": bull_pct,
            "bear_score": bear_pct,
            "adx": round(adx, 1),
            "rsi": round(rsi, 1),
            "macd_hist": round(macd_hist, 4),
            "atr": round(atr, 2),
            "ema21": round(ema21, 2),
            "ema55": round(ema55, 2),
            "ema200": round(ema200, 2),
            "has_trend": has_trend_strength
        }

        # Gráfico con indicadores integrados (últimas 60 velas)
        series_len = min(80, len(df))
        df_v = df.tail(series_len)
        chart_series = []
        for _, c in df_v.iterrows():
            chart_series.append({
                'time': int(float(c['t']) / 1000),
                'open': float(c['o']),
                'high': float(c['h']),
                'low': float(c['l']),
                'close': float(c['c']),
                'ema21': float(c['ema21']),
                'ema55': float(c['ema55']),
                'atr': float(c['atr']),
                'probBull': bull_pct / 100.0,
                'probBear': bear_pct / 100.0,
                'probNeutral': 0.0
            })

        bot_state.chart_data[symbol] = chart_series
        bot_state.strategy_metrics[symbol] = metrics

        log_msg(f"📊 {symbol} Régimen: {regime_desc} | ADX: {adx:.1f} | RSI: {rsi:.1f} | Bull: {bull_pct}% Bear: {bear_pct}% ➔ Señal: {signal}")
        return signal, confidence, 5, metrics

    def execute_order(self, symbol, signal, confidence, target_lev, current_price, atr=0.0):
        try:
            lev = target_lev
            if lev == 0: return
            self.exchange.set_leverage(lev, symbol)

            balance = self.exchange.fetch_balance()
            if self.base_currency not in balance or 'free' not in balance[self.base_currency]:
                log_msg(f"❌ Error: No capital libre en {self.base_currency}.")
                return

            usd_available = float(balance[self.base_currency]['free'])
            bot_state.balance = usd_available

            margin_per_asset = bot_state.max_margin_usd / max(1, len(self.symbols))
            margin_to_use = min(usd_available, margin_per_asset)

            if margin_to_use < 5:
                log_msg(f"❌ Error: Saldo insuficiente para {symbol} (${margin_to_use:.2f}).")
                return

            ticker_actual = self.exchange.fetch_ticker(symbol)['last']
            notional_value = margin_to_use / ((1.0 / lev) + self.kraken_fee)
            qty = float(self.exchange.amount_to_precision(symbol, notional_value / ticker_actual))
            if qty <= 0:
                log_msg(f"❌ Error {symbol}: qty=0.")
                return

            side = 'buy' if signal == "LONG" else 'sell'
            order = self.exchange.create_market_order(symbol, side, qty)
            real_entry_price = float(order.get('average') or order.get('price') or ticker_actual)

            # Cálculo de Stop Loss adaptativo por ATR si está disponible
            dynamic_sl_pct = (1.5 * atr / real_entry_price * 100.0 * lev) if atr > 0 else (self.sl_pct * 100.0)
            # Acotar SL entre 1.0% y el SL configurado por el usuario
            final_sl_pct = min(max(dynamic_sl_pct, 1.2), self.sl_pct * 100.0)

            trade = {
                'id': str(order.get('id', 'N/A'))[:8],
                'side': signal,
                'entry': real_entry_price,
                'qty': abs(qty),
                'lev': lev,
                'invested_usd': float(margin_to_use),
                'pnl_usd': 0.0,
                'pnl_pct': 0.0,
                'max_pnl_pct': 0.0,
                'current_price': real_entry_price,
                'dynamic_sl_pct': final_sl_pct,
                'status': 'OPEN',
                'timestamp': datetime.datetime.now().strftime('%d/%m %H:%M')
            }
            self.active_trades[symbol] = trade
            log_msg(f"🚀 POSICIÓN ABIERTA {symbol}: {signal} | {lev}x | Entry: {real_entry_price} | SL Dinámico: {final_sl_pct:.2f}%")

            if symbol not in bot_state.trade_histories:
                bot_state.trade_histories[symbol] = []
            bot_state.trade_histories[symbol].insert(0, trade.copy())
            if len(bot_state.trade_histories[symbol]) > 20:
                bot_state.trade_histories[symbol].pop()

        except Exception as e:
            log_msg(f"❌ Error en Kraken {symbol}: {e}")

    def manage_risk(self, symbol, current_price, rsi=None):
        if symbol not in self.active_trades: return
        trade = self.active_trades[symbol]

        entry = trade['entry']
        side = trade['side']
        qty = trade['qty']
        lev = trade.get('lev', 5)

        diff_decimal = (current_price - entry) / entry if side == "LONG" else (entry - current_price) / entry
        pnl_pct = diff_decimal * 100 * lev
        invested_usd = trade.get('invested_usd', 0)
        pnl_usd = invested_usd * diff_decimal * lev

        trade['pnl_pct'] = pnl_pct
        trade['pnl_usd'] = pnl_usd
        trade['current_price'] = current_price

        current_max_pnl = trade.get('max_pnl_pct', 0.0)
        if pnl_pct > current_max_pnl:
            trade['max_pnl_pct'] = pnl_pct
            current_max_pnl = pnl_pct

        motivo_cierre = None

        # 1. Smart Exit por Agotamiento de Impulso (RSI extremo con beneficios)
        if pnl_pct > 0.8:
            if side == "LONG" and rsi is not None and rsi > 82:
                motivo_cierre = f"Smart Exit (Agotamiento Alcista RSI {rsi:.1f})"
            elif side == "SHORT" and rsi is not None and rsi < 18:
                motivo_cierre = f"Smart Exit (Agotamiento Bajista RSI {rsi:.1f})"

        # 2. Stop Loss Dinámico / Configurado
        effective_sl = trade.get('dynamic_sl_pct', self.sl_pct * 100.0)
        tp_threshold = self.tp_pct * 100.0

        if not motivo_cierre:
            if pnl_pct <= -effective_sl:
                motivo_cierre = f"Stop Loss Dinámico ({pnl_pct:.2f}%)"
            elif pnl_pct >= tp_threshold:
                motivo_cierre = f"Take Profit Objetivo ({pnl_pct:.2f}%)"
            else:
                # 3. Trailing Stop Loss Cuantitativo
                act_threshold_pct = self.ts_act_pct * 100.0
                dist_pct = self.ts_dist_pct * 100.0
                if (current_max_pnl >= act_threshold_pct) and (pnl_pct <= current_max_pnl - dist_pct):
                    motivo_cierre = f"Trailing Stop (Protección Ganancia: Máx {current_max_pnl:.2f}%)"

        if motivo_cierre:
            log_msg(f"🔒 {symbol}: Cerrando posición ({motivo_cierre}) | PnL: {pnl_pct:.2f}% (${pnl_usd:.2f})")
            try:
                close_side = 'sell' if side == "LONG" else 'buy'
                self.exchange.create_market_order(symbol, close_side, float(qty), params={'reduceOnly': True})

                for t in bot_state.trade_histories.get(symbol, []):
                    if t.get('status') == 'OPEN' and t.get('id') == trade['id']:
                        t['status'] = 'CLOSED'
                        t['pnl_pct'] = f"{pnl_pct:.2f}%"
                        t['pnl_usd'] = f"{pnl_usd:.2f}$"
                        t['invested_usd'] = f"{invested_usd:.2f}$"
                        break

                del self.active_trades[symbol]
                log_msg(f"✅ {symbol} Posición cerrada exitosamente.")

            except Exception as e:
                log_msg(f"❌ Error al cerrar {symbol}: {e}")

    def force_close_position(self, symbol, reason):
        if symbol not in self.active_trades: return
        trade = self.active_trades[symbol]
        side = trade['side']
        qty = trade['qty']
        pnl = trade.get('pnl_pct', 0.0)
        log_msg(f"🔒 Cerrando forzosamente {symbol}: {reason}")
        try:
            close_side = 'sell' if side == "LONG" else 'buy'
            self.exchange.create_market_order(symbol, close_side, float(qty), params={'reduceOnly': True})

            for t in bot_state.trade_histories.get(symbol, []):
                if t.get('status') == 'OPEN' and t.get('id') == trade['id']:
                    t['status'] = 'CLOSED'
                    t['pnl_pct'] = f"{pnl:.2f}%" if isinstance(pnl, float) else str(pnl)
                    pnl_u = trade.get('pnl_usd', 0.0)
                    t['pnl_usd'] = f"{pnl_u:.2f}$" if isinstance(pnl_u, float) else str(pnl_u)
                    inv_u = trade.get('invested_usd', 0.0)
                    t['invested_usd'] = f"{inv_u:.2f}$" if isinstance(inv_u, float) else str(inv_u)
                    break

            del self.active_trades[symbol]
            log_msg(f"✅ {symbol} cerrada forzosamente.")
        except Exception as e:
            log_msg(f"❌ Error force close {symbol}: {e}")

# ==========================================
# HILO SECUNDARIO DEL BOT
# ==========================================
def bot_worker():
    log_msg("🤖 Motor Cuantitativo DTM-V iniciando...")
    bot_state.engine = QuantitativeTradingEngine(
        api_key=bot_state.api_key,
        api_secret=bot_state.api_secret,
        symbols=bot_state.symbols,
        timeframe=bot_state.timeframe,
        base_currency=bot_state.base_currency,
        sl_pct=bot_state.sl_pct,
        tp_pct=bot_state.tp_pct,
        ts_act_pct=bot_state.ts_activation_pct,
        ts_dist_pct=bot_state.ts_distance_pct
    )

    try:
        bal = bot_state.engine.exchange.fetch_balance()
        if bot_state.base_currency in bal and 'free' in bal[bot_state.base_currency]:
            bot_state.balance = float(bal[bot_state.base_currency]['free'])
            log_msg(f"💰 Saldo inicial: {bot_state.balance:.2f} {bot_state.base_currency}")

        for sym in bot_state.symbols:
            bot_state.engine.load_kraken_history(sym)
            open_positions = bot_state.engine.exchange.fetch_positions(symbols=[sym])
            if open_positions:
                pos = open_positions[0]
                if float(pos.get('contracts', 0)) > 0:
                    side = 'LONG' if pos.get('side') == 'long' else 'SHORT'
                    entry = float(pos.get('entryPrice', 0))
                    qty = float(pos.get('contracts', 0))
                    inv_usd = float(pos.get('initialMargin', 0))
                    
                    trade = {
                        'id': 'RECUPERADA', 'side': side, 'entry': entry, 'qty': qty, 'lev': 5,
                        'invested_usd': inv_usd, 'pnl_usd': 0.0,
                        'pnl_pct': 0.0, 'max_pnl_pct': 0.0, 'current_price': entry,
                        'status': 'OPEN', 'timestamp': datetime.datetime.now().strftime('%d/%m %H:%M')
                    }
                    bot_state.engine.active_trades[sym] = trade
                    if sym not in bot_state.trade_histories: bot_state.trade_histories[sym] = []
                    bot_state.trade_histories[sym].insert(0, trade.copy())
                    log_msg(f"🔄 {sym} posición recuperada del exchange.")

    except Exception as e:
        log_msg(f"⚠️ Aviso inicialización: {e}")

    while bot_state.is_running:
        for symbol in bot_state.symbols:
            if not bot_state.is_running: break

            bot_state.last_updates[symbol] = datetime.datetime.now().strftime('%H:%M:%S')
            try:
                df, price = bot_state.engine.get_real_data(symbol)

                rsi = float(df['rsi'].iloc[-1])
                adx = float(df['adx'].iloc[-1])
                macd_h = float(df['macd_hist'].iloc[-1])
                atr = float(df['atr'].iloc[-1])

                bot_state.indicators[symbol] = {
                    'rsi': rsi,
                    'adx': adx,
                    'macd_hist': macd_h,
                    'atr': atr,
                    'price': price
                }

                current_candle_ts = df['t'].iloc[-1]
                if symbol not in bot_state.last_candle_timestamp:
                    bot_state.last_candle_timestamp[symbol] = 0

                is_new_candle = current_candle_ts > bot_state.last_candle_timestamp[symbol]

                # Análisis institucional de mercado en cada ciclo
                signal, confidence, target_lev, metrics = bot_state.engine.analyze_market_regime(df, symbol)
                bot_state.indicators[symbol]['last_signal'] = signal

                if is_new_candle:
                    bot_state.last_candle_timestamp[symbol] = current_candle_ts
                    log_msg(f"⏱️ {symbol} Vela completada. Convicción: {confidence:.1%} [{metrics['regime']}]")

                    # Si hay cambio de tendencia estructural en contra de la posición abierta, asegurar salida
                    if symbol in bot_state.engine.active_trades:
                        current_side = bot_state.engine.active_trades[symbol]['side']
                        if (current_side == "LONG" and metrics['bear_score'] >= 75) or \
                           (current_side == "SHORT" and metrics['bull_score'] >= 75):
                            log_msg(f"🔀 {symbol} Salida por reversión estructural de tendencia.")
                            bot_state.engine.force_close_position(symbol, f"Reversión de Régimen a {metrics['regime']}")

                # Gestión activa de la posición
                if symbol in bot_state.engine.active_trades:
                    bot_state.engine.manage_risk(symbol, price, rsi=rsi)
                else:
                    # Entradas selectivas
                    if signal == "LONG":
                        bot_state.engine.execute_order(symbol, "LONG", confidence, 5, price, atr=atr)
                    elif signal == "SHORT":
                        bot_state.engine.execute_order(symbol, "SHORT", confidence, 5, price, atr=atr)

                bot_state.open_positions[symbol] = bot_state.engine.active_trades.get(symbol)

            except Exception as e:
                log_msg(f"⚠️ Error {symbol}: {e}")

        # Actualizar balance
        try:
            bal = bot_state.engine.exchange.fetch_balance()
            if bot_state.base_currency in bal:
                if 'free' in bal[bot_state.base_currency]:
                    bot_state.balance = float(bal[bot_state.base_currency]['free'])
                if 'total' in bal[bot_state.base_currency]:
                    bot_state.total_balance = float(bal[bot_state.base_currency]['total'])
        except: pass

        for _ in range(15):
            if not bot_state.is_running: break
            time.sleep(1)

    log_msg("🛑 Motor finalizado.")

# ==========================================
# RUTAS DEL SERVIDOR WEB FLASK
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start', methods=['POST'])
def start_bot():
    if bot_state.is_running:
        return jsonify({"status": "error", "message": "Ya está en ejecución."})

    data = request.json or {}
    bot_state.api_key = data.get('api_key', '')
    bot_state.api_secret = data.get('api_secret', '')
    bot_state.max_margin_usd = float(data.get('max_margin', 200))

    sym_data = data.get('symbol', 'BTC/USD:USD')
    if isinstance(sym_data, str):
        bot_state.symbols = [s.strip() for s in sym_data.split(',') if s.strip()]
    else:
        bot_state.symbols = sym_data

    bot_state.base_currency = data.get('base_currency', 'USD')
    bot_state.timeframe = data.get('timeframe', '15m')
    bot_state.sl_pct = float(data.get('sl_pct', 2.5))
    bot_state.tp_pct = float(data.get('tp_pct', 50.0))
    bot_state.ts_activation_pct = float(data.get('ts_act_pct', 1.0))
    bot_state.ts_distance_pct = float(data.get('ts_dist_pct', 0.5))

    bot_state.is_running = True
    bot_state.logs = []
    bot_state.open_positions = {}
    bot_state.trade_histories = {}
    bot_state.indicators = {}
    bot_state.last_candle_timestamp = {}
    bot_state.strategy_metrics = {}
    bot_state.last_updates = {}

    t = threading.Thread(target=bot_worker)
    t.daemon = True
    t.start()

    return jsonify({"status": "started"})

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    bot_state.is_running = False
    return jsonify({"status": "stopped"})

@app.route('/api/state')
def get_state():
    return jsonify({
        "is_running": bot_state.is_running,
        "balance": f"{bot_state.balance:.2f}",
        "total_balance": f"{getattr(bot_state, 'total_balance', bot_state.balance):.2f}",
        "base_currency": bot_state.base_currency,
        "symbols": bot_state.symbols,
        "logs": bot_state.logs,
        "open_positions": bot_state.open_positions,
        "last_updates": bot_state.last_updates,
        "strategy_metrics": bot_state.strategy_metrics,
        "current_probs_dict": {
            sym: {
                "Alcista (t+1)": m.get('bull_score', 50) / 100.0,
                "Bajista (t+1)": m.get('bear_score', 50) / 100.0,
                "regime": m.get('regime', 'Analizando...'),
                "adx": m.get('adx', 0),
                "rsi": m.get('rsi', 50),
                "atr": m.get('atr', 0),
                "macd_hist": m.get('macd_hist', 0),
                "has_trend": m.get('has_trend', False)
            }
            for sym, m in bot_state.strategy_metrics.items()
        },
        "trade_histories": bot_state.trade_histories,
        "indicators": bot_state.indicators
    })

@app.route('/api/chart_data')
def get_chart_data():
    symbol = request.args.get('symbol', '').strip()
    if not symbol: return jsonify([])

    data = bot_state.chart_data.get(symbol, [])

    if not data:
        try:
            public_ex = ccxt.krakenfutures({'enableRateLimit': True})
            ohlcv = public_ex.fetch_ohlcv(symbol, bot_state.timeframe or '15m', limit=80)
            df = pd.DataFrame(ohlcv, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            for col in ['o', 'h', 'l', 'c', 'v']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['ema21'] = df['c'].ewm(span=21, adjust=False).mean()
            df['ema55'] = df['c'].ewm(span=55, adjust=False).mean()

            chart_series = []
            for _, candle in df.iterrows():
                chart_series.append({
                    'time': int(float(candle['t']) / 1000),
                    'open': float(candle['o']),
                    'high': float(candle['h']),
                    'low': float(candle['l']),
                    'close': float(candle['c']),
                    'ema21': float(candle['ema21']),
                    'ema55': float(candle['ema55']),
                    'probBull': 0.5,
                    'probBear': 0.5,
                    'probNeutral': 0.0
                })
            return jsonify(chart_series)
        except Exception as e:
            app.logger.error(f"Fallback chart error: {e}")

    try:
        import math
        def sanitize_nans(obj):
            if isinstance(obj, dict):
                return {k: sanitize_nans(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_nans(x) for x in obj]
            elif isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return 0.0
            return obj
        return jsonify(sanitize_nans(data))
    except Exception:
        import json
        safe_json = json.dumps(data, default=lambda o: float(o) if hasattr(o, 'item') else str(o))
        safe_json = safe_json.replace('NaN', '0.0').replace('Infinity', '0.0')
        return app.response_class(response=safe_json, status=200, mimetype='application/json')

@app.route('/ping')
def ping():
    return jsonify({"status": "ok", "running": bot_state.is_running, "strategy": "DTM-V Institutional"})

def autostart_from_env():
    api_key = os.environ.get('KRAKEN_API_KEY', '')
    api_secret = os.environ.get('KRAKEN_API_SECRET', '')

    if not api_key or not api_secret:
        print("[AUTOSTART] No se encontraron KRAKEN_API_KEY/SECRET en env vars. Arranca manualmente desde el panel.")
        return

    print("[AUTOSTART] ✅ Variables de entorno detectadas. Iniciando bot automáticamente con DTM-V...")

    bot_state.api_key = api_key
    bot_state.api_secret = api_secret
    bot_state.max_margin_usd = float(os.environ.get('MAX_MARGIN', 200))
    sym_env = os.environ.get('SYMBOL', 'BTC/USD:USD')
    bot_state.symbols = [s.strip() for s in sym_env.split(',') if s.strip()]
    bot_state.base_currency = os.environ.get('BASE_CURRENCY', 'USD')
    bot_state.timeframe = os.environ.get('TIMEFRAME', '15m')
    bot_state.sl_pct = float(os.environ.get('SL_PCT', 2.5))
    bot_state.tp_pct = float(os.environ.get('TP_PCT', 50.0))
    bot_state.ts_activation_pct = float(os.environ.get('TS_ACT_PCT', 1.0))
    bot_state.ts_distance_pct = float(os.environ.get('TS_DIST_PCT', 0.5))

    bot_state.is_running = True
    t = threading.Thread(target=bot_worker)
    t.daemon = True
    t.start()

autostart_from_env()

if __name__ == '__main__':
    print("\n-----------------------------------------------------------")
    print("PANEL INSTITUCIONAL DTM-V LISTO")
    print("1. Abre tu navegador web (Chrome, Edge, Safari...)")
    print("2. Ve a: http://127.0.0.1:5000")
    print("-----------------------------------------------------------\n")
    app.run(host='127.0.0.1', port=5000, debug=False)
