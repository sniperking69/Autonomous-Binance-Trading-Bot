import torch
import numpy as np
import pandas as pd
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException
from datetime import datetime, timedelta, timezone
import os
import json
from time import sleep
from keys import api_key, api_secret
from stonevision import PredRNNpp, Num_layers, Num_hidden, In_channel, Out_channel, Period_size, Chunk_size, GRID_SIZE, marketDataImage

# --- Config ---
model_path = "pred_rnn_model.pth"
max_positions = 8
mode = 'S'  # 'S'=Simulation, 'R'=Real Trading
TRADED_BUFFER_FILE = "traded_buffer.json"
FRAMES_TO_AVERAGE = 4
supra_delta = 8

# --- Device & API ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
client = Client(api_key, api_secret)

# --- Helpers ---
def load_traded_buffer():
    if os.path.exists(TRADED_BUFFER_FILE):
        with open(TRADED_BUFFER_FILE, "r") as f:
            data = json.load(f)
            return {k: datetime.strptime(v, "%Y-%m-%d %H:%M:%S%z") for k, v in data.items()}
    return {}

def save_traded_buffer(buffer):
    data = {k: v.strftime("%Y-%m-%d %H:%M:%S%z") for k, v in buffer.items()}
    with open(TRADED_BUFFER_FILE, "w") as f:
        json.dump(data, f)

def get_token_info():
    exInfo = client.futures_exchange_info()
    return {
        s["symbol"]: {
            "quantityPrecision": s["quantityPrecision"],
            "minNotional": float(next(f["notional"] for f in s["filters"] if f["filterType"] == "MIN_NOTIONAL"))
        } for s in exInfo["symbols"] if s["contractType"] == "PERPETUAL" and "USDT" in s["symbol"] and s["status"] == "TRADING"
    }

def get_futures_balance():
    for asset in client.futures_account_balance():
        if asset['asset'] == 'USDT':
            return float(asset['balance'])
    return 0.0

def compute_volatility(token_data, tokens):
    vol_dict = {}
    for token in tokens:
        try:
            last_changes = token_data[token]["pct_change"].dropna().iloc[-FRAMES_TO_AVERAGE:]
            volatility = np.std(last_changes)
            vol_dict[token] = volatility
        except:
            vol_dict[token] = 0
    return vol_dict

def truncate(number, precision):
    return int(number * (10 ** precision)) / (10 ** precision)

def place_order(client, symbol, side, quantity, precision, mode, reduce_only=False):
    if mode == 'R':
        try:
            order = client.futures_create_order(
                symbol=symbol, type=ORDER_TYPE_MARKET, side=side,
                quantity=quantity, reduceOnly=reduce_only)
            print(f"Order placed: {symbol}, {side}, Qty: {quantity}, Order ID: {order['orderId']}")
            return order
        except BinanceAPIException as e:
            print(f"Order failed for {symbol}: {e}")
            return None
    else:
        print(f"Simulated order: {symbol}, {side}, Qty: {quantity}, ReduceOnly: {reduce_only}")
        return {'symbol': symbol, 'side': side, 'executedQty': str(quantity)}

def get_active_positions():
    return {p['symbol']: float(p['positionAmt']) for p in client.futures_position_information() if float(p['positionAmt']) != 0}

def close_position(symbol, tinfo):
    try:
        amt = float(client.futures_position_information(symbol=symbol)[0]['positionAmt'])
        if amt == 0: return
        side = SIDE_SELL if amt > 0 else SIDE_BUY
        qty = abs(amt)
        precision = tinfo[symbol]["quantityPrecision"]
        place_order(client, symbol, side, truncate(qty, precision), precision, mode, reduce_only=True)
    except Exception as e:
        print(f"Failed to close {symbol}: {e}")

def Lsafe(symbol, mrgType="ISOLATED", lvrg=2):
    try:
        client.futures_change_leverage(symbol=symbol, leverage=lvrg)
        client.futures_change_margin_type(symbol=symbol, marginType=mrgType)
    except Exception as e:
        if "-4046" not in str(e): print(f"Lsafe error: {e}")

# --- Main ---
if __name__ == "__main__":
    tinfo = get_token_info()
    traded_buffer = load_traded_buffer()
    now = datetime.now(timezone.utc)

    matrices, tokens, token_data, all_dates = marketDataImage(periods=Period_size, grid_size=GRID_SIZE)

    if matrices.shape[0] < Chunk_size:
        print("Not enough data.")
        exit(0)

    input_tensor = torch.FloatTensor(matrices[-Chunk_size:]).unsqueeze(0).to(device)

    if input_tensor.shape[1] != Chunk_size:
        print(f"Warning: input sequence length mismatch — Expected {Chunk_size}, got {input_tensor.shape[1]}")

    model = PredRNNpp(Num_layers, Num_hidden, in_channel=In_channel, out_channel=Out_channel, width=GRID_SIZE).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded.")

    # Close all positions
    for _ in range(4):
        open_positions = get_active_positions()
        if not open_positions:
            print("All positions confirmed closed.")
            break
        for symbol in open_positions:
            close_position(symbol, tinfo)
            sleep(0.2)
        print(f"Waiting for positions to close... ({len(open_positions)} open)")
        sleep(1)

    balance = get_futures_balance()
    allocation = balance * 0.6

    with torch.no_grad():
        output_seq = model(input_tensor)
        avg_pred_next = np.mean(output_seq.squeeze(0).cpu().numpy()[:FRAMES_TO_AVERAGE], axis=0)

    avg_pred_flat = avg_pred_next.flatten()
    volatility_map = compute_volatility(token_data, tokens)

    ineligible_tokens = [t for t in traded_buffer if (now - traded_buffer[t]) <= timedelta(hours=supra_delta)]
    signals = []
    for i, token in enumerate(tokens):
        if token in ineligible_tokens:
            continue
        avg_z = avg_pred_flat[i]
        volatility = volatility_map.get(token, 0)
        if abs(avg_z) < 0.5 or volatility < 0.05:
            continue
        score = abs(avg_z) * volatility
        direction = "LONG" if avg_z > 0 else "SHORT"
        signals.append((token, direction, avg_z, volatility, score))

    signals_sorted = sorted(signals, key=lambda x: x[4], reverse=True)[:max_positions]

    if not signals_sorted:
        print("No strong signals to open positions.")
        exit(0)

    allocation_per = allocation / len(signals_sorted)
    for token, direction, zscore, vol, score in signals_sorted:
        try:
            klines = client.futures_klines(symbol=token, interval=Client.KLINE_INTERVAL_1HOUR, limit=1)
            if not klines:
                raise ValueError(f"No kline data for {token}")
            opprice = float(klines[-1][4])
            precision = tinfo[token]["quantityPrecision"]
            min_notional = tinfo[token]["minNotional"]
            qty = truncate(allocation_per / opprice, precision)
            if qty * opprice < min_notional:
                qty = truncate(min_notional / opprice, precision)
            side = SIDE_BUY if direction == "LONG" else SIDE_SELL
            Lsafe(token)
            print(f"[{direction}] {token}: Avg Z-Score={zscore:.2f}, Volatility={vol:.2f}, Score={score:.2f}")
            print(f"Placing {direction} for {token} Qty={qty} Notional={qty * opprice:.2f}")
            place_order(client, token, side, qty, precision, mode)
            traded_buffer[token] = datetime.now(timezone.utc)
            sleep(0.2)
        except Exception as e:
            print(f"Error placing order for {token}: {str(e)}")

    for token in list(traded_buffer.keys()):
        if (now - traded_buffer[token]) > timedelta(hours=supra_delta):
            del traded_buffer[token]
    save_traded_buffer(traded_buffer)
