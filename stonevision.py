import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from binance.client import Client
from keys import api_key, api_secret
import math
from time import sleep
import os
import gc
import warnings
import json
import multiprocessing

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pin_memory.*no accelerator is found.*", category=UserWarning)

# CPU/GPU setup
num_cores = multiprocessing.cpu_count()
torch.set_num_threads(num_cores)
os.environ['OMP_NUM_THREADS'] = str(num_cores)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device} with {num_cores} CPU threads")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")

# Fixed grid size
GRID_SIZE = 28  # 28x28 grid = 784 slots
Chunk_size = 12  # Number of time steps in each chunk
Batch_size = 6  # Batch size for training
Num_grand_epochs = 1  # Number of grand epochs for training
Period_size = 1500

Num_layers = 4
Num_hidden = 128
In_channel = 1   # pct_change
Out_channel = 1  # predict pct_change


def marketDataImage(periods=Period_size, grid_size=GRID_SIZE, grid_map_file="grid_map.json"):
    client = Client(api_key, api_secret)
    exInfo = client.futures_exchange_info()
    tokens = [
        symbol["symbol"] for symbol in exInfo["symbols"]
        if symbol["contractType"] == "PERPETUAL" and "USDT" in symbol["symbol"]
        and symbol["status"] == "TRADING"
    ]
    total_cells = grid_size * grid_size

    # --- Load or update the grid map ---
    if os.path.exists(grid_map_file):
        with open(grid_map_file, "r") as f:
            grid_map = json.load(f)
    else:
        grid_map = {}

    # Maintain used indices
    used_indices = set(grid_map.values())
    available_indices = [i for i in range(total_cells) if i not in used_indices]

    # Add only new tokens into unused slots
    for token in tokens:
        if token not in grid_map and available_indices:
            grid_map[token] = available_indices.pop(0)

    # Save updated map
    with open(grid_map_file, "w") as f:
        json.dump(grid_map, f, indent=2)

    # --- Download market data ---
    token_data = {}
    all_dates = set()

    for symbol in grid_map:
        try:
            candles = client.futures_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_4HOUR, limit=periods)
            df = pd.DataFrame(candles, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
            if df.empty or "open_time" not in df.columns:
                continue
            df["date"] = df["open_time"].astype(int)
            df["open"] = df["open"].astype(float)
            df["close"] = df["close"].astype(float)
            df["pct_change"] = ((df["close"] - df["open"]) / df["open"].replace(0, np.nan)) * 100
            df["pct_change"] = df["pct_change"].replace([np.inf, -np.inf], 0).fillna(0)
            df = df.set_index("date")
            token_data[symbol] = df[["pct_change"]]
            all_dates.update(df.index)
            sleep(0.2)
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            continue

    all_dates = sorted(list(all_dates))[-periods:]
    for symbol in token_data:
        token_data[symbol] = token_data[symbol].reindex(all_dates, fill_value=0)
    for symbol in grid_map:
        if symbol not in token_data:
            print(f"Filling {symbol} with zeros.")
            token_data[symbol] = pd.DataFrame(0, index=all_dates, columns=["pct_change"])

    matrices = []
    for date in all_dates:
        grid = np.zeros((total_cells,))
        for token, idx in grid_map.items():
            if idx >= total_cells:
                continue
            value = token_data[token].loc[date, "pct_change"] if date in token_data[token].index else 0
            grid[idx] = value
        matrix = np.stack([grid.reshape(grid_size, grid_size)], axis=0)
        matrices.append(matrix)

    return np.array(matrices), list(grid_map.keys()), token_data, all_dates


class CausalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(CausalLSTMCell, self).__init__()
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv_x = nn.Conv2d(in_channel, num_hidden * 7, filter_size, stride, self.padding)
        self.conv_h = nn.Conv2d(num_hidden, num_hidden * 4, filter_size, stride, self.padding)
        self.conv_m = nn.Conv2d(num_hidden, num_hidden * 3, filter_size, stride, self.padding)
        self.conv_o = nn.Conv2d(num_hidden * 2, num_hidden, 1, 1, 0)
        self.layer_norm = nn.LayerNorm([num_hidden, width, width]) if layer_norm else nn.Identity()

    def forward(self, x, h, c, m):
        if h is None: h = torch.zeros_like(x[:, :self.num_hidden, ...])
        if c is None: c = torch.zeros_like(h)
        if m is None: m = torch.zeros_like(h)
        x_concat = self.conv_x(x)
        h_concat = self.conv_h(h)
        m_concat = self.conv_m(m)
        i_x, f_x, g_x, i_m, f_m, g_m, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_mh, f_mh, g_mh = torch.split(m_concat, self.num_hidden, dim=1)
        i = torch.sigmoid(i_x + i_h)
        f = torch.sigmoid(f_x + f_h + self._forget_bias)
        g = torch.tanh(g_x + g_h)
        delta_c = i * g
        c_new = f * c + delta_c
        i_m_comb = torch.sigmoid(i_m + i_mh)
        f_m_comb = torch.sigmoid(f_m + f_mh + self._forget_bias)
        g_m_comb = torch.tanh(g_m + g_mh)
        delta_m = i_m_comb * g_m_comb
        m_new = f_m_comb * m + delta_m
        mem_concat = torch.cat([c_new, m_new], dim=1)
        o = torch.sigmoid(o_x + o_h + self.conv_o(mem_concat))
        h_new = o * torch.tanh(self.layer_norm(c_new + m_new))
        return h_new, c_new, m_new, delta_c, delta_m

class PredRNNpp(nn.Module):
    def __init__(self, num_layers, num_hidden, in_channel, out_channel, width, filter_size=5):
        super(PredRNNpp, self).__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.cell_list = nn.ModuleList([
            CausalLSTMCell(in_channel if i == 0 else num_hidden, num_hidden, width, filter_size, 1, True)
            for i in range(num_layers)
        ])
        self.conv_last = nn.Conv2d(num_hidden, out_channel, 1, 1, 0)

    def forward(self, frames):
        batch, time_len, _, height, width = frames.shape
        h_t = [torch.zeros(batch, self.num_hidden, height, width, device=frames.device) for _ in range(self.num_layers)]
        c_t = [torch.zeros(batch, self.num_hidden, height, width, device=frames.device) for _ in range(self.num_layers)]
        m_t = torch.zeros(batch, self.num_hidden, height, width, device=frames.device)
        next_frames = []
        for t in range(time_len - 1):
            x = frames[:, t]
            h_t[0], c_t[0], m_t, _, _ = self.cell_list[0](x, h_t[0], c_t[0], m_t)
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], m_t, _, _ = self.cell_list[i](h_t[i-1], h_t[i], c_t[i], m_t)
            x_gen = self.conv_last(h_t[-1])
            next_frames.append(x_gen)
        return torch.stack(next_frames, dim=1)

def train_pred_rnn(period_matrices, all_dates, num_grand_epochs=Num_grand_epochs, chunk_size=Chunk_size, batch_size=Batch_size, model_path='pred_rnn_model.pth'):
    torch.cuda.empty_cache()
    matrices = np.array([period_matrices[date] for date in all_dates])
    model = PredRNNpp(Num_layers, Num_hidden, in_channel=In_channel, out_channel=Out_channel, width=GRID_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    last_avg_loss = None

    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            last_avg_loss = checkpoint.get('avg_loss', None)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            print(f"✅ Loaded model and optimizer from {model_path}, continuing training.")
            if last_avg_loss is not None:
                print(f"📉 Last saved average loss: {last_avg_loss:.5f}")
        except:
            print("⚠️ Starting training from scratch (checkpoint mismatch).")

    for grand_epoch in range(num_grand_epochs):
        print(f"\n🌐 Grand Epoch [{grand_epoch+1}/{num_grand_epochs}]")
        grand_total_loss = 0
        total_batches = 0
        for start in range(0, len(matrices)-chunk_size+1, chunk_size):
            end = start + chunk_size
            sequences = [matrices[i:i+chunk_size] for i in range(start, end-chunk_size+1)]
            if not sequences:
                continue
            sequences = torch.FloatTensor(np.array(sequences))
            dataset = torch.utils.data.TensorDataset(sequences)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
            for batch in dataloader:
                sequences_batch = batch[0]
                inputs = sequences_batch[:, :-1].to(device)
                targets = sequences_batch[:, -1].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs[:, -1], targets)
                loss.backward()
                optimizer.step()

                grand_total_loss += loss.item()
                total_batches += 1
        avg_grand_loss = grand_total_loss / total_batches
        print(f"✅ Grand Epoch [{grand_epoch+1}] Avg Loss: {avg_grand_loss:.5f}")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_loss': avg_grand_loss,
        }, model_path)
        print(f"💾 Model checkpoint saved.")

    return model

if __name__ == "__main__":
    matrices, tokens, token_data, dates = marketDataImage(periods=Period_size, grid_size=GRID_SIZE)
    model = train_pred_rnn(
        {date: matrices[i] for i, date in enumerate(dates)},
        dates,
        num_grand_epochs=Num_grand_epochs,
        chunk_size=Chunk_size,
        batch_size=Batch_size,
        model_path='pred_rnn_model.pth'
    )
    print("🏁 Training complete and model saved.")

