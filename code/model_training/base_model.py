import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader, Dataset
import math
import logging

from config import Config
from model import LSTM, MyEALSTM, MultiTCN, Transformer, iTransformer, Pyraformer

class Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class TemporalDataset(Dataset):
    def __init__(self, years):
        self.years = years
        self.input_files = [f"../../processed_data/TEM-MDM/temporal/input_{year}.npy" for year in years]
        self.output_files = [f"../../processed_data/TEM-MDM/temporal/output_{year}.npy" for year in years]
        self.data_lengths = [np.load(f, mmap_mode='r').shape[0] for f in self.input_files] 
        self.cumsum_lengths = np.cumsum([0] + self.data_lengths)  
        self.mean_vals = np.load('../../data/TEM-MDM/mean_vals.npy')
        self.std_vals = np.load('../../data/TEM-MDM/std_vals.npy')

    def __len__(self):
        return self.cumsum_lengths[-1]  

    def __getitem__(self, idx):
        year_idx = np.searchsorted(self.cumsum_lengths, idx, side="right") - 1
        file_idx = idx - self.cumsum_lengths[year_idx]

        input_data = (np.load(self.input_files[year_idx], mmap_mode='r')[file_idx] - self.mean_vals) / self.std_vals 
        output_data = np.load(self.output_files[year_idx], mmap_mode='r')[file_idx] / 23

        return torch.tensor(input_data, dtype=torch.float32).to(config.device), torch.tensor(output_data, dtype=torch.float32).to(config.device)

parser = argparse.ArgumentParser(description="Pretraining script for methane dataset")
parser.add_argument("--model", type=str, help="Model selection")
parser.add_argument("--epoch", type=int, default=10, help="Training epoches")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
args = parser.parse_args()

config = Config()
criterion = nn.MSELoss()

logging.basicConfig(
    filename=f"log/base_{args.model}.log", 
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO 
)

logger = logging.getLogger()

train_years = range(1979, 1989)
train_dataset = TemporalDataset(train_years)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

if args.model == 'lstm':
    model = LSTM(config).to(config.device)
elif args.model == 'ealstm':
    model = MyEALSTM(config).to(config.device)
elif args.model == 'tcn':
    model = MultiTCN(config).to(config.device)
elif args.model == 'transformer':
    model = Transformer(config).to(config.device)
elif args.model == 'itransformer':
    model = iTransformer(config).to(config.device)
elif args.model == 'pyraformer':
    model = Pyraformer(config).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(args.epoch):
    running_loss = 0.0
    model.train()
    for batch_X, batch_y in train_dataloader:
        if torch.isnan(batch_X).any():
            continue
        _, prediction = model(batch_X)
        loss = criterion(batch_y, prediction)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
    avg_loss = math.sqrt(running_loss / len(train_dataloader))

    logger.info(f'Epoch {epoch}: Training loss {avg_loss*23}')
    torch.save(model.state_dict(), f'../model_save/{args.model}/base_model.pth')