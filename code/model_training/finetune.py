import os
import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader, Dataset
import math
import json
import logging
from sklearn.metrics import mean_squared_error, r2_score

from config import Config
from model import LSTM, MyEALSTM, MultiTCN, Transformer, iTransformer, Pyraformer, HybridCNNLSTM

class HybridDataset(Dataset):
    def __init__(self, x_patch, x_point, y):
        self.x_patch = x_patch
        self.x_point = x_point
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_patch[idx], self.x_point[idx], self.y[idx]
    
parser = argparse.ArgumentParser(description="Pretraining script for methane dataset")
parser.add_argument("--valid_type", type=str, help="Use temporal dataset or spatial dataset")
parser.add_argument("--spatial_fold", type=int, default=0, help="The fold using as cross validation in spatial experiment")
parser.add_argument("--model", type=str, help="Model selection")
parser.add_argument("--load_pretrain", action="store_true", help="If load pretrain model")
parser.add_argument("--id", type=str, help="Experiment identifier")
parser.add_argument("--epoch", type=int, default=200, help="Training epoches")
parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
parser.add_argument("--percent", type=float, default=0, help="Training dataset available percentage")
args = parser.parse_args()
    
config = Config()
device = torch.device("cuda")
if args.valid_type == 'temporal':
    if args.percent == 0:
        train_x = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/train_data_x.npy'), dtype=torch.float32)
        train_y = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/train_data_y.npy'), dtype=torch.float32)
    else:
        train_x = torch.tensor(np.load(f'../../processed_data/FLUXNET-CH4/temporal/train_data_x_{args.percent}.npy'), dtype=torch.float32)
        train_y = torch.tensor(np.load(f'../../processed_data/FLUXNET-CH4/temporal/train_data_y_{args.percent}.npy'), dtype=torch.float32)
    test_x = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/test_data_x.npy'), dtype=torch.float32)
    test_y = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/test_data_y.npy'), dtype=torch.float32)
else:
    train_x_list, train_y_list = [], []
    for fold in range(5):
        if fold == args.spatial_fold:
            test_x = torch.tensor(np.load(f"../../processed_data/FLUXNET-CH4/spatial/data_x_{fold}.npy"), dtype=torch.float32)
            test_y = torch.tensor(np.load(f"../../processed_data/FLUXNET-CH4/spatial/data_y_{fold}.npy"), dtype=torch.float32)
        else:
            train_x_list.append(np.load(f"../../processed_data/FLUXNET-CH4/spatial/data_x_{fold}.npy"))
            train_y_list.append(np.load(f"../../processed_data/FLUXNET-CH4/spatial/data_y_{fold}.npy"))

    train_x = torch.tensor(np.concatenate(train_x_list, axis=0), dtype=torch.float32)
    train_y = torch.tensor(np.concatenate(train_y_list, axis=0), dtype=torch.float32)

mean = torch.tensor(np.load('../../data/TEM-MDM/mean_vals.npy'), dtype=torch.float32)
std = torch.tensor(np.load('../../data/TEM-MDM/std_vals.npy'), dtype=torch.float32)

train_x = (train_x - mean) / std
test_x = (test_x - mean) / std

if args.model == 'hybrid_cnn_lstm':
    train_point_x = train_x
    test_point_x = test_x

    if args.valid_type == 'temporal':
        if args.percent == 0:
            train_patch_x = torch.tensor(
                np.load('../../processed_data/FLUXNET-CH4/temporal/train_patch_x.npy'),
                dtype=torch.float32
            )
        else:
            raise ValueError("Patch files for percent-based temporal training are not prepared yet.")

        test_patch_x = torch.tensor(
            np.load('../../processed_data/FLUXNET-CH4/temporal/test_patch_x.npy'),
            dtype=torch.float32
        )
    else:
        train_patch_x_list = []
        for fold in range(5):
            if fold == args.spatial_fold:
                test_patch_x = torch.tensor(
                    np.load(f"../../processed_data/FLUXNET-CH4/spatial/data_patch_x_{fold}.npy"),
                    dtype=torch.float32
                )
            else:
                train_patch_x_list.append(
                    np.load(f"../../processed_data/FLUXNET-CH4/spatial/data_patch_x_{fold}.npy")
                )

        train_patch_x = torch.tensor(np.concatenate(train_patch_x_list, axis=0), dtype=torch.float32)

    patch_mean = mean.view(1, 1, -1, 1, 1)
    patch_std = std.view(1, 1, -1, 1, 1)

    train_patch_x = (train_patch_x - patch_mean) / patch_std
    test_patch_x = (test_patch_x - patch_mean) / patch_std

if args.model == 'hybrid_cnn_lstm':
    # 点位输入：先直接用原来的序列特征
    train_dataset = HybridDataset(
        train_patch_x.to(device),
        train_point_x.to(device),
        train_y.to(device)
    )
    test_dataset = HybridDataset(
        test_patch_x.to(device),
        test_point_x.to(device),
        test_y.to(device)
    )
else:
    # 原始单输入模型
    class SimpleDataset(Dataset):
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

    train_dataset = SimpleDataset(train_x.to(device), train_y.to(device))
    test_dataset = SimpleDataset(test_x.to(device), test_y.to(device))

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

if args.model == 'lstm':
    model = LSTM(config).to(device)
elif args.model == 'hybrid_cnn_lstm':
    model = HybridCNNLSTM(config).to(device)
elif args.model == 'ealstm':
    model = MyEALSTM(config).to(device)
elif args.model == 'tcn':
    model = MultiTCN(config).to(device)
elif args.model == 'transformer':
    model = Transformer(config).to(device)
elif args.model == 'itransformer':
    model = iTransformer(config).to(config.device)
elif args.model == 'pyraformer':
    model = Pyraformer(config).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.MSELoss()
os.makedirs(f'../model_save/{args.model}', exist_ok=True)
os.makedirs("log", exist_ok=True)

if args.percent == 0:
    if args.load_pretrain:
        model.load_state_dict(torch.load(f'../model_save/{args.model}/base_model.pth', weights_only=True))
        if args.valid_type == 'temporal':
            logging.basicConfig(
                filename=f"log/finetune_{args.model}_{args.valid_type}_{args.id}.log", 
                filemode="a",
                format="%(asctime)s - %(levelname)s - %(message)s",
                level=logging.INFO 
            )
        else:
            logging.basicConfig(
                filename=f"log/finetune_{args.model}_{args.valid_type}_fold{args.spatial_fold}.log", 
                filemode="a",
                format="%(asctime)s - %(levelname)s - %(message)s",
                level=logging.INFO 
            )
    else:
        if args.valid_type == 'temporal':
            logging.basicConfig(
                filename=f"log/scratch_{args.model}_{args.valid_type}_{args.id}.log", 
                filemode="a",
                format="%(asctime)s - %(levelname)s - %(message)s",
                level=logging.INFO 
            )
        else:
            logging.basicConfig(
                filename=f"log/scratch_{args.model}_{args.valid_type}_fold{args.spatial_fold}.log", 
                filemode="a",
                format="%(asctime)s - %(levelname)s - %(message)s",
                level=logging.INFO 
            )
else:
    if args.load_pretrain:
        model.load_state_dict(torch.load(f'../model_save/{args.model}/base_model.pth', weights_only=True))
        if args.valid_type == 'temporal':
            logging.basicConfig(
                filename=f"log/finetune_{args.model}_{args.valid_type}_{args.percent}_{args.id}.log", 
                filemode="a",
                format="%(asctime)s - %(levelname)s - %(message)s",
                level=logging.INFO 
            )
    else:
        if args.valid_type == 'temporal':
            logging.basicConfig(
                filename=f"log/scratch_{args.model}_{args.valid_type}_{args.percent}_{args.id}.log", 
                filemode="a",
                format="%(asctime)s - %(levelname)s - %(message)s",
                level=logging.INFO 
            )

logger = logging.getLogger()

logger.info("Command-line Arguments:")
logger.info(json.dumps(vars(args), indent=4)) 

logger.info("Model Configuration:")
logger.info(json.dumps(vars(config), indent=4))

for epoch in range(args.epoch):
    running_loss = 0.0
    model.train()

    if args.model == 'hybrid_cnn_lstm':
        for x_patch, x_point, batch_y in train_dataloader:
            _, prediction = model(x_patch, x_point)
            prediction = prediction.reshape(batch_y.shape)
            loss = criterion(batch_y, prediction)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
    else:
        for batch_X, batch_y in train_dataloader:
            _, prediction = model(batch_X)
            prediction = prediction.reshape(batch_y.shape)
            loss = criterion(batch_y, prediction)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

    avg_loss = math.sqrt(running_loss / len(train_dataloader))

    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        if args.model == 'hybrid_cnn_lstm':
            for x_patch, x_point, batch_y in test_dataloader:
                _, prediction = model(x_patch, x_point)
                all_predictions.append(prediction.cpu().numpy().reshape(-1))
                all_targets.append(batch_y.cpu().numpy().reshape(-1))
        else:
            for batch_X, batch_y in test_dataloader:
                _, prediction = model(batch_X)
                all_predictions.append(prediction.cpu().numpy().reshape(-1))
                all_targets.append(batch_y.cpu().numpy().reshape(-1))

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    mask = ~np.isnan(all_targets)

    rmse = np.sqrt(mean_squared_error(all_targets[mask], all_predictions[mask]))
    nrmse = rmse / np.mean(all_targets[mask])
    r2 = r2_score(all_targets[mask], all_predictions[mask])

    logger.info(f'Epoch {epoch}: Training loss {avg_loss}, testing rmse {rmse}, nrmse {nrmse}, r2 score {r2}')
    print(f'Epoch {epoch}: train RMSE {avg_loss:.4f}, test RMSE {rmse:.4f}, R2 {r2:.4f}')

if args.percent == 0:
    if args.load_pretrain:
        if args.valid_type == 'temporal':
            torch.save(model.state_dict(), f'../model_save/{args.model}/finetune_{args.valid_type}_{args.id}.pth')
        else:
            torch.save(model.state_dict(), f'../model_save/{args.model}/finetune_{args.valid_type}_fold{args.spatial_fold}.pth')
    else:
        if args.valid_type == 'temporal':
            torch.save(model.state_dict(), f'../model_save/{args.model}/scratch_{args.valid_type}_{args.id}.pth')
        else:
            torch.save(model.state_dict(), f'../model_save/{args.model}/scratch_{args.valid_type}_fold{args.spatial_fold}.pth')
else:
    if args.load_pretrain:
        if args.valid_type == 'temporal':
            torch.save(model.state_dict(), f'../model_save/{args.model}/finetune_{args.valid_type}_{args.percent}_{args.id}.pth')
    else:
        if args.valid_type == 'temporal':
            torch.save(model.state_dict(), f'../model_save/{args.model}/scratch_{args.valid_type}_{args.percent}_{args.id}.pth')