"""
Pretraining Script for X-MethaneWet Dataset
------------------------------------------
Trains deep learning models on TEM-MDM simulation data for methane flux prediction.
Supports temporal and spatial validation splits with multiple model architectures.
"""
import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader, Dataset
import math
import json
import logging
import os
import sys
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score

from config import Config
from model import LSTM, MyEALSTM, MultiTCN, Transformer, iTransformer, Pyraformer, HybridCNNLSTM


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def check_file(path):
    if not os.path.exists(path):
        log(f"WARNING: File not found: {path}")
        return False
    return True

#The original single-input Dataset
class Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

class TemporalDataset(Dataset):
    def __init__(self, years, config):
        self.years = years
        self.config = config
        self.input_files = [f"../../processed_data/TEM-MDM/temporal/input_{year}.npy" for year in years]
        self.output_files = [f"../../processed_data/TEM-MDM/temporal/output_{year}.npy" for year in years]

        for input_file, output_file in zip(self.input_files, self.output_files):
            if not check_file(input_file) or not check_file(output_file):
                raise FileNotFoundError(f"Missing files: {input_file} or {output_file}")

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

        return torch.tensor(input_data, dtype=torch.float32).to(self.config.device), torch.tensor(output_data, dtype=torch.float32).to(self.config.device)

class SpatialDataset(Dataset):
    def __init__(self, years, folds, config):
        self.years = years
        self.folds = folds
        self.config = config
        self.input_files = [f"../../processed_data/TEM-MDM/spatial/input_{year}_{fold}.npy" for year in years for fold in folds]
        self.output_files = [f"../../processed_data/TEM-MDM/spatial/output_{year}_{fold}.npy" for year in years for fold in folds]

        for input_file, output_file in zip(self.input_files, self.output_files):
            if not check_file(input_file) or not check_file(output_file):
                raise FileNotFoundError(f"Missing files: {input_file} or {output_file}")

        self.data_lengths = [np.load(f, mmap_mode='r').shape[0] for f in self.input_files]
        self.cumsum_lengths = np.cumsum([0] + self.data_lengths)
        self.mean_vals = np.load('../../data/TEM-MDM/mean_vals.npy')
        self.std_vals = np.load('../../data/TEM-MDM/std_vals.npy')

    def __len__(self):
        return self.cumsum_lengths[-1]

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cumsum_lengths, idx, side="right") - 1
        sample_idx = idx - self.cumsum_lengths[file_idx]

        input_data = (np.load(self.input_files[file_idx], mmap_mode='r')[sample_idx] - self.mean_vals) / self.std_vals
        output_data = np.load(self.output_files[file_idx], mmap_mode='r')[sample_idx] / 23

        return torch.tensor(input_data, dtype=torch.float32).to(self.config.device), torch.tensor(output_data, dtype=torch.float32).to(self.config.device)

# The Newly Added Dual-Input Hybrid Dataset
class HybridTemporalDataset(Dataset):
    def __init__(self, years, config):
        self.years = years
        self.config = config
        self.input_files = [f"../../processed_data/TEM-MDM/temporal/input_{year}.npy" for year in years]
        self.patch_files = [f"../../processed_data/TEM-MDM/temporal/patch_input_{year}.npy" for year in years]
        self.output_files = [f"../../processed_data/TEM-MDM/temporal/output_{year}.npy" for year in years]

        for f_in, f_patch, f_out in zip(self.input_files, self.patch_files, self.output_files):
            if not check_file(f_in) or not check_file(f_patch) or not check_file(f_out):
                raise FileNotFoundError(f"Missing Hybrid temporal files for year")

        self.data_lengths = [np.load(f, mmap_mode='r').shape[0] for f in self.input_files]
        self.cumsum_lengths = np.cumsum([0] + self.data_lengths)

        self.mean_vals = np.load('../../data/TEM-MDM/mean_vals.npy')
        self.std_vals = np.load('../../data/TEM-MDM/std_vals.npy')
        self.mean_patch = self.mean_vals.reshape(1, 15, 1, 1)
        self.std_patch = self.std_vals.reshape(1, 15, 1, 1)

    def __len__(self): return self.cumsum_lengths[-1]

    def __getitem__(self, idx):
        year_idx = np.searchsorted(self.cumsum_lengths, idx, side="right") - 1
        file_idx = idx - self.cumsum_lengths[year_idx]

        point_data = (np.load(self.input_files[year_idx], mmap_mode='r')[file_idx] - self.mean_vals) / self.std_vals
        patch_data = (np.load(self.patch_files[year_idx], mmap_mode='r')[file_idx] - self.mean_patch) / self.std_patch
        output_data = np.load(self.output_files[year_idx], mmap_mode='r')[file_idx] / 23

        return (torch.tensor(patch_data, dtype=torch.float32).to(self.config.device),
                torch.tensor(point_data, dtype=torch.float32).to(self.config.device),
                torch.tensor(output_data, dtype=torch.float32).to(self.config.device))

class HybridSpatialDataset(Dataset):
    def __init__(self, years, folds, config):
        self.years = years
        self.folds = folds
        self.config = config
        self.input_files = [f"../../processed_data/TEM-MDM/spatial/input_{year}_{fold}.npy" for year in years for fold in folds]
        self.patch_files = [f"../../processed_data/TEM-MDM/spatial/patch_input_{year}_{fold}.npy" for year in years for fold in folds]
        self.output_files = [f"../../processed_data/TEM-MDM/spatial/output_{year}_{fold}.npy" for year in years for fold in folds]

        for f_in, f_patch, f_out in zip(self.input_files, self.patch_files, self.output_files):
            if not check_file(f_in) or not check_file(f_patch) or not check_file(f_out):
                raise FileNotFoundError(f"Missing Hybrid spatial files")

        self.data_lengths = [np.load(f, mmap_mode='r').shape[0] for f in self.input_files]
        self.cumsum_lengths = np.cumsum([0] + self.data_lengths)

        self.mean_vals = np.load('../../data/TEM-MDM/mean_vals.npy')
        self.std_vals = np.load('../../data/TEM-MDM/std_vals.npy')
        self.mean_patch = self.mean_vals.reshape(1, 15, 1, 1)
        self.std_patch = self.std_vals.reshape(1, 15, 1, 1)

    def __len__(self): return self.cumsum_lengths[-1]

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cumsum_lengths, idx, side="right") - 1
        sample_idx = idx - self.cumsum_lengths[file_idx]

        point_data = (np.load(self.input_files[file_idx], mmap_mode='r')[sample_idx] - self.mean_vals) / self.std_vals
        patch_data = (np.load(self.patch_files[file_idx], mmap_mode='r')[sample_idx] - self.mean_patch) / self.std_patch
        output_data = np.load(self.output_files[file_idx], mmap_mode='r')[sample_idx] / 23

        return (torch.tensor(patch_data, dtype=torch.float32).to(self.config.device),
                torch.tensor(point_data, dtype=torch.float32).to(self.config.device),
                torch.tensor(output_data, dtype=torch.float32).to(self.config.device))

# Utility Functions
def setup_logging(args):
    os.makedirs("log", exist_ok=True)
    log_file = f"log/pretrain_{args.model}_{args.valid_type}_{args.id}.log" if args.valid_type == 'temporal' else f"log/pretrain_{args.model}_{args.valid_type}_fold{args.spatial_fold}.log"
    logging.basicConfig(filename=log_file, filemode="a", format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    return logging.getLogger()

def validate_args(args):
    valid_models = ['lstm', 'hybrid_cnn_lstm', 'ealstm', 'tcn', 'transformer', 'itransformer', 'pyraformer']
    if args.model not in valid_models:
        log(f"ERROR: Invalid model. Valid options: {valid_models}")
        sys.exit(1)

def create_model(args, config):
    model_map = {
        'lstm': LSTM, 'hybrid_cnn_lstm': HybridCNNLSTM, 'ealstm': MyEALSTM,
        'tcn': MultiTCN, 'transformer': Transformer, 'itransformer': iTransformer, 'pyraformer': Pyraformer
    }
    model = model_map[args.model](config).to(config.device)
    log(f"Created {args.model} model with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model

def save_model(model, args):
    model_dir = f'../model_save/{args.model}'
    os.makedirs(model_dir, exist_ok=True)
    save_path = f'{model_dir}/pretrain_{args.valid_type}_{args.id}.pth' if args.valid_type == 'temporal' else f'{model_dir}/pretrain_{args.valid_type}_fold{args.spatial_fold}.pth'
    torch.save(model.state_dict(), save_path)

def main():
    parser = argparse.ArgumentParser(description="Pretraining script for methane dataset")
    parser.add_argument("--valid_type", type=str, required=True, help="Use temporal dataset or spatial dataset")
    parser.add_argument("--spatial_fold", type=int, default=0, help="The fold using as cross validation in spatial experiment")
    parser.add_argument("--model", type=str, required=True, help="Model selection")
    parser.add_argument("--id", type=str, required=True, help="Experiment identifier")
    parser.add_argument("--epoch", type=int, default=10, help="Training epoches")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    args = parser.parse_args()

    validate_args(args)
    logger = setup_logging(args)
    log("Starting pretraining script...")
    config = Config()

    # Select a Dataset based on the model type.
    try:
        if args.model == 'hybrid_cnn_lstm':
            if args.valid_type == 'temporal':
                train_dataset = HybridTemporalDataset(range(1979, 1999), config)
                test_dataset = HybridTemporalDataset(range(1999, 2019), config)
            else:
                years = range(1979, 2019)
                train_folds = [i for i in range(5) if i != args.spatial_fold]
                train_dataset = HybridSpatialDataset(years, train_folds, config)
                test_dataset = HybridSpatialDataset(years, [args.spatial_fold], config)
        else:
            if args.valid_type == 'temporal':
                train_dataset = TemporalDataset(range(1979, 1999), config)
                test_dataset = TemporalDataset(range(1999, 2019), config)
            else:
                years = range(1979, 2019)
                train_folds = [i for i in range(5) if i != args.spatial_fold]
                train_dataset = SpatialDataset(years, train_folds, config)
                test_dataset = SpatialDataset(years, [args.spatial_fold], config)
    except Exception as e:
        log(f"ERROR creating datasets: {e}")
        sys.exit(1)

    if args.model == 'hybrid_cnn_lstm':
        batch_size = 64
    else:
        batch_size = 128

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = create_model(args, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    train_loss_list, test_loss_list, r2_list = [], [], []
    best_r2 = -float("inf")
    best_epoch = -1
    best_rmse = float("inf")

    for epoch in range(args.epoch):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for batch_idx, batch_data in enumerate(train_dataloader):
            # Clean Unpacking Logic
            if args.model == 'hybrid_cnn_lstm':
                batch_patch, batch_X, batch_y = batch_data
                if torch.isnan(batch_patch).any() or torch.isnan(batch_X).any() or torch.isnan(batch_y).any():
                    continue
            else:
                batch_X, batch_y = batch_data
                if torch.isnan(batch_X).any() or torch.isnan(batch_y).any():
                    continue

            try:
                if args.model == 'hybrid_cnn_lstm':
                    _, prediction = model(batch_patch, batch_X)
                else:
                    _, prediction = model(batch_X)

                loss = criterion(batch_y, prediction)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item()
                batch_count += 1
            except Exception as e:
                log(f"ERROR in training batch {batch_idx}: {e}")
                continue

        avg_loss = math.sqrt(running_loss / max(1, batch_count))

        model.eval()
        all_predictions, all_targets = [], []

        with torch.no_grad():
            for batch_data in test_dataloader:
                if args.model == 'hybrid_cnn_lstm':
                    batch_patch, batch_X, batch_y = batch_data
                    if torch.isnan(batch_patch).any() or torch.isnan(batch_X).any() or torch.isnan(batch_y).any():
                        continue
                    _, prediction = model(batch_patch, batch_X)
                else:
                    batch_X, batch_y = batch_data
                    if torch.isnan(batch_X).any() or torch.isnan(batch_y).any():
                        continue
                    _, prediction = model(batch_X)

                all_predictions.append(prediction.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        if all_predictions:
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            mask = ~np.isnan(all_targets)
            all_targets, all_predictions = all_targets[mask] * 23, all_predictions[mask] * 23

            rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
            nrmse = rmse / np.mean(all_targets)
            r2 = r2_score(all_targets, all_predictions)

            logger.info(f'Epoch {epoch}: Training loss {avg_loss * 23:.4f}, testing rmse {rmse:.4f}, nrmse {nrmse:.4f}, r2 score {r2:.4f}')
            print(f'Epoch {epoch}: train RMSE {avg_loss * 23:.4f}, test RMSE {rmse:.4f}, R2 {r2:.4f}')
            r2_list.append(r2)

            if r2 > best_r2:
                best_r2 = r2
                best_epoch = epoch
                best_rmse = rmse

                save_model(model, args)

                # Save an additional copy for use with `finetune.py`'s `--load_pretrain` argument.
                base_model_path = f'../model_save/{args.model}/base_model.pth'
                torch.save(model.state_dict(), base_model_path)

                print(f"New best model saved at epoch {epoch}: RMSE {rmse:.4f}, R2 {r2:.4f}")
                print(f"Best epoch: {best_epoch}, best RMSE: {best_rmse:.4f}, best R2: {best_r2:.4f}")
                logger.info(f"Best epoch: {best_epoch}, best RMSE: {best_rmse:.4f}, best R2: {best_r2:.4f}")

if __name__ == '__main__':
    main()
