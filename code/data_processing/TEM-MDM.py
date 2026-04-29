"""
TEM-MDM Data Processing Script
-----------------------------
Processes TEM-MDM simulation data into numpy arrays for machine learning.
"""
import xarray as xr
import numpy as np
import os
import sys
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def check_file(path):
    if not os.path.exists(path):
        log(f"WARNING: File not found: {path}")
        return False
    return True
def extract_patch(arr, lat_ind, lon_ind, patch_size=3):
    pad = patch_size // 2
    pad_width = [(pad, pad), (pad, pad)] + [(0, 0)] * (arr.ndim - 2)
    arr_pad = np.pad(arr, pad_width, mode='edge')

    lat_p = lat_ind + pad
    lon_p = lon_ind + pad

    slices = (
        slice(lat_p - pad, lat_p + pad + 1),
        slice(lon_p - pad, lon_p + pad + 1),
    ) + tuple(slice(None) for _ in range(arr.ndim - 2))

    return arr_pad[slices]


def convert2year_patch(data):
    # data shape: (K, K, 12, 31)
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    out = []
    for month, days in enumerate(days_in_month):
        out.append(np.transpose(data[:, :, month, :days], (2, 0, 1)))  # (days, K, K)
    return np.concatenate(out, axis=0)  # (365, K, K)


def build_patch_batch(flat_indices, year, stat_grid_dict, stat_features,
                      ch4_dict, co2_dict, year_grid_features, patch_size=3):
    patch_samples = []

    for flat_idx in flat_indices:
        lat_ind = flat_idx // 720
        lon_ind = flat_idx % 720

        patch_channels = []

        # 8 Static Features
        for feat in stat_features:
            patch = extract_patch(stat_grid_dict[feat], lat_ind, lon_ind, patch_size)
            patch_365 = np.repeat(patch[np.newaxis, :, :], 365, axis=0)  # (365,3,3)
            patch_channels.append(patch_365.astype(np.float32))

        # CH4 / CO2 Constants
        patch_channels.append(np.full((365, patch_size, patch_size), ch4_dict[year], dtype=np.float32))
        patch_channels.append(np.full((365, patch_size, patch_size), co2_dict[year], dtype=np.float32))

        # 5 Dynamic Gridded Features
        for var_name in ['PREC', 'SOLR', 'TAIR', 'VAPR', 'NPP']:
            dyn_patch = extract_patch(year_grid_features[var_name], lat_ind, lon_ind, patch_size)
            dyn_patch = convert2year_patch(dyn_patch)  # (365,3,3)
            patch_channels.append(dyn_patch.astype(np.float32))

        patch_input = np.stack(patch_channels, axis=1)  # (365,15,3,3)
        patch_samples.append(patch_input)

    return np.stack(patch_samples, axis=0).astype(np.float32)
def main():
    # Ensure output directories exist
    os.makedirs('../../processed_data/TEM-MDM/temporal', exist_ok=True)
    os.makedirs('../../processed_data/TEM-MDM/spatial', exist_ok=True)

    # static features
    stat_dict = {}
    stat_grid_dict = {}
    stat_features = ['clelev', 'clfaotxt', 'cltveg', 'phh2o', 'topsoil_bulk_density', 'vegetation_type_11', 'wetlandtype', 'climatetype']
    stat_files = ['clelev.nc', 'clfaotxt.nc', 'cltveg.nc', 'phh2o.nc', 'topsoil_bulk_density.nc', 'vegetation_type_11.nc', 'wetlandtype.nc', 'climatetype.nc']
    log('Loading static features...')
    for i, feat in enumerate(stat_features):
        filename = f'../../data/TEM-MDM/{stat_files[i]}'
        if not check_file(filename):
            continue
        try:
            feature = xr.open_dataset(filename)
            feature = feature[feat].values.astype(np.float32)
            stat_grid_dict[feat] = feature
            stat_dict[feat] = feature.reshape(-1)
        except Exception as e:
            log(f"ERROR loading {filename}: {e}")
    if not stat_dict:
        log('ERROR: No static features loaded. Exiting.')
        sys.exit(1)

    # non-spatial features
    ch4_dict, co2_dict = {}, {}
    log('Loading non-spatial features...')
    ch4_path = '../../data/TEM-MDM/ch4-1979-2018.txt'
    co2_path = '../../data/TEM-MDM/kco21979-2018.txt'
    if check_file(ch4_path):
        with open(ch4_path, 'r') as f:
            for line in f:
                year, ch4 = line.split()
                ch4_dict[int(year)] = float(ch4)
    else:
        log('ERROR: CH4 file missing.')
    if check_file(co2_path):
        with open(co2_path, 'r') as f:
            for line in f:
                year, co2 = line.split()
                co2_dict[int(year)] = float(co2)
    else:
        log('ERROR: CO2 file missing.')
    if not ch4_dict or not co2_dict:
        log('ERROR: Non-spatial features missing. Exiting.')
        sys.exit(1)

    def convert2year(data):
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        mask = np.zeros(data.shape, dtype=bool)
        for month, days in enumerate(days_in_month):
            mask[:, month, :days] = True
        valid_data = data[mask]
        valid_data = valid_data.reshape(data.shape[0], 365)
        return valid_data

    mask = np.full((360*720), False, dtype=bool)
    log('Computing mask for missing data...')
    for year in range(1979, 2019):
        output_file = f'../../data/TEM-MDM/CH4_emission_intensity_{year}.nc'
        if not check_file(output_file):
            continue
        try:
            label = xr.open_dataset(output_file, engine="netcdf4")
            y_data = convert2year(label['CH4_emission'].values.reshape(-1,12,31))
            mask = mask | np.isnan(y_data).any(axis=1)
        except Exception as e:
            log(f"ERROR loading {output_file}: {e}")

    valid_flat_indices = np.where(~mask)[0]
    indices = np.random.permutation(valid_flat_indices.shape[0])
    fold_index = np.array_split(indices, 5)

    input_vars = ['daily_ecmwf_PREC', 'daily_ecmwf_SOLR', 'daily_ecmwf_TAIR', 'daily_ecmwf_VAPR', 'monthly_NPP']
    input_vars_names = ['PREC', 'SOLR', 'TAIR', 'VAPR', 'NPP']
    shape = (360*720, 365)
    processed_years = 0
    for year in range(1979, 2019):
        log(f'Processing year {year}...')
        X_data = []
        year_grid_features = {}
        for feat in stat_dict.keys():
            feature = stat_dict[feat]
            feature = np.repeat(feature[:, np.newaxis], 365, axis=1)
            X_data.append(feature)
        if year not in ch4_dict or year not in co2_dict:
            log(f"WARNING: CH4 or CO2 data missing for year {year}, skipping.")
            continue
        feature = np.full(shape, ch4_dict[year])
        X_data.append(feature)
        feature = np.full(shape, co2_dict[year])
        X_data.append(feature)
        for i, input_var in enumerate(input_vars):
            input_file = f'../../data/TEM-MDM/{input_var}_{year}.nc'
            if not check_file(input_file):
                continue
            try:
                ds = xr.open_dataset(input_file)
                feature = ds[input_vars_names[i]].values.astype(np.float32)

                if input_var.startswith('monthly'):
                    feature = np.nan_to_num(feature, nan=0.0)
                    feature = np.repeat(feature[:, :, :, np.newaxis], 31, axis=3)  # (360,720,12,31)

                # Save the original mesh version.
                year_grid_features[input_vars_names[i]] = feature

                # Continue generating point inputs
                feature_flat = feature.reshape(-1, 12, 31)
                feature_flat = convert2year(feature_flat)
                X_data.append(feature_flat)

            except Exception as e:
                log(f"ERROR loading {input_file}: {e}")
        try:
            X_data = np.stack(X_data, axis=-1)
        except Exception as e:
            log(f"ERROR stacking X_data for year {year}: {e}")
            continue
        output_file = f'../../data/TEM-MDM/CH4_emission_intensity_{year}.nc'
        if not check_file(output_file):
            continue
        try:
            label = xr.open_dataset(output_file, engine="netcdf4")
            y_data = convert2year(label['CH4_emission'].values.reshape(-1,12,31))
        except Exception as e:
            log(f"ERROR loading label for year {year}: {e}")
            continue
        X_data = X_data[~mask]
        y_data = y_data[~mask]
        index = X_data.shape[0]
        if index == 0:
            log(f"WARNING: No valid data for year {year} after masking.")
            continue
        sample_size = max(1, int(index * 0.1))
        selected = np.random.choice(np.arange(index), sample_size, replace=False)

        temp_input = X_data[selected]
        temp_output = y_data[selected]

        # Map the post-masking indices back to the original flat grid indices.
        selected_orig_flat = valid_flat_indices[selected]

        temp_patch_input = build_patch_batch(
            selected_orig_flat,
            year,
            stat_grid_dict,
            stat_features,
            ch4_dict,
            co2_dict,
            year_grid_features
        )

        temp_mask = ~(
                np.isnan(temp_input).any(axis=(1, 2)) |
                np.isnan(temp_output).any(axis=1) |
                np.isnan(temp_patch_input).any(axis=(1, 2, 3, 4))
        )

        temp_input = temp_input[temp_mask]
        temp_patch_input = temp_patch_input[temp_mask]
        temp_output = temp_output[temp_mask]

        try:
            np.save(f'../../processed_data/TEM-MDM/temporal/input_{year}.npy', temp_input)
            np.save(f'../../processed_data/TEM-MDM/temporal/patch_input_{year}.npy', temp_patch_input)
            np.save(f'../../processed_data/TEM-MDM/temporal/output_{year}.npy', temp_output)
        except Exception as e:
            log(f"ERROR saving temporal data for year {year}: {e}")
        for i, fold in enumerate(fold_index):
            fold_input, fold_output = X_data[fold], y_data[fold]
            fold_orig_flat = valid_flat_indices[fold]

            index = fold_input.shape[0]
            if index == 0:
                log(f"WARNING: No valid fold data for year {year}, fold {i}.")
                continue

            sample_size = max(1, int(index * 0.1))
            selected = np.random.choice(np.arange(index), sample_size, replace=False)

            fold_input = fold_input[selected]
            fold_output = fold_output[selected]
            selected_orig_flat = fold_orig_flat[selected]

            fold_patch_input = build_patch_batch(
                selected_orig_flat,
                year,
                stat_grid_dict,
                stat_features,
                ch4_dict,
                co2_dict,
                year_grid_features
            )

            fold_mask = ~(
                    np.isnan(fold_input).any(axis=(1, 2)) |
                    np.isnan(fold_output).any(axis=1) |
                    np.isnan(fold_patch_input).any(axis=(1, 2, 3, 4))
            )

            fold_input = fold_input[fold_mask]
            fold_patch_input = fold_patch_input[fold_mask]
            fold_output = fold_output[fold_mask]

            log(f"Fold {i}: point {fold_input.shape}, patch {fold_patch_input.shape}, output {fold_output.shape}")

            try:
                np.save(f'../../processed_data/TEM-MDM/spatial/input_{year}_{i}.npy', fold_input)
                np.save(f'../../processed_data/TEM-MDM/spatial/patch_input_{year}_{i}.npy', fold_patch_input)
                np.save(f'../../processed_data/TEM-MDM/spatial/output_{year}_{i}.npy', fold_output)
            except Exception as e:
                log(f"ERROR saving spatial data for year {year}, fold {i}: {e}")
        processed_years += 1
    log(f'Processing complete. Years processed: {processed_years}')

if __name__ == '__main__':
    main()