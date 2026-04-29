"""
FLUXNET-CH4 Data Processing Script
---------------------------------
Processes FLUXNET-CH4 observation data into numpy arrays for machine learning.
"""
import numpy as np
import pandas as pd
import xarray as xr
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

def convert2year(data):
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    mask = np.zeros(data.shape, dtype=bool)
    for month, days in enumerate(days_in_month):
        mask[:, :, month, :days] = True 
    valid_data = data[mask]
    valid_data = valid_data.reshape(data.shape[0], data.shape[1], 365)
    return valid_data

def nearest_patch_point(lat, lon):
    lat_index = round((lat + 90) * 2) 
    lon_index = round((lon + 180) * 2)  
    return (lat_index, lon_index)
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
        # (K, K, days) -> (days, K, K)
        out.append(np.transpose(data[:, :, month, :days], (2, 0, 1)))
    return np.concatenate(out, axis=0)   # (365, K, K)


def build_patch_input(year, lat_ind, lon_ind, stat_dict, stat_features, ch4_dict, co2_dict, site, site_wet):
    patch_channels = []


    for feat in stat_features:
        patch = extract_patch(stat_dict[feat], lat_ind, lon_ind, patch_size=3)


        if feat == 'wetlandtype' and site in site_wet:
            patch = patch.copy()
            patch[1, 1] = site_wet[site]

        patch_channels.append(np.repeat(patch[np.newaxis, :, :], 365, axis=0))

    # CH4 / CO2
    patch_channels.append(np.full((365, 3, 3), ch4_dict[year], dtype=float))
    patch_channels.append(np.full((365, 3, 3), co2_dict[year], dtype=float))

    # PREC, SOLR, TAIR, VAPR
    daily_vars = [
        ('daily_ecmwf_PREC', 'PREC'),
        ('daily_ecmwf_SOLR', 'SOLR'),
        ('daily_ecmwf_TAIR', 'TAIR'),
        ('daily_ecmwf_VAPR', 'VAPR'),
    ]

    for file_stub, var_name in daily_vars:
        data_file = f'../../data/TEM-MDM/{file_stub}_{year}.nc'
        if not check_file(data_file):
            return None
        data = xr.open_dataset(data_file, engine="netcdf4")[var_name].values
        patch = extract_patch(data, lat_ind, lon_ind, patch_size=3)   # (3,3,12,31)
        patch = convert2year_patch(patch)                             # (365,3,3)
        patch_channels.append(patch)

    # 4. monthly NPP
    npp_file = f'../../data/TEM-MDM/monthly_NPP_{year}.nc'
    if not check_file(npp_file):
        return None
    npp = xr.open_dataset(npp_file, engine="netcdf4")['NPP'].values
    npp = np.nan_to_num(npp, nan=0.0)
    npp_patch = extract_patch(npp, lat_ind, lon_ind, patch_size=3)    # (3,3,12)
    npp_patch = np.repeat(npp_patch[..., np.newaxis], 31, axis=3)     # (3,3,12,31)
    npp_patch = convert2year_patch(npp_patch)                         # (365,3,3)
    patch_channels.append(npp_patch)

    # (365, 15, 3, 3)
    return np.stack(patch_channels, axis=1)
def main():
    os.makedirs('../../processed_data/FLUXNET-CH4/temporal', exist_ok=True)
    os.makedirs('../../processed_data/FLUXNET-CH4/spatial', exist_ok=True)

    # Load static features
    log('Loading static features from TEM-MDM...')
    stat_dict = {}
    stat_features = ['clelev', 'clfaotxt', 'cltveg', 'phh2o', 'topsoil_bulk_density', 'vegetation_type_11', 'wetlandtype', 'climatetype']
    stat_files = ['clelev.nc', 'clfaotxt.nc', 'cltveg.nc', 'phh2o.nc', 'topsoil_bulk_density.nc', 'vegetation_type_11.nc', 'wetlandtype.nc', 'climatetype.nc'] 
    
    for i, feat in enumerate(stat_features):
        filename = f'../../data/TEM-MDM/{stat_files[i]}' 
        if not check_file(filename):
            continue
        try:
            feature = xr.open_dataset(filename)
            feature = feature[feat].values
            stat_dict[feat] = feature
        except Exception as e:
            log(f"ERROR loading {filename}: {e}")
    
    if not stat_dict:
        log('ERROR: No static features loaded. Exiting.')
        sys.exit(1)

    # Load non-spatial features
    log('Loading non-spatial features...')
    ch4_dict, co2_dict = {}, {}
    ch4_path = '../../data/TEM-MDM/ch4-1979-2018.txt'
    co2_path = '../../data/TEM-MDM/kco21979-2018.txt'
    
    if check_file(ch4_path):
        try:
            with open(ch4_path, 'r') as f:
                for line in f:
                    year, ch4 = line.split()
                    ch4_dict[int(year)] = float(ch4)
        except Exception as e:
            log(f"ERROR loading CH4 data: {e}")
    else:
        log('ERROR: CH4 file missing.')
    
    if check_file(co2_path):
        try:
            with open(co2_path, 'r') as f:
                for line in f:
                    year, co2 = line.split()
                    co2_dict[int(year)] = float(co2)
        except Exception as e:
            log(f"ERROR loading CO2 data: {e}")
    else:
        log('ERROR: CO2 file missing.')
    
    if not ch4_dict or not co2_dict:
        log('ERROR: Non-spatial features missing. Exiting.')
        sys.exit(1)

    # Load FLUXNET site information and data
    log('Loading FLUXNET site information...')
    site_info_path = '../../data/FLUXNET-CH4/FLUXNET_CH4_2024.csv'
    fluxnet_data_path = '../../data/FLUXNET-CH4/FLUXNET_T1_DD.csv'
    
    if not check_file(site_info_path) or not check_file(fluxnet_data_path):
        log('ERROR: FLUXNET data files missing. Exiting.')
        sys.exit(1)
    
    try:
        site_info = pd.read_csv(site_info_path)
        fluxnet_data = pd.read_csv(fluxnet_data_path)
    except Exception as e:
        log(f"ERROR loading FLUXNET data: {e}")
        sys.exit(1)

    location = {}
    for index, line in site_info.iterrows():
        location[line['SITE_ID'].replace('-','.')] = (line['LOCATION_LAT'], line['LOCATION_LONG'], line["LOCATION_ELEV"], line["IGBP"], line["MAT"], line['MAP'])
    fluxnet_data['TIMESTAMP'] = pd.to_datetime(fluxnet_data['TIMESTAMP'])
    fluxnet_data['Year'] = fluxnet_data['TIMESTAMP'].dt.year
    fluxnet_data = fluxnet_data[~((fluxnet_data['TIMESTAMP'].dt.month == 2) & (fluxnet_data['TIMESTAMP'].dt.day == 29))]
    grouped = fluxnet_data.groupby(['Site', 'Year'])

    log('Calculating temporal split years...')
    splitting_year_dict = {}
    for site, group in fluxnet_data.groupby('Site'):
        unique_years = sorted(group['Year'].unique())
        midpoint = (len(unique_years)-1) * 6 // 7
        splitting_year = unique_years[midpoint] if unique_years else None
        splitting_year_dict[site] = splitting_year

    log(f'Splitting years calculated for {len(splitting_year_dict)} sites')

    site_wet = {
        "CA.SCB": 2, "CA.SCC": 1, "DE.Hte": 2, "DE.Zrk": 2, "DE.SfN": 2,
        "FI.Lom": 2, "FI.Si2": 2, "FI.Sii": 2, "FR.LGt": 2, "JP.BBY": 2,
        "NZ.Kop": 2, "RU.Ch2": 2, "RU.Cok": 2, "SE.Deg": 2, "US.A03": 2,
        "US.ICs": 2, "US.A10": 2, "US.Atq": 2, "US.Beo": 2, "US.Bes": 2,
        "US.NGB": 2, "US.BZB": 1, "US.BZF": 2, "US.Uaf": 1, "US.Ivo": 2,
        "US.Los": 2, "US.NGC": 2, "BR.Npw": 3, "BW.Gum": 4, "BW.Nxr": 5,
        "ID.Pag": 3, "MY.MLM": 3, "US.NC4": 3, "US.DPW": 4, "US.LA2": 4,
        "US.Myb": 4, "US.ORv": 3, "US.OWC": 4, "US.Sne": 4, "US.Tw1": 4,
        "US.Tw4": 4, "US.Tw5": 4, "US.WPT": 4
    }

    log('Processing temporal data...')
    train_data_x, train_patch_x, train_data_y = [], [], []
    test_data_x, test_patch_x, test_data_y = [], [], []
    test_site_list = []
    processed_samples = 0
    
    for (site, year), group in grouped:
        if site not in location.keys():
            continue
        lat, lon, elev, IGBP, mat, map = location[site]
        if group.shape[0] != 365:
            continue
        if IGBP != 'WET':
            continue
            
        try:
            temp = group.to_numpy()[:,-6:-3]
            temp = temp.astype(dtype=float)
            lat_ind, lon_ind = nearest_patch_point(lat, lon)
            
            for i, feat in enumerate(stat_features):
                if i == 0:
                    input = np.tile(elev, (365))[..., np.newaxis]
                elif feat == 'wetlandtype':
                    if site in site_wet.keys():
                        wet = site_wet[site]
                    else:
                        wet = stat_dict[feat][lat_ind, lon_ind]
                    wet = np.tile(wet, (365))[..., np.newaxis]
                    input = np.concatenate((input, wet), axis=1)
                else:
                    new_input = stat_dict[feat][lat_ind, lon_ind]
                    new_input = np.tile(new_input, (365))[..., np.newaxis]
                    input = np.concatenate((input, new_input), axis=1)
            
            if year not in ch4_dict or year not in co2_dict:
                continue
            ch4_value = np.tile(ch4_dict[year], (365))[..., np.newaxis]
            co2_value = np.tile(co2_dict[year], (365))[..., np.newaxis]
            input = np.concatenate((input, ch4_value), axis=1)
            input = np.concatenate((input, co2_value), axis=1)

            input = np.concatenate((input, temp[:,2][..., np.newaxis]), axis=1)

            solr_file = f'../../data/TEM-MDM/daily_ecmwf_SOLR_{year}.nc'
            if check_file(solr_file):
                feature = convert2year(xr.open_dataset(solr_file, engine="netcdf4")['SOLR'].values[lat_ind, lon_ind,:,:].reshape(1,1,12,31))
                input = np.concatenate((input, feature.reshape(365,1)), axis=1)
            else:
                continue

            input = np.concatenate((input, temp[:,0][..., np.newaxis]), axis=1)

            T = group['TA_F'].to_numpy()
            VPD = group['VPD_F'].to_numpy()
            vp = (0.6108 * np.exp(17.27*T/(T+237.3)))*10 - VPD
            input = np.concatenate((input, vp[..., np.newaxis]), axis=1)

            npp_file = f'../../data/TEM-MDM/monthly_NPP_{year}.nc'
            if check_file(npp_file):
                feature = xr.open_dataset(npp_file, engine="netcdf4")['NPP'].values[lat_ind, lon_ind,:].reshape(1,1,12)
                feature = np.nan_to_num(feature, nan=0.0)
                feature = np.repeat(feature[:, :, :, np.newaxis], 31, axis=3)
                feature = convert2year(feature)
                input = np.concatenate((input, feature.reshape(365,1)), axis=1)
            else:
                continue

            patch_channels = []


            for feat in stat_features:
                patch = extract_patch(stat_dict[feat], lat_ind, lon_ind, patch_size=3)


                if feat == 'wetlandtype' and site in site_wet:
                    patch = patch.copy()
                    patch[1, 1] = site_wet[site]

                patch_365 = np.repeat(patch[np.newaxis, :, :], 365, axis=0)  # (365,3,3)
                patch_channels.append(patch_365)


            patch_channels.append(np.full((365, 3, 3), ch4_dict[year], dtype=float))
            patch_channels.append(np.full((365, 3, 3), co2_dict[year], dtype=float))


            prec_file = f'../../data/TEM-MDM/daily_ecmwf_PREC_{year}.nc'
            if check_file(prec_file):
                prec_data = xr.open_dataset(prec_file, engine="netcdf4")['PREC'].values
                prec_patch = extract_patch(prec_data, lat_ind, lon_ind, patch_size=3)  # (3,3,12,31)
                prec_patch = convert2year_patch(prec_patch)  # (365,3,3)
                patch_channels.append(prec_patch)
            else:
                continue


            solr_file_patch = f'../../data/TEM-MDM/daily_ecmwf_SOLR_{year}.nc'
            if check_file(solr_file_patch):
                solr_data = xr.open_dataset(solr_file_patch, engine="netcdf4")['SOLR'].values
                solr_patch = extract_patch(solr_data, lat_ind, lon_ind, patch_size=3)
                solr_patch = convert2year_patch(solr_patch)
                patch_channels.append(solr_patch)
            else:
                continue


            tair_file = f'../../data/TEM-MDM/daily_ecmwf_TAIR_{year}.nc'
            if check_file(tair_file):
                tair_data = xr.open_dataset(tair_file, engine="netcdf4")['TAIR'].values
                tair_patch = extract_patch(tair_data, lat_ind, lon_ind, patch_size=3)
                tair_patch = convert2year_patch(tair_patch)
                patch_channels.append(tair_patch)
            else:
                continue


            vapr_file = f'../../data/TEM-MDM/daily_ecmwf_VAPR_{year}.nc'
            if check_file(vapr_file):
                vapr_data = xr.open_dataset(vapr_file, engine="netcdf4")['VAPR'].values
                vapr_patch = extract_patch(vapr_data, lat_ind, lon_ind, patch_size=3)
                vapr_patch = convert2year_patch(vapr_patch)
                patch_channels.append(vapr_patch)
            else:
                continue


            npp_file_patch = f'../../data/TEM-MDM/monthly_NPP_{year}.nc'
            if check_file(npp_file_patch):
                npp_data = xr.open_dataset(npp_file_patch, engine="netcdf4")['NPP'].values
                npp_data = np.nan_to_num(npp_data, nan=0.0)
                npp_patch = extract_patch(npp_data, lat_ind, lon_ind, patch_size=3)  # (3,3,12)
                npp_patch = np.repeat(npp_patch[..., np.newaxis], 31, axis=3)         # (3,3,12,31)
                npp_patch = convert2year_patch(npp_patch)                             # (365,3,3)
                patch_channels.append(npp_patch)
            else:
                continue

            patch_input = np.stack(patch_channels, axis=1)  # (365,15,3,3)

            if np.isnan(patch_input).any():
                continue
            if np.isnan(input).any():
                continue

            label = group['FCH4_F'].to_numpy().copy()
            mask = np.isnan(label)
            label[mask] = group['FCH4_F_ANNOPTLM'].to_numpy()[mask]

            if year > splitting_year_dict[site]:
                test_data_x.append(input)
                test_patch_x.append(patch_input)
                test_data_y.append(label)
                test_site_list.append(site)
            else:
                train_data_x.append(input)
                train_patch_x.append(patch_input)
                train_data_y.append(label)
            
            processed_samples += 1
            
        except Exception as e:
            log(f"ERROR processing {site} {year}: {e}")
            continue

    log(f'Temporal processing complete. Processed {processed_samples} samples.')

    try:
        train_data_x = np.stack(train_data_x, axis=0)
        train_patch_x = np.stack(train_patch_x, axis=0)
        train_data_y = np.stack(train_data_y, axis=0)

        test_data_x = np.stack(test_data_x, axis=0)
        test_patch_x = np.stack(test_patch_x, axis=0)
        test_data_y = np.stack(test_data_y, axis=0)

        np.save('../../processed_data/FLUXNET-CH4/temporal/train_data_x.npy', train_data_x)
        np.save('../../processed_data/FLUXNET-CH4/temporal/train_patch_x.npy', train_patch_x)
        np.save('../../processed_data/FLUXNET-CH4/temporal/train_data_y.npy', train_data_y)

        np.save('../../processed_data/FLUXNET-CH4/temporal/test_data_x.npy', test_data_x)
        np.save('../../processed_data/FLUXNET-CH4/temporal/test_patch_x.npy', test_patch_x)
        np.save('../../processed_data/FLUXNET-CH4/temporal/test_data_y.npy', test_data_y)
        
        log(f'Temporal data saved. Train: {train_data_x.shape}, Test: {test_data_x.shape}')
    except Exception as e:
        log(f"ERROR saving temporal data: {e}")

    log('Processing spatial data for cross-validation...')
    num_folds = 5
    folds = {
        0: ['BW.Gum', 'FI.Sii', 'RU.Ch2', 'US.Ivo', 'US.Srr', 'US.Bes'],
        1: ['CA.SCB', 'FI.Lom', 'US.WPT', 'US.BZF', 'US.LA2', 'US.BZB'],
        2: ['DE.Zrk', 'RU.Che', 'US.MRM', 'US.LA1', 'US.Los', 'US.NC4'],
        3: ['JP.BBY', 'FR.LGt', 'US.ICs', 'US.Atq', 'US.Myb', 'US.OWC'],
        4: ['DE.SfN', 'DE.Hte', 'US.ORv', 'US.DPW', 'FI.Si2', 'US.Beo']
    }

    for fold in range(num_folds):
        log(f'Processing fold {fold}...')
        test_sites = set(folds[fold]) 
        data_x, data_patch_x, data_y = [], [], []
        fold_samples = 0
        
        for (site, year), group in grouped:
            if site not in test_sites:
                continue
            lat, lon, elev, IGBP, mat, map = location[site]
            if group.shape[0] != 365:
                continue
            if IGBP != 'WET':
                continue
                
            try:
                temp = group.to_numpy()[:,-6:-3]
                temp = temp.astype(dtype=float)
                lat_ind, lon_ind = nearest_patch_point(lat, lon)

                for i, feat in enumerate(stat_features):
                    if i == 0:
                        input = np.tile(elev, (365))[..., np.newaxis]
                    elif feat == 'wetlandtype':
                        if site in site_wet.keys():
                            wet = site_wet[site]
                        else:
                            wet = stat_dict[feat][lat_ind, lon_ind]
                        wet = np.tile(wet, (365))[..., np.newaxis]
                        input = np.concatenate((input, wet), axis=1)
                    else:
                        new_input = stat_dict[feat][lat_ind, lon_ind]
                        new_input = np.tile(new_input, (365))[..., np.newaxis]
                        input = np.concatenate((input, new_input), axis=1)
                
                if year not in ch4_dict or year not in co2_dict:
                    continue
                ch4_value = np.tile(ch4_dict[year], (365))[..., np.newaxis]
                co2_value = np.tile(co2_dict[year], (365))[..., np.newaxis]
                input = np.concatenate((input, ch4_value), axis=1)
                input = np.concatenate((input, co2_value), axis=1)

                input = np.concatenate((input, temp[:,2][..., np.newaxis]), axis=1)

                solr_file = f'../../data/TEM-MDM/daily_ecmwf_SOLR_{year}.nc'
                if check_file(solr_file):
                    feature = convert2year(xr.open_dataset(solr_file, engine="netcdf4")['SOLR'].values[lat_ind, lon_ind,:,:].reshape(1,1,12,31))
                    input = np.concatenate((input, feature.reshape(365,1)), axis=1)
                else:
                    continue

                input = np.concatenate((input, temp[:,0][..., np.newaxis]), axis=1)
                
                T = group['TA_F'].to_numpy()
                VPD = group['VPD_F'].to_numpy()
                vp = (0.6108 * np.exp(17.27*T/(T+237.3)))*10 - VPD
                input = np.concatenate((input, vp[..., np.newaxis]), axis=1)

                npp_file = f'../../data/TEM-MDM/monthly_NPP_{year}.nc'
                if check_file(npp_file):
                    feature = xr.open_dataset(npp_file, engine="netcdf4")['NPP'].values[lat_ind, lon_ind,:].reshape(1,1,12)
                    feature = np.nan_to_num(feature, nan=0.0)
                    feature = np.repeat(feature[:, :, :, np.newaxis], 31, axis=3)
                    feature = convert2year(feature)
                    input = np.concatenate((input, feature.reshape(365,1)), axis=1)
                else:
                    continue

                patch_channels = []


                for feat in stat_features:
                    patch = extract_patch(stat_dict[feat], lat_ind, lon_ind, patch_size=3)


                    if feat == 'wetlandtype' and site in site_wet:
                        patch = patch.copy()
                        patch[1, 1] = site_wet[site]

                    patch_365 = np.repeat(patch[np.newaxis, :, :], 365, axis=0)  # (365,3,3)
                    patch_channels.append(patch_365)


                patch_channels.append(np.full((365, 3, 3), ch4_dict[year], dtype=float))
                patch_channels.append(np.full((365, 3, 3), co2_dict[year], dtype=float))


                prec_file = f'../../data/TEM-MDM/daily_ecmwf_PREC_{year}.nc'
                if check_file(prec_file):
                    prec_data = xr.open_dataset(prec_file, engine="netcdf4")['PREC'].values
                    prec_patch = extract_patch(prec_data, lat_ind, lon_ind, patch_size=3)  # (3,3,12,31)
                    prec_patch = convert2year_patch(prec_patch)  # (365,3,3)
                    patch_channels.append(prec_patch)
                else:
                    continue


                solr_file_patch = f'../../data/TEM-MDM/daily_ecmwf_SOLR_{year}.nc'
                if check_file(solr_file_patch):
                    solr_data = xr.open_dataset(solr_file_patch, engine="netcdf4")['SOLR'].values
                    solr_patch = extract_patch(solr_data, lat_ind, lon_ind, patch_size=3)
                    solr_patch = convert2year_patch(solr_patch)
                    patch_channels.append(solr_patch)
                else:
                    continue


                tair_file = f'../../data/TEM-MDM/daily_ecmwf_TAIR_{year}.nc'
                if check_file(tair_file):
                    tair_data = xr.open_dataset(tair_file, engine="netcdf4")['TAIR'].values
                    tair_patch = extract_patch(tair_data, lat_ind, lon_ind, patch_size=3)
                    tair_patch = convert2year_patch(tair_patch)
                    patch_channels.append(tair_patch)
                else:
                    continue


                vapr_file = f'../../data/TEM-MDM/daily_ecmwf_VAPR_{year}.nc'
                if check_file(vapr_file):
                    vapr_data = xr.open_dataset(vapr_file, engine="netcdf4")['VAPR'].values
                    vapr_patch = extract_patch(vapr_data, lat_ind, lon_ind, patch_size=3)
                    vapr_patch = convert2year_patch(vapr_patch)
                    patch_channels.append(vapr_patch)
                else:
                    continue


                npp_file_patch = f'../../data/TEM-MDM/monthly_NPP_{year}.nc'
                if check_file(npp_file_patch):
                    npp_data = xr.open_dataset(npp_file_patch, engine="netcdf4")['NPP'].values
                    npp_data = np.nan_to_num(npp_data, nan=0.0)
                    npp_patch = extract_patch(npp_data, lat_ind, lon_ind, patch_size=3)  # (3,3,12)
                    npp_patch = np.repeat(npp_patch[..., np.newaxis], 31, axis=3)  # (3,3,12,31)
                    npp_patch = convert2year_patch(npp_patch)  # (365,3,3)
                    patch_channels.append(npp_patch)
                else:
                    continue

                patch_input = np.stack(patch_channels, axis=1)  # (365,15,3,3)

                if np.isnan(patch_input).any():
                    continue
                if np.isnan(input).any():
                    continue
                    
                label = group['FCH4_F'].to_numpy().copy()
                mask = np.isnan(label)
                label[mask] = group['FCH4_F_ANNOPTLM'].to_numpy()[mask]
                data_x.append(input)
                data_patch_x.append(patch_input)
                data_y.append(label)
                fold_samples += 1
                
            except Exception as e:
                log(f"ERROR processing {site} {year} in fold {fold}: {e}")
                continue
                
        try:
            data_x = np.stack(data_x, axis=0)
            data_patch_x = np.stack(data_patch_x, axis=0)
            data_y = np.stack(data_y, axis=0)
            log(f"Fold {fold}: point {data_x.shape}, patch {data_patch_x.shape}, max label: {np.max(data_y):.4f}, samples: {fold_samples}")
            np.save(f'../../processed_data/FLUXNET-CH4/spatial/data_x_{fold}.npy', data_x)
            np.save(f'../../processed_data/FLUXNET-CH4/spatial/data_y_{fold}.npy', data_y)
            np.save(f'../../processed_data/FLUXNET-CH4/spatial/data_patch_x_{fold}.npy', data_patch_x)
        except Exception as e:
            log(f"ERROR saving fold {fold} data: {e}")

    log('FLUXNET-CH4 processing complete!')

if __name__ == '__main__':
    main()
