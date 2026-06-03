import os
import pandas as pd
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. Configuration & File Paths
# ==========================================
# List the exact ERA5 files you want to merge (ignoring the wave file)
ERA5_FILES = [
    "C:/Users/shadi/Downloads/wildfire prediction/era5/data_stream-oper_stepType-accum.nc",
    "C:/Users/shadi/Downloads/wildfire prediction/era5/data_stream-oper_stepType-instant.nc"
]

# The folder containing all your downloaded NASA FIRMS CSVs
FIRMS_FOLDER = "C:/Users/shadi/Downloads/wildfire prediction/dataset/" 

# Earthformer Sequence parameters
INPUT_SEQ_LEN = 12   # Look back 12 hours
PREDICT_SEQ_LEN = 6  # Predict next 6 hours

# ==========================================
# 2. Load and Align Data
# ==========================================
def prepare_data(era5_files_list, firms_folder):
    # --- A. Load ERA5 ---
    print("Loading and merging ERA5 NetCDF files...")
    # open_mfdataset automatically combines variables from multiple files
    ds_era5 = xr.open_mfdataset(era5_files_list)
    
    lats = ds_era5['latitude'].values
    lons = ds_era5['longitude'].values
    times = ds_era5['valid_time'].values
    
    print(f"Successfully merged ERA5 data. Available variables: {list(ds_era5.data_vars)}")
    
    # --- B. Load FIRMS ---
    print(f"\nSearching for CSVs in: {firms_folder}")
    csv_files = []
    for root, dirs, files in os.walk(firms_folder):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, file))
                
    if not csv_files:
        raise ValueError(f"Could not find ANY .csv files in {firms_folder}")
        
    print(f"Found {len(csv_files)} FIRMS CSV files. Loading...")
    df_list = []
    for file in csv_files:
        df_list.append(pd.read_csv(file))
        
    df_firms = pd.concat(df_list, ignore_index=True)
    print(f"Total raw fire points loaded: {len(df_firms)}")
    
    # Filter confidence (handles both MODIS numbers and VIIRS letters)
    if 'confidence' in df_firms.columns:
        if df_firms['confidence'].dtype == 'O': 
            df_firms = df_firms[df_firms['confidence'] != 'l'] # VIIRS
        else: 
            df_firms = df_firms[df_firms['confidence'] > 30]   # MODIS
            
    # Time alignment to ERA5
    df_firms['acq_time'] = df_firms['acq_time'].astype(str).str.zfill(4)
    df_firms['datetime'] = pd.to_datetime(df_firms['acq_date'] + ' ' + df_firms['acq_time'], format='%Y-%m-%d %H%M')
    df_firms['datetime'] = df_firms['datetime'].dt.round('h')

    # ==========================================
    # 3. Rasterize FIRMS to ERA5 Grid
    # ==========================================
    print("\nRasterizing FIRMS data to ERA5 grid...")
    fire_mask = np.zeros((len(times), len(lats), len(lons)), dtype=np.float32)
    
    lat_bins = np.append(lats, lats[-1] - (lats[0]-lats[1])) 
    lon_bins = np.append(lons, lons[-1] + (lons[1]-lons[0]))
    lat_bins_sorted = np.sort(lat_bins)
    lon_bins_sorted = np.sort(lon_bins)
    era5_times_pd = pd.to_datetime(times)

    for index, row in df_firms.iterrows():
        t_val = row['datetime']
        try:
            t_idx = era5_times_pd.get_loc(t_val)
        except KeyError:
            continue # Skip if outside ERA5 time bounds

        lat_idx = np.digitize(row['latitude'], lat_bins_sorted) - 1
        lon_idx = np.digitize(row['longitude'], lon_bins_sorted) - 1
        
        if lats[0] > lats[-1]:
            lat_idx = len(lats) - 1 - lat_idx
            
        if 0 <= lat_idx < len(lats) and 0 <= lon_idx < len(lons):
            fire_mask[t_idx, lat_idx, lon_idx] = 1.0 

    ds_era5['fire_mask'] = (('time', 'latitude', 'longitude'), fire_mask)

    # ==========================================
    # 4. Stack Channels and Normalize for QML
    # ==========================================
    print("\nStacking and Normalizing channels for Quantum Layer...")
    
    # *** IMPORTANT: Update these string names based on the printout of ds_era5.data_vars! ***
    # For example, 'tp' might be your accumulated precipitation, 't2m' instantaneous temp
    variables_to_use = ['t2m', 'u10', 'v10', 'tp', 'fire_mask'] 
    
    # Create the tensor: Shape (Time, Channels, Height, Width)
    data_arrays = []
    for var in variables_to_use:
        if var in ds_era5:
            data_arrays.append(ds_era5[var].values)
        else:
            print(f"Warning: Variable '{var}' not found in ERA5 data!")
            
    stacked_data = np.stack(data_arrays, axis=1)
    
    # Normalize data (CRITICAL FOR QML) -> MinMax to [0, 1]
    T, C, H, W = stacked_data.shape
    stacked_data = stacked_data.reshape(T, C, -1) 
    
    for c in range(C):
        if variables_to_use[c] == 'fire_mask':
            continue # Don't scale the binary mask
        scaler = MinMaxScaler(feature_range=(0, 1))
        stacked_data[:, c, :] = scaler.fit_transform(stacked_data[:, c, :])
        
    stacked_data = stacked_data.reshape(T, C, H, W) 
    stacked_data = np.nan_to_num(stacked_data, nan=0.0)

    print(f"Final Tensor Shape (Time, Channels, Height, Width): {stacked_data.shape}")
    return stacked_data, variables_to_use

# ==========================================
# 5. PyTorch Dataset for Earthformer
# ==========================================
class WildfireDataset(Dataset):
    def __init__(self, data_tensor, input_len, pred_len):
        self.data = data_tensor
        self.input_len = input_len
        self.pred_len = pred_len
        self.total_seq = input_len + pred_len
        self.valid_indices = range(len(data_tensor) - self.total_seq + 1)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Input sequence
        x = self.data[idx : idx + self.input_len]
        # Target sequence (Predicting ONLY the fire_mask, which is the last channel)
        y = self.data[idx + self.input_len : idx + self.total_seq, -1:] 
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ==========================================
# 6. Execution
# ==========================================
if __name__ == "__main__":
    # Run the pipeline
    data_tensor, channels = prepare_data(ERA5_FILES, FIRMS_FOLDER)
    
    # Create DataLoader
    dataset = WildfireDataset(data_tensor, INPUT_SEQ_LEN, PREDICT_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
    
    # Verify Earthformer shapes
    print("\n--- Earthformer Tensor Check ---")
    for batch_x, batch_y in dataloader:
        print(f"Input X  (Batch, Time_In, Channels, Height, Width): {batch_x.shape}")
        print(f"Target Y (Batch, Time_Out, 1, Height, Width)      : {batch_y.shape}")
        break