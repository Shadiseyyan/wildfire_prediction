import os
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. SETUP YOUR PATHS HERE
# ==========================================
# Update these to where your January 2025 files are actually located
ERA5_FILES = [
    "C:/Users/shadi/Downloads/wildfire prediction/era5/data_stream-oper_stepType-accum.nc",
    "C:/Users/shadi/Downloads/wildfire prediction/era5/data_stream-oper_stepType-instant.nc"
]

FIRMS_FOLDER = "C:/Users/shadi/Downloads/wildfire prediction/dataset/" 

# Earthformer Sequence parameters
INPUT_SEQ_LEN = 12
PREDICT_SEQ_LEN = 6

def build_numpy_dataset():
    print("="*50)
    print(" 1. Loading and Merging ERA5 NetCDF Files")
    print("="*50)
    ds_era5 = xr.open_mfdataset(ERA5_FILES)
    lats = ds_era5['latitude'].values
    lons = ds_era5['longitude'].values
    times = ds_era5['valid_time'].values
    print(f"Loaded ERA5 grid: {len(times)} hours, {len(lats)} lats, {len(lons)} lons")
    
    print("\n" + "="*50)
    print(" 2. Loading and Filtering NASA FIRMS CSVs")
    print("="*50)
    csv_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(FIRMS_FOLDER) for f in filenames if f.lower().endswith('.csv')]
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {FIRMS_FOLDER}")
        
    df_firms = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    
    # Filter confidence (MODIS vs VIIRS)
    if 'confidence' in df_firms.columns:
        if df_firms['confidence'].dtype == 'O': 
            df_firms = df_firms[df_firms['confidence'] != 'l']
        else: 
            df_firms = df_firms[df_firms['confidence'] > 30]
            
    # Align time to hourly grid
    df_firms['acq_time'] = df_firms['acq_time'].astype(str).str.zfill(4)
    df_firms['datetime'] = pd.to_datetime(df_firms['acq_date'] + ' ' + df_firms['acq_time'], format='%Y-%m-%d %H%M')
    df_firms['datetime'] = df_firms['datetime'].dt.round('h')
    print(f"Processed {len(df_firms)} high-confidence fire points.")

    print("\n" + "="*50)
    print(" 3. Rasterizing Fire Points to Weather Grid")
    print("="*50)
    fire_mask = np.zeros((len(times), len(lats), len(lons)), dtype=np.float32)
    lat_bins = np.sort(np.append(lats, lats[-1] - (lats[0]-lats[1])))
    lon_bins = np.sort(np.append(lons, lons[-1] + (lons[1]-lons[0])))
    era5_times_pd = pd.to_datetime(times)

    for _, row in df_firms.iterrows():
        try: 
            t_idx = era5_times_pd.get_loc(row['datetime'])
        except KeyError: 
            continue
        lat_idx = np.digitize(row['latitude'], lat_bins) - 1
        lon_idx = np.digitize(row['longitude'], lon_bins) - 1
        if lats[0] > lats[-1]: 
            lat_idx = len(lats) - 1 - lat_idx
        if 0 <= lat_idx < len(lats) and 0 <= lon_idx < len(lons):
            fire_mask[t_idx, lat_idx, lon_idx] = 1.0 

    ds_era5['fire_mask'] = (('time', 'latitude', 'longitude'), fire_mask)

    print("\n" + "="*50)
    print(" 4. Channel Stacking, Cropping, and Scaling")
    print("="*50)
    # Extract the variables. (Verify these names match your ERA5 file printout!)
    variables = ['t2m', 'u10', 'v10', 'tp', 'fire_mask'] 
    
    # Stack into shape: (Time, Lat, Lon, Channels) -> Channels Last!
    stacked_data = np.stack([ds_era5[v].values for v in variables], axis=-1)
    
    # Auto-Crop to nearest multiple of 8 for Earthformer Transformers
# Auto-Crop to nearest multiple of 32 (Patch Size 8 * Local Window 4 = 32)
    T, H, W, C = stacked_data.shape
    new_H = (H // 32) * 32
    new_W = (W // 32) * 32
    stacked_data = stacked_data[:, :new_H, :new_W, :]
    print(f"Cropped spatial grid from {H}x{W} to {new_H}x{new_W} to satisfy Attention windows.")
    
    # Normalize the weather data using MinMax
    stacked_data = stacked_data.reshape(T, -1, C)
    for c in range(C):
        if variables[c] == 'fire_mask': 
            continue # Skip binary mask
        scaler = MinMaxScaler()
        stacked_data[:, :, c] = scaler.fit_transform(stacked_data[:, :, c])
    stacked_data = stacked_data.reshape(T, new_H, new_W, C)
    stacked_data = np.nan_to_num(stacked_data, nan=0.0)

    print("\n" + "="*50)
    print(" 5. Generating Sliding Windows & Saving")
    print("="*50)
    X_list, Y_list = [], []
    total_seq = INPUT_SEQ_LEN + PREDICT_SEQ_LEN
    
    for i in range(len(stacked_data) - total_seq + 1):
        X_list.append(stacked_data[i : i + INPUT_SEQ_LEN])
        # Target only needs the fire mask channel (the last one)
        Y_list.append(stacked_data[i + INPUT_SEQ_LEN : i + total_seq, :, :, -1:])
        
    X_arr = np.array(X_list, dtype=np.float32)
    Y_arr = np.array(Y_list, dtype=np.float32)

    # Save to the exact folder the Earthformer is looking for
    os.makedirs("trainandtars", exist_ok=True)
    np.save("trainandtars/train_input.npy", X_arr)
    np.save("trainandtars/train_target.npy", Y_arr)
    
    print("\n✅ SUCCESS! Dataset generated and saved.")
    print(f" -> X Shape (Input):  {X_arr.shape}")
    print(f" -> Y Shape (Target): {Y_arr.shape}")

if __name__ == "__main__":  
    build_numpy_dataset()