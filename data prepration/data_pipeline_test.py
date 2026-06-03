import os
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import xarray as xr

def preprocess_multiple_nc_to_numpy(nc_paths: List[Union[str, Path]], out_dir: str):
    """Load multiple NetCDF files manually, force grid alignment, and save as NumPy array."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {len(nc_paths)} NetCDF files manually to bypass grid errors...")

    # 1. Open all files individually
    datasets = []
    for p in nc_paths:
        ds = xr.open_dataset(p)
        
        # FIX A: Force the coordinates to be sorted forwards (solves the monotonic error)
        ds = ds.sortby('latitude')
        ds = ds.sortby('longitude')
        
        datasets.append(ds)

    # 2. Extract the "Master Grid" from the very first file
    base_lat = datasets[0]['latitude'].values
    base_lon = datasets[0]['longitude'].values

    # 3. THE FIX: Interpolate (Resize) all other maps to match the Master Grid
    aligned_datasets = [datasets[0]] # The first file is already perfect
    
    for ds in datasets[1:]:
        print("Resizing a dataset to match the master grid...")
        # This magically resizes the map (e.g., from 243 down to 231) 
        # AND fixes the microscopic floating-point errors at the same time!
        aligned_ds = ds.interp(latitude=base_lat, longitude=base_lon)
        aligned_datasets.append(aligned_ds)

   # 4. Merge them securely using the newly aligned datasets
    print("Merging datasets...")
    combined_ds = xr.merge(aligned_datasets)

    # --- MAKE SURE THESE TWO LINES ARE HERE ---
    time_key = "valid_time" if "valid_time" in combined_ds.coords else "time"
    combined_ds = combined_ds.sortby(time_key)
    # ------------------------------------------

    vars = list(combined_ds.data_vars)
    print(f"Combined dataset variables: {vars}")

    # Stack variables into a channel axis (time, channel, lat, lon).
    data = np.stack([combined_ds[v].values for v in vars], axis=1)

    np.save(out_dir / "data.npy", data)
    np.save(out_dir / "times.npy", combined_ds[time_key].values)
    np.save(out_dir / "latitude.npy", combined_ds["latitude"].values)
    np.save(out_dir / "longitude.npy", combined_ds["longitude"].values)

    with open(out_dir / "channels.txt", "w", encoding="utf-8") as f:
        for v in vars:
            f.write(v + "\n")

    print(f"Saved preprocessed merged data to {out_dir} (data shape: {data.shape})")

def preprocess_fire_grids(
    csv_dir: str,
    out_dir: str,
    times: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    time_tolerance_minutes: int = 60,
):
    """Convert fire CSVs into a time-aligned grid matching the ERA5 pre_nc dataset."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and concat all fire CSVs.
    csv_dir = Path(csv_dir)
    csv_paths = sorted(csv_dir.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    dfs = []
    for p in csv_paths:
        df = pd.read_csv(
            p,
            usecols=["latitude", "longitude", "acq_date", "acq_time"],
            dtype={"latitude": float, "longitude": float, "acq_date": str, "acq_time": str},
        )
        dfs.append(df)
    fire_df = pd.concat(dfs, ignore_index=True)

    # Parse acquisition times into UTC datetime64[ns].
    times_str = fire_df["acq_date"] + " " + fire_df["acq_time"].str.zfill(4)
    fire_dt = pd.to_datetime(times_str, format="%Y-%m-%d %H%M", utc=True)

    # Match fire times to nearest ERA5 time index.
    era_times = np.asarray(times, dtype="datetime64[ns]")
    fire_np = fire_dt.values
    idx = np.searchsorted(era_times, fire_np)

    idx_lo = np.clip(idx - 1, 0, len(era_times) - 1)
    idx_hi = np.clip(idx, 0, len(era_times) - 1)

    diff_lo = np.abs(fire_np - era_times[idx_lo])
    diff_hi = np.abs(era_times[idx_hi] - fire_np)
    use_hi = diff_hi < diff_lo

    time_idx = np.where(use_hi, idx_hi, idx_lo)
    closest_delta = np.minimum(diff_lo, diff_hi)

    tol = np.timedelta64(time_tolerance_minutes, "m")
    in_range = closest_delta <= tol

    print(f"Mapped {len(fire_df)} fire records; {in_range.sum()} within {time_tolerance_minutes} min of ERA5 times")

    # Map lat/lon to the ERA5 grid.
    lat_arr = np.asarray(lat)
    lon_arr = np.asarray(lon)

    lat_sort = np.argsort(lat_arr)
    lon_sort = np.argsort(lon_arr)

    lat_sorted = lat_arr[lat_sort]
    lon_sorted = lon_arr[lon_sort]

    lat_idx0 = np.searchsorted(lat_sorted, fire_df["latitude"].values)
    lon_idx0 = np.searchsorted(lon_sorted, fire_df["longitude"].values)

    lat_idx = lat_sort[np.clip(lat_idx0, 0, len(lat_arr) - 1)]
    lon_idx = lon_sort[np.clip(lon_idx0, 0, len(lon_arr) - 1)]

    fire_grid = np.zeros((len(era_times), len(lat_arr), len(lon_arr)), dtype=np.uint16)

    np.add.at(
        fire_grid,
        (time_idx[in_range], lat_idx[in_range], lon_idx[in_range]),
        1,
    )

    fire_path = out_dir / "fire.npy"
    np.save(fire_path, fire_grid)
    print(f"Saved fire grid to {fire_path} (shape {fire_grid.shape})")


def make_earthformer_dataset(
    pre_nc_dir: str,
    out_dir: str,
    prefix: str = "test", # Allows switching between "train" and "test" easily
    window_size: int = 8,
    crop_size: tuple[int, int] = (64, 64),
    max_time: int | None = 256,
    include_fire_channel: bool = False,
):
    """Combine ERA5 and fire grids into Earthformer-style arrays."""
    pre_nc_dir = Path(pre_nc_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(pre_nc_dir / "data.npy")
    fire = np.load(pre_nc_dir / "fire.npy")

    assert data.shape[0] == fire.shape[0], "Time dimension mismatch between data and fire"

    # Convert to (time, height, width, channels)
    data_t = np.transpose(data, (0, 2, 3, 1))

    # Crop spatially
    H_full, W_full = data_t.shape[1], data_t.shape[2]
    ch, cw = crop_size
    sh = (H_full - ch) // 2
    sw = (W_full - cw) // 2
    data_t = data_t[:, sh : sh + ch, sw : sw + cw]
    fire = fire[:, sh : sh + ch, sw : sw + cw]

    if max_time is not None:
        data_t = data_t[:max_time]
        fire = fire[:max_time]

    T_total, H, W, C = data_t.shape
    N = T_total - window_size
    if N <= 0:
        raise ValueError("window_size must be smaller than total time steps")

    print(f"Building dataset: T={T_total}, H={H}, W={W}, C={C}, N={N}")

    inputs = np.zeros((N, window_size, H, W, C), dtype=np.float32)
    targets = np.zeros((N, H, W, 1), dtype=np.float32)
    times = np.zeros((N,), dtype="datetime64[ns]")

    time_arr = np.load(pre_nc_dir / "times.npy")

    for i in range(N):
        inputs[i] = data_t[i : i + window_size]
        targets[i, ..., 0] = fire[i + window_size]
        times[i] = time_arr[i + window_size]

    if include_fire_channel:
        fire_windows = fire[: N + window_size]
        fire_windows = fire_windows.reshape(N, window_size, H, W)
        fire_chan = fire_windows[..., None]
        inputs = np.concatenate([inputs, fire_chan], axis=-1)

    inputs = np.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)
    targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)

    # Saves as test_input.npy, test_target.npy, test_times.npy
    np.save(out_dir / f"{prefix}_input.npy", inputs)
    np.save(out_dir / f"{prefix}_target.npy", targets)
    np.save(out_dir / f"{prefix}_times.npy", times)

    print(
        f"Saved {prefix} data to {out_dir}: input {inputs.shape}, target {targets.shape}"
    )


if __name__ == "__main__":
    # 1. Setup Directories
    pre_nc_dir = Path("pre_nc_test")
    pre_nc_dir.mkdir(exist_ok=True)
    
    # Point this to the folder containing ALL your October and November .nc files
    nc_dataset_dir = Path("C:/Users/shadi/Downloads/wildfire prediction/test nc dataset")
    
    # THE MAGIC FIX: This automatically grabs every single .nc file in the folder, 
    # regardless of whether it is 'instant', 'accum', October, or November!
    nc_paths = sorted(list(nc_dataset_dir.rglob("*.nc")))
    
    if not nc_paths:
        raise FileNotFoundError(f"No .nc files found in {nc_dataset_dir}! Check the folder path.")

    print(f"Found {len(nc_paths)} NetCDF files. Merging them now...")

    # 2. Process NetCDF files
    # xarray will intelligently stitch time AND merge the data types (instant/accum).
    preprocess_multiple_nc_to_numpy(nc_paths, pre_nc_dir)

    # 3. Load variables for Fire grid processing
    times = np.load(pre_nc_dir / "times.npy")
    lat = np.load(pre_nc_dir / "latitude.npy")
    lon = np.load(pre_nc_dir / "longitude.npy")

    # 4. Process Fire Dataset (Update "dataset" to your FIRMS folder path if needed)
    preprocess_fire_grids("dataset", pre_nc_dir, times, lat, lon)

    # 5. Build final test tensors
    make_earthformer_dataset(
        pre_nc_dir=pre_nc_dir, 
        out_dir="test_and_tar", 
        prefix="test",       # <--- Forces it to output test_input.npy
        max_time=None        # <--- Recommend setting to None so it captures all days
    )