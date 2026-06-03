import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def preprocess_nc_to_numpy(nc_path: str, out_dir: str):
    """Load a NetCDF file and save its variables as a channel-first NumPy array.

    The output is saved into `out_dir` as:
      - data.npy: shape (time, channels, lat, lon)
      - times.npy, latitude.npy, longitude.npy
      - channels.txt: variable names (one per line)
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(nc_path)
    vars = list(ds.data_vars)
    print(f"Loaded {nc_path} with variables: {vars}")

    # Stack variables into a channel axis (time, channel, lat, lon).
    data = np.stack([ds[v].values for v in vars], axis=1)

    np.save(out_dir / "data.npy", data)
    np.save(out_dir / "times.npy", ds["valid_time"].values)
    np.save(out_dir / "latitude.npy", ds["latitude"].values)
    np.save(out_dir / "longitude.npy", ds["longitude"].values)

    with open(out_dir / "channels.txt", "w", encoding="utf-8") as f:
        for v in vars:
            f.write(v + "\n")

    print(f"Saved preprocessed data to {out_dir} (data shape: {data.shape})")


def preprocess_fire_grids(
    csv_dir: str,
    out_dir: str,
    times: np.ndarray,
    lat: np.ndarray,g
    lon: np.ndarray,
    time_tolerance_minutes: int = 60,
):
    """Convert fire CSVs into a time-aligned grid matching the ERA5 pre_nc dataset.

    The result is saved in `out_dir/fire.npy` with shape (time, lat, lon), where
    each cell contains the count of fire detections that fell into the grid cell
    at the corresponding ERA5 time step.
    """

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

    # Use searchsorted on the sorted grid (handles monotonic increasing grids).
    lat_idx0 = np.searchsorted(lat_sorted, fire_df["latitude"].values)
    lon_idx0 = np.searchsorted(lon_sorted, fire_df["longitude"].values)

    # Clip indices and map back to original ordering.
    lat_idx = lat_sort[np.clip(lat_idx0, 0, len(lat_arr) - 1)]
    lon_idx = lon_sort[np.clip(lon_idx0, 0, len(lon_arr) - 1)]

    # Build the grid: (time, lat, lon)
    fire_grid = np.zeros((len(era_times), len(lat_arr), len(lon_arr)), dtype=np.uint16)

    np.add.at(
        fire_grid,
        (time_idx[in_range], lat_idx[in_range], lon_idx[in_range]),
        1,
    )

    fire_path = out_dir / "fire.npy"
    np.save(fire_path, fire_grid)
    print(f"Saved fire grid to {fire_path} (shape {fire_grid.shape})")


def make_earthformer_train_target(
    pre_nc_dir: str,
    out_dir: str,
    window_size: int = 8,
    crop_size: tuple[int, int] = (64, 64),
    max_time: int | None = 256,
    include_fire_channel: bool = False,
):
    """Combine ERA5 and fire grids into Earthformer-style train/target arrays.

    This generates a sliding-window training dataset:
      - Input: (N, window_size, H, W, C)
      - Target: (N, H, W, 1)  (fire at next time step)

    Only a cropped spatial region and a maximum time length are used to keep
    the dataset small enough to fit in memory.
    """

    pre_nc_dir = Path(pre_nc_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(pre_nc_dir / "data.npy")
    fire = np.load(pre_nc_dir / "fire.npy")

    assert data.shape[0] == fire.shape[0], "Time dimension mismatch between data and fire"

    # Convert to (time, height, width, channels)
    data_t = np.transpose(data, (0, 2, 3, 1))

    # Crop spatially (center crop) to keep memory down.
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

    # Replace NaNs/Infs to avoid training instability.
    inputs = np.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)
    targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)

    np.save(out_dir / "train_input.npy", inputs)
    np.save(out_dir / "train_target.npy", targets)
    np.save(out_dir / "train_times.npy", times)

    print(
        f"Saved train/target data to {out_dir}: input {inputs.shape}, target {targets.shape}"
    )


if __name__ == "__main__":
    pre_nc_dir = Path("pre_nc")
    pre_nc_dir.mkdir(exist_ok=True)

    nc_path = Path("nc dataset") / "data_stream-oper_stepType-instant.nc"
    preprocess_nc_to_numpy(nc_path, pre_nc_dir)

    times = np.load(pre_nc_dir / "times.npy")
    lat = np.load(pre_nc_dir / "latitude.npy")
    lon = np.load(pre_nc_dir / "longitude.npy")

    preprocess_fire_grids("dataset", pre_nc_dir, times, lat, lon)

    make_earthformer_train_target(pre_nc_dir, "trainandtar")
