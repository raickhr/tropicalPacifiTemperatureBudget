import os
import numpy as np
import xarray as xr
import matplotlib
from datetime import datetime, timedelta

matplotlib.use("Agg")  # headless backend for clusters/servers
import matplotlib.pyplot as plt

from glob import glob
import cmocean as cmo
import imageio.v2 as imageio
from mpi4py import MPI

# =================================
# 0. MPI INITIALIZATION
# =================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


lonMin = 178.0
lonMax = 182.0

latMin = -1.0
latMax = 1.0

if rank == 0:
    print(f"[MPI] Starting with {size} ranks")


# =========================
# 1. HELPERS
# =========================

def combine_parcels_zarr(file_pattern: str) -> xr.Dataset:
    """
    Combine multiple Parcels Zarr outputs into a single Dataset with a shared time axis.

    Handles the case where particles (trajectories) do NOT share identical observation times.

    Assumes Zarr with:
      - dims: trajectory, obs
      - vars: time(trajectory, obs), lon(trajectory, obs), lat(...), etc.

    Output:
      - dims: trajectory, time
      - 'time': 1D sorted union of all valid times
      - NaNs where a trajectory has no data at that time
    """

    zarr_files = sorted(glob(file_pattern))
    if not zarr_files:
        raise ValueError(f"No Zarr files found matching pattern: {file_pattern}")

    datasets = [xr.open_dataset(p, engine="zarr", chunks="auto") for p in zarr_files]

    ds = xr.combine_nested(
        datasets,
        concat_dim="trajectory",
        combine_attrs="override"
    )

    if "obs" not in ds.dims:
        raise ValueError("Expected 'obs' dimension in Parcels output, but not found.")
    if "time" not in ds:
        raise ValueError("Expected 'time' variable in Parcels output, but not found.")

    time_2d = ds["time"]  # (trajectory, obs)

    # Global 1D time axis: union of all valid times
    if np.issubdtype(time_2d.dtype, np.datetime64):
        time_vals = time_2d.values
        mask = ~np.isnat(time_vals)
        unique_times = np.unique(time_vals[mask])
    else:
        time_vals = time_2d.values
        mask = ~np.isnan(time_vals)
        unique_times = np.unique(time_vals[mask])

    if unique_times.size == 0:
        raise ValueError("No valid time values found in Parcels output.")

    time_1d = np.sort(unique_times)

    ntraj = ds.dims["trajectory"]
    ntime = time_1d.size

    out = xr.Dataset()
    out = out.assign_coords(
        trajectory=ds["trajectory"] if "trajectory" in ds.coords else np.arange(ntraj),
        time=("time", time_1d),
    )

    # Copy over non-obs variables (e.g. trajectory metadata)
    for name, var in ds.variables.items():
        if "obs" not in var.dims and name not in ("time",):
            out[name] = var

    def map_var_to_time_axis(var2d: xr.DataArray) -> xr.DataArray:
        """Map (trajectory, obs) -> (trajectory, time) with NaN fill."""
        if var2d.dims != ("trajectory", "obs"):
            return None

        target_dtype = (
            np.float64 if np.issubdtype(var2d.dtype, np.number) else var2d.dtype
        )
        data = np.full((ntraj, ntime), np.nan, dtype=target_dtype)

        var_vals = var2d.values
        t_vals = time_2d.values

        for itraj in range(ntraj):
            t_row = t_vals[itraj, :]
            v_row = var_vals[itraj, :]

            if np.issubdtype(t_row.dtype, np.datetime64):
                valid = ~np.isnat(t_row)
            else:
                valid = ~np.isnan(t_row)

            if not np.any(valid):
                continue

            t_valid = t_row[valid]
            v_valid = v_row[valid]

            idx = np.searchsorted(time_1d, t_valid)

            good = (idx >= 0) & (idx < ntime)
            idx = idx[good]
            v_valid = v_valid[good]

            data[itraj, idx] = v_valid

        return xr.DataArray(
            data,
            dims=("trajectory", "time"),
            coords={"trajectory": out["trajectory"], "time": out["time"]},
            attrs=var2d.attrs,
        )

    # Remap all obs-based trajectory variables
    for name, var in ds.variables.items():
        if "obs" in var.dims:
            if name == "time":
                continue  # replaced by the shared 'time'
            da_mapped = map_var_to_time_axis(var)
            if da_mapped is not None:
                out[name] = da_mapped

    # Drop time steps where ~all trajectories are NaN (based on lon/lat)
    lon_all = out["lon"].to_numpy() # traj, time
    lat_all = out["lat"].to_numpy() # traj, time

    if lon_all.shape != lat_all.shape:
        raise ValueError("lon and lat shapes do not match in combined Parcels data.")

    mask = np.isnan(lon_all) | np.isnan(lat_all)
    ntraj, ntime = lon_all.shape
    num_nans_time = np.sum(mask, axis=0)
    time_keep = num_nans_time < 0.98 * ntraj  # keep times with at least 2% valid

    out = out.isel(time=time_keep)

    # Get first index along time (per trajectory) where both lon/lat are valid
    valid = (~np.isnan(lon_all)) & (~np.isnan(lat_all))   # [ntraj, ntime]
    any_valid = valid.any(axis=1)                         # [ntraj]

    # First True index per row; for rows with no True, keep a dummy 0 then mask out
    firstIndex = np.argmax(valid, axis=1)                 # [ntraj]

    # Gather first positions safely with paired fancy indexing
    row = np.arange(ntraj)
    firstLon = np.full(ntraj, np.nan, dtype=float)
    firstLat = np.full(ntraj, np.nan, dtype=float)
    good = any_valid
    firstLon[good] = lon_all[row[good], firstIndex[good]]
    firstLat[good] = lat_all[row[good], firstIndex[good]]

    # Inside-box test (inclusive)
    inside = (
        (firstLon >= lonMin) & (firstLon <= lonMax) &
        (firstLat >= latMin) & (firstLat <= latMax) &
        good
    )

    # Keep only trajectories that start inside the box
    out = out.isel(trajectory=inside)



    return out


# =========================
# 2. RANK 0 LOADS DATA
# =========================

folder = "/srv/seolab/srai/tropicalPacifiTemperatureBudget/WPWP_GLORYS_data/"

curDate = datetime(2018, 4, 1)
endDate = datetime(2019, 4, 2)
ufiles = []
while curDate <= endDate:
    ufiles.append(
        f"{folder}/GLORYS12v1_dailyAvg_withVerticalVelocities_"
        f"{curDate.year:04d}-{curDate.month:02d}-{curDate.day:02d}.nc"
    )
    curDate += timedelta(days=1)

tasks = None  # will hold per-rank work packages on rank 0

if rank == 0:
    if not ufiles:
        raise FileNotFoundError(
            "No GLORYS files found for the specified date range."
        )

    print(f"[MPI][rank 0] Loading GLORYS SST from {len(ufiles)} files...")
    fieldData = xr.open_mfdataset(ufiles, combine="by_coords")

    # Surface SST [time, y, x]
    sst_da = fieldData["thetao"].isel(depth=0)

    # Load into memory so slices work after closing fieldData
    sst_da = sst_da.load()
    sst_time = sst_da["time"].values  # numpy datetime64[ns] array
    n_sst_time = sst_da.sizes["time"]
    print('sst time range from ', sst_time[0], sst_time[-1])

    # Grid (shared)
    lon2d = sst_da["longitude"].values.astype("float32", copy=False)
    lat2d = sst_da["latitude"].values.astype("float32", copy=False)

    # Load Parcels trajectories
    print("[MPI][rank 0] Loading Parcels trajectories from Zarr...")
    ds_parcels = combine_parcels_zarr("longbox_delayed.zarr/*.zarr")

    lon_p_all = ds_parcels["lon"].values.astype("float32", copy=False)
    lat_p_all = ds_parcels["lat"].values.astype("float32", copy=False)
    timerange_all = ds_parcels["time"].values  # 1D datetime64 matching columns of lon/lat

    if lon_p_all.shape != lat_p_all.shape:
        raise ValueError("lon and lat trajectory arrays have mismatched shapes.")

    n_traj, n_frames = lon_p_all.shape

    if timerange_all.shape[0] != n_frames:
        raise ValueError(
            f"time length {timerange_all.shape[0]} does not match particle time dim {n_frames}"
        )

    print(f"[MPI][rank 0] Shapes:")
    print(f"  SST time: {n_sst_time}, grid: {lon2d.shape}")
    print(f"  Traj: {n_traj} x {n_frames} (n_traj x time)")

    # Build per-rank tasks (round-robin in frame index)
    tasks = []

    for r in range(size):
        frames_r = list(range(r, n_frames, size))

        if len(frames_r) == 0:
            # Rank gets no work but must receive a valid task dict
            task = {
                "frames": [],
                "sst_slices": np.empty((0, 0, 0), dtype="float32"),
                "sst_idx_for_frame": [],
                "lon_p": np.empty((n_traj, 0), dtype("float32")),
                "lat_p": np.empty((n_traj, 0), dtype("float32")),
                "timerange": np.empty((0,), dtype=timerange_all.dtype),
                "lon2d": lon2d,
                "lat2d": lat2d,
            }
            tasks.append(task)
            continue

        lon_p_local = lon_p_all[:, frames_r]      # [n_traj, n_local_frames]
        lat_p_local = lat_p_all[:, frames_r]
        timerange_local = timerange_all[frames_r] # [n_local_frames]

        # Compute required SST indices (global)
        sst_idx_needed = []
        for pt in timerange_local:
            idx = np.searchsorted(sst_time, pt, side="right") - 1
            # clamp to [0, n_sst_time-1] for safety
            if idx < 0:
                idx = 0
            elif idx >= n_sst_time:
                idx = n_sst_time - 1
            sst_idx_needed.append(idx)

        sst_idx_needed = sorted(set(sst_idx_needed))

        # Extract SST slices [n_sst_local, y, x]
        sst_local = sst_da.isel(time=sst_idx_needed).values.astype("float32", copy=False)

        # Build map global_idx -> local_idx
        idx_map = {g: j for j, g in enumerate(sst_idx_needed)}

        # For each local frame, which local SST slice index to use
        sst_idx_for_frame = []
        for pt in timerange_local:
            idx = np.searchsorted(sst_time, pt, side="right") - 1
            if idx < 0:
                idx = 0
            elif idx >= n_sst_time:
                idx = n_sst_time - 1
            sst_idx_for_frame.append(idx_map[idx])

        task = {
            "frames": frames_r,
            "sst_slices": sst_local,                # [n_sst_local, y, x]
            "sst_idx_for_frame": sst_idx_for_frame, # len = n_local_frames
            "lon_p": lon_p_local,                   # [n_traj, n_local_frames]
            "lat_p": lat_p_local,
            "timerange": timerange_local,
            "lon2d": lon2d,
            "lat2d": lat2d,
        }
        tasks.append(task)

    # Clean up heavy objects (all needed data is now in tasks)
    ds_parcels.close()
    fieldData.close()
    del lon_p_all, lat_p_all, timerange_all, sst_da

# =======================================
# 3. SCATTER TASKS
# =======================================

my_task = comm.scatter(tasks if rank == 0 else None, root=0)

frames = my_task["frames"]
sst_slices = my_task["sst_slices"]
sst_idx_for_frame = my_task["sst_idx_for_frame"]
lon_p = my_task["lon_p"]
lat_p = my_task["lat_p"]
timerange = my_task["timerange"]
lon2d = my_task["lon2d"]
lat2d = my_task["lat2d"]

n_local_frames = len(frames)

if rank == 0:
    total_frames = sum(len(t["frames"]) for t in tasks)
    print(f"[MPI] Work assigned. Total frames: {total_frames}")


# ==========================
# 4. PLOTTING CONFIG
# ==========================

out_dir = "images"
os.makedirs(out_dir, exist_ok=True)

cmap = cmo.cm.balance
cmap.set_bad("gray")


# ==========================================
# 5. FRAME GENERATION (PER-RANK)
# ==========================================

def make_local_frame(j: int):
    """
    Make frame for the j-th local entry in this rank's task.
    """
    global_i = frames[j]
    sst_k = sst_idx_for_frame[j]

    sst2d = sst_slices[sst_k, :, :]
    lon_part = lon_p[:, j]
    lat_part = lat_p[:, j]
    t = str(timerange[j])[:19].replace("T", " ")

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)

    pmesh = ax.pcolormesh(
        lon2d,
        lat2d,
        sst2d,
        vmin=27.5,
        vmax=29.5,
        cmap=cmap,
        shading="auto",
    )

    # Only plot valid particle positions
    valid = ~np.isnan(lon_part) & ~np.isnan(lat_part)
    ax.scatter(lon_part[valid], lat_part[valid], s=8, c="lime", alpha=1)

    ax.set_title(f"{t}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.colorbar(pmesh, ax=ax, label="SST (°C)")

    frame_path = os.path.join(out_dir, f"frame_{global_i:04d}.png")
    fig.savefig(frame_path, dpi=100)
    plt.close(fig)


def generate_frames_mpi():
    if n_local_frames > 0:
        print(f"[MPI][rank {rank}] Generating {n_local_frames} frames...")
        for j in range(n_local_frames):
            make_local_frame(j)

    # Sync all ranks before video assembly
    comm.Barrier()


# ==============================
# 6. BUILD MP4 FROM PNG FRAMES
# ==============================

def build_mp4_from_frames(
    images_pattern: str = "images/frame_*.png",
    output_file: str = "trajectories.mp4",
    fps: int = 60,
):
    frames = sorted(glob(images_pattern))
    if not frames:
        raise ValueError(f"No frames found with pattern: {images_pattern}")

    print(f"[MPI][rank 0] Writing {len(frames)} frames to {output_file} at {fps} fps...")
    with imageio.get_writer(output_file, fps=fps) as writer:
        for fpath in frames:
            img = imageio.imread(fpath)
            writer.append_data(img)

    print(f"[MPI][rank 0] Saved MP4: {output_file}")


# ===========================
# 7. MAIN
# ===========================

if __name__ == "__main__":
    generate_frames_mpi()

    if rank == 0:
        build_mp4_from_frames(
            images_pattern=os.path.join(out_dir, "frame_*.png"),
            output_file="trajectories.mp4",
            fps=60,
        )
