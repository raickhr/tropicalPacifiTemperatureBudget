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

if rank == 0:
    print(f"[MPI] Starting with {size} ranks")


# =========================
# 1. HELPERS
# =========================

def combine_parcels_zarr(file_pattern: str) -> xr.Dataset:
    """
    Combines multiple Parcels Zarr outputs into a single Dataset.

    Expects:
      - each Zarr has dims like (trajectory, obs)
      - a 2D 'time(trajectory, obs)' variable with shared timestamps along obs

    Produces:
      - one dataset with dimensions (trajectory, time)
      - 'time' is a 1D coordinate (formerly 'obs'), shared across trajectories
    """
    zarr_files = sorted(glob(file_pattern))
    if not zarr_files:
        raise ValueError(f"No Zarr files found matching pattern: {file_pattern}")

    datasets = [xr.open_dataset(path, engine="zarr", chunks="auto") for path in zarr_files]

    # Stack along 'trajectory'
    ds_combined = xr.combine_nested(datasets, concat_dim="trajectory")

    # Mask out invalid times
    valid_time_mask = ds_combined["time"].notnull()
    ds_combined = ds_combined.where(valid_time_mask, drop=False)

    # Derive 1D time coordinate by averaging along trajectory
    time_1d = ds_combined["time"].mean(dim="trajectory")

    # Drop old 2D time variable and assign new 1D time coordinate
    ds_combined = ds_combined.drop_vars("time")
    ds_combined["time"] = time_1d
    ds_combined = ds_combined.assign_coords(time=ds_combined.time)

    # Swap obs -> time and drop obs var if present
    if "obs" in ds_combined.dims:
        ds_combined = ds_combined.swap_dims({"obs": "time"})
    if "obs" in ds_combined:
        ds_combined = ds_combined.drop_vars("obs")

    return ds_combined


# =========================
# 2. RANK 0 LOADS DATA
# =========================

folder = "/srv/seolab/srai/tropicalPacifiTemperatureBudget/WPWP_GLORYS_data/"
#ufiles = sorted(glob(f"{folder}/GLORYS12v1_dailyAvg_withVerticalVelocities_2018-??-??.nc"))
curDate = datetime(2018,3,1)
endDate = datetime(2019,3,2)
ufiles = []
while curDate <= endDate:
    ufiles.append(f"{folder}/GLORYS12v1_dailyAvg_withVerticalVelocities_{curDate.year:04d}-{curDate.month:02d}-{curDate.day:02d}.nc")
    curDate += timedelta(days =1)

tasks = None  # will hold per-rank work packages on rank 0

if rank == 0:
    if not ufiles:
        raise FileNotFoundError(
            f"No GLORYS files found with pattern: "
            f"{folder}/GLORYS12v1_dailyAvg_withVerticalVelocities_2018-??-??.nc"
        )

    print(f"[MPI][rank 0] Loading GLORYS SST from {len(ufiles)} files...")
    fieldData = xr.open_mfdataset(ufiles, combine="by_coords")

    # Surface SST
    sst_da = fieldData["thetao"].isel(depth=0)  # [time, y, x]

    # Grid (shared)
    lon2d = sst_da["longitude"].values.astype("float32", copy=False)
    lat2d = sst_da["latitude"].values.astype("float32", copy=False)

    # Load Parcels trajectories
    print("[MPI][rank 0] Loading Parcels trajectories from Zarr...")
    ds_parcels = combine_parcels_zarr("longbox_corrected.zarr/*.zarr")

    lon_p_all = ds_parcels["lon"].values.astype("float32", copy=False)
    lat_p_all = ds_parcels["lat"].values.astype("float32", copy=False)
    timerange_all = ds_parcels["time"].values  # datetime64

    ds_parcels.close()
    fieldData.close()

    if lon_p_all.shape != lat_p_all.shape:
        raise ValueError("lon and lat trajectory arrays have mismatched shapes.")

    n_traj, n_frames = lon_p_all.shape
    n_sst_time = sst_da.sizes["time"]

    if timerange_all.shape[0] != n_frames:
        raise ValueError(
            f"time length {timerange_all.shape[0]} does not match particle time dim {n_frames}"
        )

    print(f"[MPI][rank 0] Shapes:")
    print(f"  SST time: {n_sst_time}, grid: {lon2d.shape}")
    print(f"  Traj: {n_traj} x {n_frames} (n_traj x time)")

    # --------------------------------------
    # Build per-rank tasks (no full SST bcast)
    # --------------------------------------
    tasks = []

    for r in range(size):
        # Global frame indices assigned to rank r
        frames_r = list(range(r, n_frames, size))

        if len(frames_r) == 0:
            # Rank gets no work
            task = {
                "frames": [],
                "sst_slices": np.empty((0, 0, 0), dtype="float32"),
                "sst_idx_for_frame": [],
                "lon_p": np.empty((n_traj, 0), dtype="float32"),
                "lat_p": np.empty((n_traj, 0), dtype="float32"),
                "timerange": np.empty((0,), dtype=timerange_all.dtype),
                "lon2d": lon2d,
                "lat2d": lat2d,
            }
            tasks.append(task)
            continue

        # Unique SST time indices needed for these frames
        sst_idx_needed = sorted(
            set(min(i // 4, n_sst_time - 1) for i in frames_r)
        )

        # Map global sst index -> local index within sst_slices
        idx_map = {g: j for j, g in enumerate(sst_idx_needed)}

        # Extract just these SST slices [n_sst_local, y, x]
        sst_local = (
            sst_da.isel(time=sst_idx_needed)
            .values
            .astype("float32", copy=False)
        )

        # Particle positions restricted to these frames [n_traj, n_local_frames]
        lon_p_local = lon_p_all[:, frames_r]
        lat_p_local = lat_p_all[:, frames_r]
        timerange_local = timerange_all[frames_r]

        # For each local frame j, which local SST slice index to use
        sst_idx_for_frame = [
            idx_map[min(i // 4, n_sst_time - 1)] for i in frames_r
        ]

        task = {
            "frames": frames_r,                    # global frame indices
            "sst_slices": sst_local,               # [n_sst_local, y, x]
            "sst_idx_for_frame": sst_idx_for_frame,# len = n_local_frames
            "lon_p": lon_p_local,                  # [n_traj, n_local_frames]
            "lat_p": lat_p_local,                  # [n_traj, n_local_frames]
            "timerange": timerange_local,          # [n_local_frames]
            "lon2d": lon2d,
            "lat2d": lat2d,
        }
        tasks.append(task)

    # Free big globals from rank 0 scope (kept only inside tasks)
    del lon_p_all, lat_p_all, timerange_all, sst_da

# =======================================
# 3. SCATTER TASKS (NO FULL SST BROADCAST)
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

    j indexes into:
      - frames[j]: global frame index
      - sst_idx_for_frame[j]: which sst_slices[k] to use
      - lon_p[:, j], lat_p[:, j]
      - timerange[j]
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
        vmin=25,
        vmax=30,
        cmap=cmap,
        shading="auto"
    )

    ax.scatter(lon_part, lat_part, s=8, c="lime", alpha=1)
    ax.set_title(f"{t}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.colorbar(pmesh, ax=ax, label="SST (°C)")

    frame_path = os.path.join(out_dir, f"frame_{global_i:04d}.png")
    fig.savefig(frame_path, dpi=100)
    plt.close(fig)


def generate_frames_mpi():
    if n_local_frames == 0:
        # No frames for this rank
        comm.Barrier()
        return

    print(f"[MPI][rank {rank}] Generating {n_local_frames} frames...")
    for j in range(n_local_frames):
        make_local_frame(j)

    comm.Barrier()


# ==============================
# 6. BUILD MP4 FROM PNG FRAMES
# ==============================

def build_mp4_from_frames(
    images_pattern: str = "images/frame_*.png",
    output_file: str = "trajectories.mp4",
    fps: int = 60
):
    frames = sorted(glob(images_pattern))
    if not frames:
        raise ValueError(f"No frames found with pattern: {images_pattern}")

    print(f"[MPI][rank 0] Writing {len(frames)} frames to {output_file} at {fps} fps...")
    with imageio.get_writer(output_file, fps=fps) as writer:
        for f in frames:
            img = imageio.imread(f)
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
            fps=60
        )
