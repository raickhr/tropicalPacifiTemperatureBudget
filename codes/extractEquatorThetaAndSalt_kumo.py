#!/usr/bin/env python

import xarray as xr
import numpy as np
import os
import sys
from datetime import datetime, timedelta

from mpi4py import MPI

# ============================================================
# MPI SETUP
# ============================================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ============================================================
# USER SETTINGS
# ============================================================
rootFolder = "/srv/cdx/hseo/Data/GLORYS12v1/PCF"
writeFolder = '/srv/seolab/srai/tropicalPacifiTemperatureBudget/WPWP_GLORYS_data'
tempFolder = os.path.join(writeFolder, "temp_eq")  # folder for per-day output
startDate = datetime(2014, 1, 1)
endDate   = datetime(2024, 12, 24)

final_out = os.path.join(writeFolder,
                         "equator_thetao_u_ssh_mld_2014_2024.nc")

# ============================================================
# RANK 0: PREPARE TEMP DIRECTORY
# ============================================================
if rank == 0:
    if not os.path.isdir(tempFolder):
        os.makedirs(tempFolder, exist_ok=True)
        print(f"[rank 0] Created temp directory: {tempFolder}")
    else:
        print(f"[rank 0] Using existing temp directory: {tempFolder}")

# Ensure all ranks wait until directory is ready
comm.Barrier()

# ============================================================
# BUILD LIST OF DATES (ON ALL RANKS)
# ============================================================
all_dates = []
curDate = startDate
while curDate <= endDate:
    all_dates.append(curDate)
    curDate += timedelta(days=1)

n_dates = len(all_dates)

if rank == 0:
    print(f"[rank 0] Total dates to process: {n_dates}")
    print(f"[rank 0] MPI size: {size} ranks")

# ============================================================
# DISTRIBUTE DATES ACROSS RANKS
# ============================================================
# Simple round-robin assignment:
#   rank 0: dates[0], dates[size], dates[2*size], ...
#   rank 1: dates[1], dates[size+1], ...
#   ...
local_dates = all_dates[rank:n_dates:size]

print(f"[rank {rank}] Assigned {len(local_dates)} dates.")

# ============================================================
# HELPER FUNCTION: PROCESS ONE DAY
# ============================================================
def process_single_day(date_obj):
    """Open a daily GLORYS file, extract equatorial section, and save to NetCDF."""
    year = date_obj.year
    yyyy_mm_dd = f"{year:04d}-{date_obj.month:02d}-{date_obj.day:02d}"

    # Input file name
    fileName = os.path.join(
        rootFolder,
        f"{year:04d}",
        f"GLORYS12v1_dailyAvg_{yyyy_mm_dd}.nc",
    )

    if not os.path.exists(fileName):
        print(f"[rank {rank}] WARNING: File not found: {fileName}")
        return

    
    # Output file name for this day
    outName = os.path.join(
        tempFolder,
        f"GLORYS12v1_dailyAvg_atEq_{yyyy_mm_dd}.nc",
    )

    # Skip if it already exists (useful for restarts)
    if os.path.exists(outName):
        # Comment out if you prefer to overwrite
        print(f"[rank {rank}] Skipping existing {outName}")
        return

    try:
        # Open dataset for this day, subset depth, and select equator (latitude ~ 0)
        # Note: method='nearest' picks nearest latitude to 0.
        ds = xr.open_dataset(fileName).sel(
            latitude=0,
            method="nearest").sel(depth=slice(0, 300))

        # Shift longitude from [-180, 180] to [0, 360), then sort and slice 106–296E
        new_lon = (ds["longitude"].values + 360) % 360
        ds = ds.assign_coords(longitude=new_lon)
        ds = ds.sortby("longitude")
        ds = ds.sel(longitude=slice(106, 296))

        # Write to NetCDF; make 'time' unlimited if present
        ds.to_netcdf(outName, unlimited_dims=("time",))

        # Close dataset to free resources
        ds.close()
        print(f'[rank {rank}] {yyyy_mm_dd} processed')

    except Exception as e:
        print(f"[rank {rank}] ERROR processing {fileName}: {e}")
        # Do not sys.exit() from nonzero ranks; just report the error.
        # Rank 0 can check success later if needed.


# ============================================================
# MAIN PER-RANK LOOP
# ============================================================
for d in local_dates:
    process_single_day(d)

# Synchronize before combining
comm.Barrier()

# ============================================================
# RANK 0: COMBINE ALL DAILY FILES INTO ONE DATASET
# ============================================================
if rank == 0:
    print("[rank 0] Combining daily equatorial files into a single dataset...")
    sys.stdout.flush()

    pattern = os.path.join(tempFolder, "GLORYS12v1_dailyAvg_atEq_*.nc")

    # Use open_mfdataset to combine all processed daily files along time
    fullDS = xr.open_mfdataset(
        pattern,
        combine="by_coords",
        # If files are big, you can add chunks here, e.g.:
        chunks={"time": 365}
    )

    print("[rank 0] Writing combined dataset to:", final_out)
    fullDS.to_netcdf(final_out)

    fullDS.close()
    print("[rank 0] Done. Combined equatorial dataset written.")


# Final barrier (optional)
comm.Barrier()
if rank == 0:
    print("[rank 0] All ranks finished.")
