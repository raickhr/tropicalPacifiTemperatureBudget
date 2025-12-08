import numpy as np
import xarray as xr
import glob
from datetime import datetime, timedelta
import sys
import os

# ============================================================
# User settings
# ============================================================
window = 121  # Running-mean window (days)
startDate = datetime(2014, 1, 1)
endDate   = datetime(2024, 12, 15)

base_path = "/proj/cmip6/data/ocean_reanalysis/glorys12v1/"
zarr_out  = (
    "/proj/seolab/srai/tropicalPacifiTemperatureBudget/"
    f"runningMeanData/rolling_mean_{window}_days.zarr"
)

# ============================================================
# Build file list (one file per day)
# Two possible naming patterns:
#   *YYYYMMDD_R*.nc
#   *YYYY-MM-DD.nc
# Stop immediately if any date is missing.
# ============================================================
fileList = []
curDate = startDate

print("Building file list...")
while curDate <= endDate:

    fname1 = f"{curDate.year:04d}/*{curDate.year:04d}{curDate.month:02d}{curDate.day:02d}_R*.nc"
    fname2 = f"{curDate.year:04d}/*{curDate.year:04d}-{curDate.month:02d}-{curDate.day:02d}.nc"

    # Use os.path.join for robustness
    readFname1 = glob.glob(os.path.join(base_path, fname1))
    readFname2 = glob.glob(os.path.join(base_path, fname2))

    if readFname1:
        if len(readFname1) > 1:
            print('more than one files for date', curDate)
            print(readFname1)
            sys.exit(1)
        readFname = sorted(readFname1)[0]  
    elif readFname2:
        if len(readFname2) > 1:
            print('more than one files for date', curDate)
            print(readFname2)
            sys.exit(1)
        readFname = sorted(readFname2)[0]
    else:
        print("ERROR: no file found for date:", curDate)
        sys.exit(1)

    fileList.append(readFname)
    curDate += timedelta(days=1)


print(f"✔ Found {len(fileList)} daily files.")


# ============================================================
# Open dataset lazily using dask
# - time chunks > window size for good rolling performance
# - spatial chunking helps memory and parallelism
# NOTE: use 'latitude' and 'longitude' here to match .sel() below.
# ============================================================

print("Opening dataset with dask...")

allDS = xr.open_mfdataset(
    fileList,
    combine="by_coords",
    chunks={
        "time": 180,         # Larger than 121-day window → efficient halos
        "latitude": 1,      # Spatial tiling; tune based on your grid size
        "longitude": 1,
        "depth": 1,       # Uncomment if needed
    },
    parallel=True,
)

# Optional: drop variables that do not need running mean to save resources
# vars_with_time = [v for v in allDS.data_vars if "time" in allDS[v].dims]
# allDS = allDS[vars_with_time]

# ============================================================
# Spatial and depth subset (still lazy)
# NOTE: If 'latitude' is descending (90 → -90), you may need
#       slice(16, -16) instead. Check allDS.latitude first.
# ============================================================

allDS = allDS.sel(latitude=slice(-16, 16))
allDS = allDS.sel(depth=slice(-1, 300))

# Ensure all longitudes are in the 0-360 range
allDS.coords['longitude'] = (allDS.coords['longitude'] + 360) % 360

# Select the desired range using .where(drop=True)
# This operation is lazy, efficient, and avoids the explicit sorting bottleneck.
allDS_subset = allDS.where(
    (allDS.longitude >= 106) & (allDS.longitude <= 296), 
    drop=True
)

print("✔ Dataset opened and sliced lazily.")
print(allDS)


# ============================================================
# Apply running mean along the time dimension
# This is lazy — computation happens only on .to_zarr().
# min_periods=window → require full 121-day window
# center=True → mean centered on the middle day of each window
# ============================================================

print(f"Applying {window}-day centered running mean...")

ds_rm = allDS_subset.rolling(
    time=window,
    center=True,
    min_periods=window,
).mean()

print("✔ Rolling mean defined (lazy).")


# ============================================================
# Save output to Zarr
# This triggers actual computation.
# ============================================================

print(f"Saving to Zarr: {zarr_out}")
ds_rm.to_zarr(zarr_out, mode="w")

print("✔ Successfully saved running mean dataset.")

