import xarray as xr
import numpy as np
import glob
import os
import gc
from datetime import datetime, timedelta

startDate = datetime(2017, 6, 1)
endDate = datetime(2020, 3, 31)
curDate = startDate

while curDate <= endDate:
    writeFname = f'/proj/cdx/hseo/Data/GLORYS12v1/PCF/GLORYS12v1_dailyAvg_{curDate:%Y-%m-%d}.nc'
    if os.path.exists(writeFname):
        curDate += timedelta(days=1)
        continue

    folder = f'/proj/cmip6/data/ocean_reanalysis/glorys12v1/{curDate.year:04d}/'
    fname1 = f'*{curDate:%Y%m%d}*.nc'
    fname2 = f'*{curDate:%Y-%m-%d}*.nc'

    readFname1 = glob.glob(folder + fname1)
    readFname2 = glob.glob(folder + fname2)

    if readFname1:
        readFname = readFname1[0]
    elif readFname2:
        readFname = readFname2[0]
    else:
        print(f'ERROR: File not found for {curDate}')
        curDate += timedelta(days=1)
        continue

    try:
        print(f'Processing {curDate}')
        ds = xr.open_dataset(readFname)  # 🚫 No Dask!

        # Subset spatial domain
        ds = ds.sel(latitude=slice(-21, 21))

        # Fast longitude shift
        new_lon = (ds['longitude'].values + 360) % 360
        ds = ds.assign_coords(longitude=new_lon)
        ds = ds.sortby('longitude')
        ds = ds.sel(longitude=slice(109, 301))

        # Write immediately
        ds.to_netcdf(writeFname, unlimited_dims='time')

        ds.close()
        del ds
        gc.collect()
    except Exception as e:
        print(f'Failed to process {curDate}: {e}')

    curDate += timedelta(days=1)

