import xarray as xr
import numpy as np
import glob
import os
import sys
import gc
from datetime import datetime, timedelta

startDate = datetime(2018, 1, 3)
endDate = datetime(2018, 1, 3)
curDate = startDate

while curDate <= endDate:
    writeFname = f'/proj/seolab/srai/tropicalPacifiTemperatureBudget/WPWP_GLORYS_data/GLORYS12v1_dailyAvg_{curDate:%Y-%m-%d}.nc'
    if os.path.exists(writeFname):
        curDate += timedelta(days=1)
        continue

    folder = f'/proj/cmip6/data/ocean_reanalysis/glorys12v1/{curDate.year:04d}/'

    fname1 = f'*{curDate.year:04d}{curDate.month:02d}{curDate.day:02d}_R*.nc'
    fname2 = f'*{curDate.year:04d}-{curDate.month:02d}-{curDate.day:02d}_R*.nc'

    readFname1 = glob.glob(folder+fname1)
    readFname2 = glob.glob(folder+fname2)
    print('fileName1 = ', readFname1, len(readFname1), len(readFname1) > 0)
    print('fileName2 = ', readFname2, len(readFname2), len(readFname2) > 0)

    if len(readFname1) > 0:
        #print(fname1, ' present')
        fname = readFname1[0]
    elif len(readFname2) > 0:
        #print(fname2, ' present')
        fname = readFname2[0]
    else:
        print('file not present at date', curDate)
        sys.exit()

    try:
        print(f'Processing {curDate}')
        ds = xr.open_dataset(fname)  # 🚫 No Dask!

        # Subset spatial domain
        ds = ds.sel(latitude=slice(-16, 16))
        ds = ds.sel(depth=slice(-1, 300))

        # Fast longitude shift
        new_lon = (ds['longitude'].values + 360) % 360
        ds = ds.assign_coords(longitude=new_lon)
        ds = ds.sortby('longitude')
        ds = ds.sel(longitude=slice(106, 296))

        # Write immediately
        ds.to_netcdf(writeFname, unlimited_dims='time')

        ds.close()
        del ds
        gc.collect()
    except Exception as e:
        print(f'Failed to process {curDate}: {e}')

    curDate += timedelta(days=1)


