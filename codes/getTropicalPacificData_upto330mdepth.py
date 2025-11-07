import xarray as xr
import numpy as np
import glob
import os
import gc
from datetime import datetime, timedelta

startDate = datetime(2017, 12, 31)
endDate = datetime(2019, 7, 1)
curDate = startDate

while curDate <= endDate:
    writeFname = f'/srv/seolab/srai/tropicalPacifiTemperatureBudget/WPWP_GLORYS_data/GLORYS12v1_dailyAvg_{curDate:%Y-%m-%d}.nc'
    if os.path.exists(writeFname):
        print('removing file', writeFname)
        os.remove(writeFname)

        #curDate += timedelta(days=1)
        #continue

    folder = f'/srv/cdx/hseo/Data/GLORYS12v1/PCF/{curDate.year:04d}/'
    fname = f'GLORYS12v1_dailyAvg_{curDate:%Y-%m-%d}.nc'
    
    readFname = glob.glob(folder + fname)
    
    if readFname:
        readFname = readFname[0]
    else:
        print(f'ERROR: File not found for {curDate}')
        curDate += timedelta(days=1)
        continue

    try:
        print(f'Processing {curDate}')
        ds = xr.open_dataset(readFname)  # 🚫 No Dask!

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


