import xarray as xr
import numpy as np
import glob
import os
import gc
from datetime import datetime, timedelta

startDate = datetime(2018,1,3)
endDate = datetime(2018,1,3)
curDate = startDate

while curDate <= endDate:
    #print(curDate)
    writeFname = f'/srv/cdx/hseo/Data/GLORYS12v1/PCF/GLORYS12v1_dailyAvg_{curDate.year:04d}-{curDate.month:02d}-{curDate.day:02d}.nc'
    folder = f'/srv/cmip6/data/ocean_reanalysis/glorys12v1/{curDate.year:04d}/'
    fname1 = f'*{curDate.year:04d}{curDate.month:02d}{curDate.day:02d}*.nc'
    fname2 = f'*{curDate.year:04d}-{curDate.month:02d}-{curDate.day:02d}*.nc'

    readFname1 = glob.glob(folder+fname1)
    readFname2 = glob.glob(folder+fname2)

    if len(readFname1) > 0:
        #print(fname1, ' present')
        readFname = readFname1[0]
    elif len(readFname2) > 0:
        #print(fname2, ' present')
        readFname = readFname2[0]
    else:
        print('ERROR', curDate)


    print(curDate)
    ds = xr.open_dataset(readFname)
    subds = ds.sel(latitude = slice(-21,21))
    
    east = subds.sel(longitude=slice(109, 180))
    west = subds.sel(longitude=slice(-180, -59))
    subds = xr.concat([east, west], dim='longitude')

    #print(subds)
    subds.to_netcdf(writeFname, unlimited_dims='time')
    curDate += timedelta(days=1)
    ds.close()
    subds.close()
    del ds, subds
    gc.collect()
