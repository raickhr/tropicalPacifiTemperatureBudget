import xarray as xr
import numpy as np
import glob
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


startDate = datetime(2010,4,1)
endDate = datetime(2023,6,1)
curDate = startDate

while curDate <= endDate:
    writeFname = f'../data/glorys12v1_{curDate.year:04d}-{curDate.month:02d}-{curDate.day:02d}.nc'
    folder = f'/proj/cmip6/data/ocean_reanalysis/glorys12v1/{curDate.year:04d}/'
    fname1 = f'*{curDate.year:04d}{curDate.month:02d}{curDate.day:02d}*.nc'
    fname2 = f'*{curDate.year:04d}-{curDate.month:02d}-{curDate.day:02d}*.nc'
    
    readFname1 = glob.glob(folder+fname1)
    readFname2 = glob.glob(folder+fname2)
    
    if os.path.isfile(readFname1[0]):
        #print(fname1, ' present')
        readFname = readFname1
    elif os.path.isfile(readFname2[0]):
        #print(fname2, ' present')
        readFname = readFname1
    else:
        print('ERROR', curDate)
        

    curDate += timedelta(days=1)
        
    ds = xr.open_dataset(readFname)
    subds = ds.sel(longitude = slice(-50, 10),
                   latitude = slice(-5,5))
    subds = subds.drop_vars(['bottomT', 'sithick', 'siconc', 'usi', 'vsi', 'thetao', 'so'])
    subds.to_netcdf(writeFname)
