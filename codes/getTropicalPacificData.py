import xarray as xr
import numpy as np
import glob
import os
import gc
from datetime import datetime, timedelta

startDate = datetime(2018,1,3)
endDate = datetime(2018,1,3)
curDate = startDate

date_Str = [
'2016-06-08',
'2016-06-15',
'2016-06-22',
'2016-06-29',
'2016-07-06',
'2016-07-13',
'2016-07-20',
'2016-07-27',
'2016-08-03',
'2016-08-10',
'2016-08-17',
'2016-08-24',
'2016-08-31',
'2016-09-07',
'2016-09-14',
'2016-09-21',
'2016-09-28',
'2016-10-05',
'2016-10-12',
'2016-10-19',
'2016-10-26',
'2016-11-02',
'2016-11-09',
'2016-11-16',
'2016-11-23',
'2016-11-30',
'2016-12-07',
'2016-12-14',
'2016-12-21',
'2016-12-28'
]

date_List = [datetime.strptime(date, '%Y-%m-%d') for date in date_Str]

#while curDate <= endDate:
for curDate in date_List:

#while curDate <= endDate:
    #print(curDate)
    writeFname = f'/proj/cdx/hseo/Data/GLORYS12v1/PCF/{curDate.year:04d}/GLORYS12v1_dailyAvg_{curDate.year:04d}-{curDate.month:02d}-{curDate.day:02d}.nc'
    folder = f'/proj/cmip6/data/ocean_reanalysis/glorys12v1/{curDate.year:04d}/'
    fname1 = f'*{curDate.year:04d}{curDate.month:02d}{curDate.day:02d}_R*.nc'
    fname2 = f'*{curDate.year:04d}-{curDate.month:02d}-{curDate.day:02d}_R*.nc'

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
    #curDate += timedelta(days=1)
    ds.close()
    subds.close()
    del ds, subds
    gc.collect()

