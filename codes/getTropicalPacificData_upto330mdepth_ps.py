import xarray as xr
import numpy as np
import glob
import os
import sys
import gc
from datetime import datetime, timedelta

startDate = datetime(2014, 1, 1)
endDate = datetime(2016, 6, 1)
curDate = startDate


# date_Str = ['2018-01-03', '2018-01-10', '2018-01-17', '2018-01-24', '2018-01-31', '2018-02-07',
#             '2018-02-14', '2018-02-21', '2018-02-28', '2018-03-07', '2018-03-14', '2018-03-21',
#             '2018-03-28', '2018-04-04', '2018-04-11', '2018-04-18', '2018-04-25', '2018-05-02',
#             '2018-05-09', '2018-05-16', '2018-05-23', '2018-05-30', '2018-06-06', '2018-06-13',
#             '2018-06-20', '2018-06-27', '2018-07-04', '2018-07-11', '2018-07-18', '2018-07-25',
#             '2018-08-01', '2018-08-08', '2018-08-15', '2018-08-22', '2018-08-29', '2018-09-05',
#             '2018-09-12', '2018-09-19', '2018-09-26', '2018-10-03', '2018-10-10', '2018-10-17',
#             '2018-10-24', '2018-10-31', '2018-11-07', '2018-11-14', '2018-11-21', '2018-11-28',
#             '2018-12-05', '2018-12-12', '2018-12-19', '2018-12-26']
# date_List = [datetime.strptime(date, '%Y-%m-%d') for date in date_Str]

while curDate <= endDate:
#for curDate in date_List:
    writeFname = f'/proj/seolab/srai/tropicalPacifiTemperatureBudget/WPWP_GLORYS_data/GLORYS12v1_dailyAvg_{curDate:%Y-%m-%d}.nc'
    if os.path.exists(writeFname):
        print('removing file', writeFname)
        os.remove(writeFname)
        #curDate += timedelta(days=1)
        #continue

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



