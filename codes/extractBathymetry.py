import xarray as xr
import numpy as np
import glob
import os
import sys
import gc
from datetime import datetime, timedelta



folder = f'/srv/cdx/hseo/Data/GLORYS12v1/PCF/'
fname = 'GLO-MFC_001_030_mask_bathy.nc'

readFname = glob.glob(folder + fname)
writeFname = f'/srv/seolab/srai/tropicalPacifiTemperatureBudget/WPWP_GLORYS_data/bath_CG.nc'

if readFname:
    readFname = readFname[0]
else:
    print(f'ERROR: File not found for {curDate}')
    sys.exit()
try:
    ds = xr.open_dataset(readFname)  # 🚫 No Dask!

    # Subset spatial domain
    ds = ds.sel(latitude=slice(-16, 16))
    # Fast longitude shift
    new_lon = (ds['longitude'].values + 360) % 360
    ds = ds.assign_coords(longitude=new_lon)
    ds = ds.sortby('longitude')
    ds = ds.sel(longitude=slice(109, 296))

    # Write immediately
    ds.to_netcdf(writeFname)

    ds.close()
    del ds
    gc.collect()
except Exception as e:
    print(f'Failed to process bathmetry from file {readFname}')


