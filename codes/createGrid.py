from netCDF4 import Dataset
import numpy as np
import xarray as xr

fldLoc = '/srv/seolab/srai/tropicalPacifiTemperatureBudget/WPWP_GLORYS_data/'
earthRad = 6.371e6
fileName = 'GLORYS12v1_dailyAvg_2017-12-31.nc'
gridFileName = fldLoc + 'glorysGrid.nc'
latVar = 'latitude'
lonVar = 'longitude'

ds = Dataset(fldLoc + fileName)

latInDeg = np.array(ds.variables[latVar])
lonInDeg = np.array(ds.variables[lonVar])

ds.close()

xlen = len(lonInDeg)
ylen = len(latInDeg)

dx = np.zeros((ylen, xlen), dtype=float)
dy = np.zeros((ylen, xlen), dtype=float)

lon = np.deg2rad(lonInDeg)
lat = np.deg2rad(latInDeg)

dlon = abs(lon[1] - lon[0])
dlat = abs(lat[1] - lat[0])

for i in range(ylen):
    R = earthRad * np.cos(lat[i])
    dx[i,:] = R*dlon
    dy[i,:] = earthRad * dlat

coords = {'latitude': latInDeg, 'longitude': lonInDeg}

xds = xr.Dataset(
    {
        'dx': (['latitude', 'longitude'], dx, {'units': 'm'}),
        'dy': (['latitude', 'longitude'], dy, {'units': 'm'}),
        'area': (['latitude', 'longitude'], dx*dy, {'units': 'm^2'}),
    },
    coords=coords
)

xds.to_netcdf(gridFileName)

xds.close()


