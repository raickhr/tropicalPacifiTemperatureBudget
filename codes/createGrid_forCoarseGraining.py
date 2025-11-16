from netCDF4 import Dataset
import numpy as np
import xarray as xr

fldLoc = '/srv/seolab/srai/tropicalPacifiTemperatureBudget/WPWP_GLORYS_data/'
earthRad = 6.371e6  # earth radius m
Omega = 7.292e-5 # earth agular velocity rad/sec

fileName = 'GLORYS12v1_dailyAvg_2017-12-31.nc'
bathymetry_fileName = 'bath_CG.nc'

gridFileName = fldLoc + 'glorysGrid_forCG.nc'
latVar = 'latitude'
lonVar = 'longitude'

ds = xr.open_dataset(fldLoc + fileName)

latInDeg = np.array(ds[latVar].to_numpy())
lonInDeg = np.array(ds[lonVar].to_numpy())
zos = ds.zos.isel(time=0).to_numpy()

waterArea = np.array(~np.isnan(zos), dtype=float)
ds.close()

ds = xr.open_dataset(fldLoc + bathymetry_fileName)
h = ds.deptho.to_numpy()
ds.close()

xlen = len(lonInDeg)
ylen = len(latInDeg)

dx = np.zeros((ylen, xlen), dtype=float)
dy = np.zeros((ylen, xlen), dtype=float)

lon = np.deg2rad(lonInDeg)
lat = np.deg2rad(latInDeg)

lon_2d, lat_2d = np.meshgrid(lonInDeg, latInDeg)
coriolis = 2 * Omega * np.sin(np.deg2rad(lat))

dlon = abs(lon[1] - lon[0])
dlat = abs(lat[1] - lat[0])

for i in range(ylen):
    R = earthRad * np.cos(lat[i])
    dx[i,:] = R*dlon
    dy[i,:] = earthRad * dlat

coords = {'latitude': latInDeg, 'longitude': lonInDeg}

xds = xr.Dataset(
    {
        'DXU': (['latitude', 'longitude'], dx, {'units': 'm', 'long_name': 'east west cell width'}),
        'DYU': (['latitude', 'longitude'], dy, {'units': 'm', 'long_name': 'north south cell width'}),
        'UAREA': (['latitude', 'longitude'], dx*dy, {'units': 'm^2', 'long_name': 'cell area'}),
        'ULAT': (['latitude', 'longitude'], lat_2d, {'units': 'deg N', 'long_name': 'latitude'}),
        'ULONG': (['latitude', 'longitude'], lon_2d, {'units': 'deg E', 'long_name': 'longitude'}),
        'KMU': (['latitude', 'longitude'], waterArea, {'units': 'sec^-1', 'long_name': 'water area 1 for water and 0 for land'}),
        'FCORU': (['latitude', 'longitude'], lon_2d, {'units': 'sec^-1', 'long_name': 'coriolis parameter'}),
        'HU': (['latitude', 'longitude'], h, {'units': 'm', 'long_name': 'ocean depth'}),
    },
    coords=coords
)

xds.to_netcdf(gridFileName)

xds.close()


