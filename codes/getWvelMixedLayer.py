import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date, date2num
from datetime import datetime, timedelta
import os
#from functions import *


def getUpwellIngVel(u, v , wtop, dx, dy, mld):
    taxis = 0
    yaxis = 1
    xaxis = 2
    
    x_p = 0.5 *(np.roll(u, -1, axis=xaxis) + u) 
    x_m = 0.5 *(np.roll(u, 1, axis=xaxis) + u)
    
    y_p = 0.5 *(np.roll(v, -1, axis=yaxis) + v)
    y_m = 0.5 *(np.roll(v, 1, axis=yaxis) + v)
    grad_x = (x_p -x_m)/(dx)
    grad_y = (y_p -y_m)/(dy)
    horiz_div = grad_x + grad_y
    w = wtop + mld * horiz_div
    return w


def writeAdditionalVariablesSingleDay(readFolder, writeFolder, curDateTime):
    fname = f'TropAtl_{curDateTime.year:04d}-{curDateTime.month:02d}-{curDateTime.day:02d}_withWtop.nc'
    wfname = f'TropAtl_{curDateTime.year:04d}-{curDateTime.month:02d}-{curDateTime.day:02d}_withWbase.nc'
    
    if not os.path.isfile(readFolder+fname):
        print(fname, ' not present')
        return 
    xds = xr.open_dataset(readFolder + fname)

    uo = xds['uo']
    vo = xds['vo']

    earthRad = 6378137 #6.371e6
    lat = xds['latitude'].to_numpy()
    lon = xds['longitude'].to_numpy()
    depth = xds['depth'].to_numpy()
    
    xlen = len(lon)
    ylen = len(lat)
    zlen = len(depth)
    
    
    dx = np.zeros((ylen, xlen), dtype=float)
    dy = np.zeros((ylen, xlen), dtype=float)
    
    depth3d = np.zeros((zlen, ylen, xlen), dtype=float)
    for i in range(zlen):
        depth3d[i,:,:] = depth[i]
    
    depthXar = xr.DataArray(depth3d, dims=['depth', 'latitude', 'longitude'],   
                            attrs= {'units':  'm', 'long_name': '3d array of depth'}, 
                            coords = {'depth': xds['depth'],
                                      'latitude':xds['latitude'],
                                      'longitude': xds['longitude']})
    
    if np.max(abs(lat)) > 2:
        latInDeg = lat.copy()
        lonInDeg = lon.copy()%360
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)
    else:
        latInDeg = np.rad2deg(lat)
        lonInDeg = np.rad2deg(lon)%360
        
        
    dlon = abs(lon[1] - lon[0])
    dlat = abs(lat[1] - lat[0])

    for i in range(ylen):
        R = earthRad * np.cos(lat[i])
        dx[i,:] = R*dlon
        dy[i,:] = earthRad * dlat

    xds['depth3d'] = depthXar
    
    uo = xds['uo']
    vo = xds['vo']
    wtop = xds['wtop']
    mld = xds['mlotst']
    
    mldMask = xds['depth'] > mld
    
    
    mask = np.isnan(uo.to_numpy())
    mask = np.logical_or(mask, np.isnan(vo.to_numpy() ))
    mask = np.logical_or(mask, abs(uo.to_numpy())> 100)
    mask = np.logical_or(mask, abs(vo.to_numpy())> 100)
    
    uo = xr.where(mask, np.nan, uo)
    vo = xr.where(mask, np.nan, vo)
    
    uo = xr.where(mldMask, np.nan, uo)
    vo = xr.where(mldMask, np.nan, vo)
    
    depthXar = xr.where(mldMask, np.nan, depthXar)

    uoDepth = uo*depthXar
    voDepth = vo*depthXar
    mldArr = depthXar.sum(dim='depth')

    avUo = uoDepth.sum(dim='depth')/depthXar.sum(dim='depth')
    avVo = voDepth.sum(dim='depth')/depthXar.sum(dim='depth')
    w = getUpwellIngVel(avUo.to_numpy(), avVo.to_numpy() , wtop.to_numpy(), dx, dy,  mldArr.to_numpy())

    dimsList = ('time','latitude', 'longitude')

    newVarsXds = xr.Dataset({
        'wbase':xr.DataArray(w, dims=dimsList,   attrs= {'units':  'm/s', 
                                                         'long_name':'vertical velcity at base of mixed layer obtained from div. U averaged over the mld' }),
    })

    wxds = xr.merge([xds, newVarsXds])
    wxds.to_netcdf(writeFolder + wfname, unlimited_dims='time')

    wxds.close()
    xds.close()
    newVarsXds.close()
    
    
def writeAdditionalVariablesAllDaysSingleFile(folder, readFname):
    fname = folder + '/' + readFname
    wfname = folder + '/' + readFname +'_added.nc'
    
    if not os.path.isfile(fname):
        print(fname, ' not present')
        return 
    
    xds = xr.open_dataset(fname)

    uo = xds['uo']
    vo = xds['vo']

    #grid
    earthRad = 6378137 #6.371e6
    lat = xds['latitude'].to_numpy()
    lon = xds['longitude'].to_numpy()
    depth = xds['depth'].to_numpy()
    
    xlen = len(lon)
    ylen = len(lat)
    zlen = len(depth)
    
    
    dx = np.zeros((ylen, xlen), dtype=float)
    dy = np.zeros((ylen, xlen), dtype=float)
    
    depth3d = np.zeros((zlen, ylen, xlen), dtype=float)
    for i in range(zlen):
        depth3d[i,:,:] = depth[i]
    
    depthXar = xr.DataArray(depth3d, dims=['depth', 'latitude', 'longitude'],   
                            attrs= {'units':  'm', 'long_name': '3d array of depth'}, 
                            coords = {'depth': xds['depth'],
                                      'latitude':xds['latitude'],
                                      'longitude': xds['longitude']})
    
    if np.max(abs(lat)) > 2:
        latInDeg = lat.copy()
        lonInDeg = lon.copy()%360
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)
    else:
        latInDeg = np.rad2deg(lat)
        lonInDeg = np.rad2deg(lon)%360
        
        
    dlon = abs(lon[1] - lon[0])
    dlat = abs(lat[1] - lat[0])

    for i in range(ylen):
        R = earthRad * np.cos(lat[i])
        dx[i,:] = R*dlon
        dy[i,:] = earthRad * dlat

    xds['depth3d'] = depthXar
    
    uo = xds['uo']
    vo = xds['vo']
    wtop = xds['wtop']
    mld = xds['mlotst']
    
    mldMask = xds['depth'] > mld
    
    
    mask = np.isnan(uo.to_numpy())
    mask = np.logical_or(mask, np.isnan(vo.to_numpy() ))
    mask = np.logical_or(mask, abs(uo.to_numpy())> 100)
    mask = np.logical_or(mask, abs(vo.to_numpy())> 100)
    
    uo = xr.where(mask, np.nan, uo)
    vo = xr.where(mask, np.nan, vo)
    
    uo = xr.where(mldMask, np.nan, uo)
    vo = xr.where(mldMask, np.nan, vo)
    
    depthXar = xr.where(mldMask, np.nan, depthXar)

    uoDepth = uo*depthXar
    voDepth = vo*depthXar
    mldArr = depthXar.sum(dim='depth')

    avUo = uoDepth.sum(dim='depth')/depthXar.sum(dim='depth')
    avVo = voDepth.sum(dim='depth')/depthXar.sum(dim='depth')
    w = getUpwellIngVel(avUo.to_numpy(), avVo.to_numpy() , wtop.to_numpy(), dx, dy,  mldArr.to_numpy())

    dimsList = ('time','latitude', 'longitude')

    newVarsXds = xr.Dataset({
        'wo':xr.DataArray(w, dims=dimsList,   attrs= {'units':  'm/s', 
                                                      'long_name':'vertical velcity at base of mixed layer obtained from div. U averaged over the mld' }),
    })

    wxds = xr.merge([xds, newVarsXds])
    wxds.to_netcdf(writeFolder + wfname, unlimited_dims='time')

    wxds.close()
    xds.close()
    newVarsXds.close()




def main():
    dataFolder = '../data/'
    #writeAdditionalVariablesAllDaysSingleFile(dataFolder,  'alldata_tropicalAtl.nc')
    for year in range(2010, 2024):
        startDate = datetime(year,4,1)
        endDate = datetime(year,6,1)
    
        curDateTime = startDate
    
        while curDateTime <= endDate:
            writeAdditionalVariablesSingleDay(dataFolder, dataFolder, curDateTime)
            curDateTime += timedelta(days=1)


if __name__ == "__main__":
    main()



