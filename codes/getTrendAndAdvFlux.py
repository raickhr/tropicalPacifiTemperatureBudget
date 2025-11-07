# process_glorys_fluxes_mpi.py
import os
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
from mpi4py import MPI

# -----------------------------------
# MPI setup
# -----------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# -----------------------------------
# CONFIG
# -----------------------------------
folder = '/srv/seolab/srai/tropicalPacifiTemperatureBudget/WPWP_GLORYS_data/'
grid_file = os.path.join(folder, 'glorysGrid.nc')

start_date = datetime(2018, 1, 1)
end_date   = datetime(2019, 7, 1)

# Seconds in one day (assumes daily means)
DT = 86400.0

# -----------------------------------
# Utilities
# -----------------------------------
def daterange_inclusive(d0, d1):
    n = (d1 - d0).days + 1
    return [d0 + timedelta(days=i) for i in range(n)]

def safe_close(*datasets):
    for ds in datasets:
        try:
            ds.close()
        except Exception:
            pass

# -----------------------------------
# LOAD GRID (each rank reads locally; file is small & shared read is OK)
# gridDS expected variables: dx(lat,lon), dy(lat,lon), area(lat,lon)
# -----------------------------------
gridDS = xr.open_dataset(grid_file)

# -----------------------------------
# Prepare the date list & distribute by striding
# Note: we need BOTH current and next-day files; last day is OK only if next-day file exists
# -----------------------------------
all_days = daterange_inclusive(start_date, end_date)
my_days = all_days[rank::size]

processed = 0
skipped   = 0
errors    = 0

for cur_date in my_days:
    next_date = cur_date + timedelta(days=1)

    fname      = f'GLORYS12v1_dailyAvg_withVerticalVelocities_{cur_date:%Y-%m-%d}.nc'
    next_fname = f'GLORYS12v1_dailyAvg_withVerticalVelocities_{next_date:%Y-%m-%d}.nc'
    path       = os.path.join(folder, fname)
    next_path  = os.path.join(folder, next_fname)

    # Require both days
    if not (os.path.exists(path) and os.path.exists(next_path)):
        if rank == 0:
            print(f'[SKIP] {cur_date:%Y-%m-%d}: missing {fname} or {next_fname}')
        skipped += 1
        continue

    ds = ds_next = None
    try:
        # Open both days
        ds      = xr.open_dataset(path)
        ds_next = xr.open_dataset(next_path)

        # Basic dims
        depth = ds['depth'].to_numpy()  # (z)
        nz = depth.size
        ny = ds.sizes['latitude']
        nx = ds.sizes['longitude']

        # -----------------------------------
        # Time tendencies (xarray math -> keeps attrs/coords nicely)
        # -----------------------------------
        dthetao_dt = (ds_next['thetao'].to_numpy() - ds['thetao'].to_numpy()) / DT  # (time, z, lat, lon)
        dso_dt     = (ds_next['so'].to_numpy()     - ds['so'].to_numpy())     / DT

        trend_Temp = xr.DataArray(
            dthetao_dt,
            dims=['time', 'depth', 'latitude', 'longitude'],
            coords={
                'time': ds['time'],
                'depth': ds['depth'],
                'latitude': ds['latitude'],
                'longitude': ds['longitude'],
            },
            attrs={
                'units': 'degC/s',
                'long_name': 'Trend of potential temperature',
                'note': f'Computed as daily difference divided by {DT} seconds',
            },
        )

        trend_Sal = xr.DataArray(
            dso_dt,
            dims=['time', 'depth', 'latitude', 'longitude'],
            coords={
                'time': ds['time'],
                'depth': ds['depth'],
                'latitude': ds['latitude'],
                'longitude': ds['longitude'],
            },
            attrs={
                'units': 'psu/s',
                'long_name': 'Trend of salinity',
                'note': f'Computed as daily difference divided by {DT} seconds',
            },
        )

        # -----------------------------------
        # Depth interfaces (top/bottom) and thickness
        # -----------------------------------
        depth_bottom_1d = np.empty_like(depth, dtype=float)
        depth_bottom_1d[:-1] = 0.5 * (depth[:-1] + depth[1:])
        depth_bottom_1d[-1]  = depth[-1] + 0.5 * (depth[-1] - depth[-2])

        depth_top_np = np.zeros((1, nz, ny, nx), dtype=np.float64)
        depth_bot_np = np.zeros((1, nz, ny, nx), dtype=np.float64)

        for k in range(nz):
            depth_bot_np[0, k, :, :] = depth_bottom_1d[k]

        ssh2d = ds['zos'].isel(time=0).to_numpy()
        depth_top_np[0, 0, :, :] = ssh2d
        for k in range(1, nz):
            depth_top_np[0, k, :, :] = depth_bottom_1d[k-1]

        dz_Xarr = xr.DataArray(
            depth_top_np - depth_bot_np,
            dims=['time', 'depth', 'latitude', 'longitude'],
            coords={
                'time': ds['time'],
                'depth': ds['depth'],
                'latitude': ds['latitude'],
                'longitude': ds['longitude'],
            },
            attrs={
                'units': 'm',
                'long_name': 'thickness of each depth layer',
                'note': 'Computed from depth interfaces derived from SSH and mid-level depths',
            },
        )

        # Cell area at T-points
        area = gridDS['area']
        dVol = dz_Xarr * area
        dVol.attrs['units'] = 'm^3'
        dVol.attrs['long_name'] = 'Volume of each grid cell'
        dVol.attrs['note'] = 'Layer thickness × horizontal area at T-points'

        # -----------------------------------
        # Prepare state variables and face values
        # -----------------------------------
        uo = ds['uo'].fillna(0.0)
        vo = ds['vo'].fillna(0.0)
        thetao = ds['thetao'].fillna(0.0)
        so = ds['so'].fillna(0.0)

        # Vertical faces (top/bottom) for tracers
        thetao_bot = 0.5 * (thetao + thetao.roll(depth=-1, roll_coords=False))
        thetao_top = thetao_bot.roll(depth=1, roll_coords=False)
        thetao_top[:, 0, :, :] = thetao[:, 0, :, :]

        so_bot = 0.5 * (so + so.roll(depth=-1, roll_coords=False))
        so_top = so_bot.roll(depth=1, roll_coords=False)
        so_top[:, 0, :, :] = so[:, 0, :, :]

        # Vertical velocities at faces
        wtop    = ds['wtop'].fillna(0.0)
        wbottom = wtop.roll(depth=-1, roll_coords=False)

        # Horizontal faces
        uEast = 0.5 * (uo.roll(longitude=-1, roll_coords=False) + uo)
        uWest = 0.5 * (uo.roll(longitude= 1, roll_coords=False) + uo)
        vNorth = 0.5 * (vo.roll(latitude=-1, roll_coords=False) + vo)
        vSouth = 0.5 * (vo.roll(latitude= 1, roll_coords=False) + vo)

        # Face metric factors (averaged to faces)
        dxNorth = 0.5 * (gridDS['dx'] + gridDS['dx'].roll(latitude=-1, roll_coords=False))
        dxSouth = 0.5 * (gridDS['dx'] + gridDS['dx'].roll(latitude= 1, roll_coords=False))
        dyEast  = 0.5 * (gridDS['dy'] + gridDS['dy'].roll(longitude=-1, roll_coords=False))
        dyWest  = 0.5 * (gridDS['dy'] + gridDS['dy'].roll(longitude= 1, roll_coords=False))

        # -----------------------------------
        # Advective fluxes (per-cell tendency contributions)
        # -----------------------------------
        # Salinity
        so_flux_east  = -(dyEast * dz_Xarr * uEast  * so) / dVol
        so_flux_west  =  (dyWest * dz_Xarr * uWest  * so) / dVol
        so_flux_north = -(dxNorth* dz_Xarr * vNorth * so) / dVol
        so_flux_south =  (dxSouth* dz_Xarr * vSouth * so) / dVol
        so_flux_horiz = so_flux_east + so_flux_west + so_flux_north + so_flux_south

        so_flux_bottom = (wbottom * dz_Xarr * so_bot) / dVol
        so_flux_top    = -(wtop    * dz_Xarr * so_top) / dVol
        so_flux_vert   = so_flux_bottom + so_flux_top
        so_flux_total  = so_flux_horiz + so_flux_vert

        # Temperature
        thetao_flux_east  = -(dyEast * dz_Xarr * uEast  * thetao) / dVol
        thetao_flux_west  =  (dyWest * dz_Xarr * uWest  * thetao) / dVol
        thetao_flux_north = -(dxNorth* dz_Xarr * vNorth * thetao) / dVol
        thetao_flux_south =  (dxSouth* dz_Xarr * vSouth * thetao) / dVol
        thetao_flux_horiz = thetao_flux_east + thetao_flux_west + thetao_flux_north + thetao_flux_south

        thetao_flux_bottom = (wbottom * dz_Xarr * thetao_bot) / dVol
        thetao_flux_top    = -(wtop    * dz_Xarr * thetao_top) / dVol
        thetao_flux_vert   = thetao_flux_bottom + thetao_flux_top
        thetao_flux_total  = thetao_flux_horiz + thetao_flux_vert

        # -----------------------------------
        # Save
        # -----------------------------------
        out_file = os.path.join(folder, f'GLORYS12v1_dailyAvg_trendWithFluxes_{cur_date:%Y-%m-%d}.nc')
        out = xr.Dataset({
            'trend_thetao': trend_Temp,          # keep original key name
            'tend_so': trend_Sal,                # keep original key name (as provided)

            'thetao_flux_horiz':  thetao_flux_horiz,
            'thetao_flux_vert':   thetao_flux_vert,
            'thetao_flux_total':  thetao_flux_total,
            'thetao_flux_east':   thetao_flux_east,
            'thetao_flux_west':   thetao_flux_west,
            'thetao_flux_north':  thetao_flux_north,
            'thetao_flux_south':  thetao_flux_south,
            'thetao_flux_bottom': thetao_flux_bottom,
            'thetao_flux_top':    thetao_flux_top,

            'so_flux_horiz':  so_flux_horiz,
            'so_flux_vert':   so_flux_vert,
            'so_flux_total':  so_flux_total,
            'so_flux_east':   so_flux_east,
            'so_flux_west':   so_flux_west,
            'so_flux_north':  so_flux_north,
            'so_flux_south':  so_flux_south,
            'so_flux_bottom': so_flux_bottom,
            'so_flux_top':    so_flux_top,

            'dVol': dVol,
        })
        # Ensure NetCDF is written per-rank without clobbering others (unique per date)
        encoding = {var: {'zlib': True, 'complevel': 1} for var in out.data_vars}
        out.to_netcdf(out_file, encoding=encoding)
        print(f'[OK][rank {rank}] {cur_date:%Y-%m-%d} -> {out_file}')
        processed += 1

    except Exception as e:
        print(f'[ERROR][rank {rank}] {cur_date:%Y-%m-%d}: {e}')
        errors += 1
    finally:
        safe_close(ds, ds_next)

# -----------------------------------
# Simple summary
# -----------------------------------
totals = comm.gather((processed, skipped, errors), root=0)
comm.Barrier()
if rank == 0:
    P = sum(t[0] for t in totals)
    S = sum(t[1] for t in totals)
    E = sum(t[2] for t in totals)
    print(f'=== MPI SUMMARY ===')
    print(f'Ranks: {size}')
    print(f'Processed: {P}  Skipped: {S}  Errors: {E}')

