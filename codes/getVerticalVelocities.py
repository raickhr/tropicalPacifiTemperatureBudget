# mpi_glorys_vertical_w.py
import os
import sys
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
from mpi4py import MPI

# ----------------------------
# CONFIG
# ----------------------------
FOLDER    = '/srv/seolab/srai/tropicalPacifiTemperatureBudget/WPWP_GLORYS_data/'
GRID_FILE = os.path.join(FOLDER, 'glorysGrid.nc')

START_DATE = datetime(2018, 1, 1)
END_DATE   = datetime(2019, 7, 1)

# Seconds in one day (assumes daily means)
DT = 86400.0

# Optional: choose a NetCDF engine explicitly (helps on some systems)
# ENGINE = "h5netcdf"
ENGINE = None  # set to "h5netcdf" if needed

# ----------------------------
# MPI SETUP
# ----------------------------
comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
size  = comm.Get_size()

def rprint(*args, **kwargs):
    """Rank-aware print."""
    prefix = f"[R{rank:02d}]"
    print(prefix, *args, **kwargs)
    sys.stdout.flush()

# ----------------------------
# BUILD DATE LIST (on rank 0) and broadcast
# ----------------------------
if rank == 0:
    dates = []
    cur = START_DATE
    while cur <= END_DATE:
        dates.append(cur)
        cur += timedelta(days=1)
else:
    dates = None

dates = comm.bcast(dates, root=0)

# ----------------------------
# LOAD GRID (per-rank, small and reused)
# Expects variables: dx(lat,lon), dy(lat,lon), area(lat,lon)
# ----------------------------
if ENGINE:
    gridDS = xr.open_dataset(GRID_FILE, engine=ENGINE)
else:
    gridDS = xr.open_dataset(GRID_FILE)

# Promote to in-memory NumPy for faster broadcasting
dx_da   = gridDS['dx']
dy_da   = gridDS['dy']
area_da = gridDS['area']

dx     = dx_da.to_numpy()
dy     = dy_da.to_numpy()
area   = area_da.to_numpy()

# Precompute face-metric averages used in divergence
# Note: roll is periodic. Adjust as needed if lat boundaries are non-periodic.
dxNorth = 0.5 * (dx + np.roll(dx, -1, axis=0))  # latitude axis = 0 if (lat,lon) order
dxSouth = 0.5 * (dx + np.roll(dx,  1, axis=0))
dyEast  = 0.5 * (dy + np.roll(dy, -1, axis=1))  # longitude axis = 1
dyWest  = 0.5 * (dy + np.roll(dy,  1, axis=1))

# ----------------------------
# PER-DAY WORK
# ----------------------------
def process_one_day(cur_date: datetime):
    """Compute vertical velocities for one day and write output NetCDF."""
    prev_date = cur_date - timedelta(days=1)

    fname0 = f'GLORYS12v1_dailyAvg_{prev_date:%Y-%m-%d}.nc'
    fname1 = f'GLORYS12v1_dailyAvg_{cur_date:%Y-%m-%d}.nc'

    path0 = os.path.join(FOLDER, fname0)
    path1 = os.path.join(FOLDER, fname1)

    # Skip if either file is missing (first day or gaps)
    if not (os.path.exists(path0) and os.path.exists(path1)):
        rprint(f"[SKIP] Missing prev/current file for {cur_date:%Y-%m-%d}")
        return False

    # Output path (unique per day; safe for parallel writes)
    out_file = os.path.join(
        FOLDER, f'GLORYS12v1_dailyAvg_withVerticalVelocities_{cur_date:%Y-%m-%d}.nc'
    )
    if os.path.exists(out_file):
        # Already done (idempotent runs)
        rprint(f"[OK] Exists: {out_file}")
        return True

    if ENGINE:
        ds0 = xr.open_dataset(path0, engine=ENGINE)
        ds1 = xr.open_dataset(path1, engine=ENGINE)
    else:
        ds0 = xr.open_dataset(path0)
        ds1 = xr.open_dataset(path1)

    try:
        # ----------------------------
        # SURFACE w = dSSH/dt
        # ----------------------------
        zos0 = ds0['zos'].isel(time=0)
        zos1 = ds1['zos'].isel(time=0)
        w_surf = (zos1 - zos0) / DT                        # (lat,lon)
        ssh2d  = zos1.to_numpy()                           # used for top interface

        # ----------------------------
        # DEPTH INTERFACES
        # ----------------------------
        depth = ds1['depth'].values  # 1D (z), meters, positive downward
        nz = depth.size
        ny = ds1.sizes['latitude']
        nx = ds1.sizes['longitude']

        depth_bottom_1d = np.empty_like(depth, dtype=float)
        depth_bottom_1d[:-1] = 0.5 * (depth[:-1] + depth[1:])
        depth_bottom_1d[-1]  = depth[-1] + 0.5 * (depth[-1] - depth[-2])

        wtop_np      = np.zeros((1, nz, ny, nx), dtype=np.float64)
        depth_top_np = np.zeros((1, nz, ny, nx), dtype=np.float64)
        depth_bot_np = np.zeros((1, nz, ny, nx), dtype=np.float64)

        # Bottom interfaces uniform by level
        for k in range(nz):
            depth_bot_np[0, k, :, :] = depth_bottom_1d[k]

        # Top interface: SSH at the surface, internal tops = previous level bottom
        depth_top_np[0, 0, :, :] =  0 #-ssh2d
        for k in range(1, nz):
            depth_top_np[0, k, :, :] = depth_bottom_1d[k-1]

        dz_np = depth_top_np - depth_bot_np  # (1,z,y,x), positive

        # ----------------------------
        # HORIZONTAL DIVERGENCE (volume flux form)
        # ----------------------------
        uo = ds1['uo'].fillna(0.0)  # (time,z,lat,lon)
        vo = ds1['vo'].fillna(0.0)

        # Face-averaged velocities (x: lon, y: lat)
        x_p = 0.5 * (uo.roll(longitude=-1) + uo)  # east face
        x_m = 0.5 * (uo.roll(longitude= 1) + uo)  # west face
        y_p = 0.5 * (vo.roll(latitude=-1) + vo)   # north face
        y_m = 0.5 * (vo.roll(latitude= 1) + vo)   # south face

        # Convert metrics to DataArrays to match coords and enable broadcasting
        # (Single time slice used below; avoid auto-alignment overhead)
        dxNorth_da = xr.DataArray(dxNorth, coords=uo.isel(time=0, depth=0).coords, dims=('latitude','longitude'))
        dxSouth_da = xr.DataArray(dxSouth, coords=uo.isel(time=0, depth=0).coords, dims=('latitude','longitude'))
        dyEast_da  = xr.DataArray(dyEast,  coords=uo.isel(time=0, depth=0).coords, dims=('latitude','longitude'))
        dyWest_da  = xr.DataArray(dyWest,  coords=uo.isel(time=0, depth=0).coords, dims=('latitude','longitude'))
        area_da    = xr.DataArray(area,    coords=uo.isel(time=0, depth=0).coords, dims=('latitude','longitude'))

        grad_x = (x_p * dyEast_da - x_m * dyWest_da) / area_da   # (time,z,lat,lon)
        grad_y = (y_p * dxNorth_da - y_m * dxSouth_da) / area_da
        horiz_div = grad_x + grad_y

        # ----------------------------
        # VERTICAL INTEGRATION (continuity)
        # w_top(k+1) = w_top(k) + div_h(k) * dz(k)
        # ----------------------------
        wtop_np[0, 0, :, :] =  0#w_surf.to_numpy()  # surface boundary condition

        div_np = horiz_div.isel(time=0).to_numpy()  # (z,y,x)
        for k in range(nz - 1):
            wtop_np[0, k + 1, :, :] = wtop_np[0, k, :, :] + div_np[k, :, :] * dz_np[0, k, :, :]

        # Centered w (cell centers)
        wtop_da = xr.DataArray(
            wtop_np,
            dims=['time', 'depth', 'latitude', 'longitude'],
            coords={'time': ds1['time'], 'depth': ds1['depth'],
                    'latitude': ds1['latitude'], 'longitude': ds1['longitude']},
            attrs={'units': 'm s-1',
                   'long_name': 'vertical velocity at top cell interface',
                   'note': 'Integrated downward: w_top(k+1)=w_top(k)+div_h(k)*dz(k)'}
        )

        wtop_da = xr.where(abs(wtop_da)>0.001, np.nan, wtop_da)
        wtop_da = wtop_da.fillna(0.0)

        wcenter = 0.5 * (wtop_da.roll(depth=-1) + wtop_da)
        wcenter[:, 0, :, :] = 0 # setting this to zero for parcels so that particles don't cross the boundary
        wcenter[:, -1, :, :] = wtop_da[:, -1, :, :]  # bottom center = top of bottom cell

        out = ds1.copy()
        out['wtop'] = wtop_da
        
        out['wo']   = wcenter.fillna(0.0)
        out['uo']   = out['uo'].fillna(0.0)
        out['vo']   = out['vo'].fillna(0.0)

        # Write result (each rank writes its own day; no parallel NetCDF needed)
        enc = {v: {'zlib': True, 'complevel': 2} for v in out.data_vars}
        if ENGINE:
            out.to_netcdf(out_file, encoding=enc, engine=ENGINE)
        else:
            out.to_netcdf(out_file, encoding=enc)

        rprint(f"[OK] {cur_date:%Y-%m-%d} -> {out_file}")
        return True

    except Exception as e:
        rprint(f"[ERROR] {cur_date:%Y-%m-%d}: {e}")
        return False

    finally:
        try:
            ds0.close()
            ds1.close()
        except Exception:
            pass

# ----------------------------
# STRIDED DATE ASSIGNMENT
# ----------------------------
# Rank r processes dates[r::size]
my_dates = dates[rank::size]

ok_count = 0
for d in my_dates:
    ok = process_one_day(d)
    ok_count += int(ok)

# ----------------------------
# SUMMARY
# ----------------------------
total_ok = comm.reduce(ok_count, op=MPI.SUM, root=0)
comm.Barrier()
if rank == 0:
    done = 0 if total_ok is None else total_ok
    print(f"[SUMMARY] Completed {done} day(s) across {size} rank(s).")

