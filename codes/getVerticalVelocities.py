import os
import xarray as xr
import numpy as np
from datetime import datetime, timedelta

# ----------------------------
# CONFIG
# ----------------------------
folder = '/srv/seolab/srai/tropicalPacifiTemperatureBudget/WPWP_GLORYS_data/'
grid_file = os.path.join(folder, 'glorysGrid.nc')

start_date = datetime(2018, 1, 1)
end_date   = datetime(2019, 7, 1)

# Seconds in one day (assumes daily means)
DT = 86400.0

# ----------------------------
# LOAD GRID (expects lat/lon metrics and area on T-grid)
# gridDS expected variables: dx(lat,lon), dy(lat,lon), area(lat,lon)
# ----------------------------
gridDS = xr.open_dataset(grid_file)

# ----------------------------
# MAIN LOOP OVER DAYS
# ----------------------------
cur_date = start_date
while cur_date <= end_date:
    prev_date = cur_date - timedelta(days=1)

    fname0 = f'GLORYS12v1_dailyAvg_{prev_date:%Y-%m-%d}.nc'
    fname1 = f'GLORYS12v1_dailyAvg_{cur_date:%Y-%m-%d}.nc'

    path0 = os.path.join(folder, fname0)
    path1 = os.path.join(folder, fname1)

    # Skip if either file is missing (first day or gaps)
    if not (os.path.exists(path0) and os.path.exists(path1)):
        print(f'[SKIP] Missing previous/current day file for {cur_date:%Y-%m-%d}')
        cur_date += timedelta(days=1)
        continue

    try:
        # Use chunking if files are large to keep memory down
        ds0 = xr.open_dataset(path0)
        ds1 = xr.open_dataset(path1)

        # ----------------------------
        # SURFACE VERTICAL VELOCITY: w_surf = d(SSH)/dt
        # SSH (zos) generally in meters; difference over one day gives m/s.
        # ----------------------------
        zos0 = ds0['zos']
        zos1 = ds1['zos']

        # Expect time dimension of size 1 in daily averages; index explicitly
        w_surf = (zos1.isel(time=0) - zos0.isel(time=0)) / DT  # (lat, lon)

        # ----------------------------
        # PREPARE DEPTH INTERFACES (z_{k-1/2}, z_{k+1/2})
        # depth is mid-level depth (positive downward). We define:
        #  - top interface at k=0 is SSH (zos at current day)
        #  - internal interfaces are midpoints between depth[k] and depth[k+1]
        #  - bottom interface for last layer is extrapolated
        # Result: depthTop[k], depthBottom[k] => thickness dz[k] = top - bottom
        # ----------------------------
        depth = ds1['depth'].values  # 1D (z), meters
        nz = depth.size
        ny = ds1.dims['latitude']
        nx = ds1.dims['longitude']

        # Compute 1D interfaces (in meters)
        depth_bottom_1d = np.empty_like(depth, dtype=float)
        # internal bottoms: midpoint between k and k+1
        depth_bottom_1d[:-1] = 0.5 * (depth[:-1] + depth[1:])
        # bottom of last cell: simple linear extrapolation
        depth_bottom_1d[-1] = depth[-1] + 0.5 * (depth[-1] - depth[-2])

        # Allocate 4D arrays for interfaces (time=1 for convenience)
        wtop_np       = np.zeros((1, nz, ny, nx), dtype=np.float64)
        depth_top_np  = np.zeros((1, nz, ny, nx), dtype=np.float64)
        depth_bot_np  = np.zeros((1, nz, ny, nx), dtype=np.float64)

        # Fill bottom interfaces (spatially uniform by level)
        for k in range(nz):
            depth_bot_np[0, k, :, :] = depth_bottom_1d[k]

        # Top interface: SSH at k = 0; else previous level's bottom
        ssh2d = ds1['zos'].isel(time=0).to_numpy()  # (lat, lon)
        depth_top_np[0, 0, :, :] = ssh2d
        for k in range(1, nz):
            depth_top_np[0, k, :, :] = depth_bottom_1d[k-1]

        # Thickness of each layer (positive meters)
        dz_np = depth_top_np - depth_bot_np  # (1, z, y, x)

        # ----------------------------
        # HORIZONTAL DIVERGENCE (volume flux form)
        # Using centered values and face-averaging to T-cell faces.
        # NOTE: .roll wraps around. This is appropriate zonally if periodic.
        # For latitude, consider replacing with padding if non-periodic boundaries.
        # ----------------------------
        # Fill NaNs to avoid propagating gaps into divergence; alternatively mask out later.
        uo = ds1['uo'].fillna(0.0)  # (time, z, lat, lon)
        vo = ds1['vo'].fillna(0.0)  # (time, z, lat, lon)

        # Average to faces (east/west for u, north/south for v)
        x_p = 0.5 * (uo.roll(longitude=-1) + uo)  # east face
        x_m = 0.5 * (uo.roll(longitude=1)  + uo)  # west face

        y_p = 0.5 * (vo.roll(latitude=-1) + vo)   # north face
        y_m = 0.5 * (vo.roll(latitude=1)  + vo)   # south face

        # Face metric factors (dy along u-flux; dx along v-flux), averaged to faces
        dxNorth = 0.5 * (gridDS['dx'] + gridDS['dx'].roll(latitude=-1))
        dxSouth = 0.5 * (gridDS['dx'] + gridDS['dx'].roll(latitude=1))
        dyEast  = 0.5 * (gridDS['dy'] + gridDS['dy'].roll(longitude=-1))
        dyWest  = 0.5 * (gridDS['dy'] + gridDS['dy'].roll(longitude=1))

        # Cell area at T-points
        area = gridDS['area']

        # Flux differences divided by area -> (1/s) horizontal divergence at T-points
        # Broadcasting works because area, dx, dy are (lat,lon) and uo/vo carry (time,z,lat,lon)
        grad_x = (x_p * dyEast - x_m * dyWest) / area  # (time,z,lat,lon)
        grad_y = (y_p * dxNorth - y_m * dxSouth) / area
        horiz_div = (grad_x + grad_y)  # (time,z,lat,lon)

        # ----------------------------
        # INTEGRATE CONTINUITY VERTICALLY:
        # w_top(k+1) = w_top(k) + div_h(k) * dz(k)
        # Sign convention: with depth positive downward and dz > 0,
        # this matches ∂w/∂z = -div_h if interfaces are chosen consistently.
        # Here we follow your original algebra.
        # ----------------------------
        # Set surface boundary condition (top of first cell): w_surf
        wtop_np[0, 0, :, :] = w_surf.to_numpy()

        # Multiply divergence by thickness (use NumPy for speed)
        # Ensure horiz_div has time=1; pick that time slice.
        div_np = horiz_div.isel(time=0).to_numpy()  # (z,y,x)

        for k in range(nz - 1):
            wtop_np[0, k + 1, :, :] = wtop_np[0, k, :, :] + div_np[k, :, :] * dz_np[0, k, :, :]

        # ----------------------------
        # PACKAGE INTO xarray + SAVE
        # ----------------------------
        wtop_da = xr.DataArray(
            wtop_np,
            dims=['time', 'depth', 'latitude', 'longitude'],
            coords={
                'time': ds1['time'],
                'depth': ds1['depth'],
                'latitude': ds1['latitude'],
                'longitude': ds1['longitude'],
            },
            attrs={
                'units': 'm s-1',
                'long_name': 'vertical velocity at top cell interface',
                'note': 'Integrated downward from surface using continuity: w_top(k+1)=w_top(k)+div_h(k)*dz(k)',
            },
        )

        out = ds1.copy()
        out['wtop'] = wtop_da


        out_file = os.path.join(folder, f'GLORYS12v1_dailyAvg_withVerticalVelocities_{cur_date:%Y-%m-%d}.nc')
        out.to_netcdf(out_file)
        print(f'[OK] Processed vertical velocities for {cur_date:%Y-%m-%d} -> {out_file}')

    except Exception as e:
        print(f'[ERROR] {cur_date:%Y-%m-%d}: {e}')

    finally:
        # Ensure files are closed promptly
        try:
            ds0.close()
            ds1.close()
        except Exception:
            pass

    cur_date += timedelta(days=1)

