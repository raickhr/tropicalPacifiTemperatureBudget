import math
import numpy as np
import trajan as ta
import xarray as xr
import parcels
import warnings
from parcels import StatusCode

from operator import attrgetter
from datetime import datetime, timedelta
from glob import glob

folder = "/srv/seolab/srai/tropicalPacifiTemperatureBudget/WPWP_GLORYS_data/"

startDate = datetime(2018,4,1,12, 0 ,0)
endDate = datetime(2019,4,2,12, 0 ,0)
ufiles = []
curDate = startDate
while curDate <= endDate:
    ufiles.append(f"{folder}/GLORYS12v1_dailyAvg_withVerticalVelocities_{curDate.year:04d}-{curDate.month:02d}-{curDate.day:02d}.nc")
    curDate += timedelta(days =1)
#ufiles = sorted(glob(f"{folder}/GLORYS12v1_dailyAvg_withVerticalVelocities_2018-??-??.nc"))

mesh_mask = f"{folder}/coordinates.nc"

filenames = {
    "U": {"lon": mesh_mask, "lat": mesh_mask, "depth": mesh_mask, "data": ufiles},
    "V": {"lon": mesh_mask, "lat": mesh_mask, "depth": mesh_mask, "data": ufiles},
    "W": {"lon": mesh_mask, "lat": mesh_mask, "depth": mesh_mask, "data": ufiles},
    # "T": {"lon": mesh_mask, "lat": mesh_mask, "depth": wfiles[0], "data": tfiles},  # Not used in this example
}

variables = {
    "U": "uo",
    "V": "vo",
    "W": "wo",
    # "T": "thetao",  # Not used in this example
}

# Note that all variables need the same dimensions in a C-Grid
c_grid_dimensions = {
    "lon": "glamf",
    "lat": "gphif",
    "depth": "depthw",
    "time": "time",
}
dimensions = {
    "U": c_grid_dimensions,
    "V": c_grid_dimensions,
    "W": c_grid_dimensions,
    # "T": c_grid_dimensions,  # Not used in this example
}

fieldset = parcels.FieldSet.from_nemo(filenames, variables, dimensions)


xpos = np.linspace(170, 210, 15)
ypos = np.linspace(-3, 3, 5)
X, Y = np.meshgrid(xpos, ypos)

nparts = len(X.flatten())
ndays = 365

xlocs = []
ylocs = []
timelocs = []


curDate = startDate 
for i in range(0, ndays, 5):
    target_datetime = curDate + timedelta(days=i)
    timeArray_array = np.full(nparts, target_datetime, dtype=object)
    xlocs.append(X.flatten())
    ylocs.append(Y.flatten())
    timelocs.append(timeArray_array)



pset = parcels.ParticleSet.from_list(
    fieldset=fieldset,  # the fields on which the particles are advected
    pclass=parcels.JITParticle,  # the type of particles (JITParticle or ScipyParticle)
    lon=xlocs,  # a vector of release longitudes
    lat=ylocs,  # a vector of release latitudes
    time=timelocs,  # a vector of release times
)

output_file = pset.ParticleFile(
    name="longbox_delayed.zarr",  # the file name
    outputdt=timedelta(hours=6),  # the time step of the outputs
)

# --- hoist grid values to constants (accessible in JIT kernels) ---
topW   = float(fieldset.W.grid.depth[0])           # top valid depth level (≈0.5 m)
# Lmin   = float(fieldset.U.grid.lon[0])             # domain min lon
# Lmax   = float(fieldset.U.grid.lon[-1])            # domain max lon
# Lrange = float(Lmax - Lmin)

fieldset.add_constant('topW', topW)
# fieldset.add_constant('Lmin', Lmin)
# fieldset.add_constant('Lmax', Lmax)
# fieldset.add_constant('Lrange', Lrange)

# --- kernels: use constants, not fieldset.*.grid.* ---
def ClampTopDepth(particle, fieldset, time):
    # Delete any particle that hit an error (>=50 covers all errors)
    if particle.state >= StatusCode.Error:          # Error == 50
        # Optional: special-case surface crossing if submerging:
        if particle.state == StatusCode.ErrorThroughSurface:
           particle.depth = fieldset.topW; 
        else:
            particle.delete()


kernels = pset.Kernel(parcels.AdvectionRK4_3D) + ClampTopDepth


#kernels = pset.Kernel(parcels.AdvectionRK4_3D)
pset.execute(kernels, 
             runtime=timedelta(days=365), 
             dt=timedelta(minutes=1),
            output_file=output_file)

