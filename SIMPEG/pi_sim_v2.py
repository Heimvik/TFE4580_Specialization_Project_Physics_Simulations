"""
3D Forward Simulation with User-Defined Waveforms
=================================================

Here we use the module *simpeg.electromagnetics.time_domain* to predict the
TDEM response for a trapezoidal waveform. We consider an airborne survey
which uses a horizontal coplanar geometry. For this tutorial, we focus
on the following:

    - How to define the transmitters and receivers
    - How to define more complicated transmitter waveforms
    - How to define the time-stepping
    - How to define the survey
    - How to solve TDEM problems on an OcTree mesh
    - How to include topography
    - The units of the conductivity model and resulting data


Please note that we have used a coarse mesh and larger time-stepping to shorten
the time of the simulation. Proper discretization in space and time is required
to simulate the fields at each time channel with sufficient accuracy.


"""

#########################################################################
# Import Modules
# --------------
#

from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz, active_from_xyz
from simpeg.utils import plot2Ddata
from simpeg import maps
import simpeg.electromagnetics.time_domain as tdem

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import json

# Import the configuration class
from tdem_config import TDEMConfig

save_file = False

# Load configuration using the TDEMConfig class
cfg = TDEMConfig('config.json')
print(cfg.summary())


###############################################################
# Defining Topography
# -------------------
#
# Here we define surface topography as an (N, 3) numpy array. Topography could
# also be loaded from a file. Here we define flat topography, however more
# complex topographies can be considered.
#

xx, yy = np.meshgrid(np.linspace(-1, 1, 101), np.linspace(-1, 1, 101))
zz = np.zeros(np.shape(xx))
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        x_val = xx[i, j]
        if cfg.grass_x_min <= x_val <= cfg.grass_x_max:
            # Normalized position within grass region (0 at edges, 1 at center)
            x_normalized = 1 - abs(2 * (x_val - cfg.grass_x_min) / (cfg.grass_x_max - cfg.grass_x_min) - 1)

            # Inverse U shape: parabolic profile (smooth bulge)
            zz[i, j] = cfg.grass_z_max * x_normalized**2
        else:
            zz[i, j] = 0.0  # Flat outside grass region

topo_xyz = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]


###############################################################
# Defining the Waveform
# ---------------------
#
# Under *simpeg.electromagnetic.time_domain.sources*
# there are a multitude of waveforms that can be defined (VTEM, Ramp-off etc...).
# Here, we consider a trapezoidal waveform, which consists of a
# linear ramp-on followed by a linear ramp-off. For each waveform, it
# is important you are cognizant of the off time!!!
#

waveform = tdem.sources.StepOffWaveform(off_time=0.0)


#####################################################################
# Create Airborne Survey
# ----------------------
#
# Here we define the survey used in our simulation. For time domain
# simulations, we must define the geometry of the source and its waveform. For
# the receivers, we define their geometry, the type of field they measure and
# the time channels at which they measure the field. For this example,
# the survey consists of a uniform grid of airborne measurements.
#

time_channels = cfg.get_time_channels()

# Defining transmitter locations
xtx, ytx, ztx = np.meshgrid([cfg.tx_x], [cfg.tx_y], [cfg.tx_z])
source_locations = np.c_[mkvc(xtx), mkvc(ytx), mkvc(ztx)]
ntx = np.size(xtx)

# Define receiver locations
xrx, yrx, zrx = np.meshgrid([cfg.rx_x], [cfg.rx_y], [cfg.rx_z])
receiver_locations = np.c_[mkvc(xrx), mkvc(yrx), mkvc(zrx)]

source_list = []  # Create empty list to store sources

# Each unique location defines a new transmitter
for ii in range(ntx):
    # Define receivers at each location.
    dbzdt_receiver = tdem.receivers.PointMagneticFluxTimeDerivative(
        receiver_locations[ii, :], time_channels, "z"
    )
    receivers_list = [
        dbzdt_receiver,
    ]  # Make a list containing all receivers even if just one

    # Must define the transmitter properties and associated receivers
    source_list.append(
        tdem.sources.CircularLoop(
            receivers_list,
            location=source_locations[ii],
            waveform=waveform,
            current=cfg.tx_current,
            radius=cfg.tx_radius,
            n_turns=cfg.tx_n_turns
        )
    )

survey = tdem.Survey(source_list)

###############################################################
# Create OcTree Mesh
# ------------------
#
# Here we define the OcTree mesh that is used for this example.
# We chose to design a coarser mesh to decrease the run time.
# When designing a mesh to solve practical time domain problems:
#
#     - Your smallest cell size should be 10%-20% the size of your smallest diffusion distance
#     - The thickness of your padding needs to be 2-3 times biggest than your largest diffusion distance
#     - The diffusion distance is ~1260*np.sqrt(rho*t)
#
#

dh = cfg.mesh_dh
dom_width = cfg.mesh_dom_width
nbc = 2 ** int(np.round(np.log(dom_width / dh) / np.log(2.0)))  # num. base cells

# Define the base mesh
h = [(dh, nbc)]
mesh = TreeMesh([h, h, h], x0="CCC")

# Mesh refinement based on topography
mesh = refine_tree_xyz(
    mesh, topo_xyz, octree_levels=[0, 0, 0, 1], method="surface", finalize=False
)

# Mesh refinement near transmitters and receivers
mesh = refine_tree_xyz(
    mesh, receiver_locations, octree_levels=[2, 4], method="radial", finalize=False
)

# High-resolution refinement around the target (aluminum can)
# Create a small box around the can for fine mesh resolution
target_refinement_box = cfg.get_target_refinement_box()
xp, yp, zp = np.meshgrid([-0.3, 0.3], [-0.3, 0.3], [-0.3, 0.3])
xyz_target = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
mesh = refine_tree_xyz(mesh, xyz_target, octree_levels=[0, 2, 4, 5], method="box", finalize=False)

mesh.finalize()

###############################################################
# Create Conductivity Model and Mapping for OcTree Mesh
# -----------------------------------------------------
#
# Here, we define the electrical properties of the Earth as a layered conductivity
# model with an aluminum can buried in a grass layer, surrounded by soil.
# Layers (from top to bottom):
#   - Air (above z=0): ~1e-8 S/m
#   - Grass layer (0 to -0.3 m depth, -0.3 to 0.3 m in x, all y): ~0.02 S/m
#   - Soil (below grass): ~0.4 S/m
#   - Aluminum can (at z=-0.1 m): ~3.5e7 S/m
#

# Active cells are cells below the surface.
ind_active = active_from_xyz(mesh, topo_xyz)
model_map = maps.InjectActiveCells(mesh, ind_active, cfg.air_conductivity)

# Get cell center coordinates
x = mesh.gridCC[ind_active, 0]
y = mesh.gridCC[ind_active, 1]
z = mesh.gridCC[ind_active, 2]

# Start with soil as the background (everything below surface)
model = cfg.soil_conductivity * np.ones(ind_active.sum())

# Define grass layer region from configuration (restricted to specific x range)
ind_grass = (
    (x >= cfg.grass_x_min) & (x <= cfg.grass_x_max) &
    (z >= cfg.grass_z_min) & (z < cfg.grass_z_max)
)
model[ind_grass] = cfg.grass_conductivity
print(f"Number of grass cells: {ind_grass.sum()} out of {ind_active.sum()} active cells")

# Define aluminum can from configuration
ind_can = (
    ((x - cfg.target_center[0])**2 + (y - cfg.target_center[1])**2 <= cfg.target_radius**2) &
    (z < cfg.target_center[2] + cfg.target_height/2) &
    (z > cfg.target_center[2] - cfg.target_height/2)
)
model[ind_can] = cfg.aluminum_conductivity


### PLOT MODEL ###
# Plot log-conductivity model
mpl.rcParams.update({"font.size": 12})
fig = plt.figure(figsize=(14, 6))

log_model = np.log10(model)

plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

# Plot cross-section at Y = 0 (through the can)
ax1 = fig.add_axes([0.08, 0.1, 0.35, 0.85])
mesh.plot_slice(
    plotting_map * log_model,
    normal="Y",
    ax=ax1,
    ind=int(mesh.h[1].size / 2),
    grid=True,
    clim=(np.log10(cfg.grass_conductivity), np.log10(cfg.aluminum_conductivity)),
)
ax1.set_title("Conductivity Model - Y=0 Cross-Section")
ax1.set_xlabel("X [m]")
ax1.set_ylabel("Z [m]")

# Plot cross-section at X = 0 (perpendicular view)
ax2 = fig.add_axes([0.50, 0.1, 0.35, 0.85])
mesh.plot_slice(
    plotting_map * log_model,
    normal="X",
    ax=ax2,
    ind=int(mesh.h[0].size / 2),
    grid=True,
    clim=(np.log10(cfg.grass_conductivity), np.log10(cfg.aluminum_conductivity)),
)
ax2.set_title("Conductivity Model - X=0 Cross-Section")
ax2.set_xlabel("Y [m]")
ax2.set_ylabel("Z [m]")

# Colorbar
ax3 = fig.add_axes([0.88, 0.1, 0.02, 0.85])
norm = mpl.colors.Normalize(vmin=np.log10(cfg.grass_conductivity), vmax=np.log10(cfg.aluminum_conductivity))
cbar = mpl.colorbar.ColorbarBase(
    ax3, norm=norm, orientation="vertical", format="$10^{%.1f}$"
)
cbar.set_label("Conductivity [S/m]", rotation=270, labelpad=15, size=12)


######################################################
# Define the Time-Stepping
# ------------------------
#
# Load time stepping configuration from cfg
#

time_steps = cfg.time_steps


#######################################################################
# Simulation: Time-Domain Response
# --------------------------------
#
# Here we define the formulation for solving Maxwell's equations. Since we are
# measuring the time-derivative of the magnetic flux density and working with
# a conductivity model, the EB formulation is the most natural. We must also
# remember to define the mapping for the conductivity model.
#

simulation = tdem.simulation.Simulation3DMagneticFluxDensity(
    mesh, survey=survey, sigmaMap=model_map, t0=cfg.simulation_t0
)

# Set the time-stepping for the simulation
simulation.time_steps = time_steps

########################################################################
# Predict Data and Plot
# ---------------------
#

# Predict data for a given model
dpred = simulation.dpred(model)

# Data were organized by location, then by time channel
dpred = np.reshape(dpred, (ntx, len(time_channels)))

fig = plt.figure(figsize=(5, 5))
ax1 = fig.add_axes([0.15, 0.15, 0.8, 0.75])
for ii in range(0, len(time_channels)):
    ax1.semilogx(
        -dpred[:, ii], receiver_locations[:, -1], "k", lw=2
    )  # -ve sign to plot -dBz/dt
ax1.set_xlabel("-dBz/dt [T/s]")
ax1.set_ylabel("Elevation [m]")
ax1.set_title("Airborne TDEM Profile")

# Response for all time channels
fig = plt.figure(figsize=(5, 5))
ax1 = fig.add_axes([0.15, 0.15, 0.8, 0.75])
ax1.plot(time_channels, -dpred[0, :], lw=2)
ax1.set_xlim((np.min(time_channels), np.max(time_channels)))
ax1.set_xlabel("time [s]")
ax1.set_ylabel("-dBz/dt [T/s]")
ax1.set_yscale("log")
ax1.set_title("Decay Curve")
ax1.legend(["First Sounding", "Last Sounding"], loc="upper right")
plt.show()


#######################################################
# Optional: Export Data
# ---------------------
#
# Write the true model, data and topography
#

if save_file:
    dir_path = os.path.dirname(tdem.__file__).split(os.path.sep)[:-3]
    dir_path.extend(["tutorials", "assets", "tdem"])
    dir_path = os.path.sep.join(dir_path) + os.path.sep

    fname = dir_path + "tdem_topo.txt"
    np.savetxt(fname, np.c_[topo_xyz], fmt="%.4e")

    # Write data with 2% noise added
    fname = dir_path + "tdem_data.obs"
    dpred = dpred + 0.02 * np.abs(dpred) * np.random.randn(len(dpred))
    t_vec = np.kron(np.ones(ntx), time_channels)
    receiver_locations = np.kron(receiver_locations, np.ones((len(time_channels), 1)))

    np.savetxt(fname, np.c_[receiver_locations, t_vec, dpred], fmt="%.4e")

    # Plot true model
    output_model = plotting_map * model
    output_model[np.isnan(output_model)] = 1e-8

    fname = dir_path + "true_model.txt"
    np.savetxt(fname, output_model, fmt="%.4e")
