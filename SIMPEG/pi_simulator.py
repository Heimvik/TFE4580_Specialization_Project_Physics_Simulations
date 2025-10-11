"""
3D Forward Simulation for Transient Response on a Cylindrical Mesh
==================================================================

Here we use the module *simpeg.electromagnetics.time_domain* to simulate the
transient response for borehole survey using a cylindrical mesh and a
radially symmetric conductivity. For this tutorial, we focus on the following:

    - How to define the transmitters and receivers
    - How to define the transmitter waveform for a step-off
    - How to define the time-stepping
    - How to define the survey
    - How to solve TDEM problems on a cylindrical mesh
    - The units of the conductivity/resistivity model and resulting data


Please note that we have used a coarse mesh larger time-stepping to shorten the
time of the simulation. Proper discretization in space and time is required to
simulate the fields at each time channel with sufficient accuracy.

Usage:
    python pi_simulator.py [config.json]

"""

#########################################################################
# Import Modules
# --------------
#

import sys
import json
import os

from discretize import CylindricalMesh
from discretize.utils import mkvc

from simpeg import maps
import simpeg.electromagnetics.time_domain as tdem

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

write_file = False

# sphinx_gallery_thumbnail_number = 2

#################################################################
# Load Configuration
# ------------------
#

# Load config from JSON if provided, otherwise use defaults
config = {}
configpath = 'config.json'
if os.path.exists(configpath):
    print(f"Loading configuration from {configpath}...")
    with open(configpath, 'r') as f:
        config = json.load(f)
    print("Configuration loaded successfully.\n")
else:
    print("No config file provided, using default parameters.\n")


#################################################################
# Defining the Waveform
# ---------------------
#
# Under *simpeg.electromagnetic.time_domain.sources*
# there are a multitude of waveforms that can be defined (VTEM, Ramp-off etc...).
# Here we simulate the response due to a step off waveform where the off-time
# begins at t=0. Other waveforms are discuss in the OcTree simulation example.
#

waveform = tdem.sources.StepOffWaveform(off_time=0.0)


#####################################################################
# Create Airborne Survey
# ----------------------
#
# Here we define the survey used in our simulation. For time domain
# simulations, we must define the geometry of the source and its waveform. For
# the receivers, we define their geometry, the type of field they measure and the time
# channels at which they measure the field. For this example,
# the survey consists of a borehold survey with a coincident loop geometry.
#

# Observation times for response (time channels)
# NOTE: Time channels must be within the simulation time range!
# For time_steps = [(5e-06, 20), (0.0001, 20), (0.001, 21)]:
#   Total sim time = 0.1e-4 + 2e-3 + 21e-3 = 0.0231 seconds
# Use logarithmic spacing to capture early and late time response
# New range: 0 to 500 microseconds (5e-4 seconds)
time_channels = np.linspace(0, 500*1e-6, 1024)  # From 10^-6 to ~5e-4 seconds (1 to 500 microseconds)

# Defining transmitter locations
xtx, ytx, ztx = np.meshgrid(config["tx_x"], config["tx_y"], config["tx_z"])
source_locations = np.c_[mkvc(xtx), mkvc(ytx), mkvc(ztx)]
ntx = np.size(xtx)

# Define receiver locations
xrx, yrx, zrx = np.meshgrid(config["rx_x"], config["rx_y"], config["rx_z"])
receiver_locations = np.c_[mkvc(xrx), mkvc(yrx), mkvc(zrx)]

source_list = []  # Create empty list to store sources

# Each unique location defines a new transmitter
for ii in range(ntx):
    # Define receivers at each location.
    dbzdt_receiver = tdem.receivers.PointMagneticFluxTimeDerivative(
        receiver_locations[ii, :], time_channels, "z"
    )
    receivers_list = [
        dbzdt_receiver
    ]  # Make a list containing all receivers even if just one

    # Must define the transmitter properties and associated receivers
    source_list.append(
        tdem.sources.CircularLoop(
            receivers_list,
            location=source_locations[ii],
            waveform=waveform,
            current=20,
            radius=0.4,
        )
    )

survey = tdem.Survey(source_list)

###############################################################
# Create Cylindrical Mesh
# -----------------------
#
# Here we create the cylindrical mesh that will be used for this tutorial
# example. We chose to design a coarser mesh to decrease the run time.
# When designing a mesh to solve practical time domain problems:
#
#     - Your smallest cell size should be 10%-20% the size of your smallest diffusion distance
#     - The thickness of your padding needs to be 2-3 times biggest than your largest diffusion distance
#     - The diffusion distance is ~1260*np.sqrt(rho*t)
#
#


hr = [(0.01, 20), (0.01, 20, 1.1)]

hphi = 1

hz = [(0.01, 20, -1.1), (0.01, 40), (0.01, 20, 1.1)]



mesh = CylindricalMesh([hr, 1, hz], x0="00C")

###############################################################
# Create Conductivity/Resistivity Model and Mapping
# -------------------------------------------------
#
# Here, we create the model that will be used to predict frequency domain
# data and the mapping from the model to the mesh. The model
# consists of several layers. For this example, we will have only flat topography.
#

# Conductivity in S/m (or resistivity in Ohm m)
air_conductivity = 1e-8
background_conductivity = 0.1
target_conductivity= 3.5*1e7
soil_conductivity = 0.4

# Find cells that are active in the forward modeling (cells below surface)
ind_active = mesh.cell_centers[:, 2] < 0

# Define mapping from model to active cells
model_map = maps.InjectActiveCells(mesh, ind_active, air_conductivity)

# Define the model
model = background_conductivity * np.ones(ind_active.sum())
ind = (mesh.cell_centers[ind_active, 2] > -0.2) & (
    mesh.cell_centers[ind_active, 2] < 0
)
model[ind] = target_conductivity
ind = (mesh.cell_centers[ind_active, 2] > -0.4) & (
    mesh.cell_centers[ind_active, 2] < -0.2
)

# Plot Conductivity Model
mpl.rcParams.update({"font.size": 14})
fig = plt.figure(figsize=(5, 6))

plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
log_model = np.log10(model)

ax1 = fig.add_axes([0.20, 0.1, 0.54, 0.85])
mesh.plot_image(
    plotting_map * log_model,
    ax=ax1,
    grid=False,
    clim=(np.log10(soil_conductivity), np.log10(target_conductivity)),
)
ax1.set_title("Conductivity Model")

ax2 = fig.add_axes([0.76, 0.1, 0.05, 0.85])
norm = mpl.colors.Normalize(
    vmin=np.log10(soil_conductivity), vmax=np.log10(target_conductivity)
)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", format="$10^{%.1f}$"
)
cbar.set_label("Conductivity [S/m]", rotation=270, labelpad=15, size=12)

######################################################
# Define the Time-Stepping
# ------------------------
#
# Stuff about time-stepping and some rule of thumb for step-off waveform
#

time_steps = [(5e-06, 20), (0.0001, 20), (0.001, 21)]


######################################################
# Define the Simulation
# ---------------------
#
# Here we define the formulation for solving Maxwell's equations. Since we are
# measuring the time-derivative of the magnetic flux density and working with
# a conductivity model, the EB formulation is the most natural. We must also
# remember to define the mapping for the conductivity model. Use *rhoMap* instead
# of *sigmaMap* if you defined a resistivity model.
#

simulation = tdem.simulation.Simulation3DMagneticFluxDensity(
    mesh, survey=survey, sigmaMap=model_map
)

# Set the time-stepping for the simulation
simulation.time_steps = time_steps

###########################################################
# Predict Data and Plot
# ---------------------
#
#

# Data are organized by transmitter, then by
# receiver then by observation time. dBdt data are in T/s.
dpred = simulation.dpred(model)

# Plot the response
dpred = np.reshape(dpred, (ntx, len(time_channels)))

# TDEM Profile
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
ax1.plot(time_channels, -dpred[0, :], "b", lw=2)
ax1.plot(time_channels, -dpred[-1, :], "r", lw=2)
ax1.set_xlim((np.min(time_channels), np.max(time_channels)))
ax1.set_xlabel("time [s]")
ax1.set_ylabel("-dBz/dt [T/s]")
ax1.set_title("Decay Curve")
ax1.legend(["First Sounding", "Last Sounding"], loc="upper right")
plt.show()
