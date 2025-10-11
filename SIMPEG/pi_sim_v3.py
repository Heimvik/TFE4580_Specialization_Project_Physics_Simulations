"""
3D Forward Simulation for Aluminum Can Detection in Grass
=========================================================

Adaptation of cylindrical mesh TDEM simulation for detecting a buried
aluminum can in a grass layer over soil. This uses the proven structure
from the original cylindrical mesh example but with realistic small-scale
geometry for metal detection.

Features:
    - Cylindrical mesh appropriate for small-scale ground surveys
    - Aluminum can target buried in grass layer
    - Configuration-based parameter management
    - Coincident loop transmitter/receiver setup


"""

#########################################################################
# Import Modules
# --------------
#

from discretize import CylindricalMesh
from discretize.utils import mkvc

from simpeg import maps
import simpeg.electromagnetics.time_domain as tdem

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Import the configuration class
from tdem_config import TDEMConfig

write_file = False

# Load configuration
cfg = TDEMConfig('config.json')
print(cfg.summary())


#################################################################
# Defining the Waveform
# ---------------------
#
# Step-off waveform where the off-time begins at t=0.
#

waveform = tdem.sources.StepOffWaveform(off_time=cfg.waveform_off_time)


#####################################################################
# Create Ground-Based Survey
# ---------------------------
#
# Ground-based survey with coincident loop geometry for metal detection.
# The survey consists of a single transmitter/receiver location.
#

# Observation times for response (time channels)
# Use logarithmic spacing to avoid t=0 and capture decay properly

time_channels = np.linspace(0, 1024e-6,1000)

# Transmitter location
xtx, ytx, ztx = np.meshgrid([0], [0], [cfg.tx_z])
source_locations = np.c_[mkvc(xtx), mkvc(ytx), mkvc(ztx)]
ntx = np.size(xtx)

# Receiver location
xrx, yrx, zrx = np.meshgrid([0], [0], [cfg.rx_z])
receiver_locations = np.c_[mkvc(xrx), mkvc(yrx), mkvc(zrx)]

source_list = []

for ii in range(ntx):
    dbzdt_receiver = tdem.receivers.PointMagneticFluxTimeDerivative(
        receiver_locations[ii, :], time_channels, "z"
    )
    receivers_list = [dbzdt_receiver]

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
# Create Cylindrical Mesh
# -----------------------
#
# Cylindrical mesh suitable for small-scale ground-based survey.
# The mesh is designed for a ~2m domain with fine resolution near the target.
#
# Diffusion distance: ~1260*sqrt(rho*t)
# For rho=10 Ohm-m, t=500μs: d ~ 0.09 m
# Cell size should be 10-20% of this: ~1-2 cm
#

# Radial direction (hr)
hr = [(0.01, 15), (0.01, 15, 1.3), (0.05, 10, 1.5)]

# Vertical direction (hz) - finer near surface where target is
hz = [(0.01, 10, -1.3), (0.01, 30), (0.01, 10, 1.3)]

mesh = CylindricalMesh([hr, 1, hz], x0="00C")

print(f"\nMesh Information:")
print(f"  Number of cells: {mesh.nC}")
print(f"  Radial extent: 0 to {mesh.nodes_x[-1]:.2f} m")
print(f"  Vertical extent: {mesh.nodes_z[0]:.2f} to {mesh.nodes_z[-1]:.2f} m")
print(f"  Smallest cell (radial): {mesh.h[0].min():.4f} m")
print(f"  Smallest cell (vertical): {mesh.h[2].min():.4f} m\n")

###############################################################
# Create Conductivity Model: Grass Layer with Buried Aluminum Can
# ----------------------------------------------------------------
#
# Layered model structure:
#   - Air (above z=0): ~1e-8 S/m
#   - Grass layer (0 to grass_z_min): cfg.grass_conductivity
#   - Soil (below grass): cfg.soil_conductivity
#   - Aluminum can (cylindrical target at target depth): cfg.target_conductivity
#

active_area_z = cfg.target_z+cfg.target_height/2
ind_active = mesh.cell_centers[:, 2] < active_area_z  # Only cells BELOW surface (negative z)

# Define mapping from model to active cells
model_map_no_target = maps.InjectActiveCells(mesh, ind_active, cfg.air_conductivity)
model_map_target = maps.InjectActiveCells(mesh, ind_active, cfg.air_conductivity)

# Get cell center coordinates
r = mesh.cell_centers[ind_active, 0]  # radial distance
z = mesh.cell_centers[ind_active, 2]  # vertical position

base_model = cfg.air_conductivity * np.ones(ind_active.sum())

ind_soil = (
    (z < cfg.target_z - cfg.target_height/2)
)
base_model[ind_soil] = cfg.soil_conductivity


model_target = base_model.copy()
model_no_target = base_model.copy()
ind_can = (
    (r <= cfg.target_radius) &
    (z < cfg.target_z + cfg.target_height/2) &
    (z > cfg.target_z - cfg.target_height/2)
)
model_target[ind_can] = cfg.aluminum_conductivity

print(f"Model Statistics:")
print(f"  Total active cells: {ind_active.sum()}")
print(f"  Aluminum can cells: {ind_can.sum()}")
print(f"  Soil cells: {ind_soil.sum()}")

# Plot Conductivity Model
mpl.rcParams.update({"font.size": 12})
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
log_model_target = np.log10(model_target)
log_model_no_target = np.log10(model_no_target)

# Plot for target model
ax1 = axes[0]
mesh.plot_image(
    plotting_map * log_model_target,
    ax=ax1,
    grid=True,
    clim=(np.log10(cfg.grass_conductivity), np.log10(cfg.aluminum_conductivity)),
)
ax1.set_title("Conductivity Model with Target (Cylindrical)")
ax1.set_xlabel("Radial Distance [m]")
ax1.set_ylabel("Elevation [m]")

# Add annotations
ax1.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Surface')
ax1.axhline(y=cfg.grass_z_min, color='g', linestyle='--', linewidth=1, alpha=0.5, label='Grass/Soil Interface')
ax1.axhline(y=cfg.target_z, color='r', linestyle=':', linewidth=1, alpha=0.7, label='Target Center')
ax1.legend(loc='lower right', fontsize=10)

# Plot for no target model
ax2 = axes[1]
mesh.plot_image(
    plotting_map * log_model_no_target,
    ax=ax2,
    grid=True,
    clim=(np.log10(cfg.grass_conductivity), np.log10(cfg.aluminum_conductivity)),
)
ax2.set_title("Conductivity Model without Target (Cylindrical)")
ax2.set_xlabel("Radial Distance [m]")
ax2.set_ylabel("Elevation [m]")

# Add annotations
ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Surface')
ax2.axhline(y=cfg.grass_z_min, color='g', linestyle='--', linewidth=1, alpha=0.5, label='Grass/Soil Interface')
ax2.legend(loc='lower right', fontsize=10)

# Shared colorbar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.87, 0.1, 0.03, 0.8])
norm = mpl.colors.Normalize(
    vmin=np.log10(cfg.grass_conductivity), 
    vmax=np.log10(cfg.aluminum_conductivity)
)
cbar = mpl.colorbar.ColorbarBase(
    cbar_ax, norm=norm, orientation="vertical", format="$10^{%.1f}$"
)
cbar.set_label("Conductivity [S/m]", rotation=270, labelpad=20, size=12)

plt.tight_layout()

######################################################
# Define the Time-Stepping
# ------------------------
#
# Time stepping must cover the observation window (up to 500 μs).
# Use the proven time steps from configuration.
#

time_steps = cfg.time_steps

# Calculate total simulation time
total_sim_time = sum([dt * n for dt, n in time_steps])
print(f"Time Stepping:")
print(f"  Steps: {time_steps}")
print(f"  Total simulation time: {total_sim_time*1e6:.1f} μs")
print(f"  Max observation time: {time_channels.max()*1e6:.1f} μs")
print(f"  Time margin: OK" if time_channels.max() < total_sim_time else "  WARNING: Observation times exceed simulation time!\n")

######################################################
# Define the Simulation
# ---------------------
#
# Simulation using the EB formulation for magnetic flux density.
#

simulation_no_target = tdem.simulation.Simulation3DMagneticFluxDensity(
    mesh, survey=survey, sigmaMap=model_map_no_target, t0=cfg.simulation_t0
)
simulation_target = tdem.simulation.Simulation3DMagneticFluxDensity(
    mesh, survey=survey, sigmaMap=model_map_target, t0=cfg.simulation_t0
)

# Set the time-stepping for the simulation
simulation_no_target.time_steps = time_steps
simulation_target.time_steps = time_steps
###########################################################
# Predict Data and Plot
# ---------------------
#
#

print("Running forward simulation...")
# Data are organized by transmitter, then by receiver, then by observation time
# dBdt data are in T/s
dpred_no_target = simulation_no_target.dpred( m=model_no_target)
dpred_target = simulation_target.dpred( m=model_target)
print("Simulation complete!\n")

# Plot the response
dpred_no_target= np.reshape(dpred_no_target, (ntx, len(time_channels)))
dpred_target = np.reshape(dpred_target, (ntx, len(time_channels)))

# Decay Curve
fig = plt.figure(figsize=(10, 5))

# Linear scale
ax1 = fig.add_subplot(121)
ax1.plot(time_channels * 1e6, -dpred_no_target[0, :], 'b', lw=2, marker='o', markersize=4)
ax1.plot(time_channels * 1e6, -dpred_target[0, :], 'r', lw=2, marker='o', markersize=4)
ax1.set_xlabel("Time [μs]")
ax1.set_ylabel("-dBz/dt [T/s]")
ax1.set_title("TDEM Decay Curve (Linear Scale)")
ax1.grid(True, alpha=0.3)
ax1.set_xlim((0, time_channels.max() * 1e6))

# Log-log scale
ax2 = fig.add_subplot(122)
ax2.loglog(time_channels * 1e6, -dpred_no_target[0, :], 'b', lw=2, marker='o', markersize=4)
ax2.loglog(time_channels * 1e6, -dpred_target[0, :], 'r', lw=2, marker='o', markersize=4)
ax2.set_xlabel("Time [μs]")
ax2.set_ylabel("-dBz/dt [T/s]")
ax2.set_title("TDEM Decay Curve (Log-Log Scale)")
ax2.grid(True, alpha=0.3, which='both')
ax2.set_xlim((time_channels.min() * 1e6, time_channels.max() * 1e6))



plt.tight_layout()

# Print some statistics
print("Response Statistics:")
print(f"  Max response: {-dpred_no_target[0, :].max():.3e} T/s at t={time_channels[np.argmax(-dpred_no_target[0, :])]*1e6:.1f} μs")
print(f"  Min response: {-dpred_no_target[0, :].min():.3e} T/s at t={time_channels[np.argmin(-dpred_no_target[0, :])]*1e6:.1f} μs")
print(f"  Response at 100 μs: {-dpred_no_target[0, np.argmin(np.abs(time_channels - 1e-4))]:.3e} T/s")

plt.show()

###########################################################
# Optional: Export Data
# ---------------------
#

if write_file:
    # Export predicted data
    output_data = np.c_[
        np.repeat(receiver_locations[0, :].reshape(1, -1), len(time_channels), axis=0),
        time_channels.reshape(-1, 1),
        dpred.T
    ]
    np.savetxt('tdem_aluminum_can_response.txt', output_data, 
               header='x[m] y[m] z[m] time[s] dBz/dt[T/s]', fmt='%.6e')
    print("\nData exported to: tdem_aluminum_can_response.txt")
