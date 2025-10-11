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

class PiSimulator:
    def __init__(self, config, loop_z_start=0.2, loop_z_increment=0.05, num_increments=0):
        self.cfg = config
        self.num_increments = num_increments
        self.loop_z_start = float(loop_z_start)
        self.loop_z_increment = float(loop_z_increment)

    def create_survey(self, time_channels, waveform, loop_z):
        # Transmitter location
        xtx, ytx, ztx = np.meshgrid([0], [0], [loop_z])
        source_locations = np.c_[mkvc(xtx), mkvc(ytx), mkvc(ztx)]
        ntx = np.size(xtx)
        # Receiver location
        xrx, yrx, zrx = np.meshgrid([0], [0], [loop_z])
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
                    current=self.cfg.tx_current,
                    radius=self.cfg.tx_radius,
                    n_turns=self.cfg.tx_n_turns
                )
            )
        return tdem.Survey(source_list)

    def create_conductivity_model(self, mesh, target_present):
        active_area_z = self.cfg.target_z + self.cfg.target_height/2
        ind_active = mesh.cell_centers[:, 2] < active_area_z

        model_map = maps.InjectActiveCells(mesh, ind_active, self.cfg.air_conductivity)

        r = mesh.cell_centers[ind_active, 0]
        z = mesh.cell_centers[ind_active, 2]

        model = self.cfg.air_conductivity * np.ones(ind_active.sum())

        ind_soil = (
            (z < self.cfg.target_z - self.cfg.target_height/2)
        )
        model[ind_soil] = self.cfg.soil_conductivity

        if target_present:
            ind_can = (
                (r <= self.cfg.target_radius) &
                (z < self.cfg.target_z + self.cfg.target_height/2) &
                (z > self.cfg.target_z - self.cfg.target_height/2)
            )
            model[ind_can] = self.cfg.aluminum_conductivity

        return {"model": model, "model_map": model_map}

    def run(self):
        print("Running simulation with the following configuration:")
        print(self.cfg.summary())

        # 1. Define the waveform to be used, in this case - always a step-off
        waveform = tdem.sources.StepOffWaveform(off_time=self.cfg.waveform_off_time)
        time_channels = np.linspace(0, 1024e-6, 1024)

        # 2. Create the survey
        surveys = []
        for i in range(self.num_increments+2):
            loop_z = self.loop_z_start + i * self.loop_z_increment
            surveys.append(self.create_survey(time_channels, waveform, loop_z))

        # 3. Create the mesh
        hr = [(0.01, 15), (0.01, 15, 1.3), (0.05, 10, 1.5)]
        hphi = 1
        hz = [(0.01, 10, -1.3), (0.01, 30), (0.01, 10, 1.3)]
        mesh = CylindricalMesh([hr, hphi, hz], x0="00C")

        # 4. Create all the different conductivity models
        models = []
        models.append(self.create_conductivity_model(mesh, target_present=False))
        models.append(self.create_conductivity_model(mesh, target_present=True))

        # 5. Create the simulation objects
        simulations = []
        
        # Simulation for baseline without the target
        simulations.append(
            tdem.simulation.Simulation3DMagneticFluxDensity(
                mesh, survey=surveys[0], sigmaMap=models[0]["model_map"], t0=self.cfg.simulation_t0
            )
        )
        simulations[0].time_steps = self.cfg.time_steps

        for i in range(0,self.num_increments+1):
            # Simulation for target present
            simulations.append(
                tdem.simulation.Simulation3DMagneticFluxDensity(
                    mesh, survey=surveys[i], sigmaMap=models[1]["model_map"], t0=self.cfg.simulation_t0
                )
            )
            simulations[i+1].time_steps = self.cfg.time_steps

        decays, times = [],[]
        for i in range(len(simulations)):
            times.append(time_channels)
            if i == 0:
                dpred = simulations[i].dpred(m=models[0]["model"])
            else:
                dpred = simulations[i].dpred(m=models[1]["model"])
            dpred = np.reshape(dpred, (1, len(time_channels)))
            decays.append(-dpred[0, :])
        return times, decays

    def plot_tdem_response(self, times, decays):
        # Decay Curve
        fig = plt.figure(figsize=(12, 5))

        # Assume all times are the same, use the first one
        time = times[0] * 1e6  # Convert to μs

        # Calculate colors and labels based on loop distance from target
        # Decay 0 is red (no target), others are shades of blue (darker = closer to target)
        colors = []
        labels = []
        
        # First decay: no target (red)
        colors.append('red')
        labels.append(f'No target (loop @ {float(self.loop_z_start):.2f}m)')
        
        # Remaining decays: with target, varying loop heights
        # Create gradient from dark blue (close) to light blue (far)
        num_with_target = len(decays) - 1
        if num_with_target > 0:
            # Use a color gradient: from dark blue (0.2, 0.2, 0.8) to light blue (0.6, 0.7, 1.0)
            for i in range(num_with_target):
                # Calculate loop height and distance from target
                loop_z = self.loop_z_start + i * self.loop_z_increment
                distance_from_target = abs(loop_z - float(self.cfg.target_z))
                
                # Color gradient: darker (closer) to lighter (farther)
                # Normalize from 0 (closest) to 1 (farthest)
                if num_with_target > 1:
                    norm_dist = i / (num_with_target - 1)
                else:
                    norm_dist = 0.5
                
                # Dark blue to light blue gradient
                r = 0.1 + norm_dist * 0.5   # 0.1 -> 0.6
                g = 0.2 + norm_dist * 0.5   # 0.2 -> 0.7
                b = 0.8 + norm_dist * 0.2   # 0.8 -> 1.0
                
                colors.append((r, g, b))
                labels.append(f'With target (loop @ {loop_z:.2f}m, dist={distance_from_target:.2f}m)')

        # Linear scale
        ax1 = fig.add_subplot(121)
        for i, decay in enumerate(decays):
            ax1.plot(time, decay, marker='x', markersize=3, color=colors[i], 
                    label=labels[i], linewidth=2, alpha=0.8)
        ax1.set_xlabel("Time [μs]")
        ax1.set_ylabel("-dBz/dt [T/s]")
        ax1.set_title("TDEM Decay Curve (Linear Scale)")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim((0, time.max()))

        # Log-log scale
        ax2 = fig.add_subplot(122)
        for i, decay in enumerate(decays):
            ax2.loglog(time, decay, marker='x', markersize=3, color=colors[i], 
                      label=labels[i], linewidth=2, alpha=0.8)
        ax2.set_xlabel("Time [μs]")
        ax2.set_ylabel("-dBz/dt [T/s]")
        ax2.set_title("TDEM Decay Curve (Log-Log Scale)")
        ax2.grid(True, alpha=0.3, which='both')
        ax2.set_xlim((time.min(), time.max()))

        # Create a shared legend below the plots
        handles, legend_labels = ax1.get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc='lower center', ncol=min(3, len(decays)), 
                  bbox_to_anchor=(0.5, -0.05), frameon=True, fontsize=9)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for legend
        plt.show()


simulator = PiSimulator(cfg,loop_z_start=cfg.tx_z, loop_z_increment=0.05, num_increments = 3)
times, decays = simulator.run()
simulator.plot_tdem_response(times, decays)

