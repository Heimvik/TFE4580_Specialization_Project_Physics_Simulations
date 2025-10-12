from discretize import CylindricalMesh
from discretize.utils import mkvc

from simpeg import maps
import simpeg.electromagnetics.time_domain as tdem

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Import the configuration class
from tdem_config import TDEMConfig


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
        
        # Prepare plotting info
        plotting_info = {
            'mesh': mesh,
            'model_with_target': models[1]["model"],
            'model_no_target': models[0]["model"],
            'ind_active': mesh.cell_centers[:, 2] < (self.cfg.target_z + self.cfg.target_height/2),
            'cfg': self.cfg,
            'loop_z_start': self.loop_z_start,
            'loop_z_increment': self.loop_z_increment,
            'num_increments': self.num_increments
        }
        
        return times, decays, plotting_info


class PiPlotter:
    def __init__(self, plotting_info, plot_linear=True, plot_loglog=True, 
                 plot_side_view=True, plot_top_view=True, plot_3d=True):
        """
        Initialize plotter with simulation info and plot selection flags.
        
        Parameters
        ----------
        plotting_info : dict
            Dictionary containing mesh, models, and configuration
        plot_linear : bool
            Plot TDEM response on linear-linear scale
        plot_loglog : bool
            Plot TDEM response on log-log scale
        plot_side_view : bool
            Plot environment from side view (X-Z plane)
        plot_top_view : bool
            Plot environment from top view (X-Y plane)
        plot_3d : bool
            Plot 3D model of entire environment
        """
        self.mesh = plotting_info['mesh']
        self.model = plotting_info['model_with_target']
        self.model_no_target = plotting_info['model_no_target']
        self.ind_active = plotting_info['ind_active']
        self.cfg = plotting_info['cfg']
        self.loop_z_start = plotting_info['loop_z_start']
        self.loop_z_increment = plotting_info['loop_z_increment']
        self.num_increments = plotting_info['num_increments']
        
        # Plot selection flags
        self.plot_linear = plot_linear
        self.plot_loglog = plot_loglog
        self.plot_side_view = plot_side_view
        self.plot_top_view = plot_top_view
        self.plot_3d = plot_3d
    
    def plot_environment(self):
        """Plot the environment based on selected views."""
        
        # Count how many plots to show
        num_plots = sum([self.plot_3d, self.plot_side_view, self.plot_top_view])
        if num_plots == 0:
            print("No environment plots selected.")
            return
        
        # Determine subplot layout
        if num_plots == 1:
            fig = plt.figure(figsize=(10, 8))
            subplot_layout = [(1, 1, 1)]
        elif num_plots == 2:
            fig = plt.figure(figsize=(14, 7))
            subplot_layout = [(1, 2, 1), (1, 2, 2)]
        else:  # 3 plots
            fig = plt.figure(figsize=(16, 12))
            subplot_layout = [(2, 2, 1), (2, 2, 2), (2, 2, 3)]
        
        plot_idx = 0
        
        # Get cell centers
        r = self.mesh.cell_centers[self.ind_active, 0]
        z = self.mesh.cell_centers[self.ind_active, 2]
        
        # 3D view
        if self.plot_3d:
            ax1 = fig.add_subplot(*subplot_layout[plot_idx], projection='3d')
            plot_idx += 1
            ax1.set_title('3D Environment View', fontsize=14, fontweight='bold')
        
            # Create theta array for full cylindrical representation
            n_theta = 36
            theta = np.linspace(0, 2*np.pi, n_theta)
            
            # Plot conductivity zones by sampling cells
            # Separate cells by conductivity type
            ind_air = self.model < 1e-6
            ind_soil = (self.model >= 0.3) & (self.model < 1e6)
            ind_aluminum = self.model > 1e6
        
            # Plot air cells (blue, transparent)
            if ind_air.sum() > 0:
                for i in np.where(ind_air)[0][::5]:  # Sample every 5th cell
                    ri, zi = r[i], z[i]
                    for tj in range(0, n_theta, 6):
                        xi = ri * np.cos(theta[tj])
                        yi = ri * np.sin(theta[tj])
                        ax1.scatter(xi, yi, zi, c='lightblue', s=1, alpha=0.1, marker='.')
            
            # Plot soil cells (brown)
            if ind_soil.sum() > 0:
                for i in np.where(ind_soil)[0][::3]:  # Sample every 3rd cell
                    ri, zi = r[i], z[i]
                    for tj in range(0, n_theta, 4):
                        xi = ri * np.cos(theta[tj])
                        yi = ri * np.sin(theta[tj])
                        ax1.scatter(xi, yi, zi, c='saddlebrown', s=3, alpha=0.3, marker='.')
            
            # Plot aluminum target cells (silver/gray)
            if ind_aluminum.sum() > 0:
                for i in np.where(ind_aluminum)[0]:
                    ri, zi = r[i], z[i]
                    for tj in theta:
                        xi = ri * np.cos(tj)
                        yi = ri * np.sin(tj)
                        ax1.scatter(xi, yi, zi, c='silver', s=20, alpha=0.8, marker='o', edgecolors='black', linewidths=0.5)
        
            # Plot transmitter/receiver loops
            loop_radius = float(self.cfg.tx_radius)
            for i in range(self.num_increments + 2):
                loop_z = self.loop_z_start + i * self.loop_z_increment
                loop_theta = np.linspace(0, 2*np.pi, 100)
                loop_x = loop_radius * np.cos(loop_theta)
                loop_y = loop_radius * np.sin(loop_theta)
                loop_z_arr = np.full_like(loop_x, loop_z)
                
                if i == 0:
                    # No target case - red
                    ax1.plot(loop_x, loop_y, loop_z_arr, 'r-', linewidth=3, label=f'Loop (no tgt) @ z={loop_z:.2f}m')
                else:
                    # With target - blue gradient
                    norm_i = (i-1) / max(1, self.num_increments)
                    color = (0.1 + norm_i*0.5, 0.2 + norm_i*0.5, 0.8 + norm_i*0.2)
                    ax1.plot(loop_x, loop_y, loop_z_arr, color=color, linewidth=2.5, 
                            label=f'Loop (w/ tgt) @ z={loop_z:.2f}m')
            
            # Plot target cylinder outline
            target_z = float(self.cfg.target_z)
            target_r = float(self.cfg.target_radius)
            target_h = float(self.cfg.target_height)
            
            # Top and bottom circles
            cyl_theta = np.linspace(0, 2*np.pi, 50)
            cyl_x = target_r * np.cos(cyl_theta)
            cyl_y = target_r * np.sin(cyl_theta)
            
            z_top = target_z + target_h/2
            z_bot = target_z - target_h/2
            ax1.plot(cyl_x, cyl_y, np.full_like(cyl_x, z_top), 'k-', linewidth=2, label='Target boundary')
            ax1.plot(cyl_x, cyl_y, np.full_like(cyl_x, z_bot), 'k-', linewidth=2)
            
            # Vertical lines
            for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
                ax1.plot([target_r*np.cos(angle)]*2, [target_r*np.sin(angle)]*2, 
                        [z_bot, z_top], 'k-', linewidth=2)
            
            # Ground plane
            ground_r = np.linspace(0, loop_radius*1.2, 20)
            ground_theta = np.linspace(0, 2*np.pi, 40)
            R, T = np.meshgrid(ground_r, ground_theta)
            X_ground = R * np.cos(T)
            Y_ground = R * np.sin(T)
            Z_ground = np.zeros_like(X_ground)
            ax1.plot_surface(X_ground, Y_ground, Z_ground, alpha=0.2, color='green', label='Ground surface')
            
            ax1.set_xlabel('X [m]', fontsize=10)
            ax1.set_ylabel('Y [m]', fontsize=10)
            ax1.set_zlabel('Z [m]', fontsize=10)
            ax1.legend(loc='upper right', fontsize=8)
            ax1.set_box_aspect([1, 1, 0.8])
        
        # Side view (X-Z plane)
        if self.plot_side_view:
            # Define target parameters outside the if block for use in all views
            target_z = float(self.cfg.target_z)
            target_r = float(self.cfg.target_radius)
            target_h = float(self.cfg.target_height)
            z_top = target_z + target_h/2
            z_bot = target_z - target_h/2
            loop_radius = float(self.cfg.tx_radius)
            
            ax2 = fig.add_subplot(*subplot_layout[plot_idx])
            plot_idx += 1
            ax2.set_title('Side View (X-Z Plane)', fontsize=12, fontweight='bold')
            
            # Plot conductivity zones in 2D
            plotting_map = maps.InjectActiveCells(self.mesh, self.ind_active, np.nan)
            log_model = np.log10(self.model)
            self.mesh.plot_image(plotting_map * log_model, ax=ax2, grid=True,
                           clim=(np.log10(self.cfg.air_conductivity), np.log10(self.cfg.aluminum_conductivity)))
        
            # Overlay features
            ax2.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Ground surface')
            ax2.axhline(y=target_z, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='Target center')
            
            # Draw target rectangle
            ax2.add_patch(plt.Rectangle((0, z_bot), target_r, target_h, 
                                        fill=False, edgecolor='black', linewidth=2, linestyle='-'))
            
            # Draw loops
            for i in range(self.num_increments + 2):
                loop_z = self.loop_z_start + i * self.loop_z_increment
                if i == 0:
                    ax2.plot([0, loop_radius], [loop_z, loop_z], 'r-', linewidth=2.5, marker='o', markersize=8)
                else:
                    norm_i = (i-1) / max(1, self.num_increments)
                    color = (0.1 + norm_i*0.5, 0.2 + norm_i*0.5, 0.8 + norm_i*0.2)
                    ax2.plot([0, loop_radius], [loop_z, loop_z], color=color, linewidth=2, marker='o', markersize=6)
            
            ax2.set_xlabel('Radial Distance [m]', fontsize=10)
            ax2.set_ylabel('Elevation [m]', fontsize=10)
            ax2.legend(loc='lower right', fontsize=8)
            ax2.grid(True, alpha=0.3)
        
        # Top view (X-Y plane)
        if self.plot_top_view:
            # Define target parameters if not already defined
            if not self.plot_side_view:
                target_z = float(self.cfg.target_z)
                target_r = float(self.cfg.target_radius)
                target_h = float(self.cfg.target_height)
                loop_radius = float(self.cfg.tx_radius)
            
            ax3 = fig.add_subplot(*subplot_layout[plot_idx])
            plot_idx += 1
            ax3.set_title('Top View (X-Y Plane)', fontsize=12, fontweight='bold')
            ax3.set_aspect('equal')
            
            # Draw loops
            for i in range(self.num_increments + 2):
                loop_z = self.loop_z_start + i * self.loop_z_increment
                circle = plt.Circle((0, 0), loop_radius, fill=False, linewidth=2)
                if i == 0:
                    circle.set_edgecolor('red')
                    circle.set_linewidth(3)
                else:
                    norm_i = (i-1) / max(1, self.num_increments)
                    color = (0.1 + norm_i*0.5, 0.2 + norm_i*0.5, 0.8 + norm_i*0.2)
                    circle.set_edgecolor(color)
                ax3.add_patch(circle)
            
            # Draw target
            target_circle = plt.Circle((0, 0), target_r, fill=True, facecolor='silver', 
                                      edgecolor='black', linewidth=2, alpha=0.7, label='Target')
            ax3.add_patch(target_circle)
            
            ax3.set_xlim([-loop_radius*1.3, loop_radius*1.3])
            ax3.set_ylim([-loop_radius*1.3, loop_radius*1.3])
            ax3.set_xlabel('X [m]', fontsize=10)
            ax3.set_ylabel('Y [m]', fontsize=10)
            ax3.legend(loc='upper right', fontsize=8)
            ax3.grid(True, alpha=0.3)
            ax3.axhline(0, color='k', linewidth=0.5, alpha=0.3)
            ax3.axvline(0, color='k', linewidth=0.5, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_tdem_response(self, times, decays):
        """Plot TDEM response based on selected plot types."""
        if not self.plot_linear and not self.plot_loglog:
            print("No TDEM response plots selected.")
            return
        
        # Determine figure layout
        if self.plot_linear and self.plot_loglog:
            fig = plt.figure(figsize=(12, 5))
            num_cols = 2
        else:
            fig = plt.figure(figsize=(7, 5))
            num_cols = 1

        time = times[0] * 1e6  # Convert to μs

        colors = []
        labels = []
        
        colors.append('red')
        labels.append(f'No target (loop @ {float(self.loop_z_start):.2f}m)')
        
        num_with_target = len(decays) - 1
        if num_with_target > 0:
            for i in range(num_with_target):
                loop_z = self.loop_z_start + i * self.loop_z_increment
                distance_from_target = abs(loop_z - float(self.cfg.target_z))
                if num_with_target > 1:
                    norm_dist = i / (num_with_target - 1)
                else:
                    norm_dist = 0.5
                
                r = 0.1 + norm_dist * 0.5   # 0.1 -> 0.6
                g = 0.2 + norm_dist * 0.5   # 0.2 -> 0.7
                b = 0.8 + norm_dist * 0.2   # 0.8 -> 1.0
                
                colors.append((r, g, b))
                labels.append(f'With target (loop @ {loop_z:.2f}m, dist={distance_from_target:.2f}m)')

        plot_idx = 1
        
        # Linear
        if self.plot_linear:
            ax1 = fig.add_subplot(1, num_cols, plot_idx)
            plot_idx += 1
            for i, decay in enumerate(decays):
                ax1.plot(time, decay, marker='x', markersize=3, color=colors[i], 
                        label=labels[i], linewidth=2, alpha=0.8)
            ax1.set_xlabel("Time [μs]")
            ax1.set_ylabel("-dBz/dt [T/s]")
            ax1.set_title("TDEM Decay Curve (Linear Scale)")
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim((0, time.max()))

        # Log-log
        if self.plot_loglog:
            ax2 = fig.add_subplot(1, num_cols, plot_idx)
            for i, decay in enumerate(decays):
                ax2.loglog(time, decay, marker='x', markersize=3, color=colors[i], 
                          label=labels[i], linewidth=2, alpha=0.8)
            ax2.set_xlabel("Time [μs]")
            ax2.set_ylabel("-dBz/dt [T/s]")
            ax2.set_title("TDEM Decay Curve (Log-Log Scale)")
            ax2.grid(True, alpha=0.3, which='both')
            ax2.set_xlim((time.min(), time.max()))

        # Shared legend
        if self.plot_linear:
            handles, legend_labels = ax1.get_legend_handles_labels()
        else:
            handles, legend_labels = ax2.get_legend_handles_labels()
            
        fig.legend(handles, legend_labels, loc='lower center', ncol=min(3, len(decays)), 
                  bbox_to_anchor=(0.5, -0.05), frameon=True, fontsize=9)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.show()

class PiConditioner:
    def __init__(self):
        pass

    def amplify(self, time, data, time_gain):
        time_gain = np.array(time_gain)
        if time_gain.size > 0:
            time_gain = time_gain[np.argsort(time_gain[:, 0])]
        gain = np.ones_like(time)
        for t, g in time_gain:
            mask = time >= t
            gain[mask] = g
        return data * gain

    def normalize(self, data):
        max = np.max(data)
        return data/max

    def quantize(self, data, depth, dtype):
        return (np.round(PiConditioner.normalize(data)* 2**depth)).astype(dtype)

simulator = PiSimulator(cfg, loop_z_start=cfg.tx_z, loop_z_increment=0.08, num_increments=5)

# Run simulation
times, data, plotting_info = simulator.run()

# Create plotter with CLI-based plot selection
# Set which plots to show (True/False for each):
plotter = PiPlotter(
    plotting_info,
    plot_linear=True,      # TDEM response linear scale
    plot_loglog=True,      # TDEM response log-log scale
    plot_side_view=True,   # Environment side view (X-Z plane)
    plot_top_view=True,    # Environment top view (X-Y plane)
    plot_3d=True           # 3D environment model
)

# Plot environment views
plotter.plot_environment()

# Plot TDEM response
plotter.plot_tdem_response(times, data)

conditioner = PiConditioner()

conditioner.amplify(data[0], times[0], time_gain={'Time': [0.02], 'Gain': [10]})
