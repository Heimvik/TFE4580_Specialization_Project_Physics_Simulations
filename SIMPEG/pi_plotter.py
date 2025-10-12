import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from simpeg import maps

class PiPlotter:
    def __init__(self, plotting_info, plot_linear=False, plot_loglog=False, 
                 plot_side_view=False, plot_top_view=False, plot_3d=False):
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
        self.enable_linear = plot_linear
        self.enable_loglog = plot_loglog
        self.enable_side_view = plot_side_view
        self.enable_top_view = plot_top_view
        self.enable_3d = plot_3d
        
        # Data for TDEM plots - now supports multiple labeled datasets
        self.datasets = []  # List of dicts with 'label', 'times', 'decays'
        
        # Color palette for different labels
        self.color_palettes = {
            'unconditioned': {'base': 'blue', 'name': 'Blue'},
            'noise_free': {'base': 'green', 'name': 'Green'},
            'noisy': {'base': 'orange', 'name': 'Orange'},
            'filtered': {'base': 'purple', 'name': 'Purple'},
            'amplified': {'base': 'cyan', 'name': 'Cyan'},
            'default': {'base': 'gray', 'name': 'Gray'}
        }

    def update_times_data(self, times, decays, label='unconditioned', replace=True):
        """
        Update times and decays data for TDEM plotting.
        
        Parameters
        ----------
        times : array or list
            Time array(s) for the data
        decays : list of arrays
            List of decay curves
        label : str
            Label for this dataset (e.g., 'unconditioned', 'noise_free', 'noisy')
            Different labels will be plotted in different colors
        replace : bool
            If True, replace all existing data. If False, append to existing datasets.
        """
        if replace:
            self.datasets = []
        
        # Store the dataset with its label
        self.datasets.append({
            'label': label,
            'times': times,
            'decays': decays
        })

    def _get_color_for_label(self, label):
        """Get the base color name for a given label."""
        if label in self.color_palettes:
            return self.color_palettes[label]['base']
        return self.color_palettes['default']['base']
    
    def _generate_color_shade(self, base_color, norm_value):
        """
        Generate a shade of the base color based on normalized value (0 to 1).
        Dark shade for norm_value=0, light shade for norm_value=1.
        """
        color_schemes = {
            'blue': lambda n: (0.1 + n * 0.5, 0.2 + n * 0.5, 0.8 + n * 0.2),
            'green': lambda n: (0.1 + n * 0.3, 0.6 + n * 0.3, 0.1 + n * 0.3),
            'orange': lambda n: (0.9 + n * 0.1, 0.4 + n * 0.4, 0.1 + n * 0.2),
            'purple': lambda n: (0.5 + n * 0.3, 0.1 + n * 0.3, 0.7 + n * 0.2),
            'cyan': lambda n: (0.1 + n * 0.3, 0.7 + n * 0.2, 0.8 + n * 0.2),
            'gray': lambda n: (0.3 + n * 0.5, 0.3 + n * 0.5, 0.3 + n * 0.5)
        }
        
        if base_color in color_schemes:
            return color_schemes[base_color](norm_value)
        else:
            return color_schemes['gray'](norm_value)
    
    def _get_colors_labels(self):
        """Generate colors and labels for all decay curves across all datasets."""
        colors = []
        labels = []
        
        for dataset in self.datasets:
            label_name = dataset['label']
            decays = dataset['decays']
            base_color = self._get_color_for_label(label_name)
            
            # First decay is always "no target" in red
            colors.append('red')
            labels.append(f'No target [{label_name}] (loop @ {float(self.loop_z_start):.2f}m)')
            
            # Remaining decays are "with target" in gradient colors
            num_with_target = len(decays) - 1
            if num_with_target > 0:
                for i in range(num_with_target):
                    loop_z = self.loop_z_start + i * self.loop_z_increment
                    distance_from_target = abs(loop_z - float(self.cfg.target_z))
                    
                    if num_with_target > 1:
                        norm_dist = i / (num_with_target - 1)
                    else:
                        norm_dist = 0.5
                    
                    color = self._generate_color_shade(base_color, norm_dist)
                    colors.append(color)
                    labels.append(f'With target [{label_name}] (loop @ {loop_z:.2f}m, dist={distance_from_target:.2f}m)')
        
        return colors, labels
    
    def plot_tdem_linear(self):
        """Plot TDEM response on linear scale."""
        if not self.datasets:
            print("No TDEM data available. Use update_times_data() first.")
            return
        
        colors, labels = self._get_colors_labels()
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        color_idx = 0
        for dataset in self.datasets:
            times = dataset['times']
            decays = dataset['decays']
            
            # Handle time as list or array
            if isinstance(times, list):
                time = times[0] * 1e6  # Convert to μs
            else:
                time = times * 1e6
            
            for decay in decays:
                ax.plot(time, decay, marker='x', markersize=3, color=colors[color_idx], 
                        label=labels[color_idx], linewidth=2, alpha=0.8)
                color_idx += 1
        
        ax.set_xlabel("Time [μs]", fontsize=12)
        ax.set_ylabel("-dBz/dt [T/s]", fontsize=12)
        ax.set_title("TDEM Decay Curve (Linear Scale)", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Get max time from all datasets
        if isinstance(self.datasets[0]['times'], list):
            max_time = self.datasets[0]['times'][0].max() * 1e6
        else:
            max_time = self.datasets[0]['times'].max() * 1e6
        ax.set_xlim((0, max_time))
        
        ax.legend(loc='best', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def plot_tdem_loglog(self):
        """Plot TDEM response on log-log scale."""
        if not self.datasets:
            print("No TDEM data available. Use update_times_data() first.")
            return
        
        colors, labels = self._get_colors_labels()
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        color_idx = 0
        min_time = float('inf')
        max_time = 0
        
        for dataset in self.datasets:
            times = dataset['times']
            decays = dataset['decays']
            
            # Handle time as list or array
            if isinstance(times, list):
                time = times[0] * 1e6  # Convert to μs
            else:
                time = times * 1e6
            
            min_time = min(min_time, time.min())
            max_time = max(max_time, time.max())
            
            for decay in decays:
                ax.loglog(time, decay, marker='x', markersize=3, color=colors[color_idx], 
                          label=labels[color_idx], linewidth=2, alpha=0.8)
                color_idx += 1
        
        ax.set_xlabel("Time [μs]", fontsize=12)
        ax.set_ylabel("-dBz/dt [T/s]", fontsize=12)
        ax.set_title("TDEM Decay Curve (Log-Log Scale)", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        
        # Set xlim only if min is positive for log scale
        if min_time > 0:
            ax.set_xlim((min_time, max_time))
        
        ax.legend(loc='best', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def plot_3d_view(self):
        """Plot 3D environment view (optimized for performance)."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('3D Environment View', fontsize=14, fontweight='bold')
        
        # Get cell centers
        r = self.mesh.cell_centers[self.ind_active, 0]
        z = self.mesh.cell_centers[self.ind_active, 2]
        
        # Use fewer theta points for performance
        n_theta = 12  # Reduced from 36
        theta = np.linspace(0, 2*np.pi, n_theta)
        
        # Separate cells by conductivity type with better thresholds
        ind_air = self.model < 1e-4
        ind_soil = (self.model >= 1e-4) & (self.model < 1e5)
        ind_aluminum = self.model >= 1e5
        
        # Plot soil cells (brown) - much more sparse sampling
        if ind_soil.sum() > 0:
            soil_indices = np.where(ind_soil)[0][::10]  # Sample every 10th cell
            for i in soil_indices:
                ri, zi = r[i], z[i]
                # Plot only a few points around circumference
                for tj in range(0, n_theta, 3):
                    xi = ri * np.cos(theta[tj])
                    yi = ri * np.sin(theta[tj])
                    ax.scatter(xi, yi, zi, c='saddlebrown', s=5, alpha=0.4, marker='.')
        
        # Plot aluminum target cells (silver/gray) - optimized
        if ind_aluminum.sum() > 0:
            al_indices = np.where(ind_aluminum)[0]
            for i in al_indices[::2]:  # Every other aluminum cell
                ri, zi = r[i], z[i]
                for tj in theta:
                    xi = ri * np.cos(tj)
                    yi = ri * np.sin(tj)
                    ax.scatter(xi, yi, zi, c='silver', s=15, alpha=0.8, marker='o', 
                             edgecolors='black', linewidths=0.3)
        
        # Plot transmitter/receiver loops
        loop_radius = float(self.cfg.tx_radius)
        for i in range(self.num_increments + 2):
            loop_z = self.loop_z_start + i * self.loop_z_increment
            loop_theta = np.linspace(0, 2*np.pi, 100)
            loop_x = loop_radius * np.cos(loop_theta)
            loop_y = loop_radius * np.sin(loop_theta)
            loop_z_arr = np.full_like(loop_x, loop_z)
            
            if i == 0:
                ax.plot(loop_x, loop_y, loop_z_arr, 'r-', linewidth=3, 
                       label=f'Loop (no tgt) @ z={loop_z:.2f}m')
            else:
                norm_i = (i-1) / max(1, self.num_increments)
                color = (0.1 + norm_i*0.5, 0.2 + norm_i*0.5, 0.8 + norm_i*0.2)
                ax.plot(loop_x, loop_y, loop_z_arr, color=color, linewidth=2.5, 
                       label=f'Loop (w/ tgt) @ z={loop_z:.2f}m')
        
        # Plot target cylinder outline
        target_z = float(self.cfg.target_z)
        target_r = float(self.cfg.target_radius)
        target_h = float(self.cfg.target_height)
        z_top = target_z + target_h/2
        z_bot = target_z - target_h/2
        
        cyl_theta = np.linspace(0, 2*np.pi, 50)
        cyl_x = target_r * np.cos(cyl_theta)
        cyl_y = target_r * np.sin(cyl_theta)
        
        ax.plot(cyl_x, cyl_y, np.full_like(cyl_x, z_top), 'k-', linewidth=2, label='Target boundary')
        ax.plot(cyl_x, cyl_y, np.full_like(cyl_x, z_bot), 'k-', linewidth=2)
        
        for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            ax.plot([target_r*np.cos(angle)]*2, [target_r*np.sin(angle)]*2, 
                   [z_bot, z_top], 'k-', linewidth=2)
        
        # Ground plane
        ground_r = np.linspace(0, loop_radius*1.2, 20)
        ground_theta = np.linspace(0, 2*np.pi, 40)
        R, T = np.meshgrid(ground_r, ground_theta)
        X_ground = R * np.cos(T)
        Y_ground = R * np.sin(T)
        Z_ground = np.zeros_like(X_ground)
        ax.plot_surface(X_ground, Y_ground, Z_ground, alpha=0.2, color='green')
        
        ax.set_xlabel('X [m]', fontsize=10)
        ax.set_ylabel('Y [m]', fontsize=10)
        ax.set_zlabel('Z [m]', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_box_aspect([1, 1, 0.8])
        
        plt.tight_layout()
        plt.show()
    
    def plot_side_view(self):
        """Plot side view (X-Z plane)."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.set_title('Side View (X-Z Plane)', fontsize=14, fontweight='bold')
        
        # Plot conductivity zones in 2D
        plotting_map = maps.InjectActiveCells(self.mesh, self.ind_active, np.nan)
        log_model = np.log10(self.model)
        self.mesh.plot_image(plotting_map * log_model, ax=ax, grid=True,
                           clim=(np.log10(self.cfg.air_conductivity), 
                                np.log10(self.cfg.aluminum_conductivity)))
        
        # Target parameters
        target_z = float(self.cfg.target_z)
        target_r = float(self.cfg.target_radius)
        target_h = float(self.cfg.target_height)
        z_top = target_z + target_h/2
        z_bot = target_z - target_h/2
        loop_radius = float(self.cfg.tx_radius)
        
        # Overlay features
        ax.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Ground surface')
        ax.axhline(y=target_z, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='Target center')
        
        # Draw target rectangle
        ax.add_patch(plt.Rectangle((0, z_bot), target_r, target_h, 
                                   fill=False, edgecolor='black', linewidth=2, linestyle='-'))
        
        # Draw loops
        for i in range(self.num_increments + 2):
            loop_z = self.loop_z_start + i * self.loop_z_increment
            if i == 0:
                ax.plot([0, loop_radius], [loop_z, loop_z], 'r-', linewidth=2.5, marker='o', markersize=8)
            else:
                norm_i = (i-1) / max(1, self.num_increments)
                color = (0.1 + norm_i*0.5, 0.2 + norm_i*0.5, 0.8 + norm_i*0.2)
                ax.plot([0, loop_radius], [loop_z, loop_z], color=color, linewidth=2, marker='o', markersize=6)
        
        ax.set_xlabel('Radial Distance [m]', fontsize=12)
        ax.set_ylabel('Elevation [m]', fontsize=12)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_top_view(self):
        """Plot top view (X-Y plane)."""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.set_title('Top View (X-Y Plane)', fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        
        target_r = float(self.cfg.target_radius)
        loop_radius = float(self.cfg.tx_radius)
        
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
            ax.add_patch(circle)
        
        # Draw target
        target_circle = plt.Circle((0, 0), target_r, fill=True, facecolor='silver', 
                                  edgecolor='black', linewidth=2, alpha=0.7, label='Target')
        ax.add_patch(target_circle)
        
        ax.set_xlim([-loop_radius*1.3, loop_radius*1.3])
        ax.set_ylim([-loop_radius*1.3, loop_radius*1.3])
        ax.set_xlabel('X [m]', fontsize=12)
        ax.set_ylabel('Y [m]', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def show_plots(self):
        """
        Interactively show plots with user selection.
        User can choose which plots to display from the enabled options.
        """
        # Build available plots dictionary
        available_plots = {}
        
        if self.enable_linear:
            available_plots['1'] = ('TDEM Linear', self.plot_tdem_linear)
        if self.enable_loglog:
            available_plots['2'] = ('TDEM Log-Log', self.plot_tdem_loglog)
        if self.enable_3d:
            available_plots['3'] = ('3D View', self.plot_3d_view)
        if self.enable_side_view:
            available_plots['4'] = ('Side View', self.plot_side_view)
        if self.enable_top_view:
            available_plots['5'] = ('Top View', self.plot_top_view)
        
        if not available_plots:
            print("No plots enabled.")
            return
        
        print(f"\n{'='*60}")
        print("PiPlotter: Interactive Plot Selection")
        print(f"{'='*60}")
        
        while True:
            print("\nAvailable plots:")
            for key, (name, _) in available_plots.items():
                print(f"  [{key}] {name}")
            print("  [q] Quit plotting")
            
            choice = input("\nSelect a plot to display (or 'q' to quit): ").strip().lower()
            
            if choice == 'q':
                print("Exiting plotter.")
                break
            
            if choice in available_plots:
                plot_name, plot_func = available_plots[choice]
                print(f"\nShowing: {plot_name}")
                plot_func()
            else:
                print(f"Invalid choice: '{choice}'. Please select from the menu.")
        
        print(f"\n{'='*60}")
        print("Plotting session ended.")
        print(f"{'='*60}\n")