import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from simpeg import maps
from discretize import CylindricalMesh
import h5py

class PiPlotter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.hdf5_file = None
        self.num_sims = 0
        self.num_present = 0
        self.num_absent = 0
        self.simulations_metadata = []
        self.mesh = None
        self.ind_active = None
        self.model_no_target = None
        print(f"PiPlotter initialized. Use load_from_hdf5() to load data.")
    
    def load_from_hdf5(self, hdf5_file):
        self.hdf5_file = hdf5_file
        
        with h5py.File(hdf5_file, 'r') as f:
            self.num_sims = f['metadata'].attrs['num_simulations']
            self.num_present = f['metadata'].attrs['num_target_present']
            self.num_absent = f['metadata'].attrs['num_target_absent']
        
        print(f"\n{'='*70}")
        print(f"Loading data from: {hdf5_file}")
        print(f"{'='*70}")
        print(f"Total simulations: {self.num_sims}")
        print(f"  - Target present: {self.num_present}")
        print(f"  - Target absent: {self.num_absent}")
        
        hr = [(0.01, 15), (0.01, 15, 1.3), (0.05, 10, 1.5)]
        hphi = 1
        hz = [(0.01, 10, -1.3), (0.01, 30), (0.01, 10, 1.3)]
        self.mesh = CylindricalMesh([hr, hphi, hz], x0="00C")
        
        active_area_z = self.cfg.separation_z
        self.ind_active = self.mesh.cell_centers[:, 2] < active_area_z
        
        r = self.mesh.cell_centers[self.ind_active, 0]
        z = self.mesh.cell_centers[self.ind_active, 2]
        self.model_no_target = self.cfg.air_conductivity * np.ones(self.ind_active.sum())
        ind_soil = (z < 0)
        self.model_no_target[ind_soil] = self.cfg.soil_conductivity
        
        self.simulations_metadata = []
        
        self._load_metadata()
        
        print(f"✓ Data loaded successfully.\n")
    
    def _load_metadata(self):
        if self.hdf5_file is None:
            print("Error: No HDF5 file loaded. Use load_from_hdf5() first.")
            return
        
        with h5py.File(self.hdf5_file, 'r') as f:
            for i in range(self.num_sims):
                sim_group = f[f'simulations/simulation_{i}']
                metadata = {
                    'index': i,
                    'loop_z': sim_group.attrs['loop_z'],
                    'target_z': sim_group.attrs.get('target_z', None),
                    'target_present': sim_group.attrs['target_present'],
                    'label': sim_group.attrs['label'],
                    'time': sim_group['time'][:],
                    'decay': sim_group['decay'][:]
                }
                if metadata['target_z'] == -999.0:
                    metadata['target_z'] = None
                self.simulations_metadata.append(metadata)
    
    def _reconstruct_model(self, target_z, target_present):

        r = self.mesh.cell_centers[self.ind_active, 0]
        z = self.mesh.cell_centers[self.ind_active, 2]
        
        model = self.model_no_target.copy()
        
        if target_present and target_z is not None:
            inner_radius = self.cfg.target_radius - self.cfg.target_thickness
            if inner_radius < 0:
                inner_radius = 0
            
            unique_r = np.unique(r)
            if len(unique_r) > 1:
                min_cell_width = np.min(np.diff(unique_r[unique_r > 0]))
            else:
                min_cell_width = 0.01
            
            top_z = target_z + self.cfg.target_height/2
            bottom_z = target_z - self.cfg.target_height/2
            
            if self.cfg.target_thickness < min_cell_width:
                ind_walls = (
                    (r >= self.cfg.target_radius - min_cell_width) &
                    (r <= self.cfg.target_radius) &
                    (z < top_z) &
                    (z > bottom_z)
                )
            else:
                ind_walls = (
                    (r >= inner_radius) &
                    (r <= self.cfg.target_radius) &
                    (z < top_z) &
                    (z > bottom_z)
                )
            
            ind_top = (r <= self.cfg.target_radius) & (np.abs(z - top_z) <= 0.01)
            ind_bottom = (r <= self.cfg.target_radius) & (np.abs(z - bottom_z) <= 0.01)
            ind_can = ind_walls | ind_top | ind_bottom
            model[ind_can] = self.cfg.aluminum_conductivity
        
        return model

    def run(self):
        if self.hdf5_file is None:
            print("Error: No data loaded. Use load_from_hdf5() first.")
            return
        
        print(f"\n{'='*60}")
        print("PiPlotter: HDF5-Based Visualization")
        print(f"{'='*60}")
        
        while True:
            print("\nAvailable plots:")
            print("  [1] Combined View (Side View + Decay Curves)")
            print("  [2] Quick Log-Log Plot (first N simulations)")
            print("  [3] Detailed Side View (model grid + configuration)")
            print("  [4] Detailed Top View (model grid + configuration)")
            print("  [q] Quit plotting")
            
            choice = input("\nSelect a plot to display (or 'q' to quit): ").strip().lower()
            
            if choice == 'q':
                print("Exiting plotter.")
                break
            
            if choice == '1':
                print("\nShowing: Combined View")
                self.plot_combined_from_hdf5()
            elif choice == '2':
                print("\nShowing: Quick Log-Log Plot")
                self._quick_loglog_plot()
            elif choice == '3':
                print("\nShowing: Detailed Side View")
                self.plot_side_view_detailed()
            elif choice == '4':
                print("\nShowing: Detailed Top View")
                self.plot_top_view_detailed()
            else:
                print(f"Invalid choice: '{choice}'. Please select from the menu.")
        
        print(f"\n{'='*60}")
        print("Plotting session ended.")
        print(f"{'='*60}\n")
    
    def _quick_loglog_plot(self):
        n = int(input(f"How many simulations to plot (max {self.num_sims})? ").strip() or "5")
        n = min(n, self.num_sims)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for i in range(n):
            meta = self.simulations_metadata[i]
            time_us = meta['time'] * 1e6
            decay = np.abs(meta['decay'])
            
            color = 'green' if meta['target_present'] else 'blue'
            ax.loglog(time_us, decay, color=color, alpha=0.7, linewidth=1.5, label=meta['label'])
        
        ax.set_xlabel('Time [μs]', fontsize=12)
        ax.set_ylabel('|dBz/dt| [T/s]', fontsize=12)
        ax.set_title(f'TDEM Decay Curves (first {n} simulations)', fontsize=14, fontweight='bold')
        
        if n <= 10:
            ax.legend(fontsize=9, ncol=2)
        else:
            legend_elements = [
                Line2D([0], [0], color='green', lw=2, label='Target Present'),
                Line2D([0], [0], color='blue', lw=2, label='Target Absent')
            ]
            ax.legend(handles=legend_elements, fontsize=10)
        
        ax.grid(True, alpha=0.3, which='both')
        plt.tight_layout()
        plt.show()
    
    def plot_combined_from_hdf5(self):

        print("\n" + "="*70)
        print("Combined View: Physical Configuration + Decay Curves")
        print("="*70)
        print("\nAvailable simulations:")
        
        for idx, meta in enumerate(self.simulations_metadata):
            if meta['target_present']:
                print(f"  [{idx+1}] {meta['label']} (Loop @ {meta['loop_z']:.3f}m, Target @ {meta['target_z']:.3f}m)")
            else:
                print(f"  [{idx+1}] {meta['label']} (Loop @ {meta['loop_z']:.3f}m, No Target)")
        
        print("\nSelect simulations to plot:")
        print("  - Enter custom selection (comma-separated, e.g., '1,3,5')")
        print("  - Enter 'all' for all simulations")
        print("  - Enter 'q' to cancel")
        
        user_input = input("Your selection: ").strip()
        
        if user_input.lower() == 'q':
            print("Plot cancelled.")
            return
        
        selected_indices = []
        if user_input.lower() == 'all':
            selected_indices = list(range(self.num_sims))
        else:
            try:
                parts = [p.strip() for p in user_input.split(',')]
                for part in parts:
                    idx = int(part) - 1
                    if 0 <= idx < self.num_sims:
                        selected_indices.append(idx)
                    else:
                        print(f"Warning: Index {part} out of range, skipping.")
            except ValueError:
                print("Invalid input format.")
                return
        
        if not selected_indices:
            print("No valid simulations selected.")
            return
        
        print(f"\nPlotting {len(selected_indices)} simulation(s)...")
        
        fig = plt.figure(figsize=(24, 11))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 3], wspace=0.25)
        ax_model = fig.add_subplot(gs[0, 0])
        ax_decay = fig.add_subplot(gs[0, 1])
        
        r = self.mesh.cell_centers[self.ind_active, 0]
        z = self.mesh.cell_centers[self.ind_active, 2]
        
        ax_model.set_title('Physical Configurations (Side View)', fontsize=18, fontweight='bold', pad=15)
        ax_model.set_aspect('equal')
        
        ax_model.axhline(0, color='darkgreen', linewidth=2, linestyle='--', alpha=0.7, label='Ground Level')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_indices)))
        
        for plot_idx, sim_idx in enumerate(selected_indices):
            meta = self.simulations_metadata[sim_idx]
            color = colors[plot_idx]
            
            if meta['target_z'] is not None:
                ax_model.scatter([self.cfg.target_radius/2], [meta['target_z']], 
                               c=[color], s=150, alpha=0.9,
                               marker='X', edgecolors='black', linewidths=2.0,
                               label=f"T{plot_idx+1}")
            
            loop_radius = float(self.cfg.tx_radius)
            ax_model.plot([0, loop_radius], [meta['loop_z'], meta['loop_z']],
                         color=color, linewidth=4, marker='o', markersize=10,
                         markerfacecolor=color, markeredgecolor='black', markeredgewidth=1.5,
                         label=f"L{plot_idx+1}")
        
        ax_model.set_xlabel('Radial Distance [m]', fontsize=15)
        ax_model.set_ylabel('Depth [m]', fontsize=15)
        ax_model.tick_params(labelsize=12)
        
        num_items = len(selected_indices) * 2 + 2
        if num_items > 15:
            ax_model.legend(loc='upper right', fontsize=7, ncol=4, framealpha=0.9, columnspacing=0.5)
        elif num_items > 10:
            ax_model.legend(loc='upper right', fontsize=7, ncol=3, framealpha=0.9)
        else:
            ax_model.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)
        
        ax_model.grid(True, alpha=0.3)
        ax_model.set_xlim([0, 0.5])
        
        ax_decay.set_title('TDEM Decay Curves (Log-Log)', fontsize=18, fontweight='bold', pad=15)
        
        for plot_idx, sim_idx in enumerate(selected_indices):
            meta = self.simulations_metadata[sim_idx]
            color = colors[plot_idx]
            
            time_us = meta['time'] * 1e6
            decay = np.abs(meta['decay'])
            
            ax_decay.loglog(time_us, decay, color=color, linewidth=2.5, alpha=0.9,
                           label=f"{plot_idx+1}: {meta['label']}", marker='.')
        
        ax_decay.set_xlabel('Time [μs]', fontsize=15)
        ax_decay.set_ylabel('|dBz/dt| [T/s]', fontsize=15)
        ax_decay.tick_params(labelsize=12)
        
        if len(selected_indices) > 15:
            ax_decay.legend(loc='lower left', fontsize=7, ncol=2, framealpha=0.9)
        elif len(selected_indices) > 8:
            ax_decay.legend(loc='lower left', fontsize=8, ncol=1, framealpha=0.9)
        else:
            ax_decay.legend(loc='lower left', fontsize=9, framealpha=0.9)
        
        ax_decay.grid(True, alpha=0.3, which='both')
        
        plt.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.08, wspace=0.25)
        plt.show()
    
    def plot_side_view_detailed(self):

        print("\n" + "="*70)
        print("Detailed Side View: Exact Mesh & Model Configuration")
        print("="*70)
        print("\nSelect a simulation to visualize:")
        
        for idx, meta in enumerate(self.simulations_metadata):
            if meta['target_present']:
                print(f"  [{idx+1}] {meta['label']} (Loop @ {meta['loop_z']:.3f}m, Target @ {meta['target_z']:.3f}m)")
            else:
                print(f"  [{idx+1}] {meta['label']} (Loop @ {meta['loop_z']:.3f}m, No Target)")
        
        try:
            choice = int(input("\nEnter simulation number: ").strip()) - 1
            if choice < 0 or choice >= self.num_sims:
                print("Invalid selection.")
                return
        except ValueError:
            print("Invalid input.")
            return
        
        meta = self.simulations_metadata[choice]
        
        model = self._reconstruct_model(meta['target_z'], meta['target_present'])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(f'Side View: {meta["label"]}', fontsize=14, fontweight='bold')
        
        plotting_map = maps.InjectActiveCells(self.mesh, self.ind_active, np.nan)
        log_model = np.log10(model)
        self.mesh.plot_image(plotting_map * log_model, ax=ax, grid=True,
                           clim=(np.log10(self.cfg.air_conductivity), 
                                np.log10(self.cfg.aluminum_conductivity)))
        
        ax.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Ground surface')
        
        loop_radius = float(self.cfg.tx_radius)
        ax.plot([0, loop_radius], [meta['loop_z'], meta['loop_z']], 
               color='blue', linewidth=2.5, marker='o', markersize=8,
               label=f'TX/RX Loop (z={meta["loop_z"]:.3f}m)')
        
        if meta['target_present'] and meta['target_z'] is not None:
            target_top = meta['target_z'] + self.cfg.target_height/2
            target_bottom = meta['target_z'] - self.cfg.target_height/2
            target_radius = self.cfg.target_radius
            
            ax.axhline(y=meta['target_z'], color='red', linestyle=':', linewidth=1.5, alpha=0.7, 
                      label=f'Target center (z={meta["target_z"]:.3f}m)')
            
            ax.add_patch(plt.Rectangle((0, target_bottom), target_radius, self.cfg.target_height,
                                      fill=False, edgecolor='black', linewidth=2, linestyle='-'))
        
        ax.set_xlabel('Radial Distance [m]', fontsize=12)
        ax.set_ylabel('Elevation [m]', fontsize=12)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, None])
        
        plt.tight_layout()
        plt.show()

    def plot_top_view_detailed(self):
        print("\n" + "="*70)
        print("Detailed Top View: Exact Mesh & Model Configuration")
        print("="*70)
        print("\nSelect a simulation to visualize:")
        
        for idx, meta in enumerate(self.simulations_metadata):
            if meta['target_present']:
                print(f"  [{idx+1}] {meta['label']} (Loop @ {meta['loop_z']:.3f}m, Target @ {meta['target_z']:.3f}m)")
            else:
                print(f"  [{idx+1}] {meta['label']} (Loop @ {meta['loop_z']:.3f}m, No Target)")
        
        try:
            choice = int(input("\nEnter simulation number: ").strip()) - 1
            if choice < 0 or choice >= self.num_sims:
                print("Invalid selection.")
                return
        except ValueError:
            print("Invalid input.")
            return
        
        meta = self.simulations_metadata[choice]
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(f'Top View: {meta["label"]}', fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        
        target_r = float(self.cfg.target_radius)
        loop_radius = float(self.cfg.tx_radius)
        
        circle = plt.Circle((0, 0), loop_radius, fill=False, linewidth=3, 
                           edgecolor='blue', label=f'TX/RX Loop (z={meta["loop_z"]:.3f}m)')
        ax.add_patch(circle)
        
        if meta['target_present'] and meta['target_z'] is not None:
            target_circle = plt.Circle((0, 0), target_r, fill=True, facecolor='silver', 
                                      edgecolor='black', linewidth=2, alpha=0.7, 
                                      label=f'Target (z={meta["target_z"]:.3f}m)')
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