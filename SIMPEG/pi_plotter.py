import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider
from simpeg import maps
from discretize import CylindricalMesh
import h5py
import os
from pi_logger import PiLogger


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
        
        print(f"Data loaded successfully.\n")
    def _load_metadata(self):
        with h5py.File(self.hdf5_file, 'r') as f:
            self.simulations_metadata = []
            for i in range(self.num_sims):
                sim_group = f[f'simulations/simulation_{i}']
                metadata = {
                    'index': i,
                    'loop_z': sim_group.attrs['loop_z'],
                    'target_type': sim_group.attrs.get('target_type', -1),
                    'target_z': sim_group.attrs.get('target_z', None),
                    'target_present': sim_group.attrs['target_present'],
                    'label': sim_group.attrs['label'],
                    'time': sim_group['time'][:],
                    'decay': sim_group['decay'][:]
                }
                if metadata['target_z'] == -999.0:
                    metadata['target_z'] = None
                self.simulations_metadata.append(metadata)

    def get_conductivity_model(self, target_type, conductivity, target_z, cfg):
        models_dir = "Models"
        models_file = os.path.join(models_dir, "conductivity_models.h5")
        
        if not os.path.exists(models_file):
            print(f"Error: Models file not found at {models_file}")
            return None
            
        with h5py.File(models_file, 'r') as f:
            if target_type == 0:
                if 'models/base_model/model' not in f:
                    print("Error: base_model not found in file")
                    return None
                base_model = f['models/base_model/model'][:]
                return base_model
            
            model_name = f"type_{target_type}_cond_{conductivity:.0e}"
            if f'models/{model_name}' not in f:
                print(f"Error: Model '{model_name}' not found in file")
                print(f"Available models: {list(f['models'].keys())}")
                return None
            
            stored_model = f[f'models/{model_name}/model'][:]
            r = f['mesh_info/cell_centers_r'][:]
            z = f['mesh_info/cell_centers_z'][:]
        
        if abs(target_z) < 0.001:
            print(f"Debug: Returning stored model at z=0 (no offset needed)")
            return stored_model
        
        model = cfg.air_conductivity * np.ones_like(stored_model)
        ind_soil = (z < 0)
        model[ind_soil] = cfg.soil_conductivity
        
        target_mask = (stored_model > cfg.soil_conductivity * 10)
        num_target_in_stored = np.sum(target_mask)
        
        if num_target_in_stored == 0:
            print(f"Warning: No target cells found in stored model")
            return stored_model
        
        print(f"Debug: Found {num_target_in_stored} target cells in stored model at z=0")
        print(f"Debug: Shifting target from z=0 to z={target_z:.3f}m")
        print(f"Debug: z range in mesh: [{np.min(z):.3f}, {np.max(z):.3f}]")
        print(f"Debug: r range in mesh: [{np.min(r):.3f}, {np.max(r):.3f}]")
        
        target_z_cells = z[target_mask]
        print(f"Debug: Target cells in stored model span z: [{np.min(target_z_cells):.3f}, {np.max(target_z_cells):.3f}]")
        print(f"Debug: After shift, target should be at z: [{np.min(target_z_cells)+target_z:.3f}, {np.max(target_z_cells)+target_z:.3f}]")
        
        copied_count = 0
        
        unique_r = np.unique(r)
        unique_z = np.unique(z)
        print(f"Debug: Unique r values: {len(unique_r)}, first few: {unique_r[:5]}")
        print(f"Debug: Unique z values: {len(unique_z)}, around target: {unique_z[(unique_z > target_z-0.1) & (unique_z < target_z+0.1)][:10]}")
        
        for i in range(len(model)):
            z_rel_to_target = z[i] - target_z
            
            j = np.where((np.abs(r - r[i]) < 0.001) & (np.abs(z - z_rel_to_target) < 0.001))[0]
            
            if len(j) > 0:
                j = j[0]
                if target_mask[j]:
                    model[i] = stored_model[j]
                    copied_count += 1
                    if copied_count <= 3:
                        print(f"  Match {copied_count}: Cell at (r={r[i]:.3f}, z={z[i]:.3f}) <- stored cell at (r={r[j]:.3f}, z={z[j]:.3f}), σ={stored_model[j]:.2e}")
        
        print(f"Debug: Copied {copied_count} target cells to offset position {target_z:.3f}m")
        
        return model
    
    def run(self):
        if self.hdf5_file is None:
            print("Error: No data loaded. Use load_from_hdf5() first.")
            return
        
        while True:
            print("\\nAvailable plots:")
            print("  [1] Combined View (Side View + Decay Curves)")
            print("  [2] Quick Log-Log Plot (first N simulations)")
            print("  [3] Plot Conductivity Models")
            print("  [4] Compare Measured vs Simulated Data")
            print("  [5] Plot Multiple Measured & Simulated Curves")
            print("  [q] Quit plotting")
            
            choice = input("\\nSelect a plot to display (or 'q' to quit): ").strip().lower()
            
            if choice == 'q':
                print("Exiting plotter.")
                break
            
            if choice == '1':
                self.plot_combined_from_hdf5()
            elif choice == '2':
                self._quick_loglog_plot()
            elif choice == '3':
                self.plot_conductivity_models()
            elif choice == '4':
                self.plot_measured_vs_simulated()
            elif choice == '5':
                self.plot_multiple_measured_and_simulated()
            else:
                print(f"Invalid choice: '{choice}'. Please select from the menu.")
    
    def _quick_loglog_plot(self):
        try:
            n = min(int(input("Number of simulations to plot [10]: ").strip() or "10"), self.num_sims)
        except ValueError:
            n = 10
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, n))
        
        with h5py.File(self.hdf5_file, 'r') as f:
            for i in range(n):
                sim_group = f[f'simulations/simulation_{i}']
                time = sim_group['time'][:]
                decay = sim_group['decay'][:]
                label = sim_group.attrs['label']
                target_present = sim_group.attrs['target_present']
                
                time_us = time * 1e6
                label_prefix = "[T]" if target_present else "[N]"
                plt.loglog(time_us, np.abs(decay), color=colors[i], 
                          linewidth=2, label=f"{label_prefix} {label}")
        
        plt.xlabel('Time [μs]', fontsize=18)
        plt.ylabel(r'$\\left|\\frac{\\partial B_z}{\\partial t}\\right|$ [T/s]', fontsize=18)
        plt.tick_params(labelsize=18)
        plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3, which='both')
        plt.tight_layout()
        plt.show()
    
    def plot_combined_from_hdf5(self):

        print("\n" + "="*70)
        print("Combined View: Physical Configuration + Decay Curves")
        print("="*70)
        print("\nAvailable simulations:")
        
        for idx, meta in enumerate(self.simulations_metadata):
            if meta['target_present']:
                print(f"  [{idx+1}] {meta['label']} (Loop @ {meta['loop_z']:.3f}m, Target at {meta['target_z']:.3f}m)")
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
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 3], wspace=0.07)
        ax_model = fig.add_subplot(gs[0, 0])
        ax_decay = fig.add_subplot(gs[0, 1])
        
        r = self.mesh.cell_centers[self.ind_active, 0]
        z = self.mesh.cell_centers[self.ind_active, 2]
        
        ax_model.set_aspect('equal')
        
        ax_model.axhline(0, color='darkgreen', linewidth=2, linestyle='--', alpha=0.7, label='Ground Level')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_indices)))
        
        for plot_idx, sim_idx in enumerate(selected_indices):
            meta = self.simulations_metadata[sim_idx]
            color = colors[plot_idx]
            
            if meta['target_z'] is not None:
                ax_model.scatter([self.cfg.target_radius/2], [meta['target_z']], 
                        c=[color], s=150, alpha=0.9,
                        marker='X', edgecolors='black', linewidths=2.0)
            
            loop_radius = float(self.cfg.tx_radius)
            ax_model.plot([0, loop_radius], [meta['loop_z'], meta['loop_z']],
                 color=color, linewidth=4, marker='o', markersize=10,
                 markerfacecolor=color, markeredgecolor='black', markeredgewidth=1.5,
                 label=f"{meta['label']}")
        
        ax_model.set_xlabel('Radial Distance [m]', fontsize=18)
        ax_model.set_ylabel('Depth [m]', fontsize=18)
        ax_model.tick_params(labelsize=18)
        
        ax_model.grid(True, alpha=0.3)
        ax_model.set_xlim([0, self.cfg.tx_radius + 0.1])
        
        for plot_idx, sim_idx in enumerate(selected_indices):
            meta = self.simulations_metadata[sim_idx]
            color = colors[plot_idx]
            
            time_us = meta['time'] * 1e6
            decay = np.abs(meta['decay'])
            
            ax_decay.loglog(time_us, decay, color=color, linewidth=2.5, alpha=0.9,
                   label=f"{meta['label']}", marker='.')
        
        ax_decay.set_xlabel('Time [μs]', fontsize=18)
        ax_decay.set_ylabel(r'$\left|\frac{\partial B_z}{\partial t}\right|$ [T/s]', fontsize=18)
        ax_decay.tick_params(labelsize=18)
        
        ax_decay.grid(True, alpha=0.3, which='both')
        
        handles, labels = ax_model.get_legend_handles_labels()
        
        n_items = len(labels)
        if n_items > 15:
            ncol = 6
        elif n_items > 10:
            ncol = 5
        elif n_items > 6:
            ncol = 4
        else:
            ncol = 3
        
        fig.legend(handles, labels, fontsize=16, loc ='lower center',
                  ncol=ncol, framealpha=0.9)
        
        plt.subplots_adjust(left=0.06, right=0.98, top=0.98, bottom=0.12, wspace=0.25)
        plt.show()
    
    def plot_conductivity_models(self):
        from pi_logger import PiLogger
        
        logger = PiLogger()
        
        time, decay_curves, labels, label_strings, metadata = logger.load_from_hdf5(self.hdf5_file)
        
        available_sims = []
        for i in range(len(labels)):
            target_type = metadata['target_types'][i]
            target_conductivity = metadata['target_conductivities'][i]
            target_z = metadata['target_z'][i]
            
            if target_type > 0 and target_z is not None:
                type_names = {1: 'Hollow Cylinder', 2: 'Shredded Can', 3: 'Solid Block'}
                type_name = type_names.get(target_type, 'Unknown')
                status = f"{type_name} (σ={target_conductivity:.1e} S/m) at z={target_z:.2f}m"
                available_sims.append((i, label_strings[i], status, target_type, target_conductivity, target_z))
        
        if not available_sims:
            print("\\nNo target simulations found.")
            return
        
        print(f"\\nAvailable simulations:")
        for idx, (_, _, status, _, _, _) in enumerate(available_sims[:10]):
            print(f"  [{idx+1:2d}] {status}")
        
        try:
            choice = input(f"\nSelect simulation [1-{min(10, len(available_sims))}]: ").strip()
            sim_data = available_sims[int(choice) - 1]
            _, _, status, target_type, target_conductivity, target_z = sim_data
            
            print(f"\\nLoading model: type={target_type}, σ={target_conductivity:.1e}, z={target_z:.3f}m")
            
            model = self.get_conductivity_model(target_type, target_conductivity, target_z, self.cfg)
            
            if model is None:
                print("Model not found. Generate models first (main menu option 3).")
                return
                
            self._plot_model_2d(model, status)
            
        except (ValueError, IndexError) as e:
            print(f"Invalid selection: {e}")
    
    def _plot_model_2d(self, model, title):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        print(f"Debug: Model shape: {model.shape}")
        print(f"Debug: Model min: {np.min(model):.2e}, max: {np.max(model):.2e}")
        print(f"Debug: Unique values in model: {len(np.unique(model))}")
        print(f"Debug: Target cells (>soil*10): {np.sum(model > self.cfg.soil_conductivity * 10)}")
        
        plotting_map = maps.InjectActiveCells(self.mesh, self.ind_active, np.nan)
        log_model = np.log10(model)
        
        print(f"Debug: log_model min: {np.min(log_model):.2f}, max: {np.max(log_model):.2f}")
        print(f"Debug: clim: ({np.log10(self.cfg.air_conductivity):.2f}, {np.log10(self.cfg.aluminum_conductivity):.2f})")
        
        im = self.mesh.plot_image(plotting_map * log_model, ax=ax, grid=True,
                            clim=(np.log10(self.cfg.air_conductivity), 
                                  np.log10(self.cfg.aluminum_conductivity)))
        
        ax.axhline(y=0, color='brown', linestyle='-', linewidth=3, alpha=0.8, label='Ground surface')
        
        ax.set_xlabel('Radial Distance [m]', fontsize=18)
        ax.set_ylabel('Elevation [m]', fontsize=18)
        ax.set_title(f'Conductivity Model: {title}', fontsize=16)
        ax.tick_params(labelsize=18)
        ax.legend(fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, None])
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)