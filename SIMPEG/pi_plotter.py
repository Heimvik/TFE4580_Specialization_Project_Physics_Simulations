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
        
        if target_z is not None:
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
            print("  [5] Compare Measured vs Simulated Data (Single)")
            print("  [6] Plot Multiple Measured & Simulated Curves")
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
            elif choice == '5':
                print("\nShowing: Measured vs Simulated Comparison")
                self.plot_measured_vs_simulated()
            elif choice == '6':
                print("\nShowing: Multiple Measured & Simulated Curves")
                self.plot_multiple_measured_and_simulated()
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
        
        ax.set_xlabel('Time [μs]', fontsize=18)
        ax.set_ylabel(r'$\left|\frac{\partial B_z}{\partial t}\right|$ [T/s]', fontsize=18)
        ax.tick_params(labelsize=18)
        
        if n <= 10:
            ax.legend(fontsize=16, ncol=2)
        else:
            legend_elements = [
                Line2D([0], [0], color='green', lw=2, label='Target Present'),
                Line2D([0], [0], color='blue', lw=2, label='Target Absent')
            ]
            ax.legend(handles=legend_elements, fontsize=16)
        
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
        
        plotting_map = maps.InjectActiveCells(self.mesh, self.ind_active, np.nan)
        log_model = np.log10(model)
        self.mesh.plot_image(plotting_map * log_model, ax=ax, grid=True,
                            clim=(np.log10(self.cfg.air_conductivity), 
                                  np.log10(self.cfg.aluminum_conductivity)))
        
        ax.axhline(y=0, color='green', linestyle='-', linewidth=2, alpha=0.7, label='Ground surface')
        
        loop_radius = float(self.cfg.tx_radius)
        ax.plot([0, loop_radius], [meta['loop_z'], meta['loop_z']], 
               color='blue', linewidth=2.5, marker='o', markersize=8,
               label=f'TX/RX coil (z={meta["loop_z"]:.3f}m)')
        
        if meta['target_z'] is not None:
            target_top = meta['target_z'] + self.cfg.target_height/2
            target_bottom = meta['target_z'] - self.cfg.target_height/2
            target_radius = self.cfg.target_radius
            
            ax.axhline(y=meta['target_z'], color='red', linestyle=':', linewidth=1.5, alpha=0.7, 
                      label=f'Target center (z={meta["target_z"]:.3f}m)')
            
            ax.add_patch(plt.Rectangle((0, target_bottom), target_radius, self.cfg.target_height,
                                      fill=False, edgecolor='black', linewidth=2, linestyle='-'))
        
        ax.set_xlabel('Radial Distance [m]', fontsize=18)
        ax.set_ylabel('Elevation [m]', fontsize=18)
        ax.tick_params(labelsize=18)
        ax.legend(fontsize=16)
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
        ax.set_xlabel('X [m]', fontsize=18)
        ax.set_ylabel('Y [m]', fontsize=18)
        ax.tick_params(labelsize=18)
        ax.legend(loc='upper right', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_measured_vs_simulated(self):
        """
        Compare measured data from a HDF5 file against a specific simulation
        from the currently loaded simulated dataset.
        """
        print("\n" + "="*70)
        print("Compare Measured vs Simulated Data")
        print("="*70)
        
        # Get measured data file path
        measured_file = input("\nEnter path to measured HDF5 file (e.g., Measure/dataset.h5): ").strip()
        
        # Check if file exists
        import os
        if not os.path.exists(measured_file):
            print(f"Error: File not found: {measured_file}")
            return
        
        # Load measured data
        print(f"\nLoading measured data from: {measured_file}")
        try:
            with h5py.File(measured_file, 'r') as f:
                # Get metadata
                num_measured = f['metadata'].attrs['num_simulations']
                print(f"Measured dataset contains {num_measured} measurement(s)")
                
                # List available measurements
                if num_measured > 1:
                    print("\nAvailable measurements:")
                    for i in range(num_measured):
                        sim = f[f'simulations/simulation_{i}']
                        label = sim.attrs.get('label', f'measurement_{i}')
                        loop_z = sim.attrs.get('loop_z', 'N/A')
                        print(f"  [{i}] {label} (Loop Z: {loop_z})")
                    
                    meas_idx = int(input(f"\nSelect measurement index [0-{num_measured-1}]: ").strip() or "0")
                else:
                    meas_idx = 0
                
                # Load selected measurement
                sim = f[f'simulations/simulation_{meas_idx}']
                measured_time = sim['time'][:]
                measured_decay = sim['decay'][:]
                measured_label = sim.attrs.get('label', 'Measured')
                measured_loop_z = sim.attrs.get('loop_z', 0.0)
                measured_target_present = sim.attrs.get('target_present', False)
        except Exception as e:
            print(f"Error reading measured file: {e}")
            return
        
        print(f"\nMeasured data loaded:")
        print(f"  Label: {measured_label}")
        print(f"  Samples: {len(measured_time)}")
        print(f"  Loop Z: {measured_loop_z}")
        print(f"  Target Present: {measured_target_present}")
        
        # Select simulation from loaded dataset
        print(f"\n{'='*70}")
        print("Select simulation from loaded dataset to compare:")
        print(f"{'='*70}")
        
        for idx, meta in enumerate(self.simulations_metadata):
            if meta['target_present']:
                print(f"  [{idx+1}] {meta['label']} (Loop @ {meta['loop_z']:.3f}m, Target @ {meta['target_z']:.3f}m)")
            else:
                print(f"  [{idx+1}] {meta['label']} (Loop @ {meta['loop_z']:.3f}m, No Target)")
        
        try:
            sim_choice = int(input(f"\nSelect simulation [1-{self.num_sims}]: ").strip()) - 1
            if sim_choice < 0 or sim_choice >= self.num_sims:
                print("Invalid selection.")
                return
        except ValueError:
            print("Invalid input.")
            return
        
        sim_meta = self.simulations_metadata[sim_choice]
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Left plot: Overlay comparison
        measured_time_us = measured_time * 1e6
        measured_decay_abs = np.abs(measured_decay)
        sim_time_us = sim_meta['time'] * 1e6
        sim_decay_abs = np.abs(sim_meta['decay'])
        
        ax1.loglog(measured_time_us, measured_decay_abs, 
                  color='red', linewidth=2.5, marker='o', markersize=4,
                  label=f'Measured: {measured_label}', alpha=0.8)
        ax1.loglog(sim_time_us, sim_decay_abs, 
                  color='blue', linewidth=2.5, marker='s', markersize=4,
                  label=f'Simulated: {sim_meta["label"]}', alpha=0.8)
        
        ax1.set_xlabel('Time [μs]', fontsize=18)
        ax1.set_ylabel(r'$\left|\frac{\partial B_z}{\partial t}\right|$ [T/s]', fontsize=18)
        ax1.tick_params(labelsize=18)
        ax1.legend(fontsize=16, loc='best')
        ax1.grid(True, alpha=0.3, which='both')
        
        # Right plot: Normalized comparison (if time arrays are compatible)
        if len(measured_time) == len(sim_meta['time']) and np.allclose(measured_time, sim_meta['time'], rtol=1e-3):
            # Same time base - show difference
            difference = measured_decay_abs - sim_decay_abs
            relative_diff = (difference / sim_decay_abs) * 100  # Percentage difference
            
            ax2.semilogx(measured_time_us, relative_diff, 
                        color='green', linewidth=2.5, marker='o', markersize=4)
            ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax2.set_xlabel('Time [μs]', fontsize=18)
            ax2.set_ylabel('Relative Difference [%]', fontsize=18)
            ax2.tick_params(labelsize=18)
            ax2.grid(True, alpha=0.3, which='both')
        else:
            # Different time bases - show normalized overlay
            measured_norm = measured_decay_abs / np.max(measured_decay_abs)
            sim_norm = sim_decay_abs / np.max(sim_decay_abs)
            
            ax2.semilogx(measured_time_us, measured_norm, 
                        color='red', linewidth=2.5, marker='o', markersize=4,
                        label='Measured (normalized)', alpha=0.8)
            ax2.semilogx(sim_time_us, sim_norm, 
                        color='blue', linewidth=2.5, marker='s', markersize=4,
                        label='Simulated (normalized)', alpha=0.8)
            ax2.set_xlabel('Time [μs]', fontsize=18)
            ax2.set_ylabel('Normalized Amplitude', fontsize=18)
            ax2.tick_params(labelsize=18)
            ax2.legend(fontsize=16, loc='best')
            ax2.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.show()
        
        # Print comparison statistics
        print(f"\n{'='*70}")
        print("Comparison Statistics:")
        print(f"{'='*70}")
        print(f"Measured:")
        print(f"  Time range: {measured_time[0]*1e6:.3f} - {measured_time[-1]*1e6:.3f} μs")
        print(f"  Amplitude range: {np.min(measured_decay_abs):.3e} - {np.max(measured_decay_abs):.3e} T/s")
        print(f"\nSimulated:")
        print(f"  Time range: {sim_meta['time'][0]*1e6:.3f} - {sim_meta['time'][-1]*1e6:.3f} μs")
        print(f"  Amplitude range: {np.min(sim_decay_abs):.3e} - {np.max(sim_decay_abs):.3e} T/s")
    
    def plot_multiple_measured_and_simulated(self):
        """
        Plot multiple measured decay curves alongside multiple simulated curves.
        """
        print("\n" + "="*70)
        print("Plot Multiple Measured & Simulated Curves")
        print("="*70)
        
        # Initialize lists to store data
        measured_data = []
        simulated_data = []
        
        # === Load Measured Data ===
        print("\n" + "-"*70)
        print("MEASURED DATA")
        print("-"*70)
        
        while True:
            add_measured = input("\nAdd measured data file? (y/n) [y]: ").strip().lower() or 'y'
            if add_measured != 'y':
                break
            
            measured_file = input("Enter path to measured HDF5 file: ").strip()
            
            import os
            if not os.path.exists(measured_file):
                print(f"Error: File not found: {measured_file}")
                continue
            
            try:
                with h5py.File(measured_file, 'r') as f:
                    num_measured = f['metadata'].attrs['num_simulations']
                    print(f"File contains {num_measured} measurement(s)")
                    
                    if num_measured > 1:
                        print("\nAvailable measurements:")
                        for i in range(num_measured):
                            sim = f[f'simulations/simulation_{i}']
                            label = sim.attrs.get('label', f'measurement_{i}')
                            print(f"  [{i}] {label}")
                        
                        indices_str = input("Enter measurement indices (comma-separated, or 'all'): ").strip()
                        if indices_str.lower() == 'all':
                            indices = list(range(num_measured))
                        else:
                            indices = [int(x.strip()) for x in indices_str.split(',')]
                    else:
                        indices = [0]
                    
                    # Load selected measurements
                    for idx in indices:
                        sim = f[f'simulations/simulation_{idx}']
                        time = sim['time'][:]
                        decay = sim['decay'][:]
                        label = sim.attrs.get('label', f'Measured_{len(measured_data)+1}')
                        
                        measured_data.append({
                            'time': time,
                            'decay': decay,
                            'label': label,
                            'file': os.path.basename(measured_file)
                        })
                        print(f"  ✓ Added: {label}")
            
            except Exception as e:
                print(f"Error reading file: {e}")
                continue
        
        if not measured_data:
            print("\nNo measured data loaded.")
        
        # === Select Simulated Data ===
        print("\n" + "-"*70)
        print("SIMULATED DATA")
        print("-"*70)
        
        if self.num_sims > 0:
            print(f"\nCurrently loaded dataset has {self.num_sims} simulations")
            add_sim = input("Add simulations from loaded dataset? (y/n) [y]: ").strip().lower() or 'y'
            
            if add_sim == 'y':
                print("\nAvailable simulations:")
                for idx, meta in enumerate(self.simulations_metadata):
                    if meta['target_present']:
                        print(f"  [{idx+1}] {meta['label']} (Loop @ {meta['loop_z']:.3f}m, Target @ {meta['target_z']:.3f}m)")
                    else:
                        print(f"  [{idx+1}] {meta['label']} (Loop @ {meta['loop_z']:.3f}m, No Target)")
                
                print("\nSelect simulations:")
                print("  - Enter indices (comma-separated, e.g., '1,3,5')")
                print("  - Enter 'all' for all simulations")
                print("  - Enter 'skip' to skip")
                
                sim_input = input("Your selection: ").strip().lower()
                
                if sim_input != 'skip':
                    if sim_input == 'all':
                        selected_indices = list(range(self.num_sims))
                    else:
                        try:
                            parts = [p.strip() for p in sim_input.split(',')]
                            selected_indices = [int(p) - 1 for p in parts if p]
                        except ValueError:
                            print("Invalid input, skipping simulations.")
                            selected_indices = []
                    
                    for idx in selected_indices:
                        if 0 <= idx < self.num_sims:
                            meta = self.simulations_metadata[idx]
                            simulated_data.append({
                                'time': meta['time'],
                                'decay': meta['decay'],
                                'label': meta['label'],
                                'target_present': meta['target_present']
                            })
                            print(f"  ✓ Added: {meta['label']}")
        else:
            print("\nNo simulated dataset loaded.")
        
        # === Create Plot ===
        if not measured_data and not simulated_data:
            print("\nNo data selected for plotting.")
            return
        
        print(f"\n{'='*70}")
        print(f"Creating plot with {len(measured_data)} measured + {len(simulated_data)} simulated curves")
        print(f"{'='*70}")
        
        # Ask about normalization
        normalize = input("\nApply baseline normalization (shift minimum to zero)? (y/n) [n]: ").strip().lower() or 'n'
        
        # Calculate global minimum if normalizing
        global_min = None
        if normalize == 'y':
            all_mins = []
            for data in measured_data:
                all_mins.append(np.min(np.abs(data['decay'])))
            for data in simulated_data:
                all_mins.append(np.min(np.abs(data['decay'])))
            global_min = min(all_mins)
            print(f"  Global minimum value: {global_min:.3e} T/s")
            print(f"  All curves will be shifted by -{global_min:.3e}")
        
        fig, ax = plt.subplots(figsize=(16, 9))
        
        # Plot measured data with solid lines
        measured_colors = plt.cm.Reds(np.linspace(0.4, 0.9, max(len(measured_data), 1)))
        for i, data in enumerate(measured_data):
            time_us = data['time'] * 1e6
            decay_abs = np.abs(data['decay'])
            
            # Apply normalization if requested
            if normalize == 'y':
                decay_abs = decay_abs - global_min
                # Avoid log(0) by replacing zeros with small positive value
                decay_abs = np.where(decay_abs <= 0, 1e-20, decay_abs)
            
            ax.loglog(time_us, decay_abs, 
                     color=measured_colors[i], linewidth=2.5, 
                     marker='o', markersize=5, markevery=max(len(time_us)//20, 1),
                     label=f"[M] {data['label']}", alpha=0.8, linestyle='-')
        
        # Plot simulated data with dashed lines
        sim_colors = plt.cm.Blues(np.linspace(0.4, 0.9, max(len(simulated_data), 1)))
        for i, data in enumerate(simulated_data):
            time_us = data['time'] * 1e6
            decay_abs = np.abs(data['decay'])
            
            # Apply normalization if requested
            if normalize == 'y':
                decay_abs = decay_abs - global_min
                # Avoid log(0) by replacing zeros with small positive value
                decay_abs = np.where(decay_abs <= 0, 1e-20, decay_abs)
            
            ax.loglog(time_us, decay_abs, 
                     color=sim_colors[i], linewidth=2.5,
                     marker='s', markersize=5, markevery=max(len(time_us)//20, 1),
                     label=f"[S] {data['label']}", alpha=0.8, linestyle='--')
        
        ax.set_xlabel('Time [μs]', fontsize=18)
        
        # Update y-label based on normalization
        if normalize == 'y':
            ax.set_ylabel(r'$\left|\frac{\partial B_z}{\partial t}\right|$ - min [T/s]', fontsize=18)
        else:
            ax.set_ylabel(r'$\left|\frac{\partial B_z}{\partial t}\right|$ [T/s]', fontsize=18)
        ax.tick_params(labelsize=18)
        ax.grid(True, alpha=0.3, which='both')
        
        # Smart legend positioning
        total_items = len(measured_data) + len(simulated_data)
        if total_items > 15:
            ncol = 3
            loc = 'upper center'
            bbox = (0.5, -0.08)
            plt.subplots_adjust(bottom=0.2)
        elif total_items > 8:
            ncol = 2
            loc = 'upper right'
            bbox = None
        else:
            ncol = 1
            loc = 'best'
            bbox = None
        
        if bbox:
            ax.legend(fontsize=16, ncol=ncol, loc=loc, bbox_to_anchor=bbox, framealpha=0.9)
        else:
            ax.legend(fontsize=16, ncol=ncol, loc=loc, framealpha=0.9)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\n{'='*70}")
        print("Plot Summary:")
        print(f"{'='*70}")
        print(f"Measured curves: {len(measured_data)}")
        for data in measured_data:
            print(f"  • {data['label']} ({len(data['time'])} points)")
        print(f"\nSimulated curves: {len(simulated_data)}")
        for data in simulated_data:
            print(f"  • {data['label']} ({len(data['time'])} points)")