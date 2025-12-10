import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import ArrowStyle
import matplotlib.patches as mpatches
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
from simpeg import maps
from discretize import CylindricalMesh
import h5py
from sklearn.metrics import roc_curve, auc

class LogitTransform(mtransforms.Transform):
    input_dims = output_dims = 1

    def transform_non_affine(self, a):
        a = np.clip(a, 1e-5, 1-1e-5)
        return np.log(a / (1 - a))

    def inverted(self):
        return InvertedLogitTransform()

class InvertedLogitTransform(mtransforms.Transform):
    input_dims = output_dims = 1

    def transform_non_affine(self, a):
        return 1 / (1 + np.exp(-a))

    def inverted(self):
        return LogitTransform()

class LogitScale(mscale.ScaleBase):
    name = 'logit_custom'

    def __init__(self, axis, **kwargs):
        super().__init__(axis)

    def get_transform(self):
        return LogitTransform()

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(FixedLocator([0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]))
        axis.set_minor_locator(NullLocator())
        axis.set_major_formatter(ScalarFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(vmin, 1e-5), min(vmax, 1-1e-5)

mscale.register_scale(LogitScale)

class BasePlotter:
    def __init__(self):
        pass

class PiemsolPlotter(BasePlotter):
    def __init__(self, cfg, simulator):
        self.cfg = cfg
        self.simulator = simulator
        self.hdf5_file = None
        self.num_sims = 0
        self.num_present = 0
        self.num_absent = 0
        self.simulations_metadata = []
        self.mesh = None
        self.ind_active = None
        self.model_no_target = None
        print(f"PiemsolPlotter initialized. Use load_from_hdf5() to load data.")
    
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
            self.skipped_indices = []
            for i in range(self.num_sims):
                try:
                    sim_group = f[f'simulations/simulation_{i}']
                    metadata = {
                        'index': i,
                        'loop_z': sim_group.attrs['loop_z'],
                        'target_type': sim_group.attrs.get('target_type', -1),
                        'target_z': sim_group.attrs.get('target_z', None),
                        'target_conductivity': sim_group.attrs.get('target_conductivity', self.cfg.aluminum_conductivity),
                        'target_present': sim_group.attrs['target_present'],
                        'label': sim_group.attrs['label'],
                        'time': sim_group['time'][:],
                        'decay': sim_group['decay'][:]
                    }
                    if metadata['target_z'] == -999.0:
                        metadata['target_z'] = None
                    self.simulations_metadata.append(metadata)
                except (OSError, KeyError) as e:
                    self.skipped_indices.append(i)
                    print(f"Warning: SkipPiemsolng missing/corrupted simulation_{i}: {e}")
                    continue
            
            if self.skipped_indices:
                print(f"\nWarning: Skipped {len(self.skipped_indices)} corrupted simulations: {self.skipped_indices}")
    
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
                plt.plot(time_us, decay, color=colors[i], 
                          linewidth=2, label=f"{label_prefix} {label}")
        
        plt.xlabel('Time [μs]', fontsize=20)
        plt.tick_params(labelsize=20)
        plt.legend(fontsize=20, bbox_to_anchor=(1.05, 1), loc='upper left')
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
            selected_indices = list(range(len(self.simulations_metadata)))
        else:
            try:
                parts = [p.strip() for p in user_input.split(',')]
                for part in parts:
                    idx = int(part) - 1
                    if 0 <= idx < len(self.simulations_metadata):
                        selected_indices.append(idx)
                    else:
                        print(f"Warning: Index {part} out of range, skipPiemsolng.")
            except ValueError:
                print("Invalid input format.")
                return
        
        if not selected_indices:
            print("No valid simulations selected.")
            return
        
        print(f"\nPlotting {len(selected_indices)} simulation(s)...")
        
        fig_model = plt.figure()
        ax_model = fig_model.add_subplot(111)
        
        r = self.mesh.cell_centers[self.ind_active, 0]
        z = self.mesh.cell_centers[self.ind_active, 2]
        
        ax_model.axhline(0, color='darkgreen', linewidth=2, linestyle='--', alpha=0.7, label='Ground level')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_indices)))
        
        for plot_idx, sim_idx in enumerate(selected_indices):
            meta = self.simulations_metadata[sim_idx]
            color = colors[plot_idx]
            meta['label'] = meta['label'].replace(' (ABOVE ground)', '').replace(' (BELOW ground)', '')
            
            if meta['target_z'] is not None:
                target_label = f"Target at {meta['target_z']:.3f}m"
                ax_model.scatter([self.cfg.target_radius/2], [meta['target_z']], 
                        c=[color], s=250, alpha=0.9,
                        marker='X', edgecolors='black', linewidths=2.0,
                        label=target_label)
            coil_label = f"Coil at {meta['loop_z']:.3f}m"
            loop_radius = float(self.cfg.tx_radius)
            ax_model.plot([0, loop_radius], [meta['loop_z'], meta['loop_z']],
                 color=color, linewidth=4, marker='o', markersize=10,
                 label=coil_label, markerfacecolor=color, markeredgecolor='black',
                 markeredgewidth=1.5)
        
        ax_model.set_xlabel('Radial distance [m]', fontsize=20)
        ax_model.set_ylabel('Depth [m]', fontsize=20)
        ax_model.tick_params(labelsize=20)
        ax_model.grid(True, alpha=0.3)
        ax_model.legend(fontsize=16, loc='best', framealpha=0.9)
        ax_model.set_ylim(-0.45,0.7)
        fig_model.tight_layout()

        fig_decay = plt.figure()
        ax_decay = fig_decay.add_subplot(111)
        
        for plot_idx, sim_idx in enumerate(selected_indices):
            meta = self.simulations_metadata[sim_idx]
            color = colors[plot_idx]
            
            time_us = meta['time'] * 1e6
            decay = np.abs(meta['decay'])
            
            ax_decay.loglog(time_us, decay, color=color, linewidth=2.5, alpha=0.9,
                   label=f"{meta['label']}", marker='.')
        
        ax_decay.set_xlabel('Time [μs]', fontsize=20)
        ax_decay.set_ylabel(r'$\frac{\partial B_{z}}{\partial t}$ [T/s]', fontsize=20)
        ax_decay.tick_params(labelsize=20)
        ax_decay.grid(True, alpha=0.3, which='both')
        ax_decay.legend(fontsize=16, loc='best', framealpha=0.9)    
        plt.show()
    
    def plot_conductivity_models(self):
        if not self.simulations_metadata:
            print("\\nNo data loaded. Run simulations first.")
            return
        
        available_sims = []
        type_names = ['No Target', 'Hollow Cylinder', 'Shredded Can', 'Solid Block']
        
        for metadata in self.simulations_metadata:
            target_type = metadata['target_type']
            target_conductivity = metadata.get('target_conductivity', self.cfg.aluminum_conductivity)
            target_z = metadata['target_z']
            loop_z = metadata['loop_z']
            
            if target_type > 0:
                status = f"Type: {type_names[target_type]}, σ={target_conductivity:.1e} S/m, z={target_z:.3f}m"
            else:
                status = f"Type: {type_names[0]} (soil + air only)"
                target_type = 0  # Ensure it's 0 for no target
                target_z = None
            
            available_sims.append((metadata['index'], loop_z, status, target_type, target_conductivity, target_z))
        
        if not available_sims:
            print("\\nNo simulations found.")
            return
        
        print(f"\\nAvailable simulations:")
        for idx, (_, _, status, _, _, _) in enumerate(available_sims[:20]):
            print(f"  [{idx+1:2d}] {status}")
        
        try:
            choice = input(f"\nSelect simulation [1-{min(20, len(available_sims))}]: ").strip()
            sim_data = available_sims[int(choice) - 1]
            _, loop_z, status, target_type, target_conductivity, target_z = sim_data
            
            if target_type > 0:
                print(f"\\nGenerating model: type={target_type}, σ={target_conductivity:.1e}, z={target_z:.3f}m")
            else:
                print(f"\\nGenerating model: No target (soil + air only)")

            model, _ = self.simulator.create_conductivity_model(self.mesh, target_z, target_type, target_conductivity)
            
            if model is None:
                print("Error generating model.")
                return
                
            self._plot_model_2d(model, status, self.mesh, self.ind_active, loop_z=loop_z)
            
        except (ValueError, IndexError) as e:
            print(f"Invalid selection: {e}")
    
    def _plot_model_2d(self, model, title, mesh=None, ind_active=None, loop_z=None):
        if mesh is None:
            mesh = self.mesh
        if ind_active is None:
            ind_active = self.ind_active
        if loop_z is None:
            loop_z = self.cfg.loop_z
            
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        num_target_cells = np.sum(model > self.cfg.soil_conductivity * 10)
        print(f"Plotting model with {num_target_cells} target cells")
        
        plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
        log_model = np.log10(model)
        
        im = mesh.plot_image(plotting_map * log_model, ax=ax, grid=True,
                            clim=(np.log10(self.cfg.air_conductivity), 
                                  np.log10(self.cfg.aluminum_conductivity)))
        
        cbar = plt.colorbar(im[0], ax=ax, pad=0.02)
        cbar.set_label(f'Logarithmic conductivity', fontsize=20)
        cbar.ax.tick_params(labelsize=18)
        
        ax.axhline(y=0, color='green', linestyle='--', linewidth=3, alpha=0.8, label='Ground level')
        
        loop_radius = float(self.cfg.tx_radius)
        ax.plot([0, loop_radius], [loop_z, loop_z], 
               color='blue', linewidth=2.5, marker='o', markersize=8,
               label=f'TX/RX coil (z={loop_z:.3f}m)')
        
        ax.set_xlabel('Radial distance [m]', fontsize=20)
        ax.set_ylabel('Elevation [m]', fontsize=20)
        ax.tick_params(labelsize=20)
        ax.legend(fontsize=18)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 0.5])
        ax.set_ylim([-0.5, 0.7])
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)

class ClassifierPlotter(BasePlotter):
    def __init__(self):
        super().__init__()
        self.model = None
    
    def set_model(self, model):
        self.model = model
    
    def plot_training_history(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(history.history['accuracy'], label='Training accuracy', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Validation accuracy', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=20)
        ax1.set_ylabel('Accuracy', fontsize=20)
        ax1.legend(fontsize=20)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=20)
        
        ax2.plot(history.history['loss'], label='Training loss', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Validation loss', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=20)
        ax2.set_ylabel('Loss', fontsize=20)
        ax2.legend(fontsize=20)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=20)
        
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, X_test, y_test, normalize=False):
        from sklearn.metrics import confusion_matrix, classification_report
        import seaborn as sns
        
        if self.model is None:
            print("Error: No model set! Use set_model() first.")
            return
        
        print("\n" + "="*70)
        print("Confusion Matrix Analysis")
        print("="*70)
        
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        cm = confusion_matrix(y_test, y_pred)
        cm = cm.T  # Transpose so rows=predicted, cols=true
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]  # Normalize along columns (true labels)
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Greys', square=True,
               xticklabels=['$C_A$', '$C_P$'],
               yticklabels=['$C_A$', '$C_P$'],
               cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
               ax=ax, annot_kws={'size': 20})
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        
        ax.set_xlabel('True class', fontsize=20)
        ax.set_ylabel('Predicted class', fontsize=20)
        
        plt.tight_layout()
        plt.show()
        print("\nClassification Report:")
        print("="*70)
        target_names = ['$C_A$', '$C_P$']
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        return cm
    
    def plot_roc_curve(self, X_test, y_test):
        if self.model is None:
            print("Error: No model set!")
            return
        
        print("\n" + "="*70)
        print("ROC Curve Analysis")
        print("="*70)
        
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred_proba_target = y_pred_proba[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_target)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr, tpr, color='darkorange', lw=3, 
               label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Probability of false alarm ($P_{FA}$)', fontsize=20)
        ax.set_ylabel('Probability of detection ($P_{D}$)', fontsize=20)
        ax.legend(loc="lower right", fontsize=20)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=20)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n✓ ROC AUC Score: {roc_auc:.4f}")
        
        return fpr, tpr, roc_auc
    
    def plot_prediction_distribution(self, X_test, y_test):
        if self.model is None:
            print("Error: No model set!")
            return
        
        print("\n" + "="*70)
        print("Prediction Probability Distribution")
        print("="*70)
        
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred_proba_target = y_pred_proba[:, 1]
        
        target_present_probs = y_pred_proba_target[y_test == 1]
        target_absent_probs = y_pred_proba_target[y_test == 0]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.hist(target_present_probs, bins=30, alpha=0.6, color='green', 
                label='$C_P$', edgecolor='black')
        ax1.hist(target_absent_probs, bins=30, alpha=0.6, color='blue', 
                label='$C_A$', edgecolor='black')
        ax1.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision threshold')
        ax1.set_xlabel('Predicted probability ($C_P$)', fontsize=20)
        ax1.set_ylabel('Frequency', fontsize=20)
        ax1.legend(fontsize=20)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=20)
        
        ax2.set_yscale('logit_custom')
        data_to_plot = [target_absent_probs, target_present_probs]
        
        positions = [1, 2]
        colors_dist = ['blue','orange']
        
        for i, (data, pos, color) in enumerate(zip(data_to_plot, positions, colors_dist)):
            if len(data) == 0:
                continue
            jitter = np.random.normal(0, 0.12, len(data))
            x_pos = np.full(len(data), pos) + jitter
            if color == 'orange':
                alpha_val = 0.15
            else:
                alpha_val = 0.1
            ax2.scatter(x_pos, data, marker='o', color=color, alpha=alpha_val, s=20, zorder=1)
        
        bp = ax2.boxplot(data_to_plot, positions=positions, widths=0.3, 
                        patch_artist=True, showfliers=False, zorder=2,
                        showmeans=True, meanprops=dict(marker='D', markerfacecolor='black', 
                                                        markeredgecolor='black', markersize=10))
        
        for patch in bp['boxes']:
            patch.set_facecolor('none')  # Transparent fill
            patch.set_edgecolor('gray')
            patch.set_linewidth(2)
        
        for whisker in bp['whiskers']:
            whisker.set_color('gray')
            whisker.set_linewidth(2)
            whisker.set_linestyle('-')
        
        for cap in bp['caps']:
            cap.set_color('gray')
            cap.set_linewidth(2)
        
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(3)
        
        ax2.set_xlim(0.5, 2.5)
        ax2.set_xticks(positions)
        ax2.set_xticklabels(['$C_A$', '$C_P$'], fontsize=20)
        ax2.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Decision threshold')
        ax2.set_ylabel('Predicted probability ($C_P$)', fontsize=20)
        ax2.set_xlabel('True class', fontsize=20)

        ax2.set_ylim(1e-5, 1-1e-5)
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='D', color='w', markerfacecolor='black', markersize=8, label='Mean'),
            Line2D([0], [0], linestyle='-', color='black', linewidth=3, label='Median'),
            Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Decision threshold')
        ]
        ax2.legend(handles=legend_elements, fontsize=16, loc='upper left')
        
        ax2.grid(True, alpha=0.3, which='both')
        ax2.tick_params(labelsize=20)
        
        plt.tight_layout()
        plt.show()

    def plot_roc_pfa_pd(self, X_test, y_test, snr_db=None, num_thresholds=100):
        if self.model is None:
            print("Error: No model set!")
            return None
        
        print("\n" + "="*70)
        print("ROC Curve: Pd vs Pfa Analysis")
        print("="*70)
        
        if snr_db is not None:
            print(f"SNR: {snr_db} dB")
        
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred_proba_target = y_pred_proba[:, 1]
        
        pfa, pd, thresholds = roc_curve(y_test, y_pred_proba_target)
        roc_auc = auc(pfa, pd)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        ax1 = axes[0]
        ax1.plot(pfa, pd, color='darkorange', lw=3, 
                label=f'ROC (AUC = {roc_auc:.4f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        pfa_markers = [0.001, 0.01, 0.05, 0.1]
        for target_pfa in pfa_markers:
            idx = np.argmin(np.abs(pfa - target_pfa))
            if pfa[idx] < 0.5:
                ax1.scatter(pfa[idx], pd[idx], s=100, zorder=5)
                ax1.annotate(f'Pfa={pfa[idx]:.3f}\nPd={pd[idx]:.3f}', 
                           xy=(pfa[idx], pd[idx]), 
                           xytext=(pfa[idx]+0.05, pd[idx]-0.1),
                           fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))
        
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('Probability of False Alarm (Pfa)', fontsize=20)
        ax1.set_ylabel('Probability of Detection (Pd)', fontsize=20)
        ax1.legend(loc="lower right", fontsize=20)
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        valid_mask = pfa > 0
        ax2.semilogx(pfa[valid_mask], pd[valid_mask], color='darkorange', lw=3,
                    label=f'ROC (AUC = {roc_auc:.4f})')
        
        ax2.set_xlim([1e-4, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Probability of False Alarm (Pfa) - Log Scale', fontsize=20)
        ax2.set_ylabel('Probability of Detection (Pd)', fontsize=20)
        ax2.legend(loc="lower right", fontsize=20)
        ax2.grid(True, alpha=0.3, which='both')
        
        ax3 = axes[2]
        
        pfa_eval_points = np.logspace(-4, 0, 50)
        pd_at_pfa = np.interp(pfa_eval_points, pfa, pd)
        
        ax3.semilogx(pfa_eval_points, pd_at_pfa, 'b-', lw=2, marker='o', markersize=4)
        ax3.axhline(0.9, color='green', linestyle='--', lw=2, label='Pd = 0.9 target')
        ax3.axhline(0.95, color='orange', linestyle='--', lw=2, label='Pd = 0.95 target')
        
        pd_targets = [0.9, 0.95, 0.99]
        for pd_target in pd_targets:
            idx = np.argmin(np.abs(pd - pd_target))
            pfa_at_pd = pfa[idx]
            if pfa_at_pd > 0:
                ax3.axvline(pfa_at_pd, color='gray', linestyle=':', alpha=0.5)
                print(f"  Pd = {pd_target:.2f} requires Pfa ≤ {pfa_at_pd:.4f} (threshold = {thresholds[idx]:.4f})")
        
        ax3.set_xlim([1e-4, 1.0])
        ax3.set_ylim([0.0, 1.05])
        ax3.set_xlabel('Probability of False Alarm (Pfa)', fontsize=20)
        ax3.set_ylabel('Probability of Detection (Pd)', fontsize=20)
        ax3.legend(loc="lower right", fontsize=20)
        ax3.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.show()
        
        print("\n" + "-"*70)
        print("Performance Summary:")
        print("-"*70)
        print(f"{'Pfa Threshold':<15} {'Pd Achieved':<15} {'Decision Threshold':<20}")
        print("-"*70)
        
        target_pfas = [0.001, 0.01, 0.05, 0.1, 0.2]
        for target_pfa in target_pfas:
            idx = np.argmin(np.abs(pfa - target_pfa))
            print(f"{pfa[idx]:<15.4f} {pd[idx]:<15.4f} {thresholds[idx]:<20.4f}")
        
        print("-"*70)
        print(f"\n✓ ROC AUC Score: {roc_auc:.4f}")
        
        return {
            'pfa': pfa,
            'pd': pd,
            'thresholds': thresholds,
            'auc': roc_auc,
            'snr_db': snr_db
        }
    
    def visualize_training_data(self, time, decay_curves, labels, label_strings, metadata):
        print("\n" + "="*70)
        print("Training Data Visualization")
        print("="*70)
        print(f"Dataset: {metadata['num_simulations']} simulations")
        print(f"  - Time samples: {len(time)}")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        time_us = time * 1e6
        
        ax = axes[0, 0]
        for i, (decay, label, label_str) in enumerate(zip(decay_curves, labels, label_strings)):
            color = 'green' if label == 1 else 'blue'
            alpha = 0.3
            ax.loglog(time_us, decay, color=color, alpha=alpha, linewidth=1)
        
        num_target_present = np.sum(labels == 1)
        num_target_absent = np.sum(labels == 0)
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, label=f'$C_P$ ({num_target_present})'),
            Line2D([0], [0], color='blue', lw=2, label=f'$C_A$ ({num_target_absent})')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=20)
        ax.set_xlabel('Time [μs]', fontsize=20)
        ax.set_ylabel('-dBz/dt [T/s]', fontsize=20)
        ax.grid(True, alpha=0.3, which='both')
        ax.tick_params(labelsize=20)
        
        ax = axes[0, 1]
        target_present_indices = np.where(labels == 1)[0][:5]
        for idx in target_present_indices:
            ax.loglog(time_us, decay_curves[idx], linewidth=2, label=label_strings[idx])
        ax.set_xlabel('Time [μs]', fontsize=20)
        ax.set_ylabel('-dBz/dt [T/s]', fontsize=20)
        ax.legend(fontsize=20)
        ax.grid(True, alpha=0.3, which='both')
        ax.tick_params(labelsize=20)
        
        ax = axes[1, 0]
        target_absent_indices = np.where(labels == 0)[0][:5]
        for idx in target_absent_indices:
            ax.loglog(time_us, decay_curves[idx], linewidth=2, label=label_strings[idx])
        ax.set_xlabel('Time [μs]', fontsize=20)
        ax.set_ylabel('-dBz/dt [T/s]', fontsize=20)
        ax.legend(fontsize=20)
        ax.grid(True, alpha=0.3, which='both')
        ax.tick_params(labelsize=20)
        
        ax = axes[1, 1]
        ax.text(0.5, 0.7, f'Total Samples: {len(decay_curves)}', 
                ha='center', va='center', fontsize=20)
        ax.text(0.5, 0.5, f'$C_P$: {num_target_present}', 
                ha='center', va='center', fontsize=20, color='green')
        ax.text(0.5, 0.3, f'$C_A$: {num_target_absent}', 
                ha='center', va='center', fontsize=20, color='blue')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()

    def plot_model_architecture(self, model, output_path='model_architecture.png'):
        print("\n" + "="*70)
        print("Model Architecture Visualization")
        print("="*70)
        
        if model is None:
            print("Error: No model provided!")
            return None
        
        layers_info = []
        for layer in model.layers:
            layer_type = layer.__class__.__name__
            
            try:
                output_shape = layer.output.shape
                shape_str = str(tuple(output_shape))
            except:
                try:
                    output_shape = layer.output_shape
                    shape_str = str(output_shape)
                except:
                    shape_str = "N/A"
            
            params = layer.count_params()
            
            details = ""
            if 'Conv1D' in layer_type:
                details = f"filters={layer.filters}, k={layer.kernel_size[0]}"
            elif 'MaxPooling' in layer_type:
                details = f"pool={layer.pool_size[0]}"
            elif 'Dense' in layer_type:
                details = f"units={layer.units}"
            elif 'Dropout' in layer_type:
                details = f"rate={layer.rate}"
            elif 'BatchNorm' in layer_type:
                details = ""
            elif 'InputLayer' in layer_type:
                config = layer.get_config()
                if 'batch_input_shape' in config:
                    shape_str = str(config['batch_input_shape'])
                elif 'batch_shape' in config:
                    shape_str = str(config['batch_shape'])
            
            layers_info.append({
                'name': layer.name,
                'type': layer_type,
                'shape': shape_str,
                'params': params,
                'details': details
            })
        
        layer_colors = {
            'InputLayer': '#E3F2FD',
            'Conv1D': '#90CAF9',
            'BatchNormalization': '#CE93D8',
            'MaxPooling1D': '#FFB74D',
            'Dropout': '#EF9A9A',
            'Dense': '#80CBC4',
            'Flatten': '#BCAAA4',
        }
        
        block_layers = {
            1: ['conv1', 'bn1', 'pool1', 'drop1'],
            2: ['conv2', 'bn2', 'pool2', 'drop2'],
            3: ['conv3', 'bn3', 'pool3', 'drop3']
        }
        
        n_layers = len(layers_info)
        box_width = 8
        box_height = 0.55
        y_spacing = 1.0
        block_spacing = 0.6
        x_center = 6
        
        total_height = n_layers * y_spacing
        
        fig_height = max(10, total_height * 0.6 + 1)
        fig, ax = plt.subplots(1, 1, figsize=(10, fig_height))
        
        y_positions = {}
        y_pos = total_height
        current_block = None
        block_y_ranges = {}
        
        for i, layer in enumerate(layers_info):
            layer_name = layer['name']
            new_block = None
            for block_num, block_layer_names in block_layers.items():
                if layer_name in block_layer_names:
                    new_block = block_num
                    break
            
            if new_block and new_block != current_block:
                current_block = new_block
                block_y_ranges[current_block] = {'start': y_pos, 'end': None}
            elif current_block and new_block is None:
                block_y_ranges[current_block]['end'] = y_pos + y_spacing - box_height/2 + 0.1
                current_block = None
            
            y_positions[i] = y_pos
            
            if current_block and new_block:
                block_y_ranges[current_block]['end'] = y_pos - box_height/2 - 0.15
            
            y_pos -= y_spacing
        
        for block_num, y_range in block_y_ranges.items():
            if y_range['start'] is not None and y_range['end'] is not None:
                static_block_height = 3 * y_spacing + box_height + 0.5
                
                block_bottom = y_range['end'] - box_height/2
                block_top = block_bottom + static_block_height
                
                block_left = x_center - box_width/2 - 0.5
                block_right = x_center + box_width/2 + 0.5
                block_width_rect = block_right - block_left
                
                block_rect = FancyBboxPatch(
                    (block_left, block_bottom),
                    block_width_rect, static_block_height,
                    boxstyle="square,pad=0.02",
                    facecolor='#f8f9fa',
                    edgecolor='#495057',
                    linewidth=1.5,
                    linestyle='-',
                    zorder=0
                )
                '''
                ax.add_patch(block_rect)
                
                ax.text(block_left + 0.15, block_bottom + 0.15,
                       f'Block {block_num}',
                       fontsize=9, fontweight='bold', ha='left', va='bottom',
                       color='#495057', fontfamily='sans-serif')
                '''
        for i, layer in enumerate(layers_info):
            y_pos = y_positions[i]
            
            color = layer_colors.get(layer['type'], '#E0E0E0')
            
            box = FancyBboxPatch(
                (x_center - box_width/2, y_pos - box_height/2),
                box_width, box_height,
                boxstyle="square,pad=0.02",
                facecolor=color,
                edgecolor='#37474F',
                linewidth=1.2,
                zorder=2
            )
            ax.add_patch(box)
            
            display_text = f"{layer['name']}({layer['type']})"
            
            ax.text(x_center, y_pos,
                   display_text,
                   fontsize=15, fontweight='bold', ha='center', va='center',
                   fontfamily='sans-serif', color='#1a1a2e', zorder=3)
            
            if i < n_layers - 1:
                next_y = y_positions[i + 1]
                arrow = FancyArrowPatch(
                    (x_center, y_pos - box_height/2 - 0.03),
                    (x_center, next_y + box_height/2 + 0.03),
                    arrowstyle=ArrowStyle('->', head_length=5, head_width=3),
                    color='#546E7A',
                    linewidth=1.2,
                    zorder=1
                )
                ax.add_patch(arrow)
        
        min_y = min(y_positions.values()) - box_height/2 - 0.3
        max_y = max(y_positions.values()) + box_height/2 + 0.3
        ax.set_xlim(x_center - box_width/2 - 1.0, x_center + box_width/2 + 1.0)
        ax.set_ylim(min_y, max_y)
        ax.axis('off')
        
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"\n✓ Model architecture saved to: {output_path}")
        
        plt.show()
        
        return output_path

    def plot_roc_multi_snr(self, model, logger, dataset_paths, snr_values):
        from sklearn.metrics import roc_curve, auc
        
        if model is None:
            print("Error: No model provided!")
            return None
        
        if len(dataset_paths) != len(snr_values):
            raise ValueError("Number of dataset paths must match number of SNR values")
        
        print("\n" + "="*70)
        print("Multi-SNR ROC Analysis")
        print("="*70)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(snr_values)))
        results = {}
        
        for i, (path, snr, color) in enumerate(zip(dataset_paths, snr_values, colors)):
            print(f"\nProcessing SNR = {snr} dB...")
            
            time, decay_curves, labels, label_strings, metadata = logger.load_from_hdf5(path)
            
            X = decay_curves.reshape(len(decay_curves), decay_curves.shape[1], 1)
            y = labels
            
            y_pred_proba = model.predict(X, verbose=0)
            y_pred_proba_target = y_pred_proba[:, 1]
            
            pfa, pd, thresholds = roc_curve(y, y_pred_proba_target)
            roc_auc = auc(pfa, pd)
            
            results[snr] = {
                'pfa': pfa,
                'pd': pd,
                'thresholds': thresholds,
                'auc': roc_auc
            }
            
            print(f"  AUC = {roc_auc:.4f}")
            
            axes[0].plot(pfa, pd, color=color, lw=2, 
                        label=f'SNR = {snr} dB (AUC = {roc_auc:.3f})')
            
            valid_mask = pfa > 0
            axes[1].semilogx(pfa[valid_mask], pd[valid_mask], color=color, lw=2,
                           label=f'SNR = {snr} dB')
        
        axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        axes[0].set_xlabel('Probability of False Alarm (Pfa)', fontsize=20)
        axes[0].set_ylabel('Probability of Detection (Pd)', fontsize=20)
        axes[0].legend(loc='lower right', fontsize=20)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([0, 1])
        axes[0].set_ylim([0, 1.05])
        axes[0].tick_params(labelsize=20)
        
        axes[1].set_xlabel('Probability of False Alarm (Pfa) - Log Scale', fontsize=20)
        axes[1].set_ylabel('Probability of Detection (Pd)', fontsize=20)
        axes[1].legend(loc='lower right', fontsize=20)
        axes[1].grid(True, alpha=0.3, which='both')
        axes[1].set_xlim([1e-4, 1])
        axes[1].set_ylim([0, 1.05])
        axes[1].tick_params(labelsize=20)
        
        plt.tight_layout()
        plt.show()
        
        return results
    
    def plot_roc_multi_snr_noise(self, model, time, X_test, y_test, snr_values, late_time, 
                                  conditioner=None, use_quantized=False, tflite_predict_fn=None):
        if model is None and not use_quantized:
            print("Error: No model provided!")
            return None
        
        if use_quantized and tflite_predict_fn is None:
            print("Error: TFLite predict function required for quantized model!")
            return None
        
        print("\n" + "="*70)
        print("Multi-SNR ROC Analysis (In-Memory Noise Addition)")
        print("="*70)
        print(f"Test samples: {len(X_test)}")
        print(f"SNR values: {snr_values} dB")
        print(f"Using quantized model: {use_quantized}")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(snr_values)))
        results = {}
        
        if X_test.ndim == 3:
            decay_curves_orig = X_test.squeeze(axis=-1)  # Remove channel dim
        else:
            decay_curves_orig = X_test
        
        print(f"\nProcessing Original (no noise)...")
        
        X_noisy = decay_curves_orig.reshape(len(decay_curves_orig), -1, 1)
        
        if use_quantized:
            y_pred_proba = tflite_predict_fn(X_noisy)
        else:
            y_pred_proba = model.predict(X_noisy, verbose=0)
        
        if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
            y_pred_proba_target = y_pred_proba[:, 1]
        else:
            y_pred_proba_target = y_pred_proba.flatten()
        
        pfa, pd, thresholds = roc_curve(y_test, y_pred_proba_target)
        roc_auc = auc(pfa, pd)
        
        results['original'] = {
            'pfa': pfa,
            'pd': pd,
            'thresholds': thresholds,
            'auc': roc_auc
        }
        
        print(f"  AUC = {roc_auc:.4f}")
        
        ax.plot(pfa, pd, color='black', lw=3, 
                label=f'No noise (AUC = {roc_auc:.3f})')
        
        for i, (snr, color) in enumerate(zip(snr_values, colors)):
            print(f"\nProcessing SNR = {snr} dB...")
            
            decay_curves_noisy = conditioner.add_noise(time, decay_curves_orig, late_time, snr)
        
            X_noisy = decay_curves_noisy.reshape(len(decay_curves_noisy), -1, 1)
            
            if use_quantized:
                y_pred_proba = tflite_predict_fn(X_noisy)
            else:
                y_pred_proba = model.predict(X_noisy, verbose=0)
            
            if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                y_pred_proba_target = y_pred_proba[:, 1]
            else:
                y_pred_proba_target = y_pred_proba.flatten()
            
            pfa, pd, thresholds = roc_curve(y_test, y_pred_proba_target)
            roc_auc = auc(pfa, pd)
            
            results[snr] = {
                'pfa': pfa,
                'pd': pd,
                'thresholds': thresholds,
                'auc': roc_auc
            }
            
            print(f"  AUC = {roc_auc:.4f}")
            
            ax.plot(pfa, pd, color=color, lw=2, 
                        label=f'SNR = {snr} dB (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        ax.set_xlabel('Probability of false alarm ($P_{FA}$)', fontsize=20)
        ax.set_ylabel('Probability of detection ($P_{D}$)', fontsize=20)
        ax.legend(loc='lower right', fontsize=20)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.tick_params(labelsize=20)
        
        plt.tight_layout()
        plt.show()
        
        print("\n" + "="*70)
        print("Summary")
        print("="*70)
        print(f"{'SNR (dB)':<12} {'AUC':<10}")
        print("-"*22)
        if 'original' in results:
            print(f"{'Original':<12} {results['original']['auc']:.4f}")
        for snr in snr_values:
            print(f"{snr:<12} {results[snr]['auc']:.4f}")
        
        return results

    def plot_multi_snr_confusion_and_distribution(self, model, time, X_test, y_test, snr_values, late_time,
                                                   conditioner=None, use_quantized=False, tflite_predict_fn=None):
        
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        if model is None and not use_quantized:
            print("Error: No model provided!")
            return None
        
        if use_quantized and tflite_predict_fn is None:
            print("Error: TFLite predict function required for quantized model!")
            return None
        
        print("\n" + "="*70)
        print("Multi-SNR Confusion Matrix and Prediction Distribution")
        print("="*70)
        print(f"Test samples: {len(X_test)}")
        print(f"SNR values: {snr_values} dB")
        print(f"Using quantized model: {use_quantized}")
        
        if X_test.ndim == 3:
            decay_curves_orig = X_test.squeeze(axis=-1)
        else:
            decay_curves_orig = X_test
        
        all_snr_labels = ['Original'] + [f'{snr} dB' for snr in snr_values]
        n_plots = len(all_snr_labels)
        
        n_cols = min(4, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig_cm, axes_cm = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_plots == 1:
            axes_cm = np.array([[axes_cm]])
        elif n_rows == 1:
            axes_cm = axes_cm.reshape(1, -1)
        elif n_cols == 1:
            axes_cm = axes_cm.reshape(-1, 1)
        
        fig_dist, axes_dist = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_plots == 1:
            axes_dist = np.array([[axes_dist]])
        elif n_rows == 1:
            axes_dist = axes_dist.reshape(1, -1)
        elif n_cols == 1:
            axes_dist = axes_dist.reshape(-1, 1)
        
        results = {}
        
        print(f"\nProcessing Original (no noise)...")
        X_input = decay_curves_orig.reshape(len(decay_curves_orig), -1, 1)
        
        if use_quantized:
            y_pred_proba = tflite_predict_fn(X_input)
        else:
            y_pred_proba = model.predict(X_input, verbose=0)
        
        if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
            y_pred_proba_target = y_pred_proba[:, 1]
        else:
            y_pred_proba_target = y_pred_proba.flatten()
        
        y_pred = (y_pred_proba_target >= 0.5).astype(int)
        accuracy = np.mean(y_pred == y_test)
        results['original'] = {'accuracy': accuracy, 'y_pred_proba': y_pred_proba_target}
        print(f"  Accuracy = {accuracy:.4f}")
        
        row, col = 0, 0
        cm = confusion_matrix(y_test, y_pred)
        cm = cm.T  # Transpose so rows=predicted, cols=true
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greys', square=True,
                   xticklabels=['$C_A$', '$C_P$'], yticklabels=['$C_A$', '$C_P$'],
                   ax=axes_cm[row, col], annot_kws={'size': 12}, cbar=False)
        axes_cm[row, col].set_title(f'Original\nAcc={accuracy:.3f}', fontsize=12)
        axes_cm[row, col].set_xlabel('True', fontsize=10)
        axes_cm[row, col].set_ylabel('Predicted', fontsize=10)
        
        target_present_probs = y_pred_proba_target[y_test == 1]
        target_absent_probs = y_pred_proba_target[y_test == 0]
        axes_dist[row, col].hist(target_present_probs, bins=20, alpha=0.6, color='green', label='$C_P$', edgecolor='black')
        axes_dist[row, col].hist(target_absent_probs, bins=20, alpha=0.6, color='blue', label='$C_A$', edgecolor='black')
        axes_dist[row, col].axvline(0.5, color='red', linestyle='--', linewidth=2)
        axes_dist[row, col].set_title(f'Original\nAcc={accuracy:.3f}', fontsize=12)
        axes_dist[row, col].set_xlabel('$P(C_P)$', fontsize=10)
        axes_dist[row, col].set_ylabel('Frequency', fontsize=10)
        axes_dist[row, col].legend(fontsize=8)
        
        for i, snr in enumerate(snr_values):
            print(f"\nProcessing SNR = {snr} dB...")
            
            decay_curves_noisy = conditioner.add_noise(time, decay_curves_orig, late_time, snr)
            X_noisy = decay_curves_noisy.reshape(len(decay_curves_noisy), -1, 1)
            
            if use_quantized:
                y_pred_proba = tflite_predict_fn(X_noisy)
            else:
                y_pred_proba = model.predict(X_noisy, verbose=0)
            
            if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                y_pred_proba_target = y_pred_proba[:, 1]
            else:
                y_pred_proba_target = y_pred_proba.flatten()
            
            y_pred = (y_pred_proba_target >= 0.5).astype(int)
            accuracy = np.mean(y_pred == y_test)
            results[snr] = {'accuracy': accuracy, 'y_pred_proba': y_pred_proba_target}
            print(f"  Accuracy = {accuracy:.4f}")
            
            plot_idx = i + 1
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            cm = confusion_matrix(y_test, y_pred)
            cm = cm.T  # Transpose so rows=predicted, cols=true
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greys', square=True,
                       xticklabels=['$C_A$', '$C_P$'], yticklabels=['$C_A$', '$C_P$'],
                       ax=axes_cm[row, col], annot_kws={'size': 12}, cbar=False)
            axes_cm[row, col].set_title(f'SNR={snr} dB\nAcc={accuracy:.3f}', fontsize=12)
            axes_cm[row, col].set_xlabel('True', fontsize=10)
            axes_cm[row, col].set_ylabel('Predicted', fontsize=10)
            
            target_present_probs = y_pred_proba_target[y_test == 1]
            target_absent_probs = y_pred_proba_target[y_test == 0]
            axes_dist[row, col].hist(target_present_probs, bins=20, alpha=0.6, color='green', label='$C_P$', edgecolor='black')
            axes_dist[row, col].hist(target_absent_probs, bins=20, alpha=0.6, color='blue', label='$C_A$', edgecolor='black')
            axes_dist[row, col].axvline(0.5, color='red', linestyle='--', linewidth=2)
            axes_dist[row, col].set_title(f'SNR={snr} dB\nAcc={accuracy:.3f}', fontsize=12)
            axes_dist[row, col].set_xlabel('$P(C_P)$', fontsize=10)
            axes_dist[row, col].set_ylabel('Frequency', fontsize=10)
            axes_dist[row, col].legend(fontsize=8)
        
        for idx in range(n_plots, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes_cm[row, col].axis('off')
            axes_dist[row, col].axis('off')
        
        fig_cm.suptitle('Confusion Matrices at Different SNR Levels', fontsize=14, fontweight='bold')
        fig_cm.tight_layout()
        fig_cm.show()
        
        fig_dist.suptitle('Prediction Distributions at Different SNR Levels', fontsize=14, fontweight='bold')
        fig_dist.tight_layout()
        fig_dist.show()
        
        print("\n" + "="*70)
        print("Summary")
        print("="*70)
        print(f"{'SNR':<15} {'Accuracy':<10}")
        print("-"*25)
        print(f"{'Original':<15} {results['original']['accuracy']:.4f}")
        for snr in snr_values:
            print(f"{f'{snr} dB':<15} {results[snr]['accuracy']:.4f}")
        
        return results

    def multi_snr_plot(self, snr_values, data_series, ylabel, ylim, original_values=None):
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for values, label, color in data_series:
            ax.plot(snr_values, values, 'o:', color=color, lw=2.5, markersize=10, label=label)
        
        if original_values:
            for key, val in original_values.items():
                if key == 'keras' or key == 'original':
                    ax.axhline(val, color='tab:orange', linestyle='-', lw=2,
                              label=f'Proposed classifier (Keras/no noise): {val:.3f}')
                elif key == 'quantized':
                    ax.axhline(val, color='tab:blue', linestyle='-', lw=2,
                              label=f'Proposed classifier (LiteRT/no noise): {val:.3f}')
        
        ax.axhline(0.5, color='grey', linestyle='--', lw=2, label='Random classifier')
        
        ax.set_xlabel('SNR [dB]', fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.legend(loc='lower right', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=20)
        
        for i, snr in enumerate(snr_values):
            for j, (values, _, color) in enumerate(data_series):
                val = values[i]
                if i == len(snr_values) - 1:
                    xytext = (0, 10)
                else:
                    xytext = (0, -20)
                ax.annotate(f'{val:.3f}', (snr, val), textcoords="offset points", 
                           xytext=xytext, ha='center', fontsize=20, color='black')
        
        plt.tight_layout()
        plt.show()
        
        return fig

    def plot_multi_snr_accuracy(self, model, time, X_test, y_test, snr_values, late_time,
                                 conditioner=None, use_quantized=False, tflite_predict_fn=None):
        
        if model is None and not use_quantized:
            print("Error: No model provided!")
            return None
        
        if use_quantized and tflite_predict_fn is None:
            print("Error: TFLite predict function required for quantized model!")
            return None
        
        print("\n" + "="*70)
        print("Multi-SNR Accuracy Analysis")
        print("="*70)
        print(f"Test samples: {len(X_test)}")
        print(f"SNR values: {snr_values} dB")
        print(f"Using quantized model: {use_quantized}")
        
        if X_test.ndim == 3:
            decay_curves_orig = X_test.squeeze(axis=-1)
        else:
            decay_curves_orig = X_test
        
        results = {}
        accuracies = []
        snr_plot_values = []
        
        print(f"\nProcessing Original (no noise)...")
        X_input = decay_curves_orig.reshape(len(decay_curves_orig), -1, 1)
        
        if use_quantized:
            y_pred_proba = tflite_predict_fn(X_input)
        else:
            y_pred_proba = model.predict(X_input, verbose=0)
        
        if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
            y_pred_proba_target = y_pred_proba[:, 1]
        else:
            y_pred_proba_target = y_pred_proba.flatten()
        
        y_pred = (y_pred_proba_target >= 0.5).astype(int)
        accuracy_original = np.mean(y_pred == y_test)
        results['original'] = accuracy_original
        print(f"  Accuracy = {accuracy_original:.4f}")
        
        for snr in snr_values:
            print(f"\nProcessing SNR = {snr} dB...")
            
            decay_curves_noisy = conditioner.add_noise(time, decay_curves_orig, late_time, snr)
            X_noisy = decay_curves_noisy.reshape(len(decay_curves_noisy), -1, 1)
            
            if use_quantized:
                y_pred_proba = tflite_predict_fn(X_noisy)
            else:
                y_pred_proba = model.predict(X_noisy, verbose=0)
            
            if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                y_pred_proba_target = y_pred_proba[:, 1]
            else:
                y_pred_proba_target = y_pred_proba.flatten()
            
            y_pred = (y_pred_proba_target >= 0.5).astype(int)
            accuracy = np.mean(y_pred == y_test)
            results[snr] = accuracy
            accuracies.append(accuracy)
            snr_plot_values.append(snr)
            print(f"  Accuracy = {accuracy:.4f}")
        
        data_series = [(accuracies, 'Proposed classifier (Keras)', 'tab:orange')]
        original_values = {'original': accuracy_original}
        fig = self.multi_snr_plot(snr_plot_values, data_series, 'Accuracy', [0.45, 1.05], original_values)
        
        print("\n" + "="*70)
        print("Summary")
        print("="*70)
        print(f"{'SNR':<15} {'Accuracy':<10}")
        print("-"*25)
        print(f"{'Original':<15} {results['original']:.4f}")
        for snr in snr_values:
            print(f"{f'{snr} dB':<15} {results[snr]:.4f}")
        
        return results

    def plot_multi_snr_auc(self, model, time, X_test, y_test, snr_values, late_time,
                           conditioner=None, use_quantized=False, tflite_predict_fn=None):
        
        if model is None and not use_quantized:
            print("Error: No model provided!")
            return None
        
        if use_quantized and tflite_predict_fn is None:
            print("Error: TFLite predict function required for quantized model!")
            return None
        
        print("\n" + "="*70)
        print("Multi-SNR AUC Analysis")
        print("="*70)
        print(f"Test samples: {len(X_test)}")
        print(f"SNR values: {snr_values} dB")
        print(f"Using quantized model: {use_quantized}")
        
        if X_test.ndim == 3:
            decay_curves_orig = X_test.squeeze(axis=-1)
        else:
            decay_curves_orig = X_test
        
        results = {}
        auc_values = []
        snr_plot_values = []
        
        print(f"\nProcessing Original (no noise)...")
        X_input = decay_curves_orig.reshape(len(decay_curves_orig), -1, 1)
        
        if use_quantized:
            y_pred_proba = tflite_predict_fn(X_input)
        else:
            y_pred_proba = model.predict(X_input, verbose=0)
        
        if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
            y_pred_proba_target = y_pred_proba[:, 1]
        else:
            y_pred_proba_target = y_pred_proba.flatten()
        
        pfa, pd, _ = roc_curve(y_test, y_pred_proba_target)
        auc_original = auc(pfa, pd)
        results['original'] = auc_original
        print(f"  AUC = {auc_original:.4f}")
        
        for snr in snr_values:
            print(f"\nProcessing SNR = {snr} dB...")
            
            decay_curves_noisy = conditioner.add_noise(time, decay_curves_orig, late_time, snr)
            X_noisy = decay_curves_noisy.reshape(len(decay_curves_noisy), -1, 1)
            
            if use_quantized:
                y_pred_proba = tflite_predict_fn(X_noisy)
            else:
                y_pred_proba = model.predict(X_noisy, verbose=0)
            
            if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                y_pred_proba_target = y_pred_proba[:, 1]
            else:
                y_pred_proba_target = y_pred_proba.flatten()
            
            pfa, pd, _ = roc_curve(y_test, y_pred_proba_target)
            auc_val = auc(pfa, pd)
            results[snr] = auc_val
            auc_values.append(auc_val)
            snr_plot_values.append(snr)
            print(f"  AUC = {auc_val:.4f}")
        
        data_series = [(auc_values, 'AUC', 'tab:orange')]
        original_values = {'original': auc_original}
        fig = self.multi_snr_plot(snr_plot_values, data_series, 'AUC', [0.45, 1.05], original_values)
        
        print("\n" + "="*70)
        print("Summary")
        print("="*70)
        print(f"{'SNR':<15} {'AUC':<10}")
        print("-"*25)
        print(f"{'Original':<15} {results['original']:.4f}")
        for snr in snr_values:
            print(f"{f'{snr} dB':<15} {results[snr]:.4f}")
        
        return results

    def plot_model_size_comparison(self, size_results):
        
        if size_results is None:
            print("Error: No size results provided!")
            return
        
        if 'quantized' not in size_results:
            print("Error: No quantized model data available for comparison!")
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        models = ['Proposed classifier (Keras)', 'Proposed classifier (LiteRT)']
        sizes_kb = [size_results['keras']['disk_kb'], size_results['quantized']['disk_kb']]
        colors = ['tab:orange', 'tab:blue']
        
        bars = ax.bar(models, sizes_kb, color=colors, width=0.6, edgecolor='black', linewidth=1.5)
        
        for bar, size in zip(bars, sizes_kb):
            height = bar.get_height()
            ax.annotate(f'{size:.2f} KB',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 5),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=20, fontweight='bold')
        
        if 'comparison' in size_results:
            ratio = size_results['comparison']['compression_ratio']
            reduction = size_results['comparison']['size_reduction_percent']
            ax.text(0.5, 0.95, f'Compression: {ratio:.2f}× ({reduction:.1f}% reduction)',
                   transform=ax.transAxes, ha='center', va='top', fontsize=18,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_ylabel('Model Size [KB]', fontsize=20)
        ax.set_xlabel('Model Type', fontsize=20)
        ax.tick_params(labelsize=20)
        ax.set_ylim([0, max(sizes_kb) * 1.25])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        return fig

    def plot_inference_time_comparison(self, inference_results):
        
        if inference_results is None:
            print("Error: No inference results provided!")
            return
        
        has_keras = 'keras' in inference_results
        has_quantized = 'quantized' in inference_results
        
        if not has_keras and not has_quantized:
            print("Error: No model timing data available!")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if has_keras:
            keras_times = inference_results['keras']['all_times_ms']
            runs_keras = np.arange(1, len(keras_times) + 1)
            ax.plot(runs_keras, keras_times, color='tab:orange', alpha=0.7, 
                    linewidth=1.5, label='Proposed classifier (Keras)')
            keras_mean = inference_results['keras']['mean_ms']
            ax.axhline(y=keras_mean, color='tab:orange', linestyle='--', 
                       linewidth=2.5, label=f'Keras average: {keras_mean:.2f} ms')
        
        if has_quantized:
            quant_times = inference_results['quantized']['all_times_ms']
            runs_quant = np.arange(1, len(quant_times) + 1)
            ax.plot(runs_quant, quant_times, color='tab:blue', alpha=0.7, 
                    linewidth=1.5, label='Proposed classifier (LiteRT)')
            quant_mean = inference_results['quantized']['mean_ms']
            ax.axhline(y=quant_mean, color='tab:blue', linestyle='--', 
                       linewidth=2.5, label=f'LiteRT average: {quant_mean:.2f} ms')
        
        ax.set_xlabel('Inference run number', fontsize=20)
        ax.set_ylabel('Inference time [ms]', fontsize=20)
        ax.tick_params(labelsize=20)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=20)

        plt.tight_layout()
        plt.show()
        
        return fig

    def plot_accuracy_comparison(self, accuracy_results):
        
        if accuracy_results is None:
            print("Error: No accuracy results provided!")
            return
        
        has_keras = 'keras' in accuracy_results
        has_quantized = 'quantized' in accuracy_results
        
        if not has_keras and not has_quantized:
            print("Error: No model accuracy data available!")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = []
        accuracies = []
        colors = []
        
        if has_keras:
            models.append('Proposed classifier (Keras)')
            accuracies.append(accuracy_results['keras']['accuracy'] * 100)
            colors.append('tab:orange')
        
        if has_quantized:
            models.append('Proposed classifier (LiteRT)')
            accuracies.append(accuracy_results['quantized']['accuracy'] * 100)
            colors.append('tab:blue')
        
        bars = ax.bar(models, accuracies, color=colors, width=0.6, 
                     edgecolor='black', linewidth=1.5)
        
        if has_keras:
            ax.axhline(accuracies[0], color='black', linestyle='--', lw=2, 
                      label=f'Proposed classifier (Keras)')
            ax.text(0, accuracies[0] - 2, f'{accuracies[0]:.2f}', 
                   fontsize=20, verticalalignment='center', horizontalalignment='left')
        
        ax.axhline(50, color='grey', linestyle='--', lw=2, 
                  label=f'Random classifier')
        
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            height = bar.get_height()
            if i == len(bars) - 1:  # Last bar (quantized)
                xytext = (0, 10)  # Place above
            else:
                xytext = (0, -20)  # Place below
            ax.annotate(f'{acc:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=xytext,
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=20, fontweight='bold')
        
        if 'comparison' in accuracy_results:
            diff = accuracy_results['comparison']['accuracy_difference'] * 100
            if abs(diff) < 0.01:
                diff_text = 'Same accuracy'
            else:
                better = 'Proposed classifier (Keras)' if diff > 0 else 'Proposed classifier (LiteRT)'
                diff_text = f'{better} is {abs(diff):.2f}% more accurate'
            ax.text(0.5, 0.95, diff_text,
                   transform=ax.transAxes, ha='center', va='top', fontsize=18,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_ylabel('Accuracy [%]', fontsize=20)
        ax.set_xlabel('Model Type', fontsize=20)
        ax.legend(loc='lower right', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        ax.tick_params(labelsize=20)
        
        plt.tight_layout()
        plt.show()
        
        return fig

    def plot_multi_snr_accuracy_comparison(self, keras_results, quantized_results, snr_values):
        
        keras_accs = [keras_results.get(snr, 0) for snr in snr_values]
        quant_accs = [quantized_results.get(snr, 0) for snr in snr_values]
        
        data_series = [
            (keras_accs, 'Proposed classifier (Keras)', 'tab:orange'),
            (quant_accs, 'Proposed classifier (LiteRT)', 'tab:blue')
        ]
        
        original_values = {}
        if 'original' in keras_results:
            original_values['keras'] = keras_results['original']
        if 'original' in quantized_results:
            original_values['quantized'] = quantized_results['original']
        
        fig = self.multi_snr_plot(snr_values, data_series, 'Accuracy', [0.45, 1.05], original_values)
        
        return fig

    def plot_multi_snr_auc_comparison(self, keras_results, quantized_results, snr_values):
        
        keras_aucs = [keras_results.get(snr, 0) for snr in snr_values]
        quant_aucs = [quantized_results.get(snr, 0) for snr in snr_values]
        
        data_series = [
            (keras_aucs, 'Proposed classifier (Keras)', 'tab:orange'),
            (quant_aucs, 'Proposed classifier (LiteRT)', 'tab:blue')
        ]
        
        original_values = {}
        if 'original' in keras_results:
            original_values['keras'] = keras_results['original']
        if 'original' in quantized_results:
            original_values['quantized'] = quantized_results['original']
        
        fig = self.multi_snr_plot(snr_values, data_series, 'AUC', [0.45, 1.05], original_values)
        
        return fig

    def plot_comprehensive_model_comparison(self, size_results, inference_results, accuracy_results):
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        colors = {'keras': 'tab:orange', 'quantized': 'tab:blue'}
        
        ax1 = axes[0, 0]
        if size_results and 'quantized' in size_results:
            models = ['Proposed classifier (Keras)', 'Proposed classifier (LiteRT)']
            sizes = [size_results['keras']['disk_kb'], size_results['quantized']['disk_kb']]
            bars = ax1.bar(models, sizes, color=[colors['keras'], colors['quantized']], 
                          width=0.6, edgecolor='black', linewidth=1.5)
            for bar, size in zip(bars, sizes):
                ax1.annotate(f'{size:.2f} KB', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 5), textcoords="offset points", ha='center', fontsize=16, fontweight='bold')
            if 'comparison' in size_results:
                ax1.set_title(f'Model Size ({size_results["comparison"]["compression_ratio"]:.2f}× compression)', fontsize=20)
            else:
                ax1.set_title('Model Size', fontsize=20)
        else:
            ax1.text(0.5, 0.5, 'No quantized model\ndata available', ha='center', va='center', fontsize=16)
            ax1.set_title('Model Size', fontsize=20)
        ax1.set_ylabel('Size [KB]', fontsize=20)
        ax1.tick_params(labelsize=18)
        ax1.grid(True, alpha=0.3, axis='y')
        
        ax2 = axes[0, 1]
        if inference_results:
            models = []
            times = []
            stds = []
            bar_colors = []
            if 'keras' in inference_results:
                models.append('Proposed classifier (Keras)')
                times.append(inference_results['keras']['mean_ms'])
                stds.append(inference_results['keras']['std_ms'])
                bar_colors.append(colors['keras'])
            if 'quantized' in inference_results:
                models.append('Proposed classifier (LiteRT)')
                times.append(inference_results['quantized']['mean_ms'])
                stds.append(inference_results['quantized']['std_ms'])
                bar_colors.append(colors['quantized'])
            
            bars = ax2.bar(models, times, yerr=stds, capsize=6, color=bar_colors,
                          width=0.6, edgecolor='black', linewidth=1.5)
            for bar, t in zip(bars, times):
                ax2.annotate(f'{t:.3f} ms', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 10), textcoords="offset points", ha='center', fontsize=16, fontweight='bold')
            if 'comparison' in inference_results:
                ax2.set_title(f'Inference Time ({inference_results["comparison"]["speedup"]:.2f}× speedup)', fontsize=20)
            else:
                ax2.set_title('Inference Time', fontsize=20)
        else:
            ax2.text(0.5, 0.5, 'No inference data\navailable', ha='center', va='center', fontsize=16)
            ax2.set_title('Inference Time', fontsize=20)
        ax2.set_ylabel('Time [ms]', fontsize=20)
        ax2.tick_params(labelsize=18)
        ax2.grid(True, alpha=0.3, axis='y')
        
        ax3 = axes[1, 0]
        if accuracy_results:
            models = []
            accs = []
            bar_colors = []
            if 'keras' in accuracy_results:
                models.append('Proposed classifier (Keras)')
                accs.append(accuracy_results['keras']['accuracy'] * 100)
                bar_colors.append(colors['keras'])
            if 'quantized' in accuracy_results:
                models.append('Proposed classifier (LiteRT)')
                accs.append(accuracy_results['quantized']['accuracy'] * 100)
                bar_colors.append(colors['quantized'])
            
            bars = ax3.bar(models, accs, color=bar_colors, width=0.6, edgecolor='black', linewidth=1.5)
            for bar, acc in zip(bars, accs):
                ax3.annotate(f'{acc:.2f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 5), textcoords="offset points", ha='center', fontsize=16, fontweight='bold')
            ax3.axhline(50, color='grey', linestyle='--', lw=2, alpha=0.5)
            ax3.set_ylim([0, 105])
            ax3.set_title('Classification Accuracy', fontsize=20)
        else:
            ax3.text(0.5, 0.5, 'No accuracy data\navailable', ha='center', va='center', fontsize=16)
            ax3.set_title('Classification Accuracy', fontsize=20)
        ax3.set_ylabel('Accuracy [%]', fontsize=20)
        ax3.tick_params(labelsize=18)
        ax3.grid(True, alpha=0.3, axis='y')
        
        ax4 = axes[1, 1]
        if size_results and 'keras' in size_results:
            params_data = {
                'Trainable': size_results['keras'].get('trainable_params', 0),
                'Non-trainable': size_results['keras'].get('non_trainable_params', 0)
            }
            total = sum(params_data.values())
            if total > 0:
                wedges, texts, autotexts = ax4.pie(params_data.values(), labels=params_data.keys(),
                                                   autopct='%1.1f%%', colors=['#64B5F6', '#81C784'],
                                                   textprops={'fontsize': 16})
                ax4.set_title(f'Parameters Distribution\n(Total: {total:,})', fontsize=20)
            else:
                ax4.text(0.5, 0.5, 'No parameter data\navailable', ha='center', va='center', fontsize=16)
                ax4.set_title('Parameters Distribution', fontsize=20)
        else:
            ax4.text(0.5, 0.5, 'No size data\navailable', ha='center', va='center', fontsize=16)
            ax4.set_title('Parameters Distribution', fontsize=20)
        
        plt.tight_layout()
        plt.show()
        
        return fig