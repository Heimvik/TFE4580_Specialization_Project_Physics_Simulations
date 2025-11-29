import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import ArrowStyle
import matplotlib.patches as mpatches
from simpeg import maps
from discretize import CylindricalMesh
import h5py


class BasePlotter:
    def __init__(self):
        pass


class PiPlotter(BasePlotter):
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
                    'target_conductivity': sim_group.attrs.get('target_conductivity', self.cfg.aluminum_conductivity),
                    'target_present': sim_group.attrs['target_present'],
                    'label': sim_group.attrs['label'],
                    'time': sim_group['time'][:],
                    'decay': sim_group['decay'][:]
                }
                if metadata['target_z'] == -999.0:
                    metadata['target_z'] = None
                self.simulations_metadata.append(metadata)
    
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
        if not self.simulations_metadata:
            print("\\nNo data loaded. Run simulations first.")
            return
        
        available_sims = []
        for metadata in self.simulations_metadata:
            target_type = metadata['target_type']
            if target_type > 0:
                target_conductivity = metadata.get('target_conductivity', self.cfg.aluminum_conductivity)
                target_z = metadata['target_z']
                type_names = ['No Target', 'Hollow Cylinder', 'Shredded Can', 'Solid Block']
                status = f"Type: {type_names[target_type]}, σ={target_conductivity:.1e} S/m, z={target_z:.3f}m"
                available_sims.append((metadata['index'], metadata['loop_z'], status, target_type, target_conductivity, target_z))
        
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
            
            print(f"\\nGenerating model: type={target_type}, σ={target_conductivity:.1e}, z={target_z:.3f}m")

            model, _ = self.simulator.create_conductivity_model(self.mesh, target_z, target_type, target_conductivity)
            
            if model is None:
                print("Error generating model.")
                return
                
            self._plot_model_2d(model, status, self.mesh, self.ind_active)
            
        except (ValueError, IndexError) as e:
            print(f"Invalid selection: {e}")
    
    def _plot_model_2d(self, model, title, mesh=None, ind_active=None):
        if mesh is None:
            mesh = self.mesh
        if ind_active is None:
            ind_active = self.ind_active
            
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        num_target_cells = np.sum(model > self.cfg.soil_conductivity * 10)
        print(f"Plotting model with {num_target_cells} target cells")
        
        plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
        log_model = np.log10(model)
        
        im = mesh.plot_image(plotting_map * log_model, ax=ax, grid=True,
                            clim=(np.log10(self.cfg.air_conductivity), 
                                  np.log10(self.cfg.aluminum_conductivity)))
        
        ax.axhline(y=0, color='brown', linestyle='-', linewidth=3, alpha=0.8, label='Ground surface')
        
        ax.set_xlabel('Radial Distance [m]', fontsize=18)
        ax.set_ylabel('Elevation [m]', fontsize=18)
        ax.set_title(f'Conductivity Model: {title}', fontsize=16)
        ax.tick_params(labelsize=18)
        ax.legend(fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 0.5])
        
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
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Accuracy', fontsize=11)
        ax1.set_title('Model Accuracy', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Loss', fontsize=11)
        ax2.set_title('Model Loss', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
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
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', square=True,
                   xticklabels=['No Target', 'Target Present'],
                   yticklabels=['No Target', 'Target Present'],
                   cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
                   ax=ax, annot_kws={'size': 14})
        
        ax.set_ylabel('True Label', fontsize=14)
        ax.set_xlabel('Predicted Label', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print("\nClassification Report:")
        print("="*70)
        target_names = ['No Target', 'Target Present']
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        return cm
    
    def plot_roc_curve(self, X_test, y_test):
        from sklearn.metrics import roc_curve, auc
        
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
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
        ax.legend(loc="lower right", fontsize=12)
        ax.grid(True, alpha=0.3)
        
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
                label='True: Target Present', edgecolor='black')
        ax1.hist(target_absent_probs, bins=30, alpha=0.6, color='blue', 
                label='True: No Target', edgecolor='black')
        ax1.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
        ax1.set_xlabel('Predicted Probability (Target Present)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        data_to_plot = [target_absent_probs, target_present_probs]
        bp = ax2.boxplot(data_to_plot, labels=['True: No Target', 'True: Target Present'],
                        patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor('blue')
        bp['boxes'][1].set_facecolor('green')
        ax2.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
        ax2.set_ylabel('Predicted Probability (Target Present)', fontsize=12)
        ax2.set_title('Prediction Confidence by True Label', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()

    def plot_roc_pfa_pd(self, X_test, y_test, snr_db=None, num_thresholds=100):
        from sklearn.metrics import roc_curve, auc
        
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
        
        # Plot 1: Standard ROC (Pd vs Pfa)
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
        ax1.set_xlabel('Probability of False Alarm (Pfa)', fontsize=12)
        ax1.set_ylabel('Probability of Detection (Pd)', fontsize=12)
        title = 'ROC Curve: Pd vs Pfa'
        if snr_db is not None:
            title += f' (SNR = {snr_db} dB)'
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend(loc="lower right", fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Semi-log ROC
        ax2 = axes[1]
        valid_mask = pfa > 0
        ax2.semilogx(pfa[valid_mask], pd[valid_mask], color='darkorange', lw=3,
                    label=f'ROC (AUC = {roc_auc:.4f})')
        
        ax2.set_xlim([1e-4, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Probability of False Alarm (Pfa) - Log Scale', fontsize=12)
        ax2.set_ylabel('Probability of Detection (Pd)', fontsize=12)
        ax2.set_title('ROC Curve (Log Pfa Scale)', fontsize=14, fontweight='bold')
        ax2.legend(loc="lower right", fontsize=11)
        ax2.grid(True, alpha=0.3, which='both')
        
        # Plot 3: Pd at specific Pfa values
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
        ax3.set_xlabel('Probability of False Alarm (Pfa)', fontsize=12)
        ax3.set_ylabel('Probability of Detection (Pd)', fontsize=12)
        ax3.set_title('Detection Performance vs Pfa', fontsize=14, fontweight='bold')
        ax3.legend(loc="lower right", fontsize=11)
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
        fig.suptitle('Training Data Overview', fontsize=16, fontweight='bold')
        
        time_us = time * 1e6
        
        # Plot 1: All decay curves colored by label
        ax = axes[0, 0]
        for i, (decay, label, label_str) in enumerate(zip(decay_curves, labels, label_strings)):
            color = 'green' if label == 1 else 'blue'
            alpha = 0.3
            ax.loglog(time_us, decay, color=color, alpha=alpha, linewidth=1)
        
        num_target_present = np.sum(labels == 1)
        num_target_absent = np.sum(labels == 0)
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, label=f'Target Present ({num_target_present})'),
            Line2D([0], [0], color='blue', lw=2, label=f'Target Absent ({num_target_absent})')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        ax.set_xlabel('Time [μs]', fontsize=11)
        ax.set_ylabel('-dBz/dt [T/s]', fontsize=11)
        ax.set_title('All Decay Curves', fontsize=12)
        ax.grid(True, alpha=0.3, which='both')
        
        # Plot 2: Sample target-present curves
        ax = axes[0, 1]
        target_present_indices = np.where(labels == 1)[0][:5]
        for idx in target_present_indices:
            ax.loglog(time_us, decay_curves[idx], linewidth=2, label=label_strings[idx])
        ax.set_xlabel('Time [μs]', fontsize=11)
        ax.set_ylabel('-dBz/dt [T/s]', fontsize=11)
        ax.set_title('Sample: Target Present', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        
        # Plot 3: Sample target-absent curves
        ax = axes[1, 0]
        target_absent_indices = np.where(labels == 0)[0][:5]
        for idx in target_absent_indices:
            ax.loglog(time_us, decay_curves[idx], linewidth=2, label=label_strings[idx])
        ax.set_xlabel('Time [μs]', fontsize=11)
        ax.set_ylabel('-dBz/dt [T/s]', fontsize=11)
        ax.set_title('Sample: Target Absent', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        
        # Plot 4: Simple stats
        ax = axes[1, 1]
        ax.text(0.5, 0.7, f'Total Samples: {len(decay_curves)}', 
                ha='center', va='center', fontsize=14)
        ax.text(0.5, 0.5, f'Target Present: {num_target_present}', 
                ha='center', va='center', fontsize=14, color='green')
        ax.text(0.5, 0.3, f'Target Absent: {num_target_absent}', 
                ha='center', va='center', fontsize=14, color='blue')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Dataset Statistics', fontsize=12)
        
        plt.tight_layout()
        plt.show()

    def plot_model_architecture(self, model, output_path='model_architecture.png'):
        print("\n" + "="*70)
        print("Model Architecture Visualization")
        print("="*70)
        
        if model is None:
            print("Error: No model provided!")
            return None
        
        # Extract layer information
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
                details = f"filters={layer.filters}, kernel={layer.kernel_size[0]}"
            elif 'MaxPooling' in layer_type:
                details = f"pool={layer.pool_size[0]}"
            elif 'Dense' in layer_type:
                details = f"units={layer.units}"
            elif 'Dropout' in layer_type:
                details = f"rate={layer.rate}"
            elif 'BatchNorm' in layer_type:
                details = "momentum=0.99"
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
            'InputLayer': '#E8F5E9',
            'Conv1D': '#BBDEFB',
            'BatchNormalization': '#F3E5F5',
            'Activation': '#FFF3E0',
            'MaxPooling1D': '#FFECB3',
            'GlobalAveragePooling1D': '#FFE0B2',
            'Dropout': '#FFCDD2',
            'Dense': '#B2DFDB',
            'Flatten': '#D7CCC8',
        }
        
        n_layers = len(layers_info)
        fig_height = max(12, n_layers * 0.8)
        fig, ax = plt.subplots(1, 1, figsize=(14, fig_height))
        
        box_width = 10
        box_height = 0.6
        y_spacing = 0.85
        x_center = 7
        
        ax.text(x_center, n_layers * y_spacing + 1.5, 
                'TDEM Pulse Induction Classifier Architecture',
                fontsize=18, fontweight='bold', ha='center', va='center',
                fontfamily='serif')
        
        ax.text(x_center, n_layers * y_spacing + 0.9,
                f'Total Parameters: {model.count_params():,}',
                fontsize=12, ha='center', va='center', style='italic',
                color='#666666')
        
        for i, layer in enumerate(layers_info):
            y_pos = (n_layers - 1 - i) * y_spacing + 0.5
            
            color = layer_colors.get(layer['type'], '#E0E0E0')
            
            box = FancyBboxPatch(
                (x_center - box_width/2, y_pos - box_height/2),
                box_width, box_height,
                boxstyle="round,pad=0.03,rounding_size=0.1",
                facecolor=color,
                edgecolor='#333333',
                linewidth=1.5,
                mutation_scale=0.5
            )
            ax.add_patch(box)
            
            ax.text(x_center - box_width/2 + 0.3, y_pos + 0.08,
                   layer['type'],
                   fontsize=11, fontweight='bold', ha='left', va='center',
                   fontfamily='monospace')
            
            name_details = layer['name']
            if layer['details']:
                name_details += f"  ({layer['details']})"
            ax.text(x_center - box_width/2 + 0.3, y_pos - 0.15,
                   name_details,
                   fontsize=9, ha='left', va='center', color='#555555')
            
            ax.text(x_center + box_width/2 - 0.3, y_pos + 0.08,
                   f"Output: {layer['shape']}",
                   fontsize=9, ha='right', va='center',
                   fontfamily='monospace', color='#333333')
            
            if layer['params'] > 0:
                ax.text(x_center + box_width/2 - 0.3, y_pos - 0.15,
                       f"Params: {layer['params']:,}",
                       fontsize=9, ha='right', va='center', color='#666666')
            
            if i < n_layers - 1:
                arrow = FancyArrowPatch(
                    (x_center, y_pos - box_height/2 - 0.02),
                    (x_center, y_pos - y_spacing + box_height/2 + 0.02),
                    arrowstyle=ArrowStyle('->', head_length=6, head_width=4),
                    color='#666666',
                    linewidth=1.5,
                    mutation_scale=1
                )
                ax.add_patch(arrow)
        
        legend_elements = []
        used_types = set(layer['type'] for layer in layers_info)
        for layer_type in ['InputLayer', 'Conv1D', 'BatchNormalization', 'Activation', 
                          'MaxPooling1D', 'GlobalAveragePooling1D', 'Dropout', 'Dense']:
            if layer_type in used_types:
                color = layer_colors.get(layer_type, '#E0E0E0')
                legend_elements.append(mpatches.Patch(facecolor=color, edgecolor='#333333',
                                                      label=layer_type, linewidth=1))
        
        ax.legend(handles=legend_elements, loc='upper left', 
                 bbox_to_anchor=(0.02, 0.98), fontsize=9,
                 title='Layer Types', title_fontsize=10)
        
        ax.set_xlim(0, 14)
        ax.set_ylim(-0.5, n_layers * y_spacing + 2)
        ax.axis('off')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"\n✓ Model architecture saved to: {output_path}")
        
        plt.show()
        
        print("\n" + "="*70)
        print("Detailed Architecture Summary:")
        print("="*70)
        model.summary()
        
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
        axes[0].set_xlabel('Probability of False Alarm (Pfa)', fontsize=12)
        axes[0].set_ylabel('Probability of Detection (Pd)', fontsize=12)
        axes[0].set_title('ROC Curves: Multiple SNR Levels', fontsize=14, fontweight='bold')
        axes[0].legend(loc='lower right', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([0, 1])
        axes[0].set_ylim([0, 1.05])
        
        axes[1].set_xlabel('Probability of False Alarm (Pfa) - Log Scale', fontsize=12)
        axes[1].set_ylabel('Probability of Detection (Pd)', fontsize=12)
        axes[1].set_title('ROC Curves (Log Pfa)', fontsize=14, fontweight='bold')
        axes[1].legend(loc='lower right', fontsize=10)
        axes[1].grid(True, alpha=0.3, which='both')
        axes[1].set_xlim([1e-4, 1])
        axes[1].set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.show()
        
        return results