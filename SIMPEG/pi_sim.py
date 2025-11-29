from discretize import CylindricalMesh
from discretize.utils import mkvc

from simpeg import maps
import simpeg.electromagnetics.time_domain as tdem

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py

from datetime import datetime
import os

from pi_config import PiConfig
from pi_plotter import PiPlotter, ClassifierPlotter
from pi_logger import PiLogger
from pi_classifier import PiClassifier
from pi_conditioner import PiConditioner


class SystemStatus:
    def __init__(self):
        self.dataset_path = None
        self.dataset_samples = 0
        self.dataset_target_present = 0
        self.dataset_target_absent = 0
        self.model_path = None
        self.model_params = 0
        self.model_trained = False
        self.train_samples = 0
        self.val_samples = 0
        self.test_samples = 0
    
    def set_dataset(self, path, samples=0, target_present=0, target_absent=0):
        self.dataset_path = path
        self.dataset_samples = samples
        self.dataset_target_present = target_present
        self.dataset_target_absent = target_absent
    
    def set_model(self, path=None, params=0, trained=False):
        self.model_path = path
        self.model_params = params
        self.model_trained = trained
    
    def set_splits(self, train=0, val=0, test=0):
        self.train_samples = train
        self.val_samples = val
        self.test_samples = test
    
    def display(self):
        print("\n┌" + "─"*68 + "┐")
        print("│" + " STATUS ".center(68) + "│")
        print("├" + "─"*68 + "┤")
        
        if self.dataset_path:
            ds_name = os.path.basename(self.dataset_path)
            if len(ds_name) > 40:
                ds_name = ds_name[:37] + "..."
            print(f"│  Dataset: {ds_name:<55} │")
            print(f"│    Samples: {self.dataset_samples:<5} (Target: {self.dataset_target_present}, No-Target: {self.dataset_target_absent})" + " "*10 + "│")
            if self.train_samples > 0:
                print(f"│    Splits: Train={self.train_samples}, Val={self.val_samples}, Test={self.test_samples}" + " "*20 + "│")
        else:
            print("│  Dataset: None loaded" + " "*45 + "│")
        
        print("├" + "─"*68 + "┤")
        
        if self.model_path:
            model_name = os.path.basename(self.model_path)
            if len(model_name) > 40:
                model_name = model_name[:37] + "..."
            status = "Trained" if self.model_trained else "Untrained"
            print(f"│  Model: {model_name:<57} │")
            print(f"│    Parameters: {self.model_params:,}, Status: {status}" + " "*25 + "│")
        elif self.model_params > 0:
            status = "Trained" if self.model_trained else "Built (untrained)"
            print(f"│  Model: In-memory ({status})" + " "*40 + "│")
            print(f"│    Parameters: {self.model_params:,}" + " "*40 + "│")
        else:
            print("│  Model: None loaded" + " "*47 + "│")
        
        print("└" + "─"*68 + "┘")


class PiSimulator:
    def __init__(self, config): 
        self.cfg = config

    def create_survey(self, time_channels, waveform, loop_z_val):        
        xtx, ytx, ztx = np.meshgrid([0], [0], [loop_z_val])
        source_locations = np.c_[mkvc(xtx), mkvc(ytx), mkvc(ztx)]
        
        xrx, yrx, zrx = np.meshgrid([0], [0], [loop_z_val])
        receiver_locations = np.c_[mkvc(xrx), mkvc(yrx), mkvc(zrx)]

        transmitters_list, receivers_list = [], []
        
        receivers_list.append(tdem.receivers.PointMagneticFluxTimeDerivative(
            receiver_locations[0, :], time_channels, "z"
        ))
        transmitters_list.append(
            tdem.sources.CircularLoop(
                receivers_list,
                location=source_locations[0],
                waveform=waveform,
                current=self.cfg.tx_current,
                radius=self.cfg.tx_radius,
                n_turns=self.cfg.tx_n_turns
            )
        )
        
        survey = tdem.Survey(transmitters_list)
        return survey

    def create_conductivity_model(self, mesh, target_z_val, target_type, target_conductivity=None):        
        ind_active = mesh.cell_centers[:, 2] < self.cfg.separation_z

        r = mesh.cell_centers[ind_active, 0]
        z = mesh.cell_centers[ind_active, 2]

        model_map = maps.InjectActiveCells(mesh, ind_active, self.cfg.air_conductivity)
        model = self.cfg.air_conductivity * np.ones(ind_active.sum())

        ind_soil = (z < 0)
        model[ind_soil] = self.cfg.soil_conductivity

        if target_type != 0:
            conductivity = target_conductivity if target_conductivity is not None else self.cfg.aluminum_conductivity
            
            top_z = target_z_val + self.cfg.target_height/2
            bottom_z = target_z_val - self.cfg.target_height/2
            
            #Target type 1: Hollow cylinder
            if target_type == 1:
                unique_r = np.unique(r)
                min_cell_width = np.min(np.diff(unique_r[unique_r > 0])) if len(unique_r) > 1 else 0.01

                inner_radius = max(0, self.cfg.target_radius - self.cfg.target_thickness)
                
                r_min_wall = self.cfg.target_radius - min_cell_width if self.cfg.target_thickness < min_cell_width else inner_radius

                ind_walls = (
                    (r >= r_min_wall) & (r <= self.cfg.target_radius) &
                    (z < top_z) & (z > bottom_z)
                )
                
                ind_top = (r <= self.cfg.target_radius) & (np.abs(z - top_z) <= min_cell_width)
                ind_bottom = (r <= self.cfg.target_radius) & (np.abs(z - bottom_z) <= min_cell_width)
                
                ind_target = ind_walls | ind_top | ind_bottom

            # Target type 2: Shredded fragments - hardcoded pattern
            elif target_type == 2:
                r_max = 0.2
                z_range = 0.1
                z_mid = target_z_val
                
                candidates = np.where(
                    (np.abs(z - z_mid) <= z_range) & 
                    (r <= r_max)
                )[0]
                
                if len(candidates) == 0:
                    ind_target = np.zeros(len(r), dtype=bool)
                else:
                    r_cand = r[candidates]
                    z_cand = z[candidates]
                    
                    unique_r = np.unique(np.round(r_cand, 4))
                    unique_z = np.unique(np.round(z_cand - z_mid, 4))
                    
                    pattern = [
                        (0, 0), (0, 1), (0, 2),
                        (1, 1), (1, 3),
                        (2, 0), (2, 2), (2, 4),
                        (3, 1), (3, 3),
                        (4, 0), (4, 2),
                        (5, 1), (5, 3), (5, 4)
                    ]
                    
                    ind_target = np.zeros(len(r), dtype=bool)
                    
                    for r_idx, z_idx in pattern:
                        if r_idx < len(unique_r) and z_idx < len(unique_z):
                            target_r = unique_r[r_idx]
                            target_z_offset = unique_z[z_idx]
                            
                            cell_idx = np.where(
                                (np.abs(r - target_r) < 0.005) & 
                                (np.abs((z - z_mid) - target_z_offset) < 0.005)
                            )[0]
                            
                            if len(cell_idx) > 0:
                                ind_target[cell_idx[0]] = True
                
            # Target type 3: Solid box
            elif target_type == 3:
                box_half_width = 2*self.cfg.target_radius
                box_height = self.cfg.target_height

                ind_target = (
                    (r <= box_half_width) & 
                    (z <= top_z) & 
                    (z >= top_z - box_height)
                )

            else:
                raise ValueError(f"Unknown target_type: {target_type}")
            
            model[ind_target] = conductivity
            
        return model, model_map

    def run(self, loop_z_range, target_type, target_z_range, mesh=None):
        target_over_soil = target_z_range is not None and target_z_range[0] >= 0
        target_under_soil = target_z_range is not None and target_z_range[0] < 0

        waveform = tdem.sources.StepOffWaveform(off_time=self.cfg.waveform_off_time)
        time_channels = np.linspace(0, 1100e-6, 1100)

        loop_z_val = np.random.uniform(loop_z_range[0], loop_z_range[1])
        while loop_z_val < self.cfg.separation_z:
            print(f"Warning: Generated loop_z={loop_z_val:.4f} < separation_z={self.cfg.separation_z}. Regenerating...")
            loop_z_val = np.random.uniform(loop_z_range[0], loop_z_range[1])
        
        if target_z_range is not None:
            target_z_val = np.random.uniform(target_z_range[0], target_z_range[1])
            while target_z_val > self.cfg.separation_z:
                print(f"Warning: Generated target_z={target_z_val:.4f} > separation_z={-self.cfg.separation_z}. Regenerating...")
                target_z_val = np.random.uniform(target_z_range[0], target_z_range[1])
        else:
            target_z_val = None

        survey = self.create_survey(time_channels, waveform, loop_z_val)

        if mesh is None:
            hr = [(0.01, 15), (0.01, 15, 1.3), (0.05, 10, 1.5)]
            hphi = 1
            hz = [(0.01, 10, -1.3), (0.01, 30), (0.01, 10, 1.3)]
            mesh = CylindricalMesh([hr, hphi, hz], origin=[0, 0, target_z_val])
            create_plotting_metadata = True
        else:
            create_plotting_metadata = False

        if (target_over_soil or target_under_soil) and target_z_range is not None:
            model, model_map = self.create_conductivity_model(mesh, target_z_val, target_type, self.cfg.aluminum_conductivity)
        else:
            print("If a target is present, target_z_range must be provided.")
            return None, None, None, None, None, None

        simulation = tdem.simulation.Simulation3DMagneticFluxDensity(
            mesh,
            survey=survey,
            sigmaMap=model_map,
            t0=self.cfg.simulation_t0
        )
        simulation.time_steps = self.cfg.time_steps

        dpred = simulation.dpred(m=model)
        dpred = np.reshape(dpred, (1, len(time_channels)))
        emf= -dpred[0, :]* (np.pi * self.cfg.tx_radius**2 * self.cfg.tx_n_turns)

        
        if target_over_soil or target_under_soil:
            label = f"Coil at {loop_z_val:.2f}, object at {target_z_val:.2f}"
        else:
            label = f"Coil at {loop_z_val:.2f}, no object present"

        simulation_metadata = {
            'loop_z': loop_z_val,
            'target_z': target_z_val if target_over_soil or target_under_soil else None,
            'target_present': target_over_soil,
            'label': label
        }

        model_params = {
            'target_type': target_type,
            'target_conductivity': self.cfg.aluminum_conductivity if target_under_soil or target_over_soil else 0.0
        }
        
        plotting_metadata = None
        if create_plotting_metadata:
            active_area_z = self.cfg.separation_z
            ind_active = mesh.cell_centers[:, 2] < active_area_z
            
            r = mesh.cell_centers[ind_active, 0]
            z = mesh.cell_centers[ind_active, 2]
            model_no_target = self.cfg.air_conductivity * np.ones(ind_active.sum())
            ind_soil = (z < 0)
            model_no_target[ind_soil] = self.cfg.soil_conductivity
            
            plotting_metadata = {
                'mesh': mesh,
                'model_no_target': model_no_target,
                'ind_active': ind_active,
                'cfg': self.cfg
            }
        return time_channels, emf, label, simulation_metadata, plotting_metadata, model_params


def run_simulations(simulator, logger, loop_z_range, target_z_range, 
                    num_target_present, num_target_absent, num_different_targets, probability_of_target_in_soil, output_file=None):    
    mesh = None
    total_sims = num_target_present + num_target_absent
    sim_index = 0
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"Simulation_{total_sims}"
        output_dir = os.path.join("Datasets", folder_name)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{timestamp}_TP{num_target_present}_TA{num_target_absent}.h5")
    
    logger.initialize_hdf5(output_file, num_target_present, num_target_absent)
    
    print(f"\n=== Running Simulations (Writing to {output_file}) ===")
    
    print()
    
    error_count = 0
    target_type = 0
    for i in range(num_target_present):
        target_type = np.random.randint(1, num_different_targets + 1)
        time, decay, label, metadata, plot_meta, model_params = simulator.run(
            loop_z_range, target_type, target_z_range, mesh=mesh
        )
        
        if mesh is None and plot_meta is not None:
            mesh = plot_meta['mesh']
        if time is None and decay is None:
            error_count += 1
            continue
        
        logger.store_to_hdf5(output_file, sim_index, time, decay, label, metadata, model_params)
        sim_index += 1
        print(f"\n\rTarget-Present: {i+1}/{num_target_present}", end='', flush=True)
    
    print()
    for i in range(num_target_absent):
        if np.random.rand() < probability_of_target_in_soil:
            target_z_range = [-simulator.cfg.separation_z, - simulator.cfg.target_height / 2]
            target_type = np.random.randint(1, num_different_targets + 1)
        else:
            target_z_range = None
            target_type = 0
        
        time, decay, label, metadata, plot_meta, model_params = simulator.run(
            loop_z_range, target_type, target_z_range, mesh=mesh
        )

        if mesh is None and decay is None:
            error_count += 1
            continue
        
        logger.store_to_hdf5(output_file, sim_index, time, decay, label, metadata, model_params)
        sim_index += 1

        print(f"\n\rTarget-Absent: {i+1}/{num_target_absent}", end='', flush=True)

    print()
    print(f"\nErrors encountered during simulation: {error_count}")
    
    logger.finalize_hdf5(output_file, len(time))
    
    print(f"\n Complete: {total_sims} simulations written to {output_file}")
    return output_file

def dataset_operations_menu(hdf5_file, cfg, simulator, logger, plotter, classifier, conditioner, status):
    try:
        with h5py.File(hdf5_file, 'r') as f:
            num_sims = f['metadata'].attrs['num_simulations']
            num_present = f['metadata'].attrs['num_target_present']
            num_absent = f['metadata'].attrs['num_target_absent']
            status.set_dataset(hdf5_file, num_sims, num_present, num_absent)
    except:
        status.set_dataset(hdf5_file)
    
    # Check for existing splits
    train_path = logger.get_split_path(hdf5_file, 'train')
    val_path = logger.get_split_path(hdf5_file, 'val')
    test_path = logger.get_split_path(hdf5_file, 'test')
    
    if train_path and val_path and test_path:
        try:
            with h5py.File(train_path, 'r') as f:
                train_count = f['metadata'].attrs['num_simulations']
            with h5py.File(val_path, 'r') as f:
                val_count = f['metadata'].attrs['num_simulations']
            with h5py.File(test_path, 'r') as f:
                test_count = f['metadata'].attrs['num_simulations']
            status.set_splits(train_count, val_count, test_count)
        except:
            pass
    
    # Update model status if classifier has a model
    if classifier.model is not None:
        status.set_model(None, classifier.model.count_params(), trained=True)
    
    while True:
        status.display()
        
        print("\n" + "="*70)
        print(f"Dataset Operations: {os.path.basename(hdf5_file)}")
        print("="*70)
        print("\nOptions:")
        print("  1. Visualize dataset")
        print("  2. Print dataset statistics")
        print("  3. Split dataset")
        print("  4. Condition dataset")
        print("  5. Train classifier")
        print("  6. Validate classifier")
        print("  7. Test classifier")
        print("  8. Classification plots")
        print("  9. Model analysis (FLOPs, parameters, architecture)")
        print("  10. Save/Load model")
        print("  [b] Back to main menu")
        
        choice = input("\nSelect option: ").strip().lower()
        
        if choice == '1':
            # Visualize dataset
            try:
                plotter.load_from_hdf5(hdf5_file)
                plotter.run()
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
        
        elif choice == '2':
            # Print statistics
            try:
                logger.print_hdf5_metadata(hdf5_file)
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '3':
            # Split and load dataset
            try:
                print("\nDataset Split Ratios:")
                train_ratio = float(input("  Training ratio [0.7]: ").strip() or "0.7")
                val_ratio = float(input("  Validation ratio [0.15]: ").strip() or "0.15")
                test_ratio = float(input("  Testing ratio [0.15]: ").strip() or "0.15")
                
                train_path, val_path, test_path = classifier.split_dataset(
                    logger, hdf5_file, train_ratio, val_ratio, test_ratio, save_to_file=True
                )
                
                # Update status with split info
                if train_path:
                    try:
                        with h5py.File(train_path, 'r') as f:
                            train_count = f['metadata'].attrs['num_simulations']
                        with h5py.File(val_path, 'r') as f:
                            val_count = f['metadata'].attrs['num_simulations']
                        with h5py.File(test_path, 'r') as f:
                            test_count = f['metadata'].attrs['num_simulations']
                        status.set_splits(train_count, val_count, test_count)
                    except:
                        pass
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
        
        elif choice == '4':
            # Condition dataset
            try:
                print(f"\n{'='*70}")
                print("CONDITION DATASET")
                print(f"{'='*70}")
                
                print("\nLoading dataset...")
                time, decay_curves, labels, label_strings, metadata = logger.load_from_hdf5(hdf5_file)
                
                print("Applying conditioning (log-scale + min-max normalization)...")
                I_0 = (np.pi * cfg.tx_radius**2 * cfg.tx_n_turns) * decay_curves
                I_1 = conditioner.condition_dataset(
                    I_0
                )
                
                # Save conditioned dataset
                base_name = os.path.basename(hdf5_file).replace('.h5', '')
                base_dir = os.path.dirname(hdf5_file)
                conditioned_file = os.path.join(base_dir, f"{base_name}_conditioned.h5")
                
                print(f"\nSaving conditioned dataset to: {conditioned_file}")
                logger.initialize_hdf5(conditioned_file, metadata['num_target_present'], 
                                     metadata['num_target_absent'])
                
                for i in range(len(decay_curves)):
                    sim_metadata = {
                        'loop_z': metadata['loop_z'][i],
                        'target_z': metadata['target_z'][i],
                        'target_present': labels[i] == 1,
                        'label': label_strings[i]
                    }
                    logger.store_to_hdf5(conditioned_file, i, time, I_1[i], 
                                       label_strings[i], sim_metadata)
                
                logger.finalize_hdf5(conditioned_file, len(time))
                
                print(f"\n✓ Conditioned dataset saved!")
                print(f"  File: {os.path.basename(conditioned_file)}")
                print(f"  Normalization: log10 + min-max")
                
                switch = input(f"\nSwitch to conditioned dataset for further operations? (y/n) [y]: ").strip().lower() or 'y'
                if switch == 'y':
                    hdf5_file = conditioned_file
                    print(f"✓ Now using: {os.path.basename(hdf5_file)}")
                else:
                    print(f"Continuing with original dataset: {os.path.basename(hdf5_file)}")
                
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
        
        elif choice == '5':
            try:
                print(f"\n{'='*70}")
                print("TRAINING MODE")
                print(f"{'='*70}")

                train_path = logger.get_split_path(hdf5_file, 'train')
                if not train_path:
                    print(f"\nNo training split found.")
                    continue
                
                print(f"Loading: {os.path.basename(train_path)}")
                time, decay_curves, labels, _, _ = logger.load_from_hdf5(train_path)
                num_samples = len(time)
                X_train = decay_curves.reshape(decay_curves.shape[0], decay_curves.shape[1], 1)
                y_train = labels
                
                val_path = logger.get_split_path(hdf5_file, 'val')
                if val_path:
                    _, decay_curves_val, labels_val, _, _ = logger.load_from_hdf5(val_path)
                    X_val = decay_curves_val.reshape(decay_curves_val.shape[0], decay_curves_val.shape[1], 1)
                    y_val = labels_val
                else:
                    X_val, y_val = None, None
                
                try:
                    epochs = int(input("\nEpochs [20]: ").strip() or "20")
                    batch_size = int(input("Batch size [32]: ").strip() or "32")
                except ValueError:
                    epochs, batch_size = 20, 32
                
                classifier.build_model(num_samples)
                classifier.train_model(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
                
                # Update status
                status.set_model(None, classifier.model.count_params(), trained=True)
                    
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
        
        elif choice == '6':
            try:
                if classifier.model is None:
                    print("\nNo model loaded. Train a model first (option 5).")
                    continue
                
                print(f"\n{'='*70}")
                print("VALIDATION MODE")
                print(f"{'='*70}")
                    
                val_path = logger.get_split_path(hdf5_file, 'val')
                if not val_path:
                    print(f"\nNo validation split found.")
                    continue
                
                print(f"Loading: {os.path.basename(val_path)}")
                _, decay_curves, labels, _, _ = logger.load_from_hdf5(val_path)
                X_val = decay_curves.reshape(decay_curves.shape[0], decay_curves.shape[1], 1)
                y_val = labels
                classifier.validate_model(X_val, y_val)
                
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
        
        elif choice == '7':
            try:
                if classifier.model is None:
                    print("\nNo model loaded. Train a model first (option 5).")
                    continue
                
                print(f"\n{'='*70}")
                print("TESTING MODE")
                print(f"{'='*70}")
                
                test_path = logger.get_split_path(hdf5_file, 'test')
                if not test_path:
                    print(f"\nNo test split found.")
                    continue
                
                print(f"Loading: {os.path.basename(test_path)}")
                _, decay_curves, labels, _, _ = logger.load_from_hdf5(test_path)
                X_test = decay_curves.reshape(decay_curves.shape[0], decay_curves.shape[1], 1)
                y_test = labels
                classifier.test_model(X_test, y_test)
                
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
        
        elif choice == '8':
            # Classification plots submenu
            try:
                if classifier.model is None:
                    print("\nNo model loaded. Train a model first (option 5) or load one (option 10).")
                    continue
                
                test_path = logger.get_split_path(hdf5_file, 'test')
                if not test_path:
                    print(f"\nNo test split found. Please split the dataset first (option 3).")
                    continue
                
                print(f"Loading test data: {os.path.basename(test_path)}")
                _, decay_curves, labels, _, _ = logger.load_from_hdf5(test_path)
                X_test = decay_curves.reshape(decay_curves.shape[0], decay_curves.shape[1], 1)
                y_test = labels
                
                # Classification plots submenu
                while True:
                    print(f"\n{'='*70}")
                    print("Classification Plots")
                    print(f"{'='*70}")
                    print("\nAvailable plots:")
                    print("  [1] Confusion Matrix")
                    print("  [2] Normalized Confusion Matrix")
                    print("  [3] ROC Curve (standard)")
                    print("  [4] ROC Curve (Pd vs Pfa - detection theory)")
                    print("  [5] Prediction Distribution")
                    print("  [6] All Classification Plots")
                    print("  [b] Back")
                    
                    plot_choice = input("\nSelect plot: ").strip().lower()
                    
                    if plot_choice == '1':
                        classifier.plot_confusion_matrix(X_test, y_test, normalize=False)
                    elif plot_choice == '2':
                        classifier.plot_confusion_matrix(X_test, y_test, normalize=True)
                    elif plot_choice == '3':
                        classifier.plot_roc_curve(X_test, y_test)
                    elif plot_choice == '4':
                        snr_input = input("Enter SNR in dB (optional, press Enter to skip): ").strip()
                        snr_db = float(snr_input) if snr_input else None
                        classifier.plot_roc_pfa_pd(X_test, y_test, snr_db=snr_db)
                    elif plot_choice == '5':
                        classifier.plot_prediction_distribution(X_test, y_test)
                    elif plot_choice == '6':
                        print("\nGenerating all classification plots...")
                        classifier.plot_confusion_matrix(X_test, y_test, normalize=False)
                        classifier.plot_confusion_matrix(X_test, y_test, normalize=True)
                        classifier.plot_roc_curve(X_test, y_test)
                        classifier.plot_roc_pfa_pd(X_test, y_test)
                        classifier.plot_prediction_distribution(X_test, y_test)
                    elif plot_choice == 'b':
                        break
                    else:
                        print("Invalid selection.")
                
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
        
        elif choice == '9':
            # Model analysis submenu
            try:
                while True:
                    print(f"\n{'='*70}")
                    print("Model Analysis")
                    print(f"{'='*70}")
                    
                    if classifier.model is not None:
                        print(f"Current model: {classifier.model.name}")
                        print(f"Input shape: {classifier.model.input_shape}")
                        print(f"Parameters: {classifier.model.count_params():,}")
                    else:
                        print("No model loaded.")
                    
                    print("\nOptions:")
                    print("  [1] Calculate FLOPs (computational cost)")
                    print("  [2] Model Architecture Diagram (PNG)")
                    print("  [3] Full LaTeX Report (document)")
                    print("  [4] TikZ Architecture Diagram")
                    print("  [5] Layer-by-layer Summary")
                    print("  [6] Build model (without training)")
                    print("  [b] Back")
                    
                    analysis_choice = input("\nSelect option: ").strip().lower()
                    
                    if analysis_choice == '1':
                        if classifier.model is None:
                            print("\nNo model loaded. Building default model...")
                            try:
                                input_len = int(input("Enter input length [50]: ").strip() or "50")
                            except ValueError:
                                input_len = 50
                            classifier.build_model(input_len)
                        
                        flops_result = classifier.calculate_flops()
                        
                        # Ask if user wants to save results
                        save_flops = input("\nSave FLOP analysis to file? (y/n) [n]: ").strip().lower()
                        if save_flops == 'y':
                            flops_file = os.path.join(os.path.dirname(hdf5_file), "flops_analysis.txt")
                            with open(flops_file, 'w') as f:
                                f.write("="*70 + "\n")
                                f.write("FLOP Analysis for PiClassifier\n")
                                f.write("="*70 + "\n\n")
                                f.write(f"Total FLOPs per inference: {flops_result['total_flops']:,}\n")
                                f.write(f"Total MFLOPs: {flops_result['mflops']:.3f}\n")
                                f.write(f"Total GFLOPs: {flops_result['gflops']:.6f}\n\n")
                                f.write("Layer Breakdown:\n")
                                f.write("-"*70 + "\n")
                                for name, info in flops_result['breakdown'].items():
                                    if info['flops'] > 0:
                                        f.write(f"{name:<25} {info['type']:<20} {info['flops']:>12,}\n")
                            print(f"✓ Saved to: {flops_file}")
                    
                    elif analysis_choice == '2':
                        if classifier.model is None:
                            try:
                                input_len = int(input("Enter input length [50]: ").strip() or "50")
                            except ValueError:
                                input_len = 50
                        else:
                            input_len = None
                        
                        output_path = os.path.join(os.path.dirname(hdf5_file), "model_architecture.png")
                        classifier.plot_model_architecture(input_length=input_len, output_path=output_path)
                    
                    elif analysis_choice == '3':
                        if classifier.model is None:
                            try:
                                input_len = int(input("Enter input length [50]: ").strip() or "50")
                            except ValueError:
                                input_len = 50
                        else:
                            input_len = None
                        
                        latex_file = os.path.join(os.path.dirname(hdf5_file), "model_report.tex")
                        classifier.generate_model_summary_latex(input_length=input_len, output_file=latex_file)
                    
                    elif analysis_choice == '4':
                        if classifier.model is None:
                            try:
                                input_len = int(input("Enter input length [50]: ").strip() or "50")
                            except ValueError:
                                input_len = 50
                            classifier.build_model(input_len)
                        
                        tikz_file = os.path.join(os.path.dirname(hdf5_file), "model_architecture.tex")
                        classifier.generate_architecture_tikz(output_file=tikz_file)
                    
                    elif analysis_choice == '5':
                        if classifier.model is None:
                            print("\nNo model loaded. Build or load a model first.")
                        else:
                            print(f"\n{'='*70}")
                            print("Layer-by-Layer Summary")
                            print(f"{'='*70}")
                            classifier.model.summary()
                    
                    elif analysis_choice == '6':
                        try:
                            input_len = int(input("Enter input length [50]: ").strip() or "50")
                        except ValueError:
                            input_len = 50
                        classifier.build_model(input_len)
                        print("\n✓ Model built successfully!")
                    
                    elif analysis_choice == 'b':
                        break
                    else:
                        print("Invalid selection.")
                        
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
        
        elif choice == '10':
            # Save/Load model submenu
            try:
                while True:
                    print(f"\n{'='*70}")
                    print("Save/Load Model")
                    print(f"{'='*70}")
                    
                    if classifier.model is not None:
                        print(f"Current model: {classifier.model.name}")
                        print(f"Parameters: {classifier.model.count_params():,}")
                    else:
                        print("No model loaded.")
                    
                    print("\nOptions:")
                    print("  [1] Save current model")
                    print("  [2] Load model from file")
                    print("  [3] List available models")
                    print("  [b] Back")
                    
                    model_choice = input("\nSelect option: ").strip().lower()
                    
                    if model_choice == '1':
                        if classifier.model is None:
                            print("\nNo model to save. Train or build a model first.")
                            continue
                        
                        default_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
                        model_name = input(f"Model filename [{default_name}]: ").strip() or default_name
                        
                        if not model_name.endswith('.keras'):
                            model_name += '.keras'
                        
                        model_path = os.path.join(os.path.dirname(hdf5_file), model_name)
                        classifier.save_model(model_path)
                    
                    elif model_choice == '2':
                        # Find .keras files
                        model_dir = os.path.dirname(hdf5_file)
                        keras_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
                        
                        if keras_files:
                            print("\nAvailable models:")
                            for i, f in enumerate(keras_files):
                                print(f"  [{i+1}] {f}")
                            print(f"  [m] Enter path manually")
                            
                            selection = input("\nSelect model: ").strip().lower()
                            
                            if selection == 'm':
                                model_path = input("Enter model path: ").strip()
                            else:
                                try:
                                    idx = int(selection) - 1
                                    if 0 <= idx < len(keras_files):
                                        model_path = os.path.join(model_dir, keras_files[idx])
                                    else:
                                        print("Invalid selection.")
                                        continue
                                except ValueError:
                                    print("Invalid selection.")
                                    continue
                        else:
                            model_path = input("Enter model path: ").strip()
                        
                        if os.path.exists(model_path):
                            classifier.load_model(model_path)
                            status.set_model(model_path, classifier.model.count_params(), trained=True)
                        else:
                            print(f"File not found: {model_path}")
                    
                    elif model_choice == '3':
                        model_dir = os.path.dirname(hdf5_file)
                        keras_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
                        
                        if keras_files:
                            print("\nAvailable models:")
                            for f in keras_files:
                                full_path = os.path.join(model_dir, f)
                                size_mb = os.path.getsize(full_path) / (1024 * 1024)
                                print(f"  - {f} ({size_mb:.2f} MB)")
                        else:
                            print("\nNo .keras files found in dataset directory.")
                    
                    elif model_choice == 'b':
                        break
                    else:
                        print("Invalid selection.")
                        
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
        
        elif choice == 'b':
            # Back to main menu
            break
        
        else:
            print("Invalid option selected.")

if __name__ == "__main__":
    cfg = PiConfig('config.json')
    simulator = PiSimulator(cfg)
    logger = PiLogger()
    plotter = PiPlotter(cfg, simulator)
    conditioner = PiConditioner(cfg)
    classifier = PiClassifier(conditioner)
    status = SystemStatus()

    num_target_present = 2000
    num_target_absent = 2000
    
    while True:
        status.display()
        
        print("\n" + "="*70)
        print("TDEM Simulation System - Main Menu")
        print("="*70)
        print("\nOptions:")
        print("  1. Generate new dataset")
        print("  2. Load existing dataset")
        print("  [q] Quit")
        
        choice = input("\nSelect option: ").strip().lower()
        
        if choice == '1':
            # Generate new dataset
            print("\n" + "="*70)
            print("Generate New Dataset")
            print("="*70)
            
            print("\nDataset Type:")
            print("  [a] Standard (magnetic flux density only)")
            print("  [b] Enhanced (with 3D magnetic field distribution)")
            dataset_type = input("\nSelect dataset type [a/b]: ").strip().lower()
            
            compute_mag_field = False
            if dataset_type == 'b':
                print("\nWarning: Computing magnetic field distribution will:")
                print("    - Significantly increase computation time (~5x slower)")
                print("    - Require more storage (~10x larger files)")
                confirm = input("\nProceed with enhanced dataset? (y/n): ").strip().lower()
                if confirm == 'y':
                    compute_mag_field = True
                else:
                    print("Switching to standard dataset...")
            
            try:
                num_tp = int(input("\nNumber of target-present simulations [3]: ").strip() or "3")
                num_ta = int(input("Number of target-absent simulations [1]: ").strip() or "1")
            except ValueError:
                print("Invalid input, using defaults (3 present, 1 absent)")
                num_tp = 3
                num_ta = 1
            
            hdf5_path = run_simulations(
                simulator, logger,
                loop_z_range=cfg.loop_z_range,
                target_z_range=cfg.target_z_range,
                num_target_present=num_tp,
                num_target_absent=num_ta,
                num_different_targets=3,
                probability_of_target_in_soil=1,
                
            )
            
            # Go to dataset operations menu
            dataset_operations_menu(hdf5_path, cfg, simulator, logger, plotter, classifier, conditioner, status)
        
        elif choice == '2':
            # Load existing dataset
            hdf5_file = logger.select_hdf5_file()
            
            if hdf5_file is None:
                print("Operation cancelled.")
            else:
                dataset_operations_menu(hdf5_file, cfg, simulator, logger, plotter, classifier, conditioner, status)
        
        elif choice == 'q':
            print("\nExiting TDEM Simulation System...")
            break
        
        else:
            print("Invalid option selected.")
