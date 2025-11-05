from discretize import CylindricalMesh
from discretize.utils import mkvc

from simpeg import maps
import simpeg.electromagnetics.time_domain as tdem

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from datetime import datetime
import os

from pi_config import PiConfig
from pi_plotter import PiPlotter
from pi_logger import PiLogger
from pi_classifier import PiClassifier
from pi_conditioner import PiConditioner


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

    def create_conductivity_model(self, mesh, target_z_val, target_in_model):        
        ind_active = mesh.cell_centers[:, 2] < self.cfg.separation_z

        r = mesh.cell_centers[ind_active, 0]
        z = mesh.cell_centers[ind_active, 2]

        model_map = maps.InjectActiveCells(mesh, ind_active, self.cfg.air_conductivity)
        
        model = self.cfg.air_conductivity * np.ones(ind_active.sum())

        ind_soil = (z < 0)
        model[ind_soil] = self.cfg.soil_conductivity

        if target_in_model: 
            inner_radius = self.cfg.target_radius - self.cfg.target_thickness
            if inner_radius < 0:
                inner_radius = 0
            
            unique_r = np.unique(r)
            if len(unique_r) > 1:
                min_cell_width = np.min(np.diff(unique_r[unique_r > 0]))
            else:
                min_cell_width = 0.01  # Default cell width
            
            top_z = target_z_val + self.cfg.target_height/2
            bottom_z = target_z_val - self.cfg.target_height/2

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
        return model, model_map

    def run(self, loop_z_range, target_z_range, mesh=None):
        target_over_soil = target_z_range is not None and target_z_range[0] >= 0
        target_under_soil = target_z_range is not None and target_z_range[0] < 0

        waveform = tdem.sources.StepOffWaveform(off_time=self.cfg.waveform_off_time)
        time_channels = np.linspace(0, 1024e-6, 1024)

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
            mesh = CylindricalMesh([hr, hphi, hz], x0="00C")
            create_plotting_metadata = True
        else:
            create_plotting_metadata = False

        model, model_map = self.create_conductivity_model(mesh, target_z_val, target_under_soil or target_over_soil)

        simulation = tdem.simulation.Simulation3DMagneticFluxDensity(
            mesh,
            survey=survey,
            sigmaMap=model_map,
            t0=self.cfg.simulation_t0
        )
        simulation.time_steps = self.cfg.time_steps

        dpred = simulation.dpred(m=model)
        dpred = np.reshape(dpred, (1, len(time_channels)))
        
        if target_over_soil or target_under_soil:
            label = f"L{loop_z_val:.2f}-T{target_z_val:.2f}"
        else:
            label = f"L{loop_z_val:.2f}-T-"

        simulation_metadata = {
            'loop_z': loop_z_val,
            'target_z': target_z_val if target_over_soil or target_under_soil else None,
            'target_present': target_over_soil,
            'label': label
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
        
        return time_channels, -dpred[0, :], label, simulation_metadata, plotting_metadata


def run_simulations(simulator, logger, loop_z_range, target_z_range, 
                    num_target_present, num_target_absent, probability_of_target_in_soil, output_file=None):    
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
    for i in range(num_target_present):
        time, decay, label, metadata, plot_meta = simulator.run(
            loop_z_range, target_z_range, mesh=mesh
        )
        
        if mesh is None and plot_meta is not None:
            mesh = plot_meta['mesh']
        
        logger.store_to_hdf5(output_file, sim_index, time, decay, label, metadata)
        sim_index += 1
        print(f"\n\rTarget-Present: {i+1}/{num_target_present}", end='', flush=True)
    
    print()
    
    for i in range(num_target_absent):
        if np.random.rand() < probability_of_target_in_soil:
            target_z_range = [-simulator.cfg.separation_z, - simulator.cfg.target_height / 2]
        else:
            target_z_range = None
        
        time, decay, label, metadata, plot_meta = simulator.run(
            loop_z_range, target_z_range, mesh=mesh
        )
        
        logger.store_to_hdf5(output_file, sim_index, time, decay, label, metadata)
        sim_index += 1

        print(f"\n\rTarget-Absent: {i+1}/{num_target_absent}", end='', flush=True)

    print()
    
    logger.finalize_hdf5(output_file, len(time))
    
    print(f"\n✓ Complete: {total_sims} simulations written to {output_file}")
    
    return output_file

def find_hdf5_files(root_dir='Datasets'):
    hdf5_files = []
    if not os.path.exists(root_dir):
        return hdf5_files
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.h5') or filename.endswith('.hdf5'):
                full_path = os.path.join(dirpath, filename)
                hdf5_files.append(full_path)
    
    return sorted(hdf5_files)

def select_hdf5_file():
    hdf5_files = find_hdf5_files('Datasets')
    
    if not hdf5_files:
        print("\nNo HDF5 files found in ./Datasets directory.")
        manual_input = input("Enter HDF5 file path manually (or 'q' to cancel): ").strip()
        if manual_input.lower() == 'q':
            return None
        return manual_input
    
    print("\n" + "="*70)
    print("Available HDF5 files:")
    print("="*70)
    
    for idx, filepath in enumerate(hdf5_files):
        file_size = os.path.getsize(filepath) / (1024 * 1024)
        modified_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        print(f"  [{idx+1}] {filepath}")
        print(f"      Size: {file_size:.2f} MB | Modified: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\nTotal files found: {len(hdf5_files)}")
    print("Enter file number, or type path manually, or 'q' to cancel")
    
    user_input = input("Your selection: ").strip()
    
    if user_input.lower() == 'q':
        return None
    
    try:
        idx = int(user_input) - 1
        if 0 <= idx < len(hdf5_files):
            return hdf5_files[idx]
        else:
            print(f"Invalid selection. Index out of range.")
            return None
    except ValueError:
        if os.path.exists(user_input):
            return user_input
        else:
            print(f"File not found: {user_input}")
            return None

def get_split_dataset_path(base_path, split_type):
    """
    Get the path to a split dataset file.
    
    Args:
        base_path: Original dataset path
        split_type: 'train', 'val', or 'test'
    
    Returns:
        Path to split dataset if it exists, otherwise None
    """
    base_without_ext = base_path.replace('.h5', '').replace('.hdf5', '')
    split_path = f"{base_without_ext}_{split_type}.h5"
    
    if os.path.exists(split_path):
        return split_path
    else:
        return None

def dataset_operations_menu(hdf5_file, cfg, simulator, logger, plotter, classifier, conditioner):
    """Menu for operations on a loaded dataset"""
    while True:
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
                time_c, decay_c, labels_c, label_strings_c, metadata_c = conditioner.condition_dataset(
                    time, decay_curves, labels, label_strings, metadata
                )
                
                # Save conditioned dataset
                base_name = os.path.basename(hdf5_file).replace('.h5', '')
                base_dir = os.path.dirname(hdf5_file)
                conditioned_file = os.path.join(base_dir, f"{base_name}_conditioned.h5")
                
                print(f"\nSaving conditioned dataset to: {conditioned_file}")
                logger.initialize_hdf5(conditioned_file, metadata_c['num_target_present'], 
                                     metadata_c['num_target_absent'])
                
                for i in range(len(decay_c)):
                    sim_metadata = {
                        'loop_z': metadata_c['loop_z'][i],
                        'target_z': metadata_c['target_z'][i],
                        'target_present': labels_c[i] == 1,
                        'label': label_strings_c[i]
                    }
                    logger.store_to_hdf5(conditioned_file, i, time_c, decay_c[i], 
                                       label_strings_c[i], sim_metadata)
                
                logger.finalize_hdf5(conditioned_file, len(time_c))
                
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
            # Train classifier
            try:
                print(f"\n{'='*70}")
                print("TRAINING MODE")
                print(f"{'='*70}")
                
                print("\nData source:")
                print("  [1] Use data from memory")
                print("  [2] Load from file")
                data_choice = input("Select data source [1]: ").strip() or "1"
                
                if data_choice == '1':
                    if classifier.X_train is None:
                        print("\nNo training data in memory. Split dataset first (option 3).")
                        continue
                    
                    num_samples = classifier.X_train.shape[1]
                    X_train, y_train = classifier.X_train, classifier.y_train
                    X_val, y_val = classifier.X_val, classifier.y_val
                    
                else:
                    train_path = get_split_dataset_path(hdf5_file, 'train')
                    if not train_path:
                        print(f"\nNo training split found.")
                        continue
                    
                    print(f"Loading: {os.path.basename(train_path)}")
                    time, decay_curves, labels, _, _ = logger.load_from_hdf5(train_path)
                    num_samples = len(time)
                    X_train = decay_curves.reshape(decay_curves.shape[0], decay_curves.shape[1], 1)
                    y_train = labels
                    
                    val_path = get_split_dataset_path(hdf5_file, 'val')
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
                    
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
        
        elif choice == '6':
            # Validate classifier
            try:
                if classifier.model is None:
                    print("\nNo model loaded. Train a model first (option 5).")
                    continue
                
                print(f"\n{'='*70}")
                print("VALIDATION MODE")
                print(f"{'='*70}")
                
                print("\nData source:")
                print("  [1] Use data from memory")
                print("  [2] Load from file")
                data_choice = input("Select data source [1]: ").strip() or "1"
                
                if data_choice == '1':
                    if classifier.X_val is None:
                        print("\nNo validation data in memory. Split dataset first (option 3).")
                        continue
                    classifier.validate_model(classifier.X_val, classifier.y_val)
                    
                else:
                    val_path = get_split_dataset_path(hdf5_file, 'val')
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
            # Test classifier
            try:
                if classifier.model is None:
                    print("\nNo model loaded. Train a model first (option 5).")
                    continue
                
                print(f"\n{'='*70}")
                print("TESTING MODE")
                print(f"{'='*70}")
                
                print("\nData source:")
                print("  [1] Use data from memory")
                print("  [2] Load from file")
                data_choice = input("Select data source [1]: ").strip() or "1"
                
                if data_choice == '1':
                    if classifier.X_test is None:
                        print("\nNo test data in memory. Split dataset first (option 3).")
                        continue
                    classifier.test_model(classifier.X_test, classifier.y_test)
                    
                else:
                    test_path = get_split_dataset_path(hdf5_file, 'test')
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
        
        elif choice == 'b':
            # Back to main menu
            break
        
        else:
            print("Invalid option selected.")

if __name__ == "__main__":
    cfg = PiConfig('config.json')
    simulator = PiSimulator(cfg)
    logger = PiLogger()
    plotter = PiPlotter(cfg)
    conditioner = PiConditioner(cfg)
    classifier = PiClassifier(conditioner)

    num_target_present = 2000
    num_target_absent = 2000
    
    while True:
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
                print("\n⚠️  Warning: Computing magnetic field distribution will:")
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
                probability_of_target_in_soil=1.0,
            )
            
            # Go to dataset operations menu
            dataset_operations_menu(hdf5_path, cfg, simulator, logger, plotter, classifier, conditioner)
        
        elif choice == '2':
            # Load existing dataset
            hdf5_file = select_hdf5_file()
            
            if hdf5_file is None:
                print("Operation cancelled.")
            else:
                # Go to dataset operations menu
                dataset_operations_menu(hdf5_file, cfg, simulator, logger, plotter, classifier, conditioner)
        
        elif choice == 'q':
            # Quit
            print("\nExiting TDEM Simulation System...")
            break
        
        else:
            print("Invalid option selected.")
