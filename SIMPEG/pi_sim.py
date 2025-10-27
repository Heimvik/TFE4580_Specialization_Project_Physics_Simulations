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


class PiSimulator:
    def __init__(self, config): 
        self.cfg = config

    def create_survey(self, time_channels, waveform, loop_z_val):        
        xtx, ytx, ztx = np.meshgrid([0], [0], [loop_z_val])
        source_locations = np.c_[mkvc(xtx), mkvc(ytx), mkvc(ztx)]
        
        xrx, yrx, zrx = np.meshgrid([0], [0], [loop_z_val])
        receiver_locations = np.c_[mkvc(xrx), mkvc(yrx), mkvc(zrx)]
        
        source_list = []
        dbzdt_receiver = tdem.receivers.PointMagneticFluxTimeDerivative(
            receiver_locations[0, :], time_channels, "z"
        )
        receivers_list = [dbzdt_receiver]
        
        source_list.append(
            tdem.sources.CircularLoop(
                receivers_list,
                location=source_locations[0],
                waveform=waveform,
                current=self.cfg.tx_current,
                radius=self.cfg.tx_radius,
                n_turns=self.cfg.tx_n_turns
            )
        )
        
        survey = tdem.Survey(source_list)
        return survey

    def create_conductivity_model(self, mesh, target_z_val, target_in_model):        
        active_area_z = self.cfg.separation_z
        ind_active = mesh.cell_centers[:, 2] < active_area_z

        model_map = maps.InjectActiveCells(mesh, ind_active, self.cfg.air_conductivity)

        r = mesh.cell_centers[ind_active, 0]
        z = mesh.cell_centers[ind_active, 2]

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


class PiConditioner:
    def __init__(self):
        pass

    def add_noise(self, time, data, late_time, snr_db):
        if isinstance(time, list):
            time = time[0]
        
        if hasattr(time, 'ndim') and time.ndim > 1:
            time = time[0]
        
        print(f"Time range: {time[0]:.6e} to {time[-1]:.6e}")
        
        if late_time < time[0]:
            raise ValueError("late_time must be within the range of time array")
        if late_time > time[-1]:
            raise ValueError("late_time must be within the range of time array")

        idx = np.where(time >= late_time)[0]
        if len(idx) == 0:
            raise ValueError("late_time is beyond the maximum time in the array")
        idx = idx[0]

        signal_power = np.mean(data[idx:]**2)
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power)

        noise = np.random.normal(0, noise_std, size=data.shape)
        return data + noise
        

    def amplify(self, time, data, time_gain):
        if isinstance(time, list):
            time = time[0]
        
        if hasattr(time, 'ndim') and time.ndim > 1:
            time = time[0]
        
        time_gain = np.array(time_gain)
        if time_gain.size > 0:
            time_gain = time_gain[np.argsort(time_gain[:, 0])]
        gain = np.ones_like(time)
        for t, g in time_gain:
            mask = time >= t
            gain[mask] = g
        return data * gain

    def normalize(self, data, max=None):
        max = np.max(np.abs(data)) if max is None else max
        return np.maximum(data / max, 0)

    def quantize(self, data, depth, dtype):
        return (np.round(data * 2**depth)).astype(dtype)


def run_simulations(simulator, logger, loop_z_range, target_z_range, 
                   num_target_present, num_target_absent, target_in_soil=True, output_file=None):    
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
        if np.random.rand() < 0.5 and target_in_soil:
            target_z_range = [-simulator.cfg.separation_z, - 0.05 - simulator.cfg.target_height / 2]
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

if __name__ == "__main__":
    cfg = PiConfig('config.json')
    simulator = PiSimulator(cfg)
    logger = PiLogger()
    plotter = PiPlotter(cfg)
    
    loop_z_range = [0.3, 0.3]
    target_z_range = [0, 0.3]
    num_target_present = 3
    num_target_absent = 1
    
    print("\n" + "="*70)
    print("TDEM Simulation System")
    print("="*70)
    print("\nOptions:")
    print("  1. Generate HDF5 dataset")
    print("  2. Visualize HDF5 data")
    print("  3. Single simulation test")
    print("  4. Print dataset statistics")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        hdf5_path = run_simulations(
            simulator, logger,
            loop_z_range=loop_z_range,
            target_z_range=target_z_range,
            num_target_present=num_target_present,
            num_target_absent=num_target_absent,
            target_in_soil=False
        )
        plotter.load_from_hdf5(hdf5_path)
        
        visualize = input("\nVisualize results from file? (y/n): ").strip().lower()
        if visualize == 'y':
            plotter.run()
        
    elif choice == '2':
        hdf5_file = select_hdf5_file()
        
        if hdf5_file is None:
            print("Operation cancelled.")
        else:
            try:
                plotter.load_from_hdf5(hdf5_file)
                plotter.run()
            except FileNotFoundError:
                print(f"Error: File '{hdf5_file}' not found.")
            except Exception as e:
                print(f"Error: {e}")
    
    elif choice == '3':
        print("\n=== Single Simulation Test ===")
        target_present = input("Include target? (y/n): ").strip().lower() == 'y'
        
        time, decay, label, metadata, _ = simulator.run(
            loop_z_range=loop_z_range,
            target_z_range=target_z_range
        )
        
        print(f"\n✓ Complete: {label}")
        print(f"  Loop: {metadata['loop_z']:.3f} m")
        print(f"  Target: {metadata['target_z']:.3f} m" if metadata['target_z'] else "  No target")
        
        visualize = input("\nVisualize? (y/n): ").strip().lower()
        if visualize == 'y':
            fig, ax = plt.subplots(figsize=(10, 6))
            time_us = time * 1e6
            color = 'green' if target_present else 'blue'
            
            ax.loglog(time_us, decay, linewidth=2, color=color, label=label)
            ax.set_xlabel('Time [μs]', fontsize=12)
            ax.set_ylabel('-dBz/dt [T/s]', fontsize=12)
            ax.set_title(f'TDEM Decay - {"Target Present" if target_present else "No Target"}', 
                        fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, which='both')
            plt.tight_layout()
            plt.show()
    
    elif choice == '4':
        hdf5_file = select_hdf5_file()
        
        if hdf5_file is None:
            print("Operation cancelled.")
        else:
            try:
                logger.print_hdf5_metadata(hdf5_file)
            except FileNotFoundError:
                print(f"Error: File '{hdf5_file}' not found.")
            except Exception as e:
                print(f"Error: {e}")
    
    else:
        print("Invalid option selected.")
