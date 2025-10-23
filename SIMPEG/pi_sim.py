from discretize import CylindricalMesh
from discretize.utils import mkvc

from simpeg import maps
import simpeg.electromagnetics.time_domain as tdem

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


import h5py
from datetime import datetime

from pi_config import PiConfig
from pi_plotter import PiPlotter
from pi_logger import PiLogger


class PiSimulator:
    def __init__(self, config): 
        self.cfg = config

    def create_survey(self, time_channels, waveform, loop_z_range):
        loop_z_val = np.random.uniform(loop_z_range[0], loop_z_range[1])
        while loop_z_val < self.cfg.separation_z:
            print(f"Warning: Generated loop_z={loop_z_val:.4f} < separation_z={self.cfg.separation_z}. Regenerating...")
            loop_z_val = np.random.uniform(loop_z_range[0], loop_z_range[1])
        
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
        return loop_z_val, survey

    def create_conductivity_model(self, mesh, target_z_range, target_present):
        target_z_val = np.random.uniform(target_z_range[0], target_z_range[1])
        while target_z_val > self.cfg.separation_z:
            print(f"Warning: Generated target_z={target_z_val:.4f} > separation_z={-self.cfg.separation_z}. Regenerating...")
            target_z_val = np.random.uniform(target_z_range[0], target_z_range[1])
        
        active_area_z = self.cfg.separation_z
        ind_active = mesh.cell_centers[:, 2] < active_area_z

        model_map = maps.InjectActiveCells(mesh, ind_active, self.cfg.air_conductivity)

        r = mesh.cell_centers[ind_active, 0]
        z = mesh.cell_centers[ind_active, 2]

        model = self.cfg.air_conductivity * np.ones(ind_active.sum())

        ind_soil = (z < 0)
        model[ind_soil] = self.cfg.soil_conductivity

        if target_present: 
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
        return target_z_val, model, model_map

    def run(self, loop_z_range, target_z_range, target_present, mesh=None):
        # 1. Define the waveform and time channels
        waveform = tdem.sources.StepOffWaveform(off_time=self.cfg.waveform_off_time)
        time_channels = np.linspace(0, 1024e-6, 1024)

        # 2. Create one survey with the range given
        loop_z_val, survey = self.create_survey(time_channels, waveform, loop_z_range)

        # 3. Create or reuse the mesh
        if mesh is None:
            hr = [(0.01, 15), (0.01, 15, 1.3), (0.05, 10, 1.5)]
            hphi = 1
            hz = [(0.01, 10, -1.3), (0.01, 30), (0.01, 10, 1.3)]
            mesh = CylindricalMesh([hr, hphi, hz], x0="00C")
            create_plotting_metadata = True
        else:
            create_plotting_metadata = False

        # 4. Create conductivity model
        target_z_val, model, model_map = self.create_conductivity_model(mesh, target_z_range, target_present)

        # 5. Run simulation based on the survey and the model        
        simulation = tdem.simulation.Simulation3DMagneticFluxDensity(
            mesh,
            survey=survey,
            sigmaMap=model_map,
            t0=self.cfg.simulation_t0
        )
        simulation.time_steps = self.cfg.time_steps

        dpred = simulation.dpred(m=model)
        dpred = np.reshape(dpred, (1, len(time_channels)))
        
        # Generate label
        if target_present:
            label = f"L{loop_z_val:.2f}-T{target_z_val:.2f}"
        else:
            label = f"L{loop_z_val:.2f}-T-"

        # Metadata related to the simulation
        simulation_metadata = {
            'loop_z': loop_z_val,
            'target_z': target_z_val if target_present else None,
            'target_present': target_present,
            'label': label
        }

        # Metadata related to the plotting (only create once)
        plotting_metadata = None
        if create_plotting_metadata:
            active_area_z = self.cfg.separation_z
            ind_active = mesh.cell_centers[:, 2] < active_area_z
            
            # Create reference model without target for plotting
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

    # Assuming additive WGN, late_time is the time the signal power is counted
    def add_noise(self, time, data, late_time, snr_db):
        # Handle if time is a list (from simulator.run())
        if isinstance(time, list):
            time = time[0]
        
        # Handle if time is 2D array, extract first row
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
        # Handle if time is a list (from simulator.run())
        if isinstance(time, list):
            time = time[0]
        
        # Handle if time is 2D array, extract first row
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
                   num_target_present, num_target_absent, output_file='tdem_dataset.h5'):
    """
    Run multiple simulations and save incrementally to HDF5.
    
    Parameters
    ----------
    simulator : PiSimulator
        Simulator instance
    logger : PiLogger
        Logger instance for HDF5 operations
    loop_z_range : list
        [min, max] range for loop height
    target_z_range : list
        [min, max] range for target depth
    num_target_present : int
        Number of simulations with target
    num_target_absent : int
        Number of simulations without target
    output_file : str
        HDF5 output filename
    
    Returns
    -------
    str
        Path to the generated HDF5 file
    """
    mesh = None
    total_sims = num_target_present + num_target_absent
    sim_index = 0
    
    # Initialize HDF5 file
    logger.initialize_hdf5(output_file, num_target_present, num_target_absent)
    
    print(f"\n=== Running Simulations (Writing to {output_file}) ===")
    
    # Run target-present simulations
    print(f"\nTarget-Present: 0/{num_target_present}", end='', flush=True)
    for i in range(num_target_present):
        time, decay, label, metadata, plot_meta = simulator.run(
            loop_z_range, target_z_range, target_present=True, mesh=mesh
        )
        
        if mesh is None and plot_meta is not None:
            mesh = plot_meta['mesh']
        
        # Write immediately to HDF5
        logger.append_to_hdf5(output_file, sim_index, time, decay, label, metadata)
        sim_index += 1
        
        print(f"\rTarget-Present: {i+1}/{num_target_present}", end='', flush=True)
    
    print()
    
    # Run target-absent simulations
    print(f"Target-Absent: 0/{num_target_absent}", end='', flush=True)
    for i in range(num_target_absent):
        time, decay, label, metadata, plot_meta = simulator.run(
            loop_z_range, target_z_range, target_present=False, mesh=mesh
        )
        
        # Write immediately to HDF5
        logger.append_to_hdf5(output_file, sim_index, time, decay, label, metadata)
        sim_index += 1
        
        print(f"\rTarget-Absent: {i+1}/{num_target_absent}", end='', flush=True)
    
    print()
    
    # Finalize metadata
    logger.finalize_hdf5(output_file, len(time))
    
    print(f"\n✓ Complete: {total_sims} simulations written to {output_file}")
    
    return output_file


def plot_from_hdf5(hdf5_file, cfg):
    """
    Plot from HDF5 file using PiPlotter - shows model reconstruction + decay curves.
    
    Parameters
    ----------
    hdf5_file : str
        Path to HDF5 file
    cfg : PiConfig
        Configuration object
    """
    from pi_plotter import plot_from_hdf5 as plotter_func
    plotter_func(hdf5_file, cfg)


if __name__ == "__main__":
    # Configuration
    cfg = PiConfig('config.json')
    simulator = PiSimulator(cfg)
    logger = PiLogger()
    
    # Define simulation parameters
    loop_z_range = [0.3, 0.6]      # Loop height range (must be >= separation_z = 0.3)
    target_z_range = [-0.5, -0.3]  # Target depth range (must be <= -separation_z = -0.3)
    num_target_present = 5          # Number of simulations with target
    num_target_absent = 5           # Number of simulations without target
    
    print("\n" + "="*70)
    print("TDEM Simulation System")
    print("="*70)
    print("\nOptions:")
    print("  1. Generate HDF5 dataset")
    print("  2. Convert HDF5 to CSV for ML")
    print("  3. Visualize HDF5 data")
    print("  4. Single simulation test")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        # Generate HDF5 dataset
        output_file = input("Output HDF5 file (default 'tdem_dataset.h5'): ").strip() or 'tdem_dataset.h5'
        
        hdf5_path = run_simulations(
            simulator, logger,
            loop_z_range=loop_z_range,
            target_z_range=target_z_range,
            num_target_present=num_target_present,
            num_target_absent=num_target_absent,
            output_file=output_file
        )
        
        # Ask if user wants to visualize
        visualize = input("\nVisualize results from file? (y/n): ").strip().lower()
        if visualize == 'y':
            plot_from_hdf5(hdf5_path, cfg)
        
    elif choice == '2':
        # Convert HDF5 to CSV
        hdf5_file = input("HDF5 file (default 'tdem_dataset.h5'): ").strip() or 'tdem_dataset.h5'
        
        try:
            # Load from HDF5
            print("\n=== Loading HDF5 Data ===")
            csv_logger = PiLogger()
            csv_logger.load_from_hdf5(hdf5_file)
            csv_logger.print_summary()
            
            # Export to CSV
            train_pct = float(input("\nTraining %% (default 70): ").strip() or "70")
            test_pct = float(input("Test %% (default 15): ").strip() or "15")
            output_dir = input("Output directory (default 'dataset'): ").strip() or "dataset"
            
            train_path, val_path, test_path = csv_logger.split_data(
                train_percent=train_pct,
                test_percent=test_pct,
                output_dir=output_dir,
                seed=42
            )
            
            print("\n✓ CSV export complete!")
            print(f"  Train: {train_path}")
            print(f"  Val: {val_path}")
            print(f"  Test: {test_path}")
            
        except FileNotFoundError:
            print(f"Error: File '{hdf5_file}' not found.")
        except Exception as e:
            print(f"Error: {e}")
    
    elif choice == '3':
        # Visualize HDF5 data
        hdf5_file = input("HDF5 file (default 'tdem_dataset.h5'): ").strip() or 'tdem_dataset.h5'
        
        try:
            plot_from_hdf5(hdf5_file, simulator.cfg)
        except FileNotFoundError:
            print(f"Error: File '{hdf5_file}' not found.")
        except Exception as e:
            print(f"Error: {e}")
    
    elif choice == '4':
        # Single simulation test
        print("\n=== Single Simulation Test ===")
        target_present = input("Include target? (y/n): ").strip().lower() == 'y'
        
        time, decay, label, metadata, _ = simulator.run(
            loop_z_range=loop_z_range,
            target_z_range=target_z_range,
            target_present=target_present
        )
        
        print(f"\n✓ Complete: {label}")
        print(f"  Loop: {metadata['loop_z']:.3f} m")
        print(f"  Target: {metadata['target_z']:.3f} m" if metadata['target_z'] else "  No target")
        
        # Quick plot
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
    
    else:
        print("Invalid option selected.")
