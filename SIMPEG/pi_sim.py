from discretize import CylindricalMesh
from discretize.utils import mkvc

from simpeg import maps
import simpeg.electromagnetics.time_domain as tdem

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from pi_config import PiConfig
from pi_plotter import PiPlotter
from pi_logger import PiLogger


class PiSimulator:
    def __init__(self, config, loop_z, target_z, decays_target_present=100, decays_target_absent=100): 
        self.cfg = config
        # These two defines the range in which the simulation data should be randomized within
        self.loop_z = loop_z
        self.target_z = target_z
        
        self.decays_target_present = decays_target_present
        self.decays_target_absent = decays_target_absent

    def create_surveys(self, time_channels, waveform):
        surveys_list = []
    
        for i in range(self.decays_target_present+self.decays_target_absent):
            loop_z_val = np.random.uniform(self.loop_z[0], self.loop_z[1])
            while loop_z_val < self.cfg.separation_z:
                print(f"Warning: Generated loop_z={loop_z_val:.4f} < separation_z={self.cfg.separation_z}. Regenerating...")
                loop_z_val = np.random.uniform(self.loop_z[0], self.loop_z[1])
            
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
            
            surveys_list.append({
                'loop_z': loop_z_val,
                'survey': survey
            })        
        print(f"Created {len(surveys_list)} surveys successfully.\n")
        return surveys_list

    def create_conductivity_models(self, mesh):
        model_list = []
        for i in range(self.decays_target_present+self.decays_target_absent):
            target_z_val = None
            if i < self.decays_target_present:
                target_z_val = np.random.uniform(self.target_z[0], self.target_z[1])
                while target_z_val > self.cfg.separation_z:
                    print(f"Warning: Generated target_z={target_z_val:.4f} > separation_z={-self.cfg.separation_z}. Regenerating...")
                    target_z_val = np.random.uniform(self.target_z[0], self.target_z[1])
            
            active_area_z = self.cfg.separation_z
            ind_active = mesh.cell_centers[:, 2] < active_area_z

            model_map = maps.InjectActiveCells(mesh, ind_active, self.cfg.air_conductivity)

            r = mesh.cell_centers[ind_active, 0]
            z = mesh.cell_centers[ind_active, 2]

            model = self.cfg.air_conductivity * np.ones(ind_active.sum())

            ind_soil = (z < 0)
            model[ind_soil] = self.cfg.soil_conductivity

            if i < self.decays_target_present:
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

            model_list.append({
                "target_z": target_z_val,
                "model": model,
                "model_map": model_map
            })
        return model_list

    def run(self):
        print("Running simulation with the following configuration:")
        try:
            print(self.cfg.summary())
        except UnicodeEncodeError:
            print("[Configuration summary skipped due to encoding issue]")
        print(f"\nGenerating {self.decays_target_present} target-present and {self.decays_target_absent} target-absent simulations...")

        # 1. Define the waveform and time channels
        waveform = tdem.sources.StepOffWaveform(off_time=self.cfg.waveform_off_time)
        time_channels = np.linspace(0, 1024e-6, 1024)

        # 2. Create all surveys (returns list with decays_target_present + decays_target_absent surveys)
        print("\n=== Creating Surveys ===")
        surveys_list = self.create_surveys(time_channels, waveform)

        # 3. Create the mesh
        print("\n=== Creating Mesh ===")
        hr = [(0.01, 15), (0.01, 15, 1.3), (0.05, 10, 1.5)]
        hphi = 1
        hz = [(0.01, 10, -1.3), (0.01, 30), (0.01, 10, 1.3)]
        mesh = CylindricalMesh([hr, hphi, hz], x0="00C")
        print(f"Mesh created with {mesh.nC} cells")

        # 4. Create all conductivity models (returns list with all models, targets in first decays_target_absent entries due to bug)
        print("\n=== Creating Conductivity Models ===")
        models_list = self.create_conductivity_models(mesh)
        print(f"Created {len(models_list)} conductivity models")

        # 5. Create and run simulations for ALL PERMUTATIONS of surveys and models
        # Target-present models are in indices 0 to decays_target_present-1
        # Target-absent models are in indices decays_target_present to end
        
        print("\n=== Running ALL Permutations (Surveys × Models) ===")
        print(f"Total surveys: {len(surveys_list)}")
        print(f"Target-present models: {self.decays_target_present}")
        print(f"Target-absent models: {self.decays_target_absent}")
        
        target_present_times = []
        target_present_decays = []
        target_present_metadata = []  # Store survey and model info
        
        target_absent_times = []
        target_absent_decays = []
        target_absent_metadata = []
        
        simulation_count = 0
        total_simulations = len(surveys_list) * len(models_list)
        
        print(f"Total simulations to run: {total_simulations}")
        
        # Permutations with target-present models
        print("\n=== Target-Present Permutations ===")
        for survey_idx in range(len(surveys_list)):
            for model_idx in range(self.decays_target_present):
                simulation = tdem.simulation.Simulation3DMagneticFluxDensity(
                    mesh, 
                    survey=surveys_list[survey_idx]['survey'], 
                    sigmaMap=models_list[model_idx]["model_map"], 
                    t0=self.cfg.simulation_t0
                )
                simulation.time_steps = self.cfg.time_steps
                
                dpred = simulation.dpred(m=models_list[model_idx]["model"])
                dpred = np.reshape(dpred, (1, len(time_channels)))
                
                loop_z_val = surveys_list[survey_idx]['loop_z']
                target_z_val = models_list[model_idx]['target_z']
                label = f"L{loop_z_val:.2f}-T{target_z_val:.2f}"
                
                target_present_times.append(time_channels)
                target_present_decays.append(-dpred[0, :])
                target_present_metadata.append({
                    'survey_idx': survey_idx,
                    'model_idx': model_idx,
                    'loop_z': loop_z_val,
                    'target_z': target_z_val,
                    'label': label
                })
                
                simulation_count += 1
                print(f"Progress: {simulation_count}/{total_simulations} - {label} (Survey {survey_idx}, Model {model_idx})")

        # Permutations with target-absent models
        print("\n=== Target-Absent Permutations ===")
        for survey_idx in range(len(surveys_list)):
            for model_idx in range(self.decays_target_present, len(models_list)):
                simulation = tdem.simulation.Simulation3DMagneticFluxDensity(
                    mesh, 
                    survey=surveys_list[survey_idx]['survey'], 
                    sigmaMap=models_list[model_idx]["model_map"], 
                    t0=self.cfg.simulation_t0
                )
                simulation.time_steps = self.cfg.time_steps
                
                dpred = simulation.dpred(m=models_list[model_idx]["model"])
                dpred = np.reshape(dpred, (1, len(time_channels)))
                
                # Generate label: L<loop_z>-T- (no target)
                loop_z_val = surveys_list[survey_idx]['loop_z']
                label = f"L{loop_z_val:.2f}-T-"
                
                target_absent_times.append(time_channels)
                target_absent_decays.append(-dpred[0, :])
                target_absent_metadata.append({
                    'survey_idx': survey_idx,
                    'model_idx': model_idx,
                    'loop_z': loop_z_val,
                    'target_z': None,  # No target in these models
                    'label': label
                })
                
                simulation_count += 1
                print(f"Progress: {simulation_count}/{total_simulations} - {label} (Survey {survey_idx}, Model {model_idx})")

        # 7. Prepare plotting info
        print("\n=== Preparing Plotting Information ===")
        # Create reference models for plotting
        active_area_z = self.cfg.separation_z
        ind_active = mesh.cell_centers[:, 2] < active_area_z
        
        # Model without target
        model_no_target = self.cfg.air_conductivity * np.ones(ind_active.sum())
        r = mesh.cell_centers[ind_active, 0]
        z = mesh.cell_centers[ind_active, 2]
        ind_soil = (z < 0)
        model_no_target[ind_soil] = self.cfg.soil_conductivity
        
        # Model with target (use first model from list)
        model_with_target = models_list[0]["model"] if len(models_list) > 0 else model_no_target
        
        plotting_info = {
            'mesh': mesh,
            'model_with_target': model_with_target,
            'model_no_target': model_no_target,
            'ind_active': ind_active,
            'cfg': self.cfg,
            'target_z_example': models_list[0]["target_z"] if len(models_list) > 0 else None,
            'loop_z_example': surveys_list[0]['loop_z'],
            'num_target_present': self.decays_target_present,
            'num_target_absent': self.decays_target_absent,
            # Legacy fields for PiPlotter compatibility
            'loop_z_start': self.loop_z[0],
            'loop_z_increment': (self.loop_z[1] - self.loop_z[0]) / max(1, self.decays_target_present + self.decays_target_absent - 1),
            'num_increments': self.decays_target_present + self.decays_target_absent - 1,
            # New fields for permutation tracking
            'surveys_list': surveys_list,
            'models_list': models_list,
            'target_present_metadata': target_present_metadata,
            'target_absent_metadata': target_absent_metadata
        }
        
        print("\n=== Simulation Complete ===")
        print(f"Generated {len(target_present_decays)} target-present decay curves")
        print(f"Generated {len(target_absent_decays)} target-absent decay curves")
        print(f"Total permutations: {len(target_present_decays) + len(target_absent_decays)}")
        
        return target_present_times, target_present_decays, target_absent_times, target_absent_decays, plotting_info

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


cfg = PiConfig('config.json')
# Define ranges for randomization: loop_z = [min, max], target_z = [min, max]
simulator = PiSimulator(
    cfg, 
    loop_z=[0.3, 0.5],      # Loop height range (must be >= separation_z = 0.3)
    target_z=[-0.5, -0.3],  # Target depth range (must be <= -separation_z = -0.3)
    decays_target_present=40,   # Reduced for testing
    decays_target_absent=40     # Reduced for testing
)
conditioner = PiConditioner()

target_present_times, target_present_decays, target_absent_times, target_absent_decays, plotting_info = simulator.run()
plotter = PiPlotter(
    plotting_info,
    plot_linear=True,
    plot_loglog=True,
    plot_side_view=True,
    plot_top_view=True,
    plot_3d=True,
    plot_combined=True  # Enable the new combined plot
)

print("\n=== Updating Plotter ===")
# Add each permutation individually with its unique label
for i, (time, decay) in enumerate(zip(target_present_times, target_present_decays)):
    label = plotting_info['target_present_metadata'][i]['label']
    plotter.update_times_data([time], [decay], label=label, replace=False)

for i, (time, decay) in enumerate(zip(target_absent_times, target_absent_decays)):
    label = plotting_info['target_absent_metadata'][i]['label']
    plotter.update_times_data([time], [decay], label=label, replace=False)

# Create logger and save dataset
print("\n=== Creating ML Dataset ===")
logger = PiLogger()

# Log all target-present data
for i, (time, decay) in enumerate(zip(target_present_times, target_present_decays)):
    metadata = plotting_info['target_present_metadata'][i]
    label = metadata['label']
    logger.append_data(decay, label, metadata)

# Log all target-absent data
for i, (time, decay) in enumerate(zip(target_absent_times, target_absent_decays)):
    metadata = plotting_info['target_absent_metadata'][i]
    label = metadata['label']
    logger.append_data(decay, label, metadata)

# Print dataset summary
logger.print_summary()

# Ask user if they want to export the dataset
print("\nWould you like to export this dataset to CSV files?")
print("This will split the data into train/validation/test sets.")
export_choice = input("Export dataset? (y/n): ").strip().lower()

if export_choice == 'y':
    # Get split percentages
    try:
        train_pct = float(input("Training set percentage (default 70): ").strip() or "70")
        test_pct = float(input("Test set percentage (default 15): ").strip() or "15")
        output_dir = input("Output directory (default 'dataset'): ").strip() or "dataset"
        
        # Split and export
        train_path, val_path, test_path = logger.split_data(
            train_percent=train_pct,
            test_percent=test_pct,
            output_dir=output_dir,
            seed=42
        )
        
        print("\n✓ Dataset exported successfully!")
        print(f"\nTo use with TensorFlow:")
        print(f"  1. Load CSV: df = pd.read_csv('{train_path}')")
        print(f"  2. Extract features: X = df[[col for col in df.columns if col.startswith('feature_')]].values")
        print(f"  3. Extract labels: y = df['binary_label'].values")
        print(f"  4. Train CNN on X (input) and y (output)")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Skipping dataset export.")
else:
    print("Dataset export skipped.")

# Show interactive plots
plotter.show_plots()
