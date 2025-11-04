import numpy as np
import h5py
from typing import Dict, Any, Tuple
from datetime import datetime


class PiLogger:
    
    def __init__(self):
        pass
    
    def initialize_hdf5(self, filename: str, num_target_present: int, num_target_absent: int, has_magnetic_field: bool = False):
        with h5py.File(filename, 'w') as f:
            f.create_group('simulations')
            meta_group = f.create_group('metadata')
            meta_group.attrs['creation_time'] = datetime.now().isoformat()
            meta_group.attrs['num_simulations'] = num_target_present + num_target_absent
            meta_group.attrs['num_target_present'] = num_target_present
            meta_group.attrs['num_target_absent'] = num_target_absent
            meta_group.attrs['has_magnetic_field'] = has_magnetic_field
    
    def store_to_hdf5(self, filename: str, sim_index: int, time: np.ndarray, 
                      decay: np.ndarray, label: str, metadata: Dict[str, Any], 
                      magnetic_field_data: Dict[str, Any] = None):
        with h5py.File(filename, 'a') as f:
            sim = f['simulations'].create_group(f'simulation_{sim_index}')
            sim.create_dataset('time', data=time, compression='gzip', compression_opts=4)
            sim.create_dataset('decay', data=decay, compression='gzip', compression_opts=4)
            sim.attrs['label'] = label
            sim.attrs['loop_z'] = metadata['loop_z']
            sim.attrs['target_present'] = metadata['target_present']
            sim.attrs['has_magnetic_field'] = metadata.get('has_magnetic_field', False)
            
            if metadata['target_z'] is not None:
                sim.attrs['target_z'] = metadata['target_z']
            else:
                sim.attrs['target_z'] = -999.0
            
            if magnetic_field_data is not None:
                mag_group = sim.create_group('magnetic_field')
                mag_group.create_dataset('selected_times', data=magnetic_field_data['selected_times'], 
                                        compression='gzip', compression_opts=4)
                
                mag_group.attrs['n_cells'] = magnetic_field_data['mesh_info']['n_cells']
                mag_group.attrs['n_faces'] = magnetic_field_data['mesh_info']['n_faces']
                
                if 'cell_centers' in magnetic_field_data['mesh_info']:
                    mag_group.create_dataset('cell_centers', data=magnetic_field_data['mesh_info']['cell_centers'],
                                            compression='gzip', compression_opts=4)
                elif 'cell_centers_active' in magnetic_field_data['mesh_info']:
                    mag_group.create_dataset('cell_centers_active', data=magnetic_field_data['mesh_info']['cell_centers_active'],
                                            compression='gzip', compression_opts=4)
                
                if 'loop_position' in magnetic_field_data['mesh_info']:
                    mag_group.create_dataset('loop_position', data=magnetic_field_data['mesh_info']['loop_position'],
                                            compression='gzip', compression_opts=4)
                    mag_group.attrs['loop_radius'] = magnetic_field_data['mesh_info']['loop_radius']
                
                snapshots_group = mag_group.create_group('snapshots')
                for idx, snapshot in enumerate(magnetic_field_data['field_snapshots']):
                    snapshots_group.create_dataset(f'field_{idx}', data=snapshot,
                                                  compression='gzip', compression_opts=4)

    def finalize_hdf5(self, filename: str, time_samples: int):
        with h5py.File(filename, 'a') as f:
            f['metadata'].attrs['time_samples'] = time_samples
    
    def load_from_hdf5(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, Dict[str, Any]]:
        print(f"Loading data from HDF5: {filepath}")
        
        with h5py.File(filepath, 'r') as f:
            num_sims = f['metadata'].attrs['num_simulations']
            has_mag_field = f['metadata'].attrs.get('has_magnetic_field', False)
            decay_curves = []
            label_strings = []
            labels = []
            loop_z_list = []
            target_z_list = []
            
            sim0 = f['simulations/simulation_0']
            time = sim0['time'][:]
            
            for i in range(num_sims):
                sim_group = f[f'simulations/simulation_{i}']
                decay = sim_group['decay'][:]
                label_str = sim_group.attrs['label']
                loop_z = sim_group.attrs['loop_z']
                target_present = sim_group.attrs['target_present']
                target_z = sim_group.attrs['target_z']
                
                if target_z == -999.0:
                    target_z = None
                
                decay_curves.append(decay)
                label_strings.append(label_str)
                labels.append(1 if target_present else 0)
                loop_z_list.append(loop_z)
                target_z_list.append(target_z)
            
            decay_curves = np.array(decay_curves)
            labels = np.array(labels)
            
            metadata = {
                'num_simulations': num_sims,
                'num_target_present': f['metadata'].attrs['num_target_present'],
                'num_target_absent': f['metadata'].attrs['num_target_absent'],
                'time_samples': f['metadata'].attrs['time_samples'],
                'creation_time': f['metadata'].attrs['creation_time'],
                'loop_z': np.array(loop_z_list),
                'target_z': np.array(target_z_list, dtype=object),
                'has_magnetic_field': has_mag_field
            }
        
        print(f"Loaded {num_sims} simulations from HDF5")
        if has_mag_field:
            print("  ✓ Magnetic field data available")
        return time, decay_curves, labels, label_strings, metadata
    
    def print_hdf5_metadata(self, filename: str):
        with h5py.File(filename, 'r') as f:
            print(f"\n{'='*70}")
            print(f"HDF5 Dataset: {filename}")
            print(f"{'='*70}")
            print(f"Total simulations: {f['metadata'].attrs['num_simulations']}")
            print(f"Target present: {f['metadata'].attrs['num_target_present']}")
            print(f"Target absent: {f['metadata'].attrs['num_target_absent']}")
            print(f"Time samples: {f['metadata'].attrs['time_samples']}")
            print(f"Created: {f['metadata'].attrs['creation_time']}")
            has_mag_field = f['metadata'].attrs.get('has_magnetic_field', False)
            print(f"Magnetic field data: {'Yes' if has_mag_field else 'No'}")
            print(f"{'='*70}\n")
