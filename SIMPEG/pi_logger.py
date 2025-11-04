import numpy as np
import h5py
from typing import Dict, Any, Tuple
from datetime import datetime


class PiLogger:
    
    def __init__(self):
        pass
    
    def initialize_hdf5(self, filename: str, num_target_present: int, num_target_absent: int):
        with h5py.File(filename, 'w') as f:
            f.create_group('simulations')
            meta_group = f.create_group('metadata')
            meta_group.attrs['creation_time'] = datetime.now().isoformat()
            meta_group.attrs['num_simulations'] = num_target_present + num_target_absent
            meta_group.attrs['num_target_present'] = num_target_present
            meta_group.attrs['num_target_absent'] = num_target_absent
    
    def store_to_hdf5(self, filename: str, sim_index: int, time: np.ndarray, 
                      decay: np.ndarray, label: str, metadata: Dict[str, Any]):
        with h5py.File(filename, 'a') as f:
            sim = f['simulations'].create_group(f'simulation_{sim_index}')
            sim.create_dataset('time', data=time, compression='gzip', compression_opts=4)
            sim.create_dataset('decay', data=decay, compression='gzip', compression_opts=4)
            sim.attrs['label'] = label
            sim.attrs['loop_z'] = metadata['loop_z']
            sim.attrs['target_present'] = metadata['target_present']
            
            if metadata['target_z'] is not None:
                sim.attrs['target_z'] = metadata['target_z']
            else:
                sim.attrs['target_z'] = -999.0

    def finalize_hdf5(self, filename: str, time_samples: int):
        with h5py.File(filename, 'a') as f:
            f['metadata'].attrs['time_samples'] = time_samples
    
    def load_from_hdf5(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, Dict[str, Any]]:
        print(f"Loading data from HDF5: {filepath}")
        
        with h5py.File(filepath, 'r') as f:
            num_sims = f['metadata'].attrs['num_simulations']
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
                'target_z': np.array(target_z_list, dtype=object)
            }
        
        print(f"Loaded {num_sims} simulations from HDF5")
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
            print(f"{'='*70}\n")
    
    def save_model(self, model, model_path: str, dataset_name: str, split_type: str):
        """
        Save a Keras model to HDF5 format with metadata.
        
        Args:
            model: Keras model to save
            model_path: Path where the model will be saved
            dataset_name: Name of the dataset used for training
            split_type: Type of split ('train', 'val', 'test', or 'full')
        """
        print(f"\n{'='*70}")
        print(f"Saving Model")
        print(f"{'='*70}")
        print(f"Model path: {model_path}")
        print(f"Dataset: {dataset_name}")
        print(f"Split type: {split_type}")
        
        # Save the model
        model.save(model_path)
        
        # Add custom metadata
        with h5py.File(model_path, 'a') as f:
            if 'model_metadata' not in f:
                meta = f.create_group('model_metadata')
            else:
                meta = f['model_metadata']
            
            meta.attrs['dataset_name'] = dataset_name
            meta.attrs['split_type'] = split_type
            meta.attrs['saved_time'] = datetime.now().isoformat()
        
        print(f"✓ Model saved successfully!")
        print(f"{'='*70}\n")
    
    def load_model(self, model_path: str):
        """
        Load a Keras model from HDF5 format and print metadata.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded Keras model
        """
        import os
        from tensorflow import keras
        
        if not os.path.exists(model_path):
            return None
        
        print(f"\n{'='*70}")
        print(f"Loading Model")
        print(f"{'='*70}")
        print(f"Model path: {model_path}")
        
        # Load the model
        model = keras.models.load_model(model_path)
        
        # Read metadata
        try:
            with h5py.File(model_path, 'r') as f:
                if 'model_metadata' in f:
                    meta = f['model_metadata']
                    print(f"Dataset: {meta.attrs.get('dataset_name', 'N/A')}")
                    print(f"Split type: {meta.attrs.get('split_type', 'N/A')}")
                    print(f"Saved: {meta.attrs.get('saved_time', 'N/A')}")
        except Exception as e:
            print(f"Warning: Could not read metadata: {e}")
        
        print(f"✓ Model loaded successfully!")
        print(f"{'='*70}\n")
        
        return model
    
    def get_model_path_for_dataset(self, dataset_path: str) -> str:
        """
        Generate the model path based on the dataset path.
        
        Args:
            dataset_path: Path to the dataset HDF5 file
        
        Returns:
            Path where the model should be saved
        """
        import os
        
        # Remove extension and add _model.h5
        base_path = dataset_path.replace('.h5', '').replace('.hdf5', '')
        model_path = f"{base_path}_model.h5"
        
        return model_path
