from tensorflow import keras 
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import h5py
from datetime import datetime


class PiClassifier():
    def __init__(self, conditioner=None):
        self.model = None
        self.conditioner = conditioner
        # Store split data in memory
        self.train_data = None  # (time, decay_curves, labels, label_strings, metadata)
        self.val_data = None
        self.test_data = None
        # Processed arrays for training
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.norm_params = None
    
    def split_dataset(self, logger, source_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, save_to_file=False): 
        print("\n" + "="*70)
        print("Conditioning Dataset: Train/Validation/Test Split")
        print("="*70)
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        
        time, decay_curves, labels, label_strings, metadata = logger.load_from_hdf5(source_path)
        
        num_samples = len(decay_curves)
        print(f"Total samples: {num_samples}")
        print(f"  - Target present: {np.sum(labels == 1)}")
        print(f"  - Target absent: {np.sum(labels == 0)}")
        
        # Shuffle indices
        indices = np.random.permutation(num_samples)
        
        # Calculate split points
        train_end = int(num_samples * train_ratio)
        val_end = train_end + int(num_samples * val_ratio)
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        print(f"\nSplit sizes:")
        print(f"  - Training: {len(train_idx)} samples ({len(train_idx)/num_samples*100:.1f}%)")
        print(f"  - Validation: {len(val_idx)} samples ({len(val_idx)/num_samples*100:.1f}%)")
        print(f"  - Testing: {len(test_idx)} samples ({len(test_idx)/num_samples*100:.1f}%)")
        
        # Store data in class (in memory)
        self.train_data = (
            time,
            decay_curves[train_idx],
            labels[train_idx],
            [label_strings[i] for i in train_idx],
            {**metadata, 'num_simulations': len(train_idx)}
        )
        
        self.val_data = (
            time,
            decay_curves[val_idx],
            labels[val_idx],
            [label_strings[i] for i in val_idx],
            {**metadata, 'num_simulations': len(val_idx)}
        )
        
        self.test_data = (
            time,
            decay_curves[test_idx],
            labels[test_idx],
            [label_strings[i] for i in test_idx],
            {**metadata, 'num_simulations': len(test_idx)}
        )
        
        # Normalize data using conditioner
        if self.conditioner:
            decay_normalized, self.norm_params = self.conditioner.normalize_for_training(decay_curves)
            print(f"\n✓ Data normalized: log-scale + min-max [range: {self.norm_params['min']:.2f} to {self.norm_params['max']:.2f}]")
        else:
            decay_normalized = decay_curves
            print(f"\n⚠️  Warning: No conditioner provided, using raw data")
        
        self.X_train = decay_normalized[train_idx].reshape(len(train_idx), decay_curves.shape[1], 1)
        self.y_train = labels[train_idx]
        self.X_val = decay_normalized[val_idx].reshape(len(val_idx), decay_curves.shape[1], 1)
        self.y_val = labels[val_idx]
        self.X_test = decay_normalized[test_idx].reshape(len(test_idx), decay_curves.shape[1], 1)
        self.y_test = labels[test_idx]
        
        print("\n✓ Dataset split complete! Data stored in memory.")
        
        # Optional: Save to files
        if save_to_file:
            print("\nSaving split datasets to files...")
            base_dir = os.path.dirname(source_path)
            base_name = os.path.basename(source_path).replace('.h5', '')
            
            # Create dataset folder
            dataset_folder = os.path.join(base_dir, base_name)
            os.makedirs(dataset_folder, exist_ok=True)
            
            train_path = os.path.join(dataset_folder, f"{base_name}_train.h5")
            val_path = os.path.join(dataset_folder, f"{base_name}_val.h5")
            test_path = os.path.join(dataset_folder, f"{base_name}_test.h5")
            
            self._save_split_dataset(source_path, train_path, train_idx, "Training")
            self._save_split_dataset(source_path, val_path, val_idx, "Validation")
            self._save_split_dataset(source_path, test_path, test_idx, "Testing")
            
            print(f"\n✓ Files saved in: {dataset_folder}")
            print(f"  - Training set: {os.path.basename(train_path)}")
            print(f"  - Validation set: {os.path.basename(val_path)}")
            print(f"  - Testing set: {os.path.basename(test_path)}")
            
            return train_path, val_path, test_path
        
        return None, None, None
    
    def _save_split_dataset(self, source_path, dest_path, indices, split_name):
        """Helper function to save a subset of the dataset"""
        with h5py.File(source_path, 'r') as f_src:
            with h5py.File(dest_path, 'w') as f_dst:
                # Copy metadata
                metadata_group = f_dst.create_group('metadata')
                src_meta = f_src['metadata']
                
                # Count target present/absent in this split
                num_target_present = 0
                num_target_absent = 0
                for idx in indices:
                    sim = f_src[f'simulations/simulation_{idx}']
                    if sim.attrs['target_present']:
                        num_target_present += 1
                    else:
                        num_target_absent += 1
                
                metadata_group.attrs['num_simulations'] = len(indices)
                metadata_group.attrs['num_target_present'] = num_target_present
                metadata_group.attrs['num_target_absent'] = num_target_absent
                metadata_group.attrs['time_samples'] = src_meta.attrs['time_samples']
                metadata_group.attrs['creation_time'] = src_meta.attrs['creation_time']
                metadata_group.attrs['split_type'] = split_name
                metadata_group.attrs['parent_dataset'] = source_path
                
                if 'has_magnetic_field' in src_meta.attrs:
                    metadata_group.attrs['has_magnetic_field'] = src_meta.attrs['has_magnetic_field']
                
                # Create simulations group
                sims_group = f_dst.create_group('simulations')
                
                # Copy selected simulations
                for new_idx, orig_idx in enumerate(indices):
                    src_sim = f_src[f'simulations/simulation_{orig_idx}']
                    dst_sim = sims_group.create_group(f'simulation_{new_idx}')
                    
                    # Copy datasets
                    for key in src_sim.keys():
                        if isinstance(src_sim[key], h5py.Dataset):
                            src_sim.copy(key, dst_sim)
                        elif isinstance(src_sim[key], h5py.Group):
                            # Handle nested groups (e.g., magnetic_field)
                            src_sim.copy(key, dst_sim)
                    
                    # Copy attributes
                    for attr_name, attr_value in src_sim.attrs.items():
                        dst_sim.attrs[attr_name] = attr_value
        
        print(f"  ✓ {split_name} set saved: {dest_path} ({len(indices)} samples)")
    
    def build_model(self, num_samples):
        inputs = keras.Input(shape=(num_samples, 1))
        x = layers.Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Conv1D(filters=128, kernel_size=3, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(2, activation='softmax')(x)
        self.model = keras.Model(inputs, outputs)
        return self.model
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, epochs=20, batch_size=32):
        print("\n" + "="*70)
        print("Training Classifier")
        print("="*70)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Target present: {np.sum(y_train == 1)}, Target absent: {np.sum(y_train == 0)}")
        
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        if validation_data:
            print(f"Validation samples: {len(X_val)}")
        
        optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        self.model.compile(optimizer=optimizer,
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=1
        )
        
        self.plot_training_history(history)
        print("\n✓ Training complete!")
        
        return history
    
    def validate_model(self, X_val, y_val):
        print("\n" + "="*70)
        print("Validating Model")
        print("="*70)
        
        print(f"Validation samples: {len(X_val)}")
        val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=1)
        
        print(f"\nLoss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        return val_loss, val_accuracy
    
    def test_model(self, X_test, y_test):
        print("\n" + "="*70)
        print("Testing Model")
        print("="*70)
        
        print(f"Test samples: {len(X_test)}")
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=1)
        
        print(f"\nLoss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
        return test_loss, test_accuracy
    
    def visualize_training_data(self, time, decay_curves, labels, label_strings, metadata):
        """Visualize the loaded training data"""
        print("\n" + "="*70)
        print("Training Data Visualization")
        print("="*70)
        print(f"Dataset: {metadata['num_simulations']} simulations")
        print(f"  - Time samples: {len(time)}")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Training Data Overview', fontsize=16, fontweight='bold')
        
        time_us = time * 1e6  # Convert to microseconds
        
        # Plot 1: All decay curves colored by label
        ax = axes[0, 0]
        for i, (decay, label, label_str) in enumerate(zip(decay_curves, labels, label_strings)):
            color = 'green' if label == 1 else 'blue'
            alpha = 0.3
            ax.loglog(time_us, decay, color=color, alpha=alpha, linewidth=1)
        
        # Add legend
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
        target_present_indices = np.where(labels == 1)[0][:5]  # First 5
        for idx in target_present_indices:
            ax.loglog(time_us, decay_curves[idx], linewidth=2, label=label_strings[idx])
        ax.set_xlabel('Time [μs]', fontsize=11)
        ax.set_ylabel('-dBz/dt [T/s]', fontsize=11)
        ax.set_title('Sample: Target Present', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        
        # Plot 3: Sample target-absent curves
        ax = axes[1, 0]
        target_absent_indices = np.where(labels == 0)[0][:5]  # First 5
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

    def plot_training_history(self, history):
        """Plot training and validation accuracy/loss"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Accuracy', fontsize=11)
        ax1.set_title('Model Accuracy', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Loss', fontsize=11)
        ax2.set_title('Model Loss', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

