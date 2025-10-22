"""
TDEM Data Logger for Machine Learning
======================================

This module provides logging and dataset management for TDEM simulation data,
with support for TensorFlow-compatible data export and visualization.

Author: TDEM Simulation Project
Date: October 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime


class PiLogger:
    """
    Data logger for TDEM simulations with ML dataset management.
    
    Supports logging data with target presence/absence labels, random splitting
    into train/validation/test sets, and export to CSV format for TensorFlow.
    """
    
    def __init__(self):
        """
        Initialize the logger with empty data structures.
        """
        # Data storage: list of dictionaries containing data and metadata
        self.data_entries = []
        
        # Metadata tracking
        self.num_target_present = 0
        self.num_target_absent = 0
        self.data_shape = None  # Will be set on first append
        
        print("PiLogger initialized. Ready to log TDEM data.")
    
    def append_data(self, data: np.ndarray, label: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Append TDEM data with configuration label.
        
        Parameters
        ----------
        data : np.ndarray
            TDEM decay curve data. Can be 1D array (single curve) or 2D array (multiple curves).
            If 2D, each row is treated as a separate data sample.
        label : str
            Configuration label from simulator (e.g., 'L0.41-T-0.34' or 'L0.45-T-')
            Format: L<loop_z>-T<target_z> or L<loop_z>-T- (no target)
        metadata : dict, optional
            Additional metadata to store with this data entry (e.g., loop_z, target_z, etc.)
        
        Raises
        ------
        ValueError
            If data format is incompatible
        """
        # Convert data to numpy array if it's a list
        if isinstance(data, list):
            data = np.array(data)
        
        # Ensure data is at least 2D (samples × features)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim > 2:
            raise ValueError(f"Data must be 1D or 2D array, got shape {data.shape}")
        
        # Check data shape consistency
        if self.data_shape is None:
            self.data_shape = data.shape[1]  # Number of features (time samples)
            print(f"Data shape set to: {self.data_shape} time samples per curve")
        elif data.shape[1] != self.data_shape:
            raise ValueError(
                f"Data shape mismatch: expected {self.data_shape} features, got {data.shape[1]}"
            )
        
        # Determine target class from label (T- means no target)
        has_target = not label.endswith('T-')
        target_class = 'target_present' if has_target else 'target_absent'
        binary_label = 1 if has_target else 0
        
        # Parse label for loop_z and target_z
        try:
            parts = label.split('-')
            loop_z_str = parts[0].replace('L', '')
            target_z_str = parts[1].replace('T', '') if len(parts) > 1 else ''
            
            loop_z = float(loop_z_str) if loop_z_str else None
            target_z = float(target_z_str) if target_z_str and target_z_str != '' else None
        except (ValueError, IndexError):
            loop_z = None
            target_z = None
        
        # Store each sample as a separate entry
        num_samples = data.shape[0]
        for i in range(num_samples):
            entry = {
                'data': data[i, :],
                'label': label,  # Configuration label (e.g., L0.41-T-0.34)
                'target_class': target_class,
                'binary_label': binary_label,  # 1=present, 0=absent for CNN training
                'loop_z': loop_z,
                'target_z': target_z,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata if metadata is not None else {}
            }
            self.data_entries.append(entry)
            
            # Update counters
            if has_target:
                self.num_target_present += 1
            else:
                self.num_target_absent += 1
        
        print(f"Appended {num_samples} sample(s) with label '{label}' (class: {target_class})")
        print(f"Total: {len(self.data_entries)} samples "
              f"({self.num_target_present} present, {self.num_target_absent} absent)")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of logged data.
        
        Returns
        -------
        summary : dict
            Dictionary containing data statistics
        """
        summary = {
            'total_samples': len(self.data_entries),
            'target_present': self.num_target_present,
            'target_absent': self.num_target_absent,
            'feature_dimension': self.data_shape,
            'class_balance': (
                self.num_target_present / len(self.data_entries) 
                if len(self.data_entries) > 0 else 0
            )
        }
        return summary
    
    def print_summary(self):
        """Print a formatted summary of logged data."""
        summary = self.get_summary()
        print("\n" + "=" * 60)
        print("PiLogger Data Summary")
        print("=" * 60)
        print(f"Total samples: {summary['total_samples']}")
        print(f"  Target present: {summary['target_present']}")
        print(f"  Target absent: {summary['target_absent']}")
        print(f"Feature dimension: {summary['feature_dimension']}")
        print(f"Class balance: {summary['class_balance']:.2%} present")
        print("=" * 60 + "\n")
    
    def split_data(self, train_percent: float, test_percent: float, 
                   output_dir: str = 'dataset', seed: Optional[int] = None) -> Tuple[str, str, str]:
        """
        Split logged data into train/validation/test sets and export to CSV files.
        
        Parameters
        ----------
        train_percent : float
            Percentage of data for training set (0-100)
        test_percent : float
            Percentage of data for test set (0-100)
        output_dir : str, optional
            Directory to save the CSV files (default: 'dataset')
        seed : int, optional
            Random seed for reproducibility
        
        Returns
        -------
        train_path : str
            Path to training data CSV file
        val_path : str
            Path to validation data CSV file
        test_path : str
            Path to test data CSV file
        
        Raises
        ------
        ValueError
            If percentages are invalid or no data is logged
        """
        # Validate inputs
        if len(self.data_entries) == 0:
            raise ValueError("No data logged. Cannot split empty dataset.")
        
        if not (0 < train_percent < 100 and 0 < test_percent < 100):
            raise ValueError("Percentages must be between 0 and 100")
        
        if train_percent + test_percent >= 100:
            raise ValueError("Sum of train_percent and test_percent must be less than 100")
        
        # Calculate validation percentage
        val_percent = 100 - train_percent - test_percent
        
        print(f"\nSplitting data: {train_percent}% train, {val_percent}% validation, {test_percent}% test")
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Create random permutation of indices
        num_samples = len(self.data_entries)
        indices = np.random.permutation(num_samples)
        
        # Calculate split sizes
        train_size = int(num_samples * train_percent / 100)
        test_size = int(num_samples * test_percent / 100)
        val_size = num_samples - train_size - test_size
        
        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        print(f"Split sizes: train={train_size}, val={val_size}, test={test_size}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filenames with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_path = os.path.join(output_dir, f'train_data_{timestamp}.csv')
        val_path = os.path.join(output_dir, f'val_data_{timestamp}.csv')
        test_path = os.path.join(output_dir, f'test_data_{timestamp}.csv')
        
        # Export each split
        self._export_split(train_indices, train_path, 'train')
        self._export_split(val_indices, val_path, 'validation')
        self._export_split(test_indices, test_path, 'test')
        
        # Save metadata
        metadata_path = os.path.join(output_dir, f'dataset_info_{timestamp}.json')
        self._save_metadata(metadata_path, train_size, val_size, test_size, 
                           train_percent, test_percent, val_percent, seed)
        
        print(f"\nDataset exported successfully to: {output_dir}")
        print(f"  Training data: {train_path}")
        print(f"  Validation data: {val_path}")
        print(f"  Test data: {test_path}")
        print(f"  Metadata: {metadata_path}\n")
        
        return train_path, val_path, test_path
    
    def _export_split(self, indices: np.ndarray, filepath: str, split_name: str):
        """
        Export a data split to CSV file in TensorFlow-compatible format.
        
        CSV Format:
        -----------
        config_label, loop_z, target_z, binary_label, feature_0, feature_1, ..., feature_N
        
        Where:
        - config_label: Configuration identifier (e.g., L0.41-T-0.34)
        - loop_z: Loop height in meters
        - target_z: Target depth in meters (or empty for no target)
        - binary_label: 1=target present, 0=target absent (for CNN classification)
        - feature_i: TDEM decay curve values (time series data)
        
        Parameters
        ----------
        indices : np.ndarray
            Indices of samples to include in this split
        filepath : str
            Path to output CSV file
        split_name : str
            Name of the split (for logging)
        """
        # Gather data for this split
        data_matrix = []
        config_labels = []
        binary_labels = []
        loop_zs = []
        target_zs = []
        
        for idx in indices:
            entry = self.data_entries[idx]
            data_matrix.append(entry['data'])
            config_labels.append(entry['label'])
            binary_labels.append(entry['binary_label'])
            loop_zs.append(entry.get('loop_z', ''))
            target_zs.append(entry.get('target_z', ''))
        
        # Convert to numpy arrays
        data_matrix = np.array(data_matrix)
        binary_labels = np.array(binary_labels)
        
        # Create DataFrame with all columns
        # First: metadata columns
        df = pd.DataFrame({
            'config_label': config_labels,
            'loop_z': loop_zs,
            'target_z': target_zs,
            'binary_label': binary_labels
        })
        
        # Then: feature columns (TDEM decay curve)
        feature_cols = [f'feature_{i}' for i in range(data_matrix.shape[1])]
        feature_df = pd.DataFrame(data_matrix, columns=feature_cols)
        
        # Concatenate metadata and features
        df = pd.concat([df, feature_df], axis=1)
        
        # Export to CSV
        df.to_csv(filepath, index=False)
        
        # Print split statistics
        num_present = np.sum(binary_labels == 1)
        num_absent = np.sum(binary_labels == 0)
        print(f"  {split_name}: {len(binary_labels)} samples ({num_present} present, {num_absent} absent)")
    
    def _save_metadata(self, filepath: str, train_size: int, val_size: int, test_size: int,
                      train_percent: float, test_percent: float, val_percent: float, seed: Optional[int]):
        """Save dataset metadata to JSON file."""
        metadata = {
            'creation_timestamp': datetime.now().isoformat(),
            'total_samples': len(self.data_entries),
            'feature_dimension': self.data_shape,
            'split_sizes': {
                'train': train_size,
                'validation': val_size,
                'test': test_size
            },
            'split_percentages': {
                'train': train_percent,
                'validation': val_percent,
                'test': test_percent
            },
            'class_distribution': {
                'target_present': self.num_target_present,
                'target_absent': self.num_target_absent
            },
            'random_seed': seed,
            'csv_format': {
                'columns': [
                    'config_label: Configuration identifier (e.g., L0.41-T-0.34)',
                    'loop_z: Loop height in meters',
                    'target_z: Target depth in meters (empty if no target)',
                    'binary_label: 1=target present, 0=target absent (use for CNN training)',
                    'feature_0 to feature_N: TDEM decay curve time series data'
                ],
                'label_format': 'L<loop_z>-T<target_z> or L<loop_z>-T- (no target)',
                'tensorflow_usage': 'Load CSV, extract feature_* columns as input, binary_label as output'
            },
            'data_description': {
                'input_features': f'{self.data_shape} time samples of TDEM decay curve (-dBz/dt)',
                'output_classes': '2 (binary classification: target present/absent)',
                'recommended_model': 'CNN for 1D time series classification'
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=4)
    
    def plot_csv_data(self, csv_filepath: str, time_axis: Optional[np.ndarray] = None,
                     num_samples: int = 5, save_fig: Optional[str] = None):
        """
        Plot TDEM decay curves from CSV file for verification.
        
        Parameters
        ----------
        csv_filepath : str
            Path to CSV file to plot
        time_axis : np.ndarray, optional
            Time axis in microseconds. If None, will use indices.
        num_samples : int, optional
            Number of random samples to plot (default: 5)
        save_fig : str, optional
            If provided, save figure to this filepath
        """
        # Load CSV data
        df = pd.read_csv(csv_filepath)
        
        # Extract features and labels
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        data_matrix = df[feature_cols].values
        binary_labels = df['binary_label'].values
        config_labels = df['config_label'].values
        
        # Create time axis if not provided
        if time_axis is None:
            time_axis = np.arange(len(feature_cols))
            time_label = 'Sample Index'
        else:
            time_label = 'Time [μs]'
        
        # Randomly select samples to plot
        num_total = len(binary_labels)
        if num_samples > num_total:
            num_samples = num_total
        
        plot_indices = np.random.choice(num_total, size=num_samples, replace=False)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot target present samples
        present_indices = [idx for idx in plot_indices if binary_labels[idx] == 1]
        colors_present = plt.cm.tab10(np.linspace(0, 1, len(present_indices)))
        
        for i, idx in enumerate(present_indices):
            label = config_labels[idx]
            ax1.plot(time_axis, data_matrix[idx, :], alpha=0.8, linewidth=2, 
                    color=colors_present[i], label=label)
        
        ax1.set_xlabel(time_label, fontsize=12)
        ax1.set_ylabel('-dBz/dt [T/s]', fontsize=12)
        ax1.set_title(f'Target Present Samples (n={len(present_indices)})', 
                     fontsize=14, fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3, which='both')
        ax1.legend(loc='best', fontsize=9)
        
        # Plot target absent samples
        absent_indices = [idx for idx in plot_indices if binary_labels[idx] == 0]
        colors_absent = plt.cm.tab10(np.linspace(0, 1, len(absent_indices)))
        
        for i, idx in enumerate(absent_indices):
            label = config_labels[idx]
            ax2.plot(time_axis, data_matrix[idx, :], alpha=0.8, linewidth=2,
                    color=colors_absent[i], label=label)
        
        ax2.set_xlabel(time_label, fontsize=12)
        ax2.set_ylabel('-dBz/dt [T/s]', fontsize=12)
        ax2.set_title(f'Target Absent Samples (n={len(absent_indices)})', 
                     fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3, which='both')
        ax2.legend(loc='best', fontsize=9)
        
        # Add overall title
        filename = os.path.basename(csv_filepath)
        fig.suptitle(f'TDEM Dataset Verification: {filename}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save or show
        if save_fig:
            plt.savefig(save_fig, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_fig}")
        else:
            plt.show()
    
    def clear_data(self):
        """Clear all logged data."""
        self.data_entries = []
        self.num_target_present = 0
        self.num_target_absent = 0
        self.data_shape = None
        print("All data cleared from logger.")
    
    def save_logger_state(self, filepath: str):
        """
        Save current logger state to file for later loading.
        
        Parameters
        ----------
        filepath : str
            Path to save the logger state
        """
        state = {
            'data_entries': self.data_entries,
            'num_target_present': self.num_target_present,
            'num_target_absent': self.num_target_absent,
            'data_shape': self.data_shape
        }
        
        np.save(filepath, state, allow_pickle=True)
        print(f"Logger state saved to: {filepath}")
    
    def load_logger_state(self, filepath: str):
        """
        Load logger state from file.
        
        Parameters
        ----------
        filepath : str
            Path to the saved logger state
        """
        state = np.load(filepath, allow_pickle=True).item()
        
        self.data_entries = state['data_entries']
        self.num_target_present = state['num_target_present']
        self.num_target_absent = state['num_target_absent']
        self.data_shape = state['data_shape']
        
        print(f"Logger state loaded from: {filepath}")
        self.print_summary()


# Example usage
if __name__ == "__main__":
    # Create logger
    logger = PiLogger()
    
    # Simulate some data with configuration labels
    print("\n--- Simulating data logging ---")
    
    # Generate mock TDEM data (1024 time samples)
    time_samples = 1024
    
    # Add target present samples with labels
    for i in range(50):
        data = np.random.randn(time_samples) * np.exp(-np.linspace(0, 5, time_samples))
        label = f"L{0.3 + i*0.01:.2f}-T{-0.3 - i*0.01:.2f}"
        metadata = {'loop_z': 0.3 + i*0.01, 'target_z': -0.3 - i*0.01}
        logger.append_data(data, label, metadata)
    
    # Add target absent samples with labels
    for i in range(50):
        data = np.random.randn(time_samples) * np.exp(-np.linspace(0, 3, time_samples)) * 0.1
        label = f"L{0.3 + i*0.01:.2f}-T-"
        metadata = {'loop_z': 0.3 + i*0.01, 'target_z': None}
        logger.append_data(data, label, metadata)
    
    # Print summary
    logger.print_summary()
    
    # Split and export data
    print("\n--- Splitting and exporting data ---")
    train_path, val_path, test_path = logger.split_data(
        train_percent=70,
        test_percent=15,
        output_dir='dataset',
        seed=42
    )
    
    # Create time axis for plotting
    time_axis = np.linspace(0, 1024, time_samples)  # microseconds
    
    # Plot validation data
    print("\n--- Plotting validation data ---")
    logger.plot_csv_data(val_path, time_axis=time_axis, num_samples=3)
