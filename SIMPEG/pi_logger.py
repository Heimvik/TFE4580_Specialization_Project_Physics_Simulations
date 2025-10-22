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
    
    def append_data(self, data: np.ndarray, target_class: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Append TDEM data with target presence annotation.
        
        Parameters
        ----------
        data : np.ndarray
            TDEM decay curve data. Can be 1D array (single curve) or 2D array (multiple curves).
            If 2D, each row is treated as a separate data sample.
        target_class : str
            Target classification: 'target_present' or 'target_absent'
            - 'target_present': Target is in grass layer (shallow burial)
            - 'target_absent': Target in ground or not present at all
        metadata : dict, optional
            Additional metadata to store with this data entry (e.g., depth, conductivity, etc.)
        
        Raises
        ------
        ValueError
            If target_class is not valid or data format is incompatible
        """
        # Validate target class
        valid_classes = ['target_present', 'target_absent']
        if target_class not in valid_classes:
            raise ValueError(f"target_class must be one of {valid_classes}, got '{target_class}'")
        
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
        
        # Store each sample as a separate entry
        num_samples = data.shape[0]
        for i in range(num_samples):
            entry = {
                'data': data[i, :],
                'target_class': target_class,
                'label': 1 if target_class == 'target_present' else 0,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata if metadata is not None else {}
            }
            self.data_entries.append(entry)
            
            # Update counters
            if target_class == 'target_present':
                self.num_target_present += 1
            else:
                self.num_target_absent += 1
        
        print(f"Appended {num_samples} sample(s) with class '{target_class}'")
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
        labels = []
        
        for idx in indices:
            entry = self.data_entries[idx]
            data_matrix.append(entry['data'])
            labels.append(entry['label'])
        
        # Convert to numpy arrays
        data_matrix = np.array(data_matrix)
        labels = np.array(labels)
        
        # Create DataFrame with feature columns and label
        feature_cols = [f'feature_{i}' for i in range(data_matrix.shape[1])]
        df = pd.DataFrame(data_matrix, columns=feature_cols)
        df['label'] = labels
        
        # Export to CSV
        df.to_csv(filepath, index=False)
        
        # Print split statistics
        num_present = np.sum(labels == 1)
        num_absent = np.sum(labels == 0)
        print(f"  {split_name}: {len(labels)} samples ({num_present} present, {num_absent} absent)")
    
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
            'data_format': 'CSV with feature columns and binary label (1=present, 0=absent)'
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=4)
    
    def plot_csv_data(self, csv_filepath: str, sampling_rate: float, 
                     num_samples: int = 5, save_fig: Optional[str] = None):
        """
        Plot TDEM decay curves from CSV file for verification.
        
        Parameters
        ----------
        csv_filepath : str
            Path to CSV file to plot
        sampling_rate : float
            Sampling rate in Hz (samples per second) to compute time axis
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
        labels = df['label'].values
        
        # Compute time axis from sampling rate
        num_points = len(feature_cols)
        time_step = 1.0 / sampling_rate  # seconds
        time_axis = np.arange(num_points) * time_step * 1e6  # Convert to microseconds
        
        # Randomly select samples to plot
        num_total = len(labels)
        if num_samples > num_total:
            num_samples = num_total
        
        plot_indices = np.random.choice(num_total, size=num_samples, replace=False)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot target present samples
        present_indices = [idx for idx in plot_indices if labels[idx] == 1]
        for idx in present_indices:
            ax1.plot(time_axis, data_matrix[idx, :], alpha=0.7, linewidth=2)
        
        ax1.set_xlabel('Time [μs]', fontsize=12)
        ax1.set_ylabel('Signal Amplitude', fontsize=12)
        ax1.set_title(f'Target Present Samples (n={len(present_indices)})', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend([f'Sample {idx}' for idx in present_indices], loc='best')
        
        # Plot target absent samples
        absent_indices = [idx for idx in plot_indices if labels[idx] == 0]
        for idx in absent_indices:
            ax2.plot(time_axis, data_matrix[idx, :], alpha=0.7, linewidth=2)
        
        ax2.set_xlabel('Time [μs]', fontsize=12)
        ax2.set_ylabel('Signal Amplitude', fontsize=12)
        ax2.set_title(f'Target Absent Samples (n={len(absent_indices)})', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend([f'Sample {idx}' for idx in absent_indices], loc='best')
        
        # Add overall title
        filename = os.path.basename(csv_filepath)
        fig.suptitle(f'TDEM Data Verification: {filename}\nSampling Rate: {sampling_rate} Hz', 
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
    
    # Simulate some data
    print("\n--- Simulating data logging ---")
    
    # Generate mock TDEM data (1024 time samples)
    time_samples = 1024
    
    # Add target present samples
    for i in range(50):
        data = np.random.randn(time_samples) * np.exp(-np.linspace(0, 5, time_samples))
        metadata = {'depth': 0.05, 'conductivity': 3.5e7}
        logger.append_data(data, 'target_present', metadata)
    
    # Add target absent samples
    for i in range(50):
        data = np.random.randn(time_samples) * np.exp(-np.linspace(0, 3, time_samples)) * 0.1
        metadata = {'depth': None, 'conductivity': 0.4}
        logger.append_data(data, 'target_absent', metadata)
    
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
    
    # Plot validation data
    print("\n--- Plotting validation data ---")
    logger.plot_csv_data(val_path, sampling_rate=1e6, num_samples=3)
