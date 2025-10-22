"""
Example script for loading and using TDEM dataset with TensorFlow/Keras
=========================================================================

This script demonstrates how to:
1. Load the CSV dataset exported by PiLogger
2. Prepare data for TensorFlow/Keras CNN training
3. Build a simple 1D CNN for binary classification
4. Train and evaluate the model

Author: TDEM Simulation Project
Date: October 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import os

# Uncomment these lines when TensorFlow is installed
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers


def load_tdem_dataset(train_path, val_path, test_path):
    """
    Load TDEM dataset from CSV files.
    
    Parameters
    ----------
    train_path : str
        Path to training data CSV
    val_path : str
        Path to validation data CSV
    test_path : str
        Path to test data CSV
    
    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test : np.ndarray
        Training, validation, and test data with labels
    feature_cols : list
        List of feature column names
    metadata : dict
        Metadata DataFrames for each split
    """
    # Load CSV files
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Loaded datasets:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    # Extract feature columns (TDEM decay curves)
    feature_cols = [col for col in train_df.columns if col.startswith('feature_')]
    print(f"\nNumber of features: {len(feature_cols)}")
    
    # Extract features (X) and labels (y)
    X_train = train_df[feature_cols].values
    y_train = train_df['binary_label'].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df['binary_label'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['binary_label'].values
    
    # Store metadata
    metadata = {
        'train': train_df[['config_label', 'loop_z', 'target_z']],
        'val': val_df[['config_label', 'loop_z', 'target_z']],
        'test': test_df[['config_label', 'loop_z', 'target_z']]
    }
    
    # Print class distribution
    print(f"\nClass distribution:")
    print(f"  Train: {np.sum(y_train == 1)} present, {np.sum(y_train == 0)} absent")
    print(f"  Val: {np.sum(y_val == 1)} present, {np.sum(y_val == 0)} absent")
    print(f"  Test: {np.sum(y_test == 1)} present, {np.sum(y_test == 0)} absent")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, metadata


def preprocess_data(X_train, X_val, X_test, normalize=True):
    """
    Preprocess TDEM data for CNN training.
    
    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray
        Raw feature data
    normalize : bool
        Whether to apply standardization
    
    Returns
    -------
    X_train, X_val, X_test : np.ndarray
        Preprocessed data ready for CNN input
    scaler : StandardScaler or None
        Fitted scaler (if normalize=True)
    """
    if normalize:
        # Standardize features (zero mean, unit variance)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        print("\nData normalized (StandardScaler)")
    else:
        scaler = None
    
    # Reshape for 1D CNN: (samples, timesteps, channels)
    # For 1D time series, channels = 1
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    print(f"Data shape for CNN: {X_train.shape}")
    print(f"  Samples: {X_train.shape[0]}")
    print(f"  Timesteps: {X_train.shape[1]}")
    print(f"  Channels: {X_train.shape[2]}")
    
    return X_train, X_val, X_test, scaler


def build_cnn_model(input_shape, num_classes=2):
    """
    Build a 1D CNN for TDEM binary classification.
    
    This is a simple example architecture. You should tune it based on your data.
    
    Parameters
    ----------
    input_shape : tuple
        Shape of input data (timesteps, channels)
    num_classes : int
        Number of output classes (2 for binary classification)
    
    Returns
    -------
    model : keras.Model
        Compiled Keras model
    """
    # Uncomment when TensorFlow is available
    """
    model = keras.Sequential([
        # First convolutional block
        layers.Conv1D(64, kernel_size=7, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Second convolutional block
        layers.Conv1D(128, kernel_size=5, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Third convolutional block
        layers.Conv1D(256, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.4),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel architecture:")
    model.summary()
    
    return model
    """
    
    print("\nNote: TensorFlow not imported. Install TensorFlow to build and train the model.")
    print("      pip install tensorflow")
    return None


def visualize_predictions(X_test, y_test, y_pred, metadata_df, num_samples=4):
    """
    Visualize model predictions with actual TDEM curves.
    
    Parameters
    ----------
    X_test : np.ndarray
        Test data (samples, timesteps, channels)
    y_test : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    metadata_df : pd.DataFrame
        Metadata for test samples
    num_samples : int
        Number of samples to visualize
    """
    # Flatten X_test if needed
    if X_test.ndim == 3:
        X_test_flat = X_test.reshape(X_test.shape[0], X_test.shape[1])
    else:
        X_test_flat = X_test
    
    # Create time axis
    time_axis = np.arange(X_test_flat.shape[1])
    
    # Select random samples
    indices = np.random.choice(len(y_test), size=min(num_samples, len(y_test)), replace=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        
        # Get data
        curve = X_test_flat[idx, :]
        true_label = y_test[idx]
        pred_label = y_pred[idx]
        config = metadata_df.iloc[idx]['config_label']
        
        # Plot
        ax.plot(time_axis, curve, linewidth=2)
        
        # Title with prediction result
        color = 'green' if true_label == pred_label else 'red'
        result = "✓ Correct" if true_label == pred_label else "✗ Wrong"
        ax.set_title(f"{config}\nTrue: {true_label}, Pred: {pred_label} {result}", 
                    fontsize=11, fontweight='bold', color=color)
        
        ax.set_xlabel('Time Index', fontsize=10)
        ax.set_ylabel('Signal Amplitude', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main execution function."""
    print("="*70)
    print("TDEM Dataset Loading Example for TensorFlow/Keras CNN")
    print("="*70)
    
    # Define dataset paths
    # TODO: Update these paths to your actual dataset files
    dataset_dir = 'dataset'
    
    # Find the most recent dataset files
    csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"\nError: No CSV files found in '{dataset_dir}'")
        print("Please run pi_sim.py first to generate the dataset.")
        return
    
    # Group files by timestamp
    train_files = [f for f in csv_files if f.startswith('train_data_')]
    val_files = [f for f in csv_files if f.startswith('val_data_')]
    test_files = [f for f in csv_files if f.startswith('test_data_')]
    
    if not (train_files and val_files and test_files):
        print("\nError: Incomplete dataset. Need train, val, and test files.")
        return
    
    # Use the most recent files
    train_path = os.path.join(dataset_dir, sorted(train_files)[-1])
    val_path = os.path.join(dataset_dir, sorted(val_files)[-1])
    test_path = os.path.join(dataset_dir, sorted(test_files)[-1])
    
    print(f"\nUsing dataset files:")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")
    print(f"  Test: {test_path}\n")
    
    # Step 1: Load data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, metadata = load_tdem_dataset(
        train_path, val_path, test_path
    )
    
    # Step 2: Preprocess data
    X_train, X_val, X_test, scaler = preprocess_data(X_train, X_val, X_test, normalize=True)
    
    # Step 3: Build model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, channels)
    model = build_cnn_model(input_shape, num_classes=2)
    
    if model is None:
        print("\nSkipping training (TensorFlow not available)")
        return
    
    # Step 4: Train model
    # Uncomment when TensorFlow is available
    """
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Step 5: Evaluate model
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Step 6: Make predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Step 7: Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Target Absent', 'Target Present']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Step 8: Visualize predictions
    visualize_predictions(X_test, y_test, y_pred, metadata['test'], num_samples=4)
    """
    
    print("\n" + "="*70)
    print("Example complete! Install TensorFlow to train the CNN model.")
    print("="*70)


if __name__ == "__main__":
    main()
