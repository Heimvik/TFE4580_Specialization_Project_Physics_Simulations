# TDEM ML Dataset Generation and Usage Guide

## Overview

This system generates labeled TDEM (Time-Domain Electromagnetic) simulation data for training Convolutional Neural Networks (CNNs) to detect buried metal targets.

## System Components

### 1. **pi_sim.py** - Simulation Engine
- Generates all permutations of surveys (loop positions) × conductivity models (target configurations)
- Creates unique labels for each permutation: `L<loop_z>-T<target_z>` or `L<loop_z>-T-`
- Tracks metadata for each simulation (loop height, target depth, configuration)

### 2. **pi_logger.py** - Dataset Management
- Logs simulation data with configuration labels
- Splits data into train/validation/test sets
- Exports to TensorFlow-ready CSV format
- Provides visualization tools for data verification

### 3. **load_dataset_example.py** - TensorFlow Integration
- Example code for loading CSV data
- CNN architecture for binary classification
- Training and evaluation pipeline
- Prediction visualization

## Workflow

### Step 1: Generate Simulation Data

Run the simulator:
```bash
python pi_sim.py
```

This will:
1. Generate randomized surveys (loop positions)
2. Generate randomized conductivity models (target depths)
3. Simulate ALL permutations (surveys × models)
4. Create labeled datasets with format: `L<loop_z>-T<target_z>`

**Label Format:**
- **With target**: `L0.41-T-0.34` (Loop @ 0.41m, Target @ -0.34m)
- **Without target**: `L0.45-T-` (Loop @ 0.45m, No Target)

### Step 2: Export Dataset

After simulation completes, you'll be prompted:
```
Would you like to export this dataset to CSV files? (y/n):
```

Enter `y` and provide:
- **Training percentage** (default: 70%)
- **Test percentage** (default: 15%)
- **Output directory** (default: 'dataset')

The validation percentage is calculated automatically: `100 - train - test`

### Step 3: Understand CSV Format

Each CSV file contains:

| Column | Description | Example |
|--------|-------------|---------|
| `config_label` | Configuration identifier | `L0.41-T-0.34` |
| `loop_z` | Loop height in meters | `0.41` |
| `target_z` | Target depth in meters | `-0.34` (or empty) |
| `binary_label` | **CNN training label** | `1` (present) or `0` (absent) |
| `feature_0` to `feature_N` | TDEM decay curve time series | Floating point values |

**Key Points:**
- `binary_label` is what you use for CNN training: 1=target present, 0=target absent
- Features are the raw TDEM decay curve samples (-dBz/dt values)
- All metadata is preserved for analysis and debugging

### Step 4: Load Data with TensorFlow

Use the example script:
```bash
python load_dataset_example.py
```

Or integrate into your own code:

```python
import pandas as pd
import numpy as np

# Load training data
df = pd.read_csv('dataset/train_data_YYYYMMDD_HHMMSS.csv')

# Extract features (input for CNN)
feature_cols = [col for col in df.columns if col.startswith('feature_')]
X = df[feature_cols].values

# Extract labels (output for CNN)
y = df['binary_label'].values

# X shape: (num_samples, num_timesteps)
# y shape: (num_samples,)
```

### Step 5: Preprocess for CNN

```python
from sklearn.preprocessing import StandardScaler

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Reshape for 1D CNN: (samples, timesteps, channels)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
```

### Step 6: Build and Train CNN

```python
from tensorflow import keras
from tensorflow.keras import layers

# Build 1D CNN
model = keras.Sequential([
    layers.Conv1D(64, kernel_size=7, activation='relu', input_shape=(num_timesteps, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),
    
    layers.Conv1D(128, kernel_size=5, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')  # 2 classes: present/absent
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32
)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")
```

## Configuration Parameters

### In pi_sim.py

```python
simulator = PiSimulator(
    cfg, 
    loop_z=[0.3, 0.5],           # Loop height range [min, max] meters
    target_z=[-0.5, -0.3],       # Target depth range [min, max] meters
    decays_target_present=10,    # Number of target-present models
    decays_target_absent=10      # Number of target-absent models
)
```

**Total simulations** = `(decays_target_present + decays_target_absent)² × 2`

Example: 10+10 = 20 models → 20×20 = 400 permutations total

**WARNING**: Large values create exponential growth in simulation time!
- 10+10 models → 400 permutations (~2 hours)
- 40+40 models → 6,400 permutations (~32 hours)
- 100+100 models → 40,000 permutations (~200 hours)

### Recommended Ranges

For **quick testing** (minutes):
```python
decays_target_present=2
decays_target_absent=2
# Total: 16 permutations
```

For **small dataset** (hours):
```python
decays_target_present=10
decays_target_absent=10
# Total: 400 permutations
```

For **full dataset** (days):
```python
decays_target_present=50
decays_target_absent=50
# Total: 10,000 permutations
```

## Dataset Metadata

Each export creates a `dataset_info_YYYYMMDD_HHMMSS.json` file with:

```json
{
    "creation_timestamp": "2025-10-22T...",
    "total_samples": 400,
    "feature_dimension": 1024,
    "split_sizes": {
        "train": 280,
        "validation": 60,
        "test": 60
    },
    "class_distribution": {
        "target_present": 200,
        "target_absent": 200
    },
    "csv_format": {
        "columns": [...],
        "tensorflow_usage": "..."
    }
}
```

## Visualization

### View Simulation Results
After simulation, use the interactive plotter:
```
=== PiPlotter: Interactive Plot Selection ===
Available plots:
  1: TDEM Linear
  2: TDEM Log-Log
  3: 3D View
  4: Side View
  5: Top View
  6: TDEM Log-Log + Side View  <-- Select permutations to compare

Enter plot number (or 'q' to quit): 6
```

Option 6 allows you to:
- Select specific permutations by number
- View side-by-side: physical configuration + TDEM response
- Verify correspondence between setup and data

### Verify CSV Data
```python
from pi_logger import PiLogger

logger = PiLogger()
logger.plot_csv_data(
    'dataset/train_data_20251022_143045.csv',
    time_axis=np.linspace(0, 1024, 1024),  # microseconds
    num_samples=5
)
```

## Best Practices

1. **Start Small**: Test with 2×2 models (16 permutations) first
2. **Check Balance**: Ensure equal target-present and target-absent samples
3. **Monitor Progress**: Simulation prints progress every 10 samples
4. **Save Checkpoints**: Export datasets at different scales for comparison
5. **Normalize Data**: Always standardize features before CNN training
6. **Use Validation Set**: Monitor overfitting during training
7. **Analyze Errors**: Use config_label to identify difficult cases

## Troubleshooting

### Problem: Simulation too slow
**Solution**: Reduce model counts (e.g., 10×10 instead of 40×40)

### Problem: Imbalanced classes
**Solution**: Set equal `decays_target_present` and `decays_target_absent`

### Problem: CNN overfitting
**Solution**: 
- Increase dataset size
- Add more dropout layers
- Use data augmentation

### Problem: Low accuracy
**Solution**:
- Check data normalization
- Increase model complexity
- Tune hyperparameters (learning rate, batch size)
- Verify data quality with visualization

## File Structure

```
SIMPEG/
├── pi_sim.py                    # Simulation engine
├── pi_logger.py                 # Dataset management
├── pi_plotter.py                # Visualization
├── pi_config.py                 # Configuration
├── load_dataset_example.py      # TensorFlow integration example
├── dataset/                     # Output directory
│   ├── train_data_*.csv        # Training data
│   ├── val_data_*.csv          # Validation data
│   ├── test_data_*.csv         # Test data
│   └── dataset_info_*.json     # Metadata
└── DATASET_README.md           # This file
```

## Citation

If you use this dataset generation system in your research, please cite:

```
TFE4580 Specialization Project: TDEM Target Detection
Author: [Your Name]
Institution: [Your Institution]
Date: October 2025
```

## Support

For issues or questions:
1. Check this README
2. Review `load_dataset_example.py` for TensorFlow usage
3. Inspect `dataset_info_*.json` for dataset statistics
4. Use visualization tools to verify data quality

## License

[Add your license information here]
