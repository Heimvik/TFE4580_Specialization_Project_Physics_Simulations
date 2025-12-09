# PIEMSOL - Pulse Induction Electromagnetic Solver

PIEMSOL is a machine learning framework for TDEM (Time-Domain Electromagnetic) pulse induction signal classification and analysis.

## Features

- **Data Management**: HDF5-based dataset creation, conditioning, and splitting
- **CNN Classifier**: 1D convolutional neural network for binary classification (target present/absent)
- **Model Quantization**: INT8 quantization for edge deployment
- **Performance Analysis**: Comprehensive inference profiling, accuracy metrics, and ROC analysis
- **Multi-SNR Testing**: Evaluate classifier performance under various noise conditions
- **Visualization**: Training curves, confusion matrices, ROC curves, and architecture diagrams

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run the main simulator:

```bash
python piemsol_sim.py
```

## Main Components

### piemsol_sim.py
Main entry point providing interactive menu for:
- Dataset generation and management
- Model training and evaluation
- Quantization and deployment
- Visualization and analysis

### piemsol_classifier.py
`PiemsolClassifier` class handles:
- Model architecture (CNN with 3 conv blocks + fully connected layers)
- Training, validation, and testing
- Keras and TFLite model management
- Inference profiling (FP32 vs INT8)

### piemsol_conditioner.py
`PiemsolConditioner` class provides:
- Noise addition for SNR testing
- Log-scale conditioning pipeline
- Data normalization and quantization

### piemsol_plotter.py
`ClassifierPlotter` class generates:
- Training history plots
- Confusion matrices and ROC curves
- Multi-SNR performance analysis
- Model architecture diagrams

### piemsol_logger.py
`PiemsolLogger` class manages:
- HDF5 dataset creation and storage
- Dataset splitting (train/val/test)
- Metadata tracking

### piemsol_config.py
`PiemsolConfig` class loads simulation parameters from `config.json`

## Workflow

1. **Generate Dataset**: Use option 2 to create simulation dataset
2. **Condition Data**: Apply log10 transform and normalization (option 3)
3. **Split Dataset**: Create train/val/test splits (option 4)
4. **Train Model**: Build and train CNN classifier (option 5)
5. **Quantize Model**: Convert to INT8 TFLite format (option 6)
6. **Analyze**: Use options 7-11 for comprehensive performance analysis

## Model Architecture

- **Input**: (1024, 1) - conditioned time-series decay curves
- **Conv Block 1**: 16 filters, kernel=7, pool=4, dropout=0.2
- **Conv Block 2**: 32 filters, kernel=5, pool=4, dropout=0.2
- **Conv Block 3**: 16 filters, kernel=3, pool=4, dropout=0.2
- **FC Layer**: 32 units, dropout=0.4
- **Output**: 2 classes (softmax)

## Performance Metrics

- **Accuracy**: Classification accuracy on test set
- **AUC**: Area under ROC curve
- **Inference Time**: Per-sample latency (ms)
- **Model Size**: Memory footprint (KB)
- **Compression Ratio**: FP32 vs INT8 size reduction

## Multi-SNR Analysis

Test classifier robustness by:
1. Adding white Gaussian noise to raw signals
2. Applying conditioning pipeline
3. Running inference and computing metrics
4. Comparing FP32 vs INT8 performance

## Configuration

Edit `config.json` to customize:
- Loop geometry and parameters
- Target dimensions and conductivity
- Time channels and mesh settings
- Waveform configuration

## Output Files

- **Models/**: Trained Keras and TFLite models
- **Datasets/**: HDF5 files with simulation data
- **Images/**: Generated plots and visualizations
- **Split/**: Train/val/test dataset splits

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy, Matplotlib, scikit-learn
- h5py for HDF5 file handling

## License

See LICENSE file for details.
