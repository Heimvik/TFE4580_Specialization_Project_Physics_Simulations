import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
import numpy as np
import os
import h5py
from datetime import datetime

from pi_plotter import ClassifierPlotter


class PiClassifier():
    def __init__(self, conditioner=None):
        self.model = None
        self.conditioner = conditioner
        self.input_shape = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.norm_params = None
        self.plotter = ClassifierPlotter()
    
    def load_model(self, model_path):
        print("\n" + "="*70)
        print("Loading Pre-trained Model")
        print("="*70)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = keras.models.load_model(model_path)
        self.input_shape = self.model.input_shape[1:]  # Exclude batch dimension
        
        print(f"✓ Model loaded from: {model_path}")
        print(f"  - Input shape: {self.input_shape}")
        print(f"  - Output classes: {self.model.output_shape[-1]}")
        print(f"  - Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def save_model(self, model_path):
        if self.model is None:
            raise ValueError("No model to save. Build or load a model first.")
        
        print("\n" + "="*70)
        print("Saving Model")
        print("="*70)
        
        self.model.save(model_path)
        print(f"✓ Model saved to: {model_path}")
        
        return model_path
    
    def calculate_flops(self, input_length=None, verbose=True):
        print("\n" + "="*70)
        print("FLOP Calculation for Single Inference")
        print("="*70)
        
        if self.model is None:
            if input_length is None:
                raise ValueError("No model loaded and no input_length provided. "
                               "Either load a model or provide input_length.")
            # Build a temporary model for FLOP calculation
            print("No model loaded. Building model for FLOP calculation...")
            self.build_model(input_length)
        
        flops_breakdown = {}
        total_flops = 0
        current_shape = list(self.model.input_shape[1:])  # [time_samples, channels]
        
        if verbose:
            print(f"\nInput shape: {current_shape}")
            print("-" * 70)
            print(f"{'Layer':<25} {'Type':<20} {'Output Shape':<20} {'FLOPs':>12}")
            print("-" * 70)
        
        for layer in self.model.layers:
            layer_name = layer.name
            layer_type = layer.__class__.__name__
            layer_flops = 0
            
            if isinstance(layer, layers.Conv1D):
                # Conv1D FLOPs: 2 * K * C_in * C_out * L_out
                # Where: K = kernel_size, C_in = input_channels, C_out = filters
                # L_out = output_length, factor of 2 for multiply-add
                kernel_size = layer.kernel_size[0]
                in_channels = current_shape[-1]
                out_channels = layer.filters
                
                if layer.padding == 'same':
                    out_length = current_shape[0]
                else:  # 'valid'
                    out_length = current_shape[0] - kernel_size + 1
                
                # Multiply-adds (each is 2 FLOPs: 1 mult + 1 add)
                layer_flops = 2 * kernel_size * in_channels * out_channels * out_length
                
                # Add bias if present
                if layer.use_bias:
                    layer_flops += out_channels * out_length
                
                current_shape = [out_length, out_channels]
                
            elif isinstance(layer, layers.BatchNormalization):
                # BatchNorm FLOPs: 4 * elements (subtract mean, divide std, scale, shift)
                num_elements = np.prod(current_shape)
                layer_flops = 4 * num_elements
                
            elif isinstance(layer, layers.MaxPooling1D) or isinstance(layer, layers.AveragePooling1D):
                # Pooling FLOPs: comparisons for max, additions for average
                pool_size = layer.pool_size[0] if hasattr(layer.pool_size, '__len__') else layer.pool_size
                strides = layer.strides[0] if hasattr(layer.strides, '__len__') else layer.strides
                
                out_length = current_shape[0] // strides
                num_channels = current_shape[-1]
                
                # Max pooling: (pool_size - 1) comparisons per output
                # Average pooling: pool_size additions + 1 division per output
                if isinstance(layer, layers.MaxPooling1D):
                    layer_flops = (pool_size - 1) * out_length * num_channels
                else:
                    layer_flops = (pool_size + 1) * out_length * num_channels
                
                current_shape = [out_length, num_channels]
                
            elif isinstance(layer, layers.GlobalAveragePooling1D):
                # GlobalAvgPool: sum all elements + divide
                sequence_length = current_shape[0]
                num_channels = current_shape[-1]
                layer_flops = sequence_length * num_channels + num_channels  # additions + divisions
                current_shape = [num_channels]
            
            elif isinstance(layer, layers.Flatten):
                # Flatten: 0 FLOPs (just reshaping)
                layer_flops = 0
                current_shape = [np.prod(current_shape)]
                
            elif isinstance(layer, layers.Dense):
                # Dense FLOPs: 2 * input_size * output_size (multiply-add)
                in_features = current_shape[-1] if isinstance(current_shape, list) else current_shape
                out_features = layer.units
                layer_flops = 2 * in_features * out_features
                
                # Add bias
                if layer.use_bias:
                    layer_flops += out_features
                
                current_shape = [out_features]
                
            elif isinstance(layer, layers.Dropout):
                # Dropout: 0 FLOPs during inference (no-op)
                layer_flops = 0
                
            elif isinstance(layer, layers.InputLayer):
                # Input: 0 FLOPs
                layer_flops = 0
            
            # Add activation FLOPs (ReLU: 1 comparison per element)
            if hasattr(layer, 'activation') and layer.activation is not None:
                activation_name = layer.activation.__name__ if hasattr(layer.activation, '__name__') else str(layer.activation)
                if activation_name == 'relu':
                    num_elements = np.prod(current_shape) if isinstance(current_shape, list) else current_shape
                    layer_flops += num_elements  # 1 comparison per element
                elif activation_name == 'softmax':
                    # Softmax: exp(x), sum, divide for each element
                    num_elements = np.prod(current_shape) if isinstance(current_shape, list) else current_shape
                    layer_flops += 3 * num_elements
            
            flops_breakdown[layer_name] = {
                'type': layer_type,
                'flops': layer_flops,
                'output_shape': list(current_shape) if isinstance(current_shape, list) else [current_shape]
            }
            total_flops += layer_flops
            
            if verbose and layer_flops > 0:
                print(f"{layer_name:<25} {layer_type:<20} {str(current_shape):<20} {layer_flops:>12,}")
        
        if verbose:
            print("-" * 70)
            print(f"{'TOTAL':<45} {'':<20} {total_flops:>12,}")
            print("="*70)
            
            # Summary statistics
            print(f"\n📊 Summary:")
            print(f"  - Total FLOPs per inference: {total_flops:,}")
            print(f"  - Total MFLOPs: {total_flops / 1e6:.3f}")
            print(f"  - Total GFLOPs: {total_flops / 1e9:.6f}")
            
            # Breakdown by layer type
            print(f"\n📈 FLOPs by Layer Type:")
            type_flops = {}
            for name, info in flops_breakdown.items():
                ltype = info['type']
                if ltype not in type_flops:
                    type_flops[ltype] = 0
                type_flops[ltype] += info['flops']
            
            for ltype, flops in sorted(type_flops.items(), key=lambda x: -x[1]):
                if flops > 0:
                    pct = flops / total_flops * 100
                    print(f"    {ltype:<25} {flops:>12,} ({pct:>5.1f}%)")
        
        return {
            'breakdown': flops_breakdown,
            'total_flops': total_flops,
            'mflops': total_flops / 1e6,
            'gflops': total_flops / 1e9
        }
    
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
        
        # Prepare arrays for training - data should already be conditioned before split
        self.X_train = decay_curves[train_idx].reshape(len(train_idx), decay_curves.shape[1], 1)
        self.y_train = labels[train_idx]
        self.X_val = decay_curves[val_idx].reshape(len(val_idx), decay_curves.shape[1], 1)
        self.y_val = labels[val_idx]
        self.X_test = decay_curves[test_idx].reshape(len(test_idx), decay_curves.shape[1], 1)
        self.y_test = labels[test_idx]
        
        print("\n✓ Dataset split complete! Data stored in memory.")
        
        # Optional: Save to files
        if save_to_file:
            print("\nSaving split datasets to files...")
            base_dir = os.path.dirname(source_path)
            base_name = os.path.basename(source_path).replace('.h5', '')
            
            # Save split files in the same directory as the source file
            train_path = os.path.join(base_dir, f"{base_name}_train.h5")
            val_path = os.path.join(base_dir, f"{base_name}_val.h5")
            test_path = os.path.join(base_dir, f"{base_name}_test.h5")
            
            self._save_split_dataset(source_path, train_path, train_idx, "Training")
            self._save_split_dataset(source_path, val_path, val_idx, "Validation")
            self._save_split_dataset(source_path, test_path, test_idx, "Testing")
            
            print(f"\n✓ Files saved in: {base_dir}")
            print(f"  - Training set: {os.path.basename(train_path)}")
            print(f"  - Validation set: {os.path.basename(val_path)}")
            print(f"  - Testing set: {os.path.basename(test_path)}")
            
            return train_path, val_path, test_path
        
        return None, None, None
    
    def _save_split_dataset(self, source_path, dest_path, indices, split_name):
        # Ensure the destination directory exists
        dest_dir = os.path.dirname(os.path.abspath(dest_path))
        try:
            os.makedirs(dest_dir, exist_ok=True)
        except Exception as e:
            print(f"  Warning: Could not create directory {dest_dir}: {e}")
            raise
        
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
        inputs = keras.Input(shape=(num_samples, 1), name='input')
        
        x = layers.Conv1D(filters=16, kernel_size=7, padding='same', activation='relu', name='conv1')(inputs)
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.MaxPooling1D(pool_size=4, name='pool1')(x)
        x = layers.Dropout(0.2, name='drop1')(x)

        x = layers.Conv1D(filters=32, kernel_size=5, padding='same', activation='relu', name='conv2')(x)
        x = layers.BatchNormalization(name='bn2')(x)
        x = layers.MaxPooling1D(pool_size=4, name='pool2')(x)
        x = layers.Dropout(0.2, name='drop2')(x)

        x = layers.Conv1D(filters=16, kernel_size=3, padding='same', activation='relu', name='conv3')(x)
        x = layers.BatchNormalization(name='bn3')(x)
        x = layers.MaxPooling1D(pool_size=4, name='pool3')(x)
        x = layers.Dropout(0.2, name='drop3')(x)

        x = layers.Flatten(name='flatten')(x)

        x = layers.Dense(32, activation='relu', name='fc1')(x)
        x = layers.Dropout(0.4, name='drop4')(x)
        outputs = layers.Dense(2, activation='softmax', name='output')(x)

        self.model = keras.Model(inputs, outputs, name='TDEM_Classifier')
        
        print("\n" + "="*70)
        print("Model Architecture Summary")
        print("="*70)
        self.model.summary()
        
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
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=1
        )
        
        self.plotter.plot_training_history(history)
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
    
    def quantize_model(self, representative_data=None, output_path=None):
        if self.model is None:
            print("Error: No model to quantize. Build or load a model first.")
            return None
        
        print("\n" + "="*70)
        print("Quantizing Model to INT8 (Byte Precision)")
        print("="*70)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        converter.target_spec.supported_types = [tf.int8]
        
        if representative_data is not None:
            def representative_dataset():
                for i in range(min(100, len(representative_data))):
                    sample = representative_data[i:i+1].astype(np.float32)
                    yield [sample]
            
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            print("Using full integer quantization with representative dataset")
        else:
            print("Using dynamic range quantization (weights only)")
        
        try:
            self.quantized_model = converter.convert()
            
            original_size = sum(w.numpy().nbytes for w in self.model.weights)
            quantized_size = len(self.quantized_model)
            compression_ratio = original_size / quantized_size
            
            print(f"\nQuantization Results:")
            print(f"  Original model size: {original_size / 1024:.2f} KB")
            print(f"  Quantized model size: {quantized_size / 1024:.2f} KB")
            print(f"  Compression ratio: {compression_ratio:.2f}x")
            
            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(self.quantized_model)
                print(f"  Saved to: {output_path}")
            
            print("="*70)
            return self.quantized_model
            
        except Exception as e:
            print(f"Quantization failed: {e}")
            return None
    
    def visualize_training_data(self, time, decay_curves, labels, label_strings, metadata):
        self.plotter.visualize_training_data(time, decay_curves, labels, label_strings, metadata)

    def plot_confusion_matrix(self, X_test, y_test, normalize=False):
        self.plotter.set_model(self.model)
        return self.plotter.plot_confusion_matrix(X_test, y_test, normalize)
    
    def plot_roc_curve(self, X_test, y_test):
        self.plotter.set_model(self.model)
        return self.plotter.plot_roc_curve(X_test, y_test)
    
    def plot_prediction_distribution(self, X_test, y_test):
        self.plotter.set_model(self.model)
        return self.plotter.plot_prediction_distribution(X_test, y_test)
    
    def plot_roc_pfa_pd(self, X_test, y_test, snr_db=None, num_thresholds=100):
        self.plotter.set_model(self.model)
        return self.plotter.plot_roc_pfa_pd(X_test, y_test, snr_db, num_thresholds)
    
    def plot_model_architecture(self, input_length=None, output_path='model_architecture.png'):
        if self.model is None and input_length is not None:
            self.build_model(input_length)
        return self.plotter.plot_model_architecture(self.model, output_path)
    
    def plot_roc_multi_snr(self, logger, dataset_paths, snr_values):
        self.plotter.set_model(self.model)
        return self.plotter.plot_roc_multi_snr(self.model, logger, dataset_paths, snr_values)
    
    def generate_model_summary_latex(self, input_length=None, output_file=None):
        if self.model is None:
            if input_length is None:
                print("No model loaded. Building default model with 50 time samples...")
                input_length = 50
            self.build_model(input_length)
        
        print("\n" + "="*70)
        print("Generating LaTeX Model Summary")
        print("="*70)
        
        trainable_params = int(np.sum([np.prod(v.shape) for v in self.model.trainable_weights]))
        non_trainable_params = int(np.sum([np.prod(v.shape) for v in self.model.non_trainable_weights]))
        total_params = trainable_params + non_trainable_params
        
        flops_info = self.calculate_flops(verbose=False)
        
        latex_doc = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}
\usepackage{amsmath}
\usepackage{geometry}
\usepackage{xcolor}
\usepackage{colortbl}
\geometry{margin=2.5cm}

\title{TDEM Pulse Induction Classifier\\Model Architecture Report}
\author{Generated by PiClassifier}
\date{\today}

\begin{document}
\maketitle

\section{Model Overview}
This document describes the neural network architecture for Time-Domain Electromagnetic (TDEM) 
pulse induction signal classification. The model is designed for binary classification 
to detect the presence or absence of metallic targets.

\subsection{Architecture Summary}
\begin{table}[H]
\centering
\caption{Model Parameters Summary}
\label{tab:params_summary}
\begin{tabular}{@{}lr@{}}
\toprule
\textbf{Parameter Category} & \textbf{Count} \\
\midrule
Trainable Parameters & """ + f"{trainable_params:,}" + r""" \\
Non-trainable Parameters & """ + f"{non_trainable_params:,}" + r""" \\
\midrule
\textbf{Total Parameters} & \textbf{""" + f"{total_params:,}" + r"""} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Computational Complexity}
\begin{table}[H]
\centering
\caption{Computational Cost per Inference}
\label{tab:flops}
\begin{tabular}{@{}lr@{}}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Total FLOPs & """ + f"{flops_info['total_flops']:,}" + r""" \\
MFLOPs & """ + f"{flops_info['mflops']:.3f}" + r""" \\
GFLOPs & """ + f"{flops_info['gflops']:.6f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}

\section{Layer-by-Layer Architecture}

\begin{table}[H]
\centering
\caption{Detailed Layer Configuration}
\label{tab:layers}
\begin{tabular}{@{}llrrl@{}}
\toprule
\textbf{Layer Name} & \textbf{Type} & \textbf{Output Shape} & \textbf{Params} & \textbf{Configuration} \\
\midrule
"""
        for layer in self.model.layers:
            layer_name = layer.name.replace('_', r'\_')
            layer_type = layer.__class__.__name__
            layer_params = layer.count_params()
            
            try:
                output_shape = str(layer.output.shape)
            except:
                output_shape = str(layer.output_shape)
            
            config = ""
            if 'Conv1D' in layer_type:
                config = f"filters={layer.filters}, k={layer.kernel_size[0]}"
            elif 'MaxPooling' in layer_type:
                config = f"pool={layer.pool_size[0]}"
            elif 'Dense' in layer_type:
                config = f"units={layer.units}"
            elif 'Dropout' in layer_type:
                config = f"rate={layer.rate}"
            elif 'BatchNorm' in layer_type:
                config = "axis=-1"
            
            latex_doc += f"{layer_name} & {layer_type} & {output_shape} & {layer_params:,} & {config} \\\\\n"
        
        latex_doc += r"""\bottomrule
\end{tabular}
\end{table}

\section{FLOP Breakdown by Layer Type}

\begin{table}[H]
\centering
\caption{Computational Cost by Layer Type}
\label{tab:flops_breakdown}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Layer Type} & \textbf{FLOPs} & \textbf{Percentage} \\
\midrule
"""
        type_flops = {}
        for name, info in flops_info['breakdown'].items():
            ltype = info['type']
            if ltype not in type_flops:
                type_flops[ltype] = 0
            type_flops[ltype] += info['flops']
        
        for ltype, flops in sorted(type_flops.items(), key=lambda x: -x[1]):
            if flops > 0:
                pct = flops / flops_info['total_flops'] * 100
                latex_doc += f"{ltype} & {flops:,} & {pct:.1f}\\% \\\\\n"
        
        latex_doc += r"""\midrule
\textbf{Total} & \textbf{""" + f"{flops_info['total_flops']:,}" + r"""} & \textbf{100\%} \\
\bottomrule
\end{tabular}
\end{table}

\section{Model Architecture Diagram}
The network follows a sequential convolutional architecture:

\begin{enumerate}
    \item \textbf{Input Layer}: 1D time-series signal
    \item \textbf{Feature Extraction}: Three Conv1D blocks with filters (16$\rightarrow$32$\rightarrow$16)
    \item \textbf{Regularization}: BatchNormalization and Dropout after each conv block
    \item \textbf{Pooling}: MaxPooling1D to reduce temporal dimension
    \item \textbf{Classification}: Flatten followed by Dense layers
    \item \textbf{Output}: Softmax activation for binary classification
\end{enumerate}

\begin{figure}[H]
\centering
\fbox{\parbox{0.8\textwidth}{
\centering
\textbf{Model Flow}\\[0.5em]
Input $(N, 1)$ $\rightarrow$ Conv1D(16) $\rightarrow$ BN $\rightarrow$ MaxPool(4) $\rightarrow$ Dropout\\
$\downarrow$\\
Conv1D(32) $\rightarrow$ BN $\rightarrow$ MaxPool(4) $\rightarrow$ Dropout\\
$\downarrow$\\
Conv1D(16) $\rightarrow$ BN $\rightarrow$ MaxPool(4) $\rightarrow$ Dropout\\
$\downarrow$\\
Flatten $\rightarrow$ Dense(32) $\rightarrow$ Dropout $\rightarrow$ Dense(2, softmax)
}}
\caption{Simplified model architecture flow diagram}
\label{fig:model_flow}
\end{figure}

\end{document}
"""
        
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Non-trainable Parameters: {non_trainable_params:,}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Total FLOPs: {flops_info['total_flops']:,}")
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(latex_doc)
            print(f"\nLaTeX document saved to: {output_file}")
        
        print("="*70)
        
        return latex_doc
    
    def generate_architecture_tikz(self, output_file=None):
        if self.model is None:
            print("No model loaded.")
            return None
        
        print("\n" + "="*70)
        print("Generating TikZ Architecture Diagram")
        print("="*70)
        
        tikz_code = r"""\documentclass[border=10pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{positioning,shapes.geometric,arrows.meta,fit,backgrounds}

\begin{document}
\begin{tikzpicture}[
    node distance=0.8cm,
    layer/.style={rectangle, draw, rounded corners, minimum width=3cm, minimum height=0.6cm, align=center, font=\small},
    conv/.style={layer, fill=blue!20},
    bn/.style={layer, fill=purple!15, minimum width=2.5cm},
    pool/.style={layer, fill=yellow!25, minimum width=2.5cm},
    drop/.style={layer, fill=red!15, minimum width=2.5cm},
    dense/.style={layer, fill=green!20},
    input/.style={layer, fill=gray!15},
    arrow/.style={-{Stealth[scale=0.8]}, thick}
]

% Input
\node[input] (input) {Input: (N, 1)};

% Block 1
\node[conv, below=of input] (conv1) {Conv1D: 16 filters, k=7};
\node[bn, below=of conv1] (bn1) {BatchNorm};
\node[pool, below=of bn1] (pool1) {MaxPool: 4};
\node[drop, below=of pool1] (drop1) {Dropout: 0.2};

% Block 2
\node[conv, below=of drop1] (conv2) {Conv1D: 32 filters, k=5};
\node[bn, below=of conv2] (bn2) {BatchNorm};
\node[pool, below=of bn2] (pool2) {MaxPool: 4};
\node[drop, below=of pool2] (drop2) {Dropout: 0.2};

% Block 3
\node[conv, below=of drop2] (conv3) {Conv1D: 16 filters, k=3};
\node[bn, below=of conv3] (bn3) {BatchNorm};
\node[pool, below=of bn3] (pool3) {MaxPool: 4};
\node[drop, below=of pool3] (drop3) {Dropout: 0.3};

% Flatten and dense
\node[pool, below=of drop3] (flatten) {Flatten};
\node[dense, below=of flatten] (fc1) {Dense: 32, ReLU};
\node[drop, below=of fc1] (drop4) {Dropout: 0.4};
\node[dense, below=of drop4] (output) {Dense: 2, Softmax};

% Arrows
\foreach \i/\j in {input/conv1, conv1/bn1, bn1/pool1, pool1/drop1,
                   drop1/conv2, conv2/bn2, bn2/pool2, pool2/drop2,
                   drop2/conv3, conv3/bn3, bn3/pool3, pool3/drop3,
                   drop3/flatten, flatten/fc1, fc1/drop4, drop4/output} {
    \draw[arrow] (\i) -- (\j);
}

% Block labels
\begin{scope}[on background layer]
    \node[draw=blue!50, dashed, rounded corners, fit=(conv1)(bn1)(pool1)(drop1), inner sep=3pt, label={[blue!70]left:Block 1}] {};
    \node[draw=blue!50, dashed, rounded corners, fit=(conv2)(bn2)(pool2)(drop2), inner sep=3pt, label={[blue!70]left:Block 2}] {};
    \node[draw=blue!50, dashed, rounded corners, fit=(conv3)(bn3)(pool3)(drop3), inner sep=3pt, label={[blue!70]left:Block 3}] {};
\end{scope}

\end{tikzpicture}
\end{document}
"""
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(tikz_code)
            print(f"TikZ diagram saved to: {output_file}")
        
        print("="*70)
        return tikz_code



