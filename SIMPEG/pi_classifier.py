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
    
    def load_quantized_model(self, model_path):
        """Load a quantized TFLite model for inference."""
        print("\n" + "="*70)
        print("Loading Quantized TFLite Model")
        print("="*70)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load TFLite model
        self.tflite_interpreter = tf.lite.Interpreter(model_path=model_path)
        self.tflite_interpreter.allocate_tensors()
        
        # Get input/output details
        self.tflite_input_details = self.tflite_interpreter.get_input_details()
        self.tflite_output_details = self.tflite_interpreter.get_output_details()
        
        input_shape = self.tflite_input_details[0]['shape']
        input_dtype = self.tflite_input_details[0]['dtype']
        output_shape = self.tflite_output_details[0]['shape']
        
        # Get model size
        model_size = os.path.getsize(model_path)
        
        print(f"✓ Quantized model loaded from: {model_path}")
        print(f"  - Input shape: {input_shape}")
        print(f"  - Input dtype: {input_dtype}")
        print(f"  - Output shape: {output_shape}")
        print(f"  - Model size: {model_size / 1024:.2f} KB")
        
        self.quantized_model_path = model_path
        return self.tflite_interpreter
    
    def predict_quantized(self, X):
        """Run inference using the quantized TFLite model."""
        if not hasattr(self, 'tflite_interpreter') or self.tflite_interpreter is None:
            raise ValueError("No quantized model loaded. Load a TFLite model first.")
        
        input_details = self.tflite_input_details[0]
        output_details = self.tflite_output_details[0]
        
        # Handle quantization parameters if using integer quantization
        input_scale = input_details.get('quantization_parameters', {}).get('scales', [1.0])
        input_zero_point = input_details.get('quantization_parameters', {}).get('zero_points', [0])
        
        predictions = []
        
        for i in range(len(X)):
            sample = X[i:i+1].astype(np.float32)
            
            # Quantize input if needed
            if input_details['dtype'] == np.int8:
                if len(input_scale) > 0 and input_scale[0] != 0:
                    sample = sample / input_scale[0] + input_zero_point[0]
                sample = sample.astype(np.int8)
            
            self.tflite_interpreter.set_tensor(input_details['index'], sample)
            self.tflite_interpreter.invoke()
            output = self.tflite_interpreter.get_tensor(output_details['index'])
            
            # Dequantize output if needed
            if output_details['dtype'] == np.int8:
                output_scale = output_details.get('quantization_parameters', {}).get('scales', [1.0])
                output_zero_point = output_details.get('quantization_parameters', {}).get('zero_points', [0])
                if len(output_scale) > 0 and output_scale[0] != 0:
                    output = (output.astype(np.float32) - output_zero_point[0]) * output_scale[0]
            
            predictions.append(output[0])
        
        return np.array(predictions)
    
    def has_quantized_model(self):
        """Check if a quantized model is loaded."""
        return hasattr(self, 'tflite_interpreter') and self.tflite_interpreter is not None
    
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
        
        # One-hot encode labels for binary crossentropy
        y_train_onehot = keras.utils.to_categorical(y_train, num_classes=2)
        
        if X_val is not None and y_val is not None:
            y_val_onehot = keras.utils.to_categorical(y_val, num_classes=2)
            validation_data = (X_val, y_val_onehot)
            print(f"Validation samples: {len(X_val)}")
        else:
            validation_data = None
        
        optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        self.model.compile(optimizer=optimizer,
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        
        history = self.model.fit(
            X_train, y_train_onehot,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            shuffle=True,
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
        
        # Convert labels to one-hot encoding for binary_crossentropy
        y_val_onehot = keras.utils.to_categorical(y_val, num_classes=2)
        
        val_loss, val_accuracy = self.model.evaluate(X_val, y_val_onehot, verbose=1)
        
        print(f"\nLoss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        return val_loss, val_accuracy
    
    def test_model(self, X_test, y_test):
        print("\n" + "="*70)
        print("Testing Model")
        print("="*70)
        
        print(f"Test samples: {len(X_test)}")
        
        # Convert labels to one-hot encoding for binary_crossentropy
        y_test_onehot = keras.utils.to_categorical(y_test, num_classes=2)
        
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test_onehot, verbose=1)
        
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
    
    def plot_roc_multi_snr_noise(self, time, X_test, y_test, snr_values, late_time, use_quantized=False):
        """
        Plot ROC curves for multiple SNR values by adding noise in memory.
        
        Parameters:
        -----------
        time : np.ndarray
            Time array for the decay curves
        X_test : np.ndarray
            Test data (decay curves reshaped for model input)
        y_test : np.ndarray
            Test labels
        snr_values : list
            List of SNR values in dB to test
        late_time : float
            Late time value for noise calculation
        use_quantized : bool
            Whether to use quantized (TFLite) model for inference
        """
        if use_quantized:
            if not self.has_quantized_model():
                raise ValueError("No quantized model loaded. Load a TFLite model first.")
            tflite_predict_fn = self.predict_quantized
            model = None
        else:
            if self.model is None:
                raise ValueError("No Keras model loaded. Train or load a model first.")
            tflite_predict_fn = None
            model = self.model
        
        self.plotter.set_model(model)
        return self.plotter.plot_roc_multi_snr_noise(
            model=model,
            time=time,
            X_test=X_test,
            y_test=y_test,
            snr_values=snr_values,
            late_time=late_time,
            conditioner=self.conditioner,
            use_quantized=use_quantized,
            tflite_predict_fn=tflite_predict_fn
        )
    
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
        
        layer_colors = {
            'InputLayer': 'gray!15',
            'Conv1D': 'blue!15',
            'BatchNormalization': 'purple!12',
            'MaxPooling1D': 'orange!15',
            'Dropout': 'red!12',
            'Dense': 'teal!15',
            'Flatten': 'brown!12',
        }
        
        latex_doc = r"""\documentclass[11pt,a4paper]{article}
        \usepackage[utf8]{inputenc}
        \usepackage{booktabs}
        \usepackage{graphicx}
        \usepackage{float}
        \usepackage{amsmath}
        \usepackage{geometry}
        \usepackage{xcolor}
        \usepackage{colortbl}
        \usepackage{multirow}
        \geometry{margin=2.5cm}

        \title{TDEM Pulse Induction Classifier\\Model Architecture Report}
        \author{Generated by PiemsolClassifier}
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
        \caption{Layer-wise Parameter Breakdown of the TDEM Classifier}
        \label{tab:layer_params}
        \resizebox{\textwidth}{!}{%
        \begin{tabular}{|l|l|r|p{2cm}|r|r|}
        \hline
        \rowcolor{gray!30}
        \textbf{Block} & \textbf{Layer Name} & \textbf{Output Shape} & \textbf{Filter/Pool Size} & \textbf{Stride} & \textbf{Parameters} \\
        \hline
        """

        block_assignments = {
            'input': ('Input', 'input'),
            'conv1': ('Block 1', 'conv1'), 'bn1': ('Block 1', 'bn1'), 'pool1': ('Block 1', 'pool1'), 'drop1': ('Block 1', 'drop1'),
            'conv2': ('Block 2', 'conv2'), 'bn2': ('Block 2', 'bn2'), 'pool2': ('Block 2', 'pool2'), 'drop2': ('Block 2', 'drop2'),
            'conv3': ('Block 3', 'conv3'), 'bn3': ('Block 3', 'bn3'), 'pool3': ('Block 3', 'pool3'), 'drop3': ('Block 3', 'drop3'),
            'flatten': ('Classification Head', 'flatten'), 'fc1': ('Classification Head', 'fc1'), 'drop4': ('Classification Head', 'drop4'), 'output': ('Classification Head', 'output')
        }

        block_counts = {}
        for layer in self.model.layers:
            layer_name = layer.name
            block_name, _ = block_assignments.get(layer_name, ('Unknown', layer_name))
            if block_name not in block_counts:
                block_counts[block_name] = 0
            block_counts[block_name] += 1

        current_block = None
        block_row_count = {}

        for layer in self.model.layers:
            layer_name = layer.name
            layer_type = layer.__class__.__name__
            layer_params = layer.count_params()
            
            try:
                output_shape = layer.output.shape[1:]
                shape_str = "$(" + ", ".join(str(d) if d is not None else "N" for d in output_shape) + ")$"
            except:
                try:
                    output_shape = layer.output_shape[1:]
                    shape_str = "$(" + ", ".join(str(d) if d is not None else "N" for d in output_shape) + ")$"
                except:
                    shape_str = "N/A"
            
            filter_pool_size = "--"
            stride = "--"
            
            if 'Conv1D' in layer_type:
                filter_pool_size = str(layer.kernel_size[0])
                stride = str(layer.strides[0])
            elif 'MaxPooling' in layer_type or 'AveragePooling' in layer_type:
                pool_size = layer.pool_size[0] if hasattr(layer.pool_size, '__len__') else layer.pool_size
                strides = layer.strides[0] if hasattr(layer.strides, '__len__') else layer.strides
                filter_pool_size = str(pool_size)
                stride = str(strides) if strides else str(pool_size)
            
            block_name, _ = block_assignments.get(layer_name, ('Unknown', layer_name))
    
        # -- Logic for BOLD Separators between blocks --
        if block_name != current_block:
            if current_block is not None:
                latex_doc += r"\noalign{\hrule height 2pt}" + "\n"
            current_block = block_name
            block_row_count[block_name] = block_counts[block_name]
        
        row_color = layer_colors.get(layer_type, 'white')
        layer_name_escaped = layer_name.replace('_', r'\_')
        
        latex_doc += f"\\rowcolor{{{row_color}}}\n"
        
        # -- FIX: USE NEGATIVE MULTIROW ON THE LAST ROW --
        # We determine if this is the last row of the block
        is_last_row = (block_row_count[block_name] == 1)
        total_rows = block_counts[block_name]
        
        if total_rows == 1:
            # Case: Single row block (like Input), just print the name directly
            first_col = f"\\cellcolor{{white}}{block_name}"
        elif is_last_row:
            # Case: Last row of a multi-row block. 
            # We use NEGATIVE multirow (e.g., -4) to project text upwards.
            # This ensures text sits ON TOP of the white cells defined in previous loop iterations.
            first_col = f"\\cellcolor{{white}}\\multirow{{-{total_rows}}}{{*}}{{{block_name}}}"
        else:
            # Case: Upper rows of a multi-row block.
            # We print an empty cell with white background to mask the row color.
            first_col = f"\\cellcolor{{white}}"

        latex_doc += f"{first_col} & {layer_name_escaped}({layer_type}) & {shape_str} & {filter_pool_size} & {stride} & {layer_params:,} \\\\\n"
        
        block_row_count[block_name] -= 1

        latex_doc += r"""\hline
        \rowcolor{gray!30}
        \multicolumn{5}{|l|}{\textbf{Trainable Parameters}} & \textbf{""" + f"{trainable_params:,}" + r"""} \\
        \rowcolor{gray!30}
        \multicolumn{5}{|l|}{\textbf{Non-trainable Parameters}} & \textbf{""" + f"{non_trainable_params:,}" + r"""} \\
        \hline
        \rowcolor{gray!40}
        \multicolumn{5}{|l|}{\textbf{Total Parameters}} & \textbf{""" + f"{total_params:,}" + r"""} \\
        \hline
        \end{tabular}%
        }
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



