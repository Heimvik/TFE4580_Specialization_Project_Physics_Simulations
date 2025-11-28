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
        x = layers.MaxPooling1D(pool_size=2, name='pool1')(x)
        x = layers.Dropout(0.2, name='drop1')(x)

        x = layers.Conv1D(filters=32, kernel_size=5, padding='same', activation='relu', name='conv2')(x)
        x = layers.BatchNormalization(name='bn2')(x)
        x = layers.MaxPooling1D(pool_size=2, name='pool2')(x)
        x = layers.Dropout(0.2, name='drop2')(x)

        x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', name='conv3')(x)
        x = layers.BatchNormalization(name='bn3')(x)
        x = layers.MaxPooling1D(pool_size=2, name='pool3')(x)
        x = layers.Dropout(0.3, name='drop3')(x)

        x = layers.GlobalAveragePooling1D(name='global_pool')(x)

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

    def plot_confusion_matrix(self, X_test, y_test, normalize=False):
        from sklearn.metrics import confusion_matrix, classification_report
        import seaborn as sns
        
        if self.model is None:
            print("Error: No model loaded!")
            return
        
        print("\\n" + "="*70)
        print("Confusion Matrix Analysis")
        print("="*70)
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', square=True,
                   xticklabels=['No Target', 'Target Present'],
                   yticklabels=['No Target', 'Target Present'],
                   cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
                   ax=ax, annot_kws={'size': 14})
        
        ax.set_ylabel('True Label', fontsize=14)
        ax.set_xlabel('Predicted Label', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Print classification report
        print("\\nClassification Report:")
        print("="*70)
        target_names = ['No Target', 'Target Present']
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        return cm
    
    def generate_model_summary_latex(self):
        if self.model is None:
            print("Error: No model loaded!")
            return
        
        print("\\n" + "="*70)
        print("Model Parameters Summary (LaTeX)")
        print("="*70)
        
        # Count parameters
        trainable_params = np.sum([np.prod(v.shape) for v in self.model.trainable_weights])
        non_trainable_params = np.sum([np.prod(v.shape) for v in self.model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params
        
        # Generate LaTeX table
        latex_code = r"""
\\begin{table}[h]
\\centering
\\caption{Model Architecture Parameters}
\\label{tab:model_params}
\\begin{tabular}{|l|r|}
\\hline
\\textbf{Parameter Type} & \\textbf{Count} \\\\
\\hline
Trainable Parameters & """ + f"{trainable_params:,}" + r""" \\
Non-trainable Parameters & """ + f"{non_trainable_params:,}" + r""" \\
\\hline
Total Parameters & """ + f"{total_params:,}" + r""" \\
\\hline
\\end{tabular}
\\end{table}

\\begin{table}[h]
\\centering
\\caption{Layer-wise Parameter Breakdown}
\\label{tab:layer_params}
\\begin{tabular}{|l|l|r|}
\\hline
\\textbf{Layer Name} & \\textbf{Layer Type} & \\textbf{Parameters} \\\\
\\hline
"""
        
        # Add layer details
        for layer in self.model.layers:
            layer_params = layer.count_params()
            layer_name = layer.name.replace('_', '\\_')
            layer_type = layer.__class__.__name__
            latex_code += f"{layer_name} & {layer_type} & {layer_params:,} \\\\\\ \\n"
        
        latex_code += r"""\\hline
\\end{tabular}
\\end{table}
"""
        
        print(latex_code)
        print("\\n" + "="*70)
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Non-trainable Parameters: {non_trainable_params:,}")
        print(f"Total Parameters: {total_params:,}")
        print("="*70)
        
        return latex_code
    
    def plot_model_architecture(self):
        from tensorflow.keras.utils import plot_model
        
        if self.model is None:
            print("Error: No model loaded!")
            return
        
        print("\\n" + "="*70)
        print("Model Architecture Visualization")
        print("="*70)
        
        # Save architecture diagram
        output_path = 'model_architecture.png'
        plot_model(self.model, to_file=output_path, show_shapes=True, 
                  show_layer_names=True, rankdir='TB', expand_nested=True,
                  dpi=150, show_layer_activations=True)
        
        print(f"\\n✓ Model architecture saved to: {output_path}")
        
        # Also create a detailed text representation
        print("\\nDetailed Architecture:")
        print("="*70)
        self.model.summary()
        
        # Display the image
        from PIL import Image
        img = Image.open(output_path)
        plt.figure(figsize=(12, 16))
        plt.imshow(img)
        plt.axis('off')
        plt.title('TDEM Classifier Architecture', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
        return output_path
    
    def plot_roc_curve(self, X_test, y_test):
        from sklearn.metrics import roc_curve, auc
        
        if self.model is None:
            print("Error: No model loaded!")
            return
        
        print("\\n" + "="*70)
        print("ROC Curve Analysis")
        print("="*70)
        
        # Get prediction probabilities
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred_proba_target = y_pred_proba[:, 1]  # Probability of target present
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_target)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr, tpr, color='darkorange', lw=3, 
               label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
        ax.legend(loc="lower right", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\\n✓ ROC AUC Score: {roc_auc:.4f}")
        
        return fpr, tpr, roc_auc
    
    def plot_prediction_distribution(self, X_test, y_test):
        if self.model is None:
            print("Error: No model loaded!")
            return
        
        print("\\n" + "="*70)
        print("Prediction Probability Distribution")
        print("="*70)
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred_proba_target = y_pred_proba[:, 1]
        
        # Separate by true label
        target_present_probs = y_pred_proba_target[y_test == 1]
        target_absent_probs = y_pred_proba_target[y_test == 0]
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(target_present_probs, bins=30, alpha=0.6, color='green', 
                label='True: Target Present', edgecolor='black')
        ax1.hist(target_absent_probs, bins=30, alpha=0.6, color='blue', 
                label='True: No Target', edgecolor='black')
        ax1.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
        ax1.set_xlabel('Predicted Probability (Target Present)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        data_to_plot = [target_absent_probs, target_present_probs]
        bp = ax2.boxplot(data_to_plot, labels=['True: No Target', 'True: Target Present'],
                        patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor('blue')
        bp['boxes'][1].set_facecolor('green')
        ax2.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
        ax2.set_ylabel('Predicted Probability (Target Present)', fontsize=12)
        ax2.set_title('Prediction Confidence by True Label', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()


class PiMultiClassClassifier():
    
    def __init__(self, conditioner=None):
        self.model = None
        self.conditioner = conditioner
        self.num_classes = 4
        self.class_names = ['No Target', 'Hollow Cylinder', 'Shredded Can', 'Solid Block']
        
        # Store split data
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
    
    def split_dataset(self, logger, source_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, save_to_file=False):
        print("\\n" + "="*70)
        print("Multi-Class Dataset Split")
        print("="*70)
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        
        time, decay_curves, labels, label_strings, metadata = logger.load_from_hdf5(source_path)
        
        target_types = metadata.get('target_types', np.full(len(labels), 0))
        multi_labels = target_types.astype(int)
        
        num_samples = len(decay_curves)
        print(f"Total samples: {num_samples}")
        for class_idx in range(self.num_classes):
            count = np.sum(multi_labels == class_idx)
            print(f"  - {self.class_names[class_idx]}: {count}")
        
        # Shuffle indices
        indices = np.random.permutation(num_samples)
        
        # Calculate split points
        train_end = int(num_samples * train_ratio)
        val_end = train_end + int(num_samples * val_ratio)
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        print(f"\\nSplit sizes:")
        print(f"  - Training: {len(train_idx)} samples ({len(train_idx)/num_samples*100:.1f}%)")
        print(f"  - Validation: {len(val_idx)} samples ({len(val_idx)/num_samples*100:.1f}%)")
        print(f"  - Testing: {len(test_idx)} samples ({len(test_idx)/num_samples*100:.1f}%)")
        
        # Prepare arrays
        self.X_train = decay_curves[train_idx].reshape(len(train_idx), decay_curves.shape[1], 1)
        self.y_train = multi_labels[train_idx]
        self.X_val = decay_curves[val_idx].reshape(len(val_idx), decay_curves.shape[1], 1)
        self.y_val = multi_labels[val_idx]
        self.X_test = decay_curves[test_idx].reshape(len(test_idx), decay_curves.shape[1], 1)
        self.y_test = multi_labels[test_idx]
        
        print("\\n✓ Multi-class dataset split complete!")
        
        return train_idx, val_idx, test_idx
    
    def build_model(self, num_samples):
        inputs = keras.Input(shape=(num_samples, 1), name='input')
        
        # Block 1
        x = layers.Conv1D(filters=24, kernel_size=7, padding='same', activation='relu', name='conv1')(inputs)
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.MaxPooling1D(pool_size=2, name='pool1')(x)
        x = layers.Dropout(0.2, name='drop1')(x)
        
        # Block 2
        x = layers.SeparableConv1D(filters=48, kernel_size=5, padding='same', activation='relu', name='sepconv1')(x)
        x = layers.BatchNormalization(name='bn2')(x)
        x = layers.MaxPooling1D(pool_size=2, name='pool2')(x)
        x = layers.Dropout(0.2, name='drop2')(x)
        
        # Block 3
        x = layers.SeparableConv1D(filters=96, kernel_size=3, padding='same', activation='relu', name='sepconv2')(x)
        x = layers.BatchNormalization(name='bn3')(x)
        x = layers.MaxPooling1D(pool_size=2, name='pool3')(x)
        x = layers.Dropout(0.3, name='drop3')(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D(name='global_pool')(x)
        
        # Classifier head
        x = layers.Dense(64, activation='relu', name='fc1')(x)
        x = layers.Dropout(0.4, name='drop4')(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        self.model = keras.Model(inputs, outputs, name='TDEM_MultiClass_Classifier')
        
        print("\\n" + "="*70)
        print("Multi-Class Model Architecture")
        print("="*70)
        self.model.summary()
        
        return self.model
    
    def train_model(self, epochs=20, batch_size=32):
        print("\\n" + "="*70)
        print("Training Multi-Class Classifier")
        print("="*70)
        
        if self.X_train is None:
            print("Error: No training data. Run split_dataset() first.")
            return None
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        validation_data = None
        if self.X_val is not None and self.y_val is not None:
            validation_data = (self.X_val, self.y_val)
        
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=1
        )
        
        self.plot_training_history(history)
        print("\\n✓ Training complete!")
        
        return history
    
    def validate_model(self):
        if self.X_val is None:
            print("Error: No validation data.")
            return None, None
        
        print("\\n" + "="*70)
        print("Validating Multi-Class Model")
        print("="*70)
        
        val_loss, val_accuracy = self.model.evaluate(self.X_val, self.y_val, verbose=1)
        print(f"\\nLoss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        
        return val_loss, val_accuracy
    
    def test_model(self):
        if self.X_test is None:
            print("Error: No test data.")
            return None, None
        
        print("\\n" + "="*70)
        print("Testing Multi-Class Model")
        print("="*70)
        
        test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=1)
        print(f"\\nLoss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
        
        return test_loss, test_accuracy
    
    def plot_training_history(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Multi-Class Training History', fontsize=16, fontweight='bold')
        
        ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in history.history:
            ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Accuracy', fontsize=11)
        ax1.set_title('Model Accuracy', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history.history:
            ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Loss', fontsize=11)
        ax2.set_title('Model Loss', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, normalize=False):
        from sklearn.metrics import confusion_matrix, classification_report
        import seaborn as sns
        
        if self.model is None or self.X_test is None:
            print("Error: No model or test data!")
            return
        
        print("\\n" + "="*70)
        print("Multi-Class Confusion Matrix")
        print("="*70)
        
        # Get predictions
        y_pred_proba = self.model.predict(self.X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Compute confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', square=True,
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
                   ax=ax, annot_kws={'size': 12})
        
        ax.set_ylabel('True Label', fontsize=14)
        ax.set_xlabel('Predicted Label', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.show()
        
        # Print classification report
        print("\\nClassification Report:")
        print("="*70)
        print(classification_report(self.y_test, y_pred, target_names=self.class_names))
        
        return cm



