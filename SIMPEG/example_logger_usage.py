"""
Example: Using PiLogger with PiSimulator and PiConditioner
============================================================

This script demonstrates how to use PiLogger to collect and manage
TDEM simulation data for machine learning applications.
"""

import numpy as np
from pi_config import PiConfig
from pi_sim import PiSimulator, PiConditioner
from pi_logger import PiLogger


def main():
    # Initialize components
    cfg = PiConfig('config.json')
    logger = PiLogger()
    conditioner = PiConditioner()
    
    print("=" * 60)
    print("TDEM Data Collection for Machine Learning")
    print("=" * 60)
    
    # ========================================================================
    # Example 1: Collect data with target present (in grass layer)
    # ========================================================================
    print("\n--- Collecting TARGET PRESENT data ---")
    
    # Configure for shallow burial (target in grass layer)
    cfg.target_z = 0.05  # 5 cm depth (in grass layer)
    
    # Run multiple simulations with variations
    for i in range(10):
        print(f"\nSimulation {i+1}/10 with target present...")
        
        # Vary loop height slightly
        loop_height = 0.2 + i * 0.01
        simulator = PiSimulator(cfg, loop_z_start=loop_height, 
                               loop_z_increment=0.0, num_increments=0)
        
        # Run simulation
        time, unconditioned_data, plotting_info = simulator.run()
        
        # Apply conditioning to each decay curve
        for decay in unconditioned_data:
            # Add noise
            noisy = conditioner.add_noise(time, decay, late_time=10e-6, snr_db=20)
            # Amplify
            noisy = conditioner.amplify(time, noisy, time_gain=[[50e-6, 10]])
            # Normalize
            noisy = conditioner.normalize(noisy, 0.7)
            # Quantize
            processed = conditioner.quantize(noisy, depth=8, dtype=np.uint8)
            
            # Log the data with metadata
            metadata = {
                'target_depth': float(cfg.target_z),
                'loop_height': loop_height,
                'target_radius': float(cfg.target_radius),
                'soil_conductivity': float(cfg.soil_conductivity)
            }
            logger.append_data(processed, 'target_present', metadata)
    
    # ========================================================================
    # Example 2: Collect data with target absent (deeper or no target)
    # ========================================================================
    print("\n\n--- Collecting TARGET ABSENT data ---")
    
    # Configure for deep burial or no target
    cfg.target_z = -0.5  # 50 cm depth (deep in soil, not in grass)
    
    for i in range(10):
        print(f"\nSimulation {i+1}/10 with target absent...")
        
        loop_height = 0.2 + i * 0.01
        simulator = PiSimulator(cfg, loop_z_start=loop_height,
                               loop_z_increment=0.0, num_increments=0)
        
        time, unconditioned_data, plotting_info = simulator.run()
        
        for decay in unconditioned_data:
            noisy = conditioner.add_noise(time, decay, late_time=10e-6, snr_db=20)
            noisy = conditioner.amplify(time, noisy, time_gain=[[50e-6, 10]])
            noisy = conditioner.normalize(noisy, 0.7)
            processed = conditioner.quantize(noisy, depth=8, dtype=np.uint8)
            
            metadata = {
                'target_depth': float(cfg.target_z),
                'loop_height': loop_height,
                'target_radius': float(cfg.target_radius),
                'soil_conductivity': float(cfg.soil_conductivity)
            }
            logger.append_data(processed, 'target_absent', metadata)
    
    # ========================================================================
    # Print summary of collected data
    # ========================================================================
    logger.print_summary()
    
    # ========================================================================
    # Split data into train/validation/test sets
    # ========================================================================
    print("\n--- Splitting data for machine learning ---")
    train_path, val_path, test_path = logger.split_data(
        train_percent=70,
        test_percent=15,
        output_dir='ml_dataset',
        seed=42  # For reproducibility
    )
    
    # ========================================================================
    # Visualize the data to verify quality
    # ========================================================================
    print("\n--- Visualizing training data ---")
    logger.plot_csv_data(
        csv_filepath=train_path,
        sampling_rate=1e6,  # 1 MHz (1 microsecond per sample)
        num_samples=5,
        save_fig='ml_dataset/training_data_preview.png'
    )
    
    print("\n--- Visualizing validation data ---")
    logger.plot_csv_data(
        csv_filepath=val_path,
        sampling_rate=1e6,
        num_samples=3,
        save_fig='ml_dataset/validation_data_preview.png'
    )
    
    # ========================================================================
    # Optional: Save logger state for later use
    # ========================================================================
    logger.save_logger_state('ml_dataset/logger_state.npy')
    
    print("\n" + "=" * 60)
    print("Data collection complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Use the CSV files with TensorFlow/Keras:")
    print("   train_df = pd.read_csv('ml_dataset/train_data_*.csv')")
    print("2. Load features and labels:")
    print("   X_train = train_df.drop('label', axis=1).values")
    print("   y_train = train_df['label'].values")
    print("3. Build and train your neural network!")


if __name__ == "__main__":
    main()
