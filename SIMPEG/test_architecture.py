#!/usr/bin/env python3
"""
Test script to verify the updated architecture diagram with static block heights.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from pi_plotter import PiPlotter
from tensorflow.keras.models import load_model

def test_architecture_diagram():
    """Test the architecture diagram generation with static block heights."""

    # Initialize plotter
    plotter = PiPlotter()

    # Load the trained model
    try:
        model = load_model('trained_classifier.keras')
        print(f"Model loaded successfully. Summary:")
        model.summary()

        # Generate architecture diagram
        print("\nGenerating architecture diagram...")
        plotter.plot_model_architecture(model, save_path='test_architecture_updated.png')
        print("Architecture diagram saved as 'test_architecture_updated.png'")

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_architecture_diagram()
    if success:
        print("Test completed successfully!")
    else:
        print("Test failed!")