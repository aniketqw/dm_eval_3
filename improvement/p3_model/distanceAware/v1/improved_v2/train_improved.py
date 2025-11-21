"""
Train Improved Distance-Aware Chronos with Enhanced Loss and Architecture
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from distance_aware_chronos_v2 import ImprovedDistanceAwareChronos, train_improved_model
import numpy as np
import pickle
from pathlib import Path

def main():
    print("="*60)
    print("Training Improved Distance-Aware Chronos Model")
    print("="*60)
    
    # Load data
    data_path = Path("../chronos_data/train/timeseries.pkl")
    print(f"\nLoading data from: {data_path}")
    
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)
    
    print(f"Loaded {len(all_data)} time series")
    
    # Split train/val
    split_idx = int(0.9 * len(all_data))
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Initialize improved model
    print("\nInitializing Improved Distance-Aware Chronos...")
    model = ImprovedDistanceAwareChronos(
        model_name="amazon/chronos-t5-small",
        num_bins=4096,
        use_enhanced_output=True  # Use enhanced output with attention
    )
    
    # Count parameters
    trainable_params = sum(p.numel() for p in model.distance_output.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    
    # Train
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    train_losses, val_losses = train_improved_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        epochs=15,  # More epochs for better convergence
        batch_size=8,
        learning_rate=3e-4,
        checkpoint_dir="./checkpoints"
    )
    
    # Plot training curves
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Improved Distance-Aware Chronos Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./checkpoints/training_curves.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Training curves saved to: ./checkpoints/training_curves.png")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
