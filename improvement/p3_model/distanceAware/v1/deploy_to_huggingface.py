# deploy_to_huggingface.py

import torch
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import T5ForConditionalGeneration
import shutil
from datetime import datetime

from distance_aware_chronos import DistanceAwareChronos


class HuggingFaceDeployer:
    """Deploy Distance-Aware Chronos to HuggingFace Hub"""
    
    def __init__(
        self,
        checkpoint_dir: str,
        repo_name: str,
        token: str = None
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.repo_name = repo_name
        self.token = token
        
        # Initialize HF API
        self.api = HfApi(token=token)
        
    def prepare_for_upload(self, output_dir: str = "./hf_model"):
        """Prepare model files for HuggingFace upload"""
        print("Preparing model for HuggingFace...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load checkpoint
        checkpoint_path = self.checkpoint_dir / "checkpoint.pt"
        config_path = self.checkpoint_dir / "config.json"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load model
        print("Loading trained model...")
        model = DistanceAwareChronos(
            model_name="amazon/chronos-t5-small",
            num_bins=config['num_bins'],
            device='cpu'
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Save model components
        print("Saving model components...")
        
        # Save the base Chronos/T5 model
        model.chronos.save_pretrained(output_path / "base_model")
        
        # Save distance-aware components
        torch.save(
            model.distance_output.state_dict(),
            output_path / "distance_output.pt"
        )
        
        # Save configuration
        model_config = {
            'model_type': 'distance_aware_chronos',
            'base_model': 'amazon/chronos-t5-small',
            'num_bins': config['num_bins'],
            'training_epoch': config['epoch'],
            'val_loss': config['val_loss'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path / "config.json", 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # Create model card
        self._create_model_card(output_path, model_config)
        
        # Create usage example
        self._create_usage_example(output_path)
        
        print(f"✓ Model prepared at: {output_path}")
        return output_path
    
    def _create_model_card(self, output_path, config):
        """Create README.md model card"""
        model_card = f"""---
license: apache-2.0
tags:
- time-series
- forecasting
- chronos
- distance-aware
library_name: transformers
---

# Distance-Aware Chronos

This is a distance-aware enhancement of the Chronos time series forecasting model.

## Model Description

This model extends the original [Chronos](https://github.com/amazon-science/chronos-forecasting) 
architecture with distance-aware loss functions and output layers that explicitly consider the 
ordinal nature of quantized time series bins.

**Base Model:** {config['base_model']}  
**Number of Bins:** {config['num_bins']}  
**Training Epoch:** {config['training_epoch']}  
**Validation Loss:** {config['val_loss']:.4f}

## Key Features

- **Distance-Aware Loss:** Combines ordinal cross-entropy, smooth label loss, and Earth Mover's Distance
- **Ordinal Output Layer:** Uses Gaussian kernels and sinusoidal position encodings
- **Improved Bin Predictions:** Better handling of nearby bin relationships

## Installation
```bash
pip install torch transformers chronos
```

## Usage
```python
from distance_aware_chronos import DistanceAwareChronos
import numpy as np

# Load model
model = DistanceAwareChronos.from_pretrained("{self.repo_name}")

# Prepare your time series
context = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Your historical data

# Generate forecasts
predictions = model.predict(context, horizon=24, num_samples=100)

print(f"Forecast shape: {{predictions.shape}}")
```

## Training Data

Trained on the [Chronos datasets](https://huggingface.co/datasets/autogluon/chronos_datasets) 
from HuggingFace.

## Citation

If you use this model, please cite:
```bibtex
@article{{chronos2024,
  title={{Chronos: Learning the Language of Time Series}},
  author={{Ansari, Abdul Fatir et al.}},
  journal={{Transactions on Machine Learning Research}},
  year={{2024}}
}}
```

## License

Apache 2.0
"""
        
        with open(output_path / "README.md", 'w') as f:
            f.write(model_card)
        
        print("✓ Model card created")
    
    def _create_usage_example(self, output_path):
        """Create example usage script"""
        example = """# Example: Using Distance-Aware Chronos

import numpy as np
from distance_aware_chronos import DistanceAwareChronos

# Load the model
model = DistanceAwareChronos.from_pretrained("YOUR_USERNAME/distance-aware-chronos")

# Example 1: Simple forecasting
context = np.random.randn(100)  # Your time series
forecast = model.predict(context, horizon=24, num_samples=100)

print(f"Forecast shape: {forecast.shape}")
print(f"Mean forecast: {forecast.mean()}")

# Example 2: With visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(range(len(context)), context, label='Historical', color='blue')
plt.plot(range(len(context), len(context) + len(forecast)), 
         forecast, label='Forecast', color='red', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Time Series Forecast')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
"""
        
        with open(output_path / "example_usage.py", 'w') as f:
            f.write(example)
        
        print("✓ Usage example created")
    
    def upload_to_hub(self, model_dir: str):
        """Upload model to HuggingFace Hub"""
        print(f"\nUploading to HuggingFace Hub: {self.repo_name}")
        
        try:
            # Create repository
            print("Creating repository...")
            create_repo(
                repo_id=self.repo_name,
                token=self.token,
                private=False,
                exist_ok=True
            )
            
            # Upload folder
            print("Uploading files...")
            upload_folder(
                folder_path=model_dir,
                repo_id=self.repo_name,
                token=self.token,
                commit_message=f"Upload Distance-Aware Chronos model"
            )
            
            print(f"\n✓ Model uploaded successfully!")
            print(f"✓ View at: https://huggingface.co/{self.repo_name}")
            
        except Exception as e:
            print(f"\n✗ Upload failed: {e}")
            print("\nMake sure you have:")
            print("1. A HuggingFace account")
            print("2. Generated an access token (Settings -> Access Tokens)")
            print("3. Logged in: huggingface-cli login")


def main():
    """Main deployment execution"""
    print("="*60)
    print("HUGGINGFACE DEPLOYMENT")
    print("="*60)
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy Distance-Aware Chronos to HuggingFace")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--repo", type=str, required=True, help="HuggingFace repo name (username/model-name)")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token (or set HF_TOKEN env var)")
    
    args = parser.parse_args()
    
    # Get token from multiple sources
    token = args.token or os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    
    # If still no token, try to read from HuggingFace CLI config
    if not token:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except:
            pass
    
    if not token:
        print("\n⚠ No HuggingFace token provided!")
        print("Please either:")
        print("  1. Pass --token YOUR_TOKEN")
        print("  2. Set HF_TOKEN environment variable")
        print("  3. Run: huggingface-cli login")
        return
    
    # Initialize deployer
    deployer = HuggingFaceDeployer(
        checkpoint_dir=args.checkpoint,
        repo_name=args.repo,
        token=token
    )
    
    # Prepare model
    model_dir = deployer.prepare_for_upload()
    
    # Upload
    response = input("\nUpload to HuggingFace Hub? (y/n): ")
    if response.lower() == 'y':
        deployer.upload_to_hub(model_dir)
    else:
        print(f"\nModel prepared but not uploaded.")
        print(f"Files are in: {model_dir}")
        print(f"\nTo upload later, run:")
        print(f"  python deploy_to_huggingface.py --checkpoint {args.checkpoint} --repo {args.repo}")


if __name__ == "__main__":
    main()
#     python deploy_to_huggingface.py \
#   --checkpoint ./checkpoints/epoch_8 \
#   --repo phoenix21/distance-aware-chronos