"""Quick test to see why predictions are NaN"""
import numpy as np
import torch
from distance_aware_chronos import DistanceAwareChronos
from huggingface_hub import hf_hub_download
import json

print("Loading model...")
config_path = hf_hub_download(
    repo_id="Phoenix21/distance-aware-chronos-t",
    filename="config.json",
    repo_type="model"
)
distance_output_path = hf_hub_download(
    repo_id="Phoenix21/distance-aware-chronos-t",
    filename="distance_output.pt",
    repo_type="model"
)

with open(config_path, 'r') as f:
    da_config = json.load(f)

model = DistanceAwareChronos(
    model_name=da_config.get('base_model', 'amazon/chronos-t5-small'),
    num_bins=da_config.get('num_bins', 4096),
    device='cuda'
)

state_dict = torch.load(distance_output_path, map_location='cuda')
model.distance_output.load_state_dict(state_dict)
model.eval()

print("✓ Model loaded")

# Simple test
test_context = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 10)
print(f"Test context shape: {test_context.shape}")
print(f"Test context sample: {test_context[:10]}")

try:
    print("\nTesting prediction...")
    forecast = model.predict(test_context, horizon=5, num_samples=10)
    print(f"Forecast: {forecast}")
    print(f"Forecast shape: {forecast.shape}")
    print(f"Has NaN: {np.any(np.isnan(forecast))}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
