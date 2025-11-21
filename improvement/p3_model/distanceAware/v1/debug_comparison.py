"""Debug comparison to see what's failing"""
import numpy as np
import torch
from distance_aware_chronos import DistanceAwareChronos
from chronos import ChronosPipeline
from huggingface_hub import hf_hub_download
import json

# Load Distance-Aware model
print("Loading Distance-Aware model...")
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
da_model = DistanceAwareChronos(
    model_name=da_config.get('base_model', 'amazon/chronos-t5-small'),
    num_bins=da_config.get('num_bins', 4096),
    device=device
)

state_dict = torch.load(distance_output_path, map_location=device)
da_model.distance_output.load_state_dict(state_dict)
print("✓ Distance-Aware loaded")

# Load Original model
print("Loading Original Chronos...")
orig_model = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map=device,
    torch_dtype=torch.float32
)
print("✓ Original loaded")

# Test with simple data
print("\nTesting with simple series...")
test_series = np.random.randn(200).cumsum() + 100
context = test_series[:150]
truth = test_series[150:158]

print(f"Context: shape={context.shape}, mean={context.mean():.2f}, std={context.std():.2f}")
print(f"Truth: {truth[:5]}")

# Test Distance-Aware
print("\n1. Testing Distance-Aware predict()...")
try:
    da_pred = da_model.predict(context, horizon=8, num_samples=20)
    print(f"   ✓ Prediction: {da_pred}")
    print(f"   Has NaN: {np.any(np.isnan(da_pred))}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test Original
print("\n2. Testing Original Chronos predict()...")
try:
    orig_forecast = orig_model.predict(
        torch.tensor(context[np.newaxis, :], dtype=torch.float32),
        8,
        num_samples=20
    )
    orig_pred = np.median(orig_forecast.cpu().numpy()[0], axis=0)
    print(f"   ✓ Prediction: {orig_pred}")
    print(f"   Has NaN: {np.any(np.isnan(orig_pred))}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")
