---
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

**Base Model:** amazon/chronos-t5-small  
**Number of Bins:** 4096  
**Training Epoch:** 10  
**Validation Loss:** 2.4703

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
model = DistanceAwareChronos.from_pretrained("Phoenix21/distance-aware-chronos-t")

# Prepare your time series
context = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Your historical data

# Generate forecasts
predictions = model.predict(context, horizon=24, num_samples=100)

print(f"Forecast shape: {predictions.shape}")
```

## Training Data

Trained on the [Chronos datasets](https://huggingface.co/datasets/autogluon/chronos_datasets) 
from HuggingFace.

## Citation

If you use this model, please cite:
```bibtex
@article{chronos2024,
  title={Chronos: Learning the Language of Time Series},
  author={Ansari, Abdul Fatir et al.},
  journal={Transactions on Machine Learning Research},
  year={2024}
}
```

## License

Apache 2.0
