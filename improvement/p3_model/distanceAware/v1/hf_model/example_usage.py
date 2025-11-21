# Example: Using Distance-Aware Chronos

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
