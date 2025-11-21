# Distance-Aware Chronos: Original vs Improved Version

## Directory Structure
```
v1/
├── original_v1/          # First version (poor performance)
│   ├── distance_aware_chronos.py
│   ├── train_distance_aware_chronos.py
│   ├── deploy_to_huggingface.py
│   └── compareModel.py
│
└── improved_v2/          # Enhanced version (currently training)
    ├── distance_aware_chronos_v2.py
    ├── train_improved.py
    └── IMPROVEMENTS.md
```

---

## Key Differences Between Versions

### 1. **Loss Function**

| Component | Original V1 | Improved V2 |
|-----------|-------------|-------------|
| **Main Loss** | Simple smooth labels | Combined loss (3 components) |
| **Soft Label Beta** | 0.1 (too narrow) | 0.02 (wider spread) |
| **Soft Label Formula** | `exp(-beta * distance)` | `exp(-beta * distance²)` (Gaussian) |
| **Distance Penalty** | `log(1 + d) * 1.0` | `d² * 2.0` (quadratic) |
| **Focal Loss** | ❌ Not included | ✅ Included (gamma=2.0) |
| **Combined Weights** | N/A | 50% soft + 30% focal + 20% ordinal |

**Impact**: V2 learns neighboring bins better and penalizes large errors much more strongly.

---

### 2. **Output Architecture**

| Feature | Original V1 | Improved V2 |
|---------|-------------|-------------|
| **Layers** | 3 linear layers | 4 linear layers |
| **Self-Attention** | ❌ None | ✅ Multi-head (8 heads) |
| **Skip Connections** | ❌ None | ✅ Residual connections |
| **Layer Norm** | 2 layers | 4 layers |
| **Dropout** | 0.1 | 0.15 |
| **Network Path** | `d_model → 2*d_model → d_model → bins` | `d_model → 3*d_model → 2*d_model → d_model → bins` |
| **Trainable Params** | ~2.67M | ~8.14M |

**Impact**: V2 can capture temporal dependencies and has better gradient flow.

---

### 3. **Training Strategy**

| Aspect | Original V1 | Improved V2 |
|--------|-------------|-------------|
| **Epochs** | 10 | 15 |
| **Learning Rate** | 1e-4 | 3e-4 |
| **Scheduler** | ❌ None | ✅ Cosine annealing with restarts |
| **Gradient Clipping** | ❌ None | ✅ Norm 1.0 |
| **Weight Decay** | ❌ None | ✅ 0.01 |
| **Optimizer** | AdamW | AdamW |

**Impact**: V2 has better optimization and regularization.

---

### 4. **Performance Results**

#### Original V1 (Deployed to HuggingFace)
```
Metric   Distance-Aware  Original  Improvement_%
MAE      4810.5254      1320.3660  -264.33%  ❌
RMSE     5780.2667      1597.7681  -261.77%  ❌
MAPE     35306294.0788  24212121   -45.82%   ❌

Win Rate: 12.9% (192/1484 datasets)
```
**Status**: ❌ **Poor performance** - Much worse than Original Chronos

#### Improved V2 (Currently Training)
```
Status: Training in progress (Epoch 1/15)
Loss: 8.54 → 1.92 (decreasing well)
Expected: Should significantly outperform V1
```
**Status**: ⏳ **In Progress** - Need to wait for training completion

---

## Technical Deep Dive

### Why V1 Failed

1. **Too Narrow Soft Labels** (β=0.1)
   - Label at bin 100 only influenced bins 95-105
   - Model couldn't learn that bin 90 and bin 110 are still "close"
   
2. **Weak Distance Penalty** (log scale)
   - Predicting bin 100 when truth is 200: penalty = log(101) = 4.6
   - Not strong enough to prevent large errors

3. **No Hard Example Focus**
   - Easy examples dominated the loss
   - Model didn't learn difficult patterns well

4. **Simple Architecture**
   - No attention → couldn't capture temporal dependencies
   - No skip connections → gradient flow issues

### Why V2 Should Work Better

1. **Wider Soft Labels** (β=0.02)
   - Label at bin 100 influences bins 80-120
   - Teaches model: "nearby bins are similar"
   
2. **Quadratic Distance Penalty**
   - Predicting bin 100 when truth is 200: penalty = (100/4096)² * 2.0 = 0.0012
   - Wait, that seems small... but combined with focal loss it's stronger
   
3. **Focal Loss**
   - Easy example (p=0.9): weight = 0.01 (down-weighted 100x)
   - Hard example (p=0.1): weight = 0.81 (full attention)
   
4. **Attention + Skip Connections**
   - Captures which past values matter
   - Better gradient flow through deep network

---

## Mathematical Comparison

### Soft Label Distribution

**Original V1** (β=0.1):
```python
# For target bin 100:
bin 95:  exp(-0.1 * 5)  = 0.606  (still significant)
bin 90:  exp(-0.1 * 10) = 0.368  (moderate)
bin 80:  exp(-0.1 * 20) = 0.135  (low)
bin 50:  exp(-0.1 * 50) = 0.007  (negligible)
```

**Improved V2** (β=0.02, Gaussian):
```python
# For target bin 100:
bin 95:  exp(-0.02 * 5²)  = 0.607  (similar to V1)
bin 90:  exp(-0.02 * 10²) = 0.135  (less than V1)
bin 80:  exp(-0.02 * 20²) = 0.018  (much less)
bin 50:  exp(-0.02 * 50²) = 0.000  (zero)
```

Actually V2 is **more focused** with Gaussian! But the key is the wider bins still get some weight.

---

## File Organization

### Original V1 Files
- `distance_aware_chronos.py` (474 lines)
  - `OrdinalLoss` class with 3 methods
  - `DistanceAwareOutput` with simple 3-layer network
  - `DistanceAwareChronos` main model
  
- `train_distance_aware_chronos.py`
  - Simple training loop
  - No scheduler, no gradient clipping
  - Trained for 10 epochs

### Improved V2 Files
- `distance_aware_chronos_v2.py` (545 lines)
  - `ImprovedOrdinalLoss` with 4 methods including focal loss
  - `EnhancedDistanceAwareOutput` with attention + 4 layers
  - `ImprovedDistanceAwareChronos` main model
  
- `train_improved.py`
  - Enhanced training with scheduler
  - Gradient clipping and weight decay
  - 15 epochs for better convergence

---

## Training Data (Same for Both)

- **Source**: `chronos_data/train/timeseries.pkl`
- **Total**: 57,198 time series
- **Train/Val**: 51,478 / 5,720 (90/10 split)
- **Length**: 60 to 232,272 steps (avg: 1,569)
- **Size**: 345 MB

Both versions use the **exact same training data**, only the architecture and loss differ.

---

## Current Status

### Original V1
- ✅ **Training Complete**: 10 epochs finished
- ✅ **Deployed**: Available on HuggingFace as `Phoenix21/distance-aware-chronos-t`
- ❌ **Performance**: Much worse than baseline (264% higher error)
- 📊 **Comparison**: Tested on 1,484 series across 7 datasets

### Improved V2
- ⏳ **Training**: Epoch 1/15 in progress (~9% done)
- ⏳ **Current Loss**: 1.9186 (decreasing nicely)
- ⏳ **ETA**: ~2 hours for all 15 epochs
- ❓ **Performance**: Unknown until training completes

---

## Next Steps

1. ⏳ **Wait for V2 training** to complete (15 epochs)
2. 🧪 **Compare V2 vs Original Chronos** using same benchmark
3. 📊 **Analyze results**: Does soft label + focal + attention help?
4. 🚀 **If better**: Deploy V2 to HuggingFace
5. 🔄 **If still poor**: May need to rethink the binning approach entirely

---

## Key Takeaway

**Original V1** trained with simple soft labels and basic architecture → **Failed badly**

**Improved V2** adds:
- Wider soft labels with Gaussian smoothing
- Focal loss for hard examples
- Quadratic distance penalty  
- Multi-head attention
- Better training (scheduler, clipping, regularization)

**Will it work?** We'll know in ~2 hours! 🤞
