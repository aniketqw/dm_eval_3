# Improvements to Distance-Aware Chronos

## Current Performance Issue
The baseline model showed poor performance:
- MAE: 4810.53 vs 1320.37 (Original) - **264% worse**
- RMSE: 5780.27 vs 1597.77 (Original) - **262% worse**  
- Win rate: Only 12.9% (192/1484 datasets)

## Key Improvements Made

### 1. **Enhanced Soft Label Smoothing**
**Problem**: Original used `beta=0.1` which created too narrow distributions
**Solution**: 
- Changed to `beta=0.02` for wider Gaussian smoothing
- Formula: `exp(-beta * distance^2)` instead of `exp(-beta * distance)`
- **Impact**: Allows model to learn neighboring bins better, reducing over-confident predictions

### 2. **Focal Loss Addition**
**New Component**: Added focal loss to focus on hard examples
```python
focal_weight = (1 - p_t)^gamma
loss = focal_weight * cross_entropy + alpha * distance^2
```
- **Purpose**: Down-weights easy examples, focuses on difficult predictions
- **Parameters**: `gamma=2.0` (standard focal loss parameter)

### 3. **Quadratic Distance Penalty**
**Problem**: Original used `log(1 + distance)` - too weak
**Solution**: Changed to `distance^2 * alpha` with `alpha=2.0`
- **Impact**: Strongly penalizes predictions far from ground truth
- Quadratic penalty grows much faster than logarithmic

### 4. **Combined Loss Function**
**New Approach**: Weighted combination of 3 losses
```python
combined_loss = 0.5 * soft_label + 0.3 * focal + 0.2 * ordinal
```
- **Soft labels**: Main learning signal with smooth targets
- **Focal loss**: Hard example mining
- **Ordinal loss**: Distance-aware penalty

### 5. **Enhanced Output Architecture**
**Added**:
- **Multi-head Self-Attention**: Captures temporal dependencies
  - 8 attention heads
  - Learns relationships across time steps
- **Deeper Network**: 4 layers instead of 3
  - `d_model → 3*d_model → 2*d_model → d_model → num_bins`
- **Skip Connections**: Residual connections for better gradient flow
- **Layer Normalization**: After each layer for stable training
- **Increased Dropout**: 0.15 (from 0.1) for better regularization

### 6. **Improved Training Strategy**
- **Learning Rate Scheduler**: Cosine annealing with warm restarts
- **Gradient Clipping**: Clip to norm of 1.0
- **Weight Decay**: 0.01 for L2 regularization
- **More Epochs**: 15 instead of 10 for better convergence

## Architecture Comparison

| Component | Original | Improved |
|-----------|----------|----------|
| Loss Function | Simple smooth labels | Combined (soft + focal + ordinal) |
| Beta (smoothing) | 0.1 (narrow) | 0.02 (wide) |
| Alpha (distance penalty) | 1.0 | 2.0 |
| Distance penalty | log(1+d) | d^2 |
| Output layers | 3 linear layers | 4 layers + attention |
| Skip connections | No | Yes (residual) |
| Dropout | 0.1 | 0.15 |
| Attention | No | Multi-head (8 heads) |
| Trainable params | ~2.67M | ~11.5M |

## Expected Improvements

### 1. **Better Ordinal Awareness**
- Quadratic penalty forces model to respect bin ordering
- Soft labels with wider spread teach neighborhood relationships

### 2. **Reduced Overfitting**
- Focal loss prevents over-confidence on easy examples
- Higher dropout and weight decay improve generalization

### 3. **Better Temporal Modeling**
- Self-attention captures dependencies across context
- Helps model understand trends and patterns

### 4. **More Robust Predictions**
- Combined loss provides multiple learning signals
- Skip connections improve gradient flow

## How to Train

```bash
cd /home/h20250169/study/modelTraining/dm_eval_3/improvement/p3_model/distanceAware/v1
/home/h20250169/miniconda3/envs/chronos_env/bin/python train_improved.py
```

**Training Configuration**:
- Epochs: 15
- Batch size: 8
- Learning rate: 3e-4 with cosine annealing
- Gradient clipping: 1.0
- Weight decay: 0.01

**Checkpoints**: Saved to `./checkpoints_v2/`

## Testing the Improved Model

After training, update `compareModel.py` to use:
```python
from distance_aware_chronos_v2 import ImprovedDistanceAwareChronos

model = ImprovedDistanceAwareChronos(
    model_name="amazon/chronos-t5-small",
    num_bins=4096,
    use_enhanced_output=True
)
# Load checkpoint
model.distance_output.load_state_dict(torch.load('checkpoints_v2/best_model.pt'))
```

## Why These Changes Should Help

1. **Soft Labels with β=0.02**: Creates smoother target distributions
   - Original β=0.1: Label at bin 100 only affects bins 95-105
   - New β=0.02: Label at bin 100 affects bins 80-120
   - **Result**: Model learns "nearness" better

2. **Quadratic Distance Penalty**: Stronger penalization
   - Off by 10 bins: 1.0 → 0.006 (penalty 0.006 vs 0.0003)
   - Off by 100 bins: 1.0 → 0.61 (penalty 0.61 vs 0.095)
   - **Result**: Large errors heavily penalized

3. **Focal Loss**: Adaptive difficulty
   - Easy example (p=0.9): Weight = 0.01 (down-weighted 100x)
   - Hard example (p=0.1): Weight = 0.81 (nearly full weight)
   - **Result**: Focuses on improving hard predictions

4. **Attention Mechanism**: Context awareness
   - Learns which past values are important
   - Captures trends, seasonality, outliers
   - **Result**: Better understanding of temporal patterns

## Next Steps

1. **Train the improved model** using `train_improved.py`
2. **Compare results** with Original Chronos
3. **If successful**: Deploy to HuggingFace
4. **If still poor**: May need to reconsider the binning approach entirely
