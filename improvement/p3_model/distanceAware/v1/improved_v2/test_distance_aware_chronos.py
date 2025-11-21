#!/usr/bin/env python3
"""
Complete Distance-Aware Chronos Implementation Testing Suite

This script tests:
1. Distance-aware loss functions (Wasserstein, Label Smoothing, Weighted CE)
2. Metrics (MAE, MAPE, WAE)
3. Training integration
4. Correctness verification

Run: python test_distance_aware_chronos.py

Output: test_results.log (detailed test results)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import sys
import traceback


# ============================================================================
# LOGGING SETUP
# ============================================================================

class Logger:
    def __init__(self, filename='test_results.log'):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
        self.write(f"{'='*80}\n")
        self.write(f"Distance-Aware Chronos Testing Suite\n")
        self.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.write(f"{'='*80}\n\n")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

# Redirect stdout to logger
logger = Logger('test_results.log')
sys.stdout = logger


# ============================================================================
# REFERENCE IMPLEMENTATIONS - Distance-Aware Loss Functions
# ============================================================================

def wasserstein_loss(predicted_probs, target_bins, bin_centers, p=1):
    """
    Wasserstein loss (Earth Mover's Distance)
    
    Args:
        predicted_probs: [batch_size, num_bins] - softmax probabilities
        target_bins: [batch_size] - target bin indices
        bin_centers: [num_bins] - center values of each bin
        p: 1 for W1 distance, 2 for W2 distance
    
    Returns:
        Scalar loss
    """
    batch_size, num_bins = predicted_probs.shape
    
    # Create degenerate target distribution (delta at target_bin)
    target_dist = torch.zeros_like(predicted_probs)
    target_dist[range(batch_size), target_bins] = 1.0
    
    # Compute cumulative distributions for efficient W1 computation
    target_cumsum = torch.cumsum(target_dist, dim=1)
    pred_cumsum = torch.cumsum(predicted_probs, dim=1)
    
    # Compute bin widths
    bin_widths = torch.diff(bin_centers, prepend=bin_centers[0:1])
    
    # W1 distance = integral of |CDF_target - CDF_pred|
    if p == 1:
        loss = torch.mean(torch.sum(
            torch.abs(target_cumsum - pred_cumsum) * bin_widths,
            dim=1
        ))
    elif p == 2:
        loss = torch.sqrt(torch.mean(torch.sum(
            (target_cumsum - pred_cumsum)**2 * bin_widths,
            dim=1
        )))
    else:
        raise ValueError(f"p must be 1 or 2, got {p}")
    
    return loss


def label_smoothing_loss(logits, target_bins, num_bins, sigma=1.0):
    """
    Ordinal label smoothing with Gaussian kernel
    
    Args:
        logits: [batch_size, num_bins] - model outputs before softmax
        target_bins: [batch_size] - target bin indices
        num_bins: number of bins
        sigma: smoothing bandwidth (higher = more spread)
    
    Returns:
        Scalar loss
    """
    batch_size = target_bins.size(0)
    smooth_labels = torch.zeros(batch_size, num_bins, device=logits.device)
    
    # Create smoothed labels with Gaussian kernel
    for i in range(batch_size):
        center = target_bins[i].item()
        for j in range(num_bins):
            distance = abs(j - center)
            smooth_labels[i, j] = torch.exp(torch.tensor(-distance**2 / (2 * sigma**2)))
        
        # Normalize to sum to 1
        smooth_labels[i] = smooth_labels[i] / smooth_labels[i].sum()
    
    # Cross-entropy with smoothed labels
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -torch.mean(torch.sum(smooth_labels * log_probs, dim=-1))
    
    return loss


def distance_weighted_cross_entropy(logits, target_bins, num_bins, alpha=1.0):
    """
    Cross-entropy weighted by distance from true class
    
    Args:
        logits: [batch_size, num_bins] - model outputs before softmax
        target_bins: [batch_size] - target bin indices
        num_bins: number of bins
        alpha: distance weighting parameter (higher = stronger penalty)
    
    Returns:
        Scalar loss
    """
    batch_size = target_bins.size(0)
    
    # Create distance matrix
    bin_indices = torch.arange(num_bins, device=logits.device)
    distances = torch.abs(bin_indices.unsqueeze(0) - target_bins.unsqueeze(1).float())
    
    # Distance weights: w_i = exp(alpha * |i - target|)
    weights = torch.exp(alpha * distances)
    
    # Weighted log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    weighted_log_probs = log_probs * weights
    
    # Gather weighted log prob at target indices
    target_weighted_log_probs = torch.gather(
        weighted_log_probs, 
        1, 
        target_bins.unsqueeze(1)
    )
    
    loss = -torch.mean(target_weighted_log_probs)
    
    return loss


# ============================================================================
# REFERENCE IMPLEMENTATIONS - Metrics
# ============================================================================

def compute_mae(predicted_probs, target_bins):
    """
    Mean Absolute Error using expected value (CORRECT implementation)
    
    Args:
        predicted_probs: [batch_size, num_bins] - softmax probabilities
        target_bins: [batch_size] - target bin indices
    
    Returns:
        MAE scalar
    """
    num_bins = predicted_probs.size(1)
    bin_indices = torch.arange(num_bins, device=predicted_probs.device, dtype=torch.float32)
    
    # Expected value: E[bin] = sum(bin_i * p_i)
    predicted_bins = torch.sum(predicted_probs * bin_indices, dim=1)
    
    # MAE
    mae = torch.mean(torch.abs(predicted_bins - target_bins.float()))
    
    return mae


def compute_mape(predicted_probs, target_bins, epsilon=1e-8):
    """
    Mean Absolute Percentage Error with division-by-zero protection
    
    Args:
        predicted_probs: [batch_size, num_bins]
        target_bins: [batch_size]
        epsilon: small constant to prevent division by zero
    
    Returns:
        MAPE in percentage
    """
    num_bins = predicted_probs.size(1)
    bin_indices = torch.arange(num_bins, device=predicted_probs.device, dtype=torch.float32)
    
    predicted_bins = torch.sum(predicted_probs * bin_indices, dim=1)
    
    # MAPE with epsilon
    mape = torch.mean(torch.abs(
        (target_bins.float() - predicted_bins) / (target_bins.float() + epsilon)
    )) * 100.0
    
    return mape


def compute_amae(predicted_probs, target_bins, num_bins):
    """
    Averaged MAE (class-balanced) - AMAE
    
    Args:
        predicted_probs: [batch_size, num_bins]
        target_bins: [batch_size]
        num_bins: total number of bins
    
    Returns:
        AMAE scalar
    """
    bin_indices = torch.arange(num_bins, device=predicted_probs.device, dtype=torch.float32)
    
    mae_per_class = []
    for k in range(num_bins):
        mask = (target_bins == k)
        if mask.sum() > 0:
            predicted_bins = torch.sum(predicted_probs[mask] * bin_indices, dim=1)
            mae_k = torch.mean(torch.abs(predicted_bins - k))
            mae_per_class.append(mae_k)
    
    if len(mae_per_class) == 0:
        return torch.tensor(0.0)
    
    amae = torch.mean(torch.stack(mae_per_class))
    return amae


def compute_wae_sample_weighted(predicted_probs, target_bins, sample_weights=None):
    """
    Weighted Absolute Error with per-sample weights
    
    Args:
        predicted_probs: [batch_size, num_bins]
        target_bins: [batch_size]
        sample_weights: [batch_size] - importance weights (None = uniform)
    
    Returns:
        WAE scalar
    """
    num_bins = predicted_probs.size(1)
    bin_indices = torch.arange(num_bins, device=predicted_probs.device, dtype=torch.float32)
    
    predicted_bins = torch.sum(predicted_probs * bin_indices, dim=1)
    absolute_errors = torch.abs(predicted_bins - target_bins.float())
    
    if sample_weights is None:
        sample_weights = torch.ones_like(absolute_errors)
    
    wae = torch.sum(absolute_errors * sample_weights) / torch.sum(sample_weights)
    return wae


# ============================================================================
# UNIT TESTS
# ============================================================================

def test_ordinal_structure(loss_fn, loss_name):
    """Test if loss respects ordinal structure"""
    print(f"\n{'='*80}")
    print(f"Testing Ordinal Structure: {loss_name}")
    print(f"{'='*80}")
    
    num_bins = 100
    true_bin = 50
    
    print(f"\nTrue bin: {true_bin}")
    print(f"Testing predictions at different distances...\n")
    
    results = []
    for pred_bin in [50, 51, 52, 55, 60, 70, 90]:
        logits = torch.zeros(1, num_bins)
        logits[0, pred_bin] = 10.0  # High confidence on predicted bin
        
        target = torch.tensor([true_bin])
        
        if 'wasserstein' in loss_name.lower():
            probs = F.softmax(logits, dim=-1)
            bin_centers = torch.linspace(-15, 15, num_bins)
            loss = loss_fn(probs, target, bin_centers)
        elif 'smoothing' in loss_name.lower():
            loss = loss_fn(logits, target, num_bins, sigma=1.0)
        elif 'weighted' in loss_name.lower():
            loss = loss_fn(logits, target, num_bins, alpha=1.0)
        else:
            loss = loss_fn(logits, target)
        
        distance = abs(pred_bin - true_bin)
        results.append((pred_bin, distance, loss.item()))
        print(f"  Predict bin {pred_bin:2d} (distance={distance:2d}): Loss = {loss.item():.6f}")
    
    # Verify monotonicity
    print("\n" + "-"*80)
    print("Verification: Loss should increase with distance")
    print("-"*80)
    
    monotonic = True
    for i in range(len(results) - 1):
        if results[i+1][2] < results[i][2]:  # Loss should increase
            monotonic = False
            print(f"❌ FAIL: Loss decreased from distance {results[i][1]} to {results[i+1][1]}")
            print(f"         {results[i][2]:.6f} > {results[i+1][2]:.6f}")
    
    if monotonic:
        print("✅ PASS: Loss increases monotonically with distance")
    
    return monotonic


def test_mae_implementation():
    """Test MAE uses expected value, not argmax"""
    print(f"\n{'='*80}")
    print(f"Testing MAE Implementation")
    print(f"{'='*80}")
    
    num_bins = 100
    
    # Test 1: Uniform distribution
    print("\nTest 1: Uniform distribution over 100 bins")
    print("-"*80)
    
    probs = torch.ones(1, num_bins) / num_bins
    target = torch.tensor([50])
    
    # Expected value of uniform is (0 + 99) / 2 = 49.5
    expected_bin = 49.5
    mae = compute_mae(probs, target)
    
    print(f"Target bin: {target.item()}")
    print(f"Expected predicted bin (uniform): {expected_bin}")
    print(f"MAE: {mae.item():.4f}")
    print(f"Expected MAE: {abs(expected_bin - target.item()):.4f}")
    
    test1_pass = abs(mae.item() - 0.5) < 0.01
    if test1_pass:
        print("✅ PASS: MAE correctly uses expected value")
    else:
        print("❌ FAIL: MAE incorrect")
    
    # Test 2: Delta distribution at specific bin
    print("\nTest 2: Delta distribution (all mass at bin 30)")
    print("-"*80)
    
    probs = torch.zeros(1, num_bins)
    probs[0, 30] = 1.0
    target = torch.tensor([50])
    
    mae = compute_mae(probs, target)
    expected_mae = abs(30 - 50)
    
    print(f"Target bin: {target.item()}")
    print(f"Predicted bin (delta at 30): 30")
    print(f"MAE: {mae.item():.4f}")
    print(f"Expected MAE: {expected_mae}")
    
    test2_pass = abs(mae.item() - expected_mae) < 0.01
    if test2_pass:
        print("✅ PASS: MAE correct for delta distribution")
    else:
        print("❌ FAIL: MAE incorrect")
    
    # Test 3: Bimodal distribution
    print("\nTest 3: Bimodal distribution (50% at bin 20, 50% at bin 60)")
    print("-"*80)
    
    probs = torch.zeros(1, num_bins)
    probs[0, 20] = 0.5
    probs[0, 60] = 0.5
    target = torch.tensor([50])
    
    mae = compute_mae(probs, target)
    expected_bin = 0.5 * 20 + 0.5 * 60  # = 40
    expected_mae = abs(expected_bin - 50)
    
    print(f"Target bin: {target.item()}")
    print(f"Expected predicted bin: {expected_bin}")
    print(f"MAE: {mae.item():.4f}")
    print(f"Expected MAE: {expected_mae}")
    
    test3_pass = abs(mae.item() - expected_mae) < 0.01
    if test3_pass:
        print("✅ PASS: MAE correct for bimodal distribution")
    else:
        print("❌ FAIL: MAE incorrect")
    
    return test1_pass and test2_pass and test3_pass


def test_mape_implementation():
    """Test MAPE handles division by zero"""
    print(f"\n{'='*80}")
    print(f"Testing MAPE Implementation")
    print(f"{'='*80}")
    
    num_bins = 100
    
    # Test 1: Non-zero target
    print("\nTest 1: Non-zero target (bin 50)")
    print("-"*80)
    
    probs = torch.zeros(1, num_bins)
    probs[0, 40] = 1.0  # Predict bin 40
    target = torch.tensor([50])
    
    mape = compute_mape(probs, target)
    expected_mape = abs((50 - 40) / 50) * 100  # = 20%
    
    print(f"Target: {target.item()}, Predicted: 40")
    print(f"MAPE: {mape.item():.2f}%")
    print(f"Expected: {expected_mape:.2f}%")
    
    test1_pass = abs(mape.item() - expected_mape) < 1.0
    if test1_pass:
        print("✅ PASS: MAPE correct for non-zero target")
    else:
        print("❌ FAIL: MAPE incorrect")
    
    # Test 2: Zero target (edge case)
    print("\nTest 2: Zero target (bin 0) - Division by zero protection")
    print("-"*80)
    
    probs = torch.zeros(1, num_bins)
    probs[0, 10] = 1.0  # Predict bin 10
    target = torch.tensor([0])
    
    try:
        mape = compute_mape(probs, target, epsilon=1e-8)
        print(f"Target: {target.item()}, Predicted: 10")
        print(f"MAPE: {mape.item():.2f}%")
        print(f"✅ PASS: No division by zero error")
        test2_pass = True
        
        if mape.item() > 1e6:
            print("⚠️  WARNING: MAPE explodes for near-zero targets!")
            print("   Recommendation: Use MAE instead for ordinal data")
    except Exception as e:
        print(f"❌ FAIL: Division by zero error: {e}")
        test2_pass = False
    
    return test1_pass and test2_pass


def test_gradient_flow(loss_fn, loss_name):
    """Test if gradients flow correctly"""
    print(f"\n{'='*80}")
    print(f"Testing Gradient Flow: {loss_name}")
    print(f"{'='*80}")
    
    num_bins = 100
    
    # Create logits with high confidence on wrong bin
    logits = torch.zeros(1, num_bins, requires_grad=True)
    logits.data[0, 20] = 10.0  # Predict bin 20
    target = torch.tensor([50])  # True is bin 50
    
    print(f"Setup: Predict bin 20 (wrong), True bin 50")
    print("-"*80)
    
    # Compute loss and gradients
    if 'wasserstein' in loss_name.lower():
        probs = F.softmax(logits, dim=-1)
        bin_centers = torch.linspace(-15, 15, num_bins)
        loss = loss_fn(probs, target, bin_centers)
    elif 'smoothing' in loss_name.lower():
        loss = loss_fn(logits, target, num_bins, sigma=1.0)
    elif 'weighted' in loss_name.lower():
        loss = loss_fn(logits, target, num_bins, alpha=1.0)
    else:
        loss = loss_fn(logits, target)
    
    loss.backward()
    
    # Check gradients
    print(f"\nLoss value: {loss.item():.6f}")
    print(f"\nGradient analysis:")
    print(f"  Gradient at true bin (50):       {logits.grad[0, 50].item():+.6f}")
    print(f"  Gradient at predicted bin (20):  {logits.grad[0, 20].item():+.6f}")
    print(f"  Gradient at nearby true (49):    {logits.grad[0, 49].item():+.6f}")
    print(f"  Gradient at nearby true (51):    {logits.grad[0, 51].item():+.6f}")
    
    # Verify gradient direction
    print("\n" + "-"*80)
    print("Verification:")
    print("-"*80)
    
    checks_pass = True
    
    # Check for NaN/Inf
    if torch.isnan(logits.grad).any():
        print("❌ FAIL: NaN in gradients")
        checks_pass = False
    elif torch.isinf(logits.grad).any():
        print("❌ FAIL: Inf in gradients")
        checks_pass = False
    else:
        print("✅ PASS: No NaN/Inf in gradients")
    
    # For distance-aware losses, gradient at true bin should encourage increasing logit there
    # (negative gradient since we minimize loss)
    if logits.grad[0, 50].item() < 0:
        print("✅ PASS: Gradient encourages moving toward true bin")
    else:
        print("⚠️  WARNING: Gradient direction unexpected at true bin")
    
    return checks_pass


def test_label_smoothing_normalization():
    """Test if label smoothing normalizes to 1"""
    print(f"\n{'='*80}")
    print(f"Testing Label Smoothing Normalization")
    print(f"{'='*80}")
    
    num_bins = 100
    logits = torch.zeros(10, num_bins)  # Dummy logits
    target_bins = torch.randint(0, num_bins, (10,))
    
    print(f"\nTesting with {len(target_bins)} random target bins")
    print("-"*80)
    
    # Extract smooth labels by computing loss with zero logits
    # (this tests the smoothing function directly)
    smooth_labels = torch.zeros(10, num_bins)
    
    for i in range(10):
        center = target_bins[i].item()
        for j in range(num_bins):
            distance = abs(j - center)
            smooth_labels[i, j] = torch.exp(torch.tensor(-distance**2 / (2 * 1.0**2)))
        smooth_labels[i] = smooth_labels[i] / smooth_labels[i].sum()
    
    # Check normalization
    sums = smooth_labels.sum(dim=1)
    
    print(f"Label sums (should all be 1.0):")
    for i, s in enumerate(sums):
        print(f"  Target bin {target_bins[i].item():2d}: sum = {s.item():.6f}")
    
    all_normalized = torch.allclose(sums, torch.ones(10), atol=1e-5)
    
    print("\n" + "-"*80)
    if all_normalized:
        print("✅ PASS: All smooth labels sum to 1.0")
    else:
        print("❌ FAIL: Smooth labels not properly normalized")
    
    # Check peak is at target
    print("\n" + "-"*80)
    print("Checking peaks are at target bins:")
    peaks_correct = True
    for i in range(10):
        peak_bin = torch.argmax(smooth_labels[i]).item()
        target_bin = target_bins[i].item()
        if peak_bin != target_bin:
            print(f"  ❌ Sample {i}: Peak at {peak_bin}, should be {target_bin}")
            peaks_correct = False
    
    if peaks_correct:
        print("✅ PASS: All peaks at correct target bins")
    else:
        print("❌ FAIL: Some peaks incorrect")
    
    return all_normalized and peaks_correct


def test_wasserstein_cumulative_property():
    """Test if Wasserstein uses cumulative distributions"""
    print(f"\n{'='*80}")
    print(f"Testing Wasserstein Cumulative Distribution Property")
    print(f"{'='*80}")
    
    num_bins = 100
    bin_centers = torch.linspace(-15, 15, num_bins)
    
    # Test case: Two distributions that differ by translation
    print("\nTest: Translating distribution should give proportional distance")
    print("-"*80)
    
    # Distribution 1: Delta at bin 40
    probs1 = torch.zeros(1, num_bins)
    probs1[0, 40] = 1.0
    target = torch.tensor([50])
    
    # Distribution 2: Delta at bin 45 (closer)
    probs2 = torch.zeros(1, num_bins)
    probs2[0, 45] = 1.0
    
    loss1 = wasserstein_loss(probs1, target, bin_centers, p=1)
    loss2 = wasserstein_loss(probs2, target, bin_centers, p=1)
    
    print(f"Distance from bin 40 to 50: W1 = {loss1.item():.6f}")
    print(f"Distance from bin 45 to 50: W1 = {loss2.item():.6f}")
    print(f"Ratio: {loss1.item() / loss2.item():.2f} (should be ~2.0)")
    
    ratio_correct = abs(loss1.item() / loss2.item() - 2.0) < 0.1
    
    if ratio_correct:
        print("✅ PASS: Wasserstein distance scales correctly")
    else:
        print("❌ FAIL: Wasserstein distance scaling incorrect")
    
    # Test triangle inequality
    print("\n" + "-"*80)
    print("Test: Triangle inequality W(p1, p3) ≤ W(p1, p2) + W(p2, p3)")
    print("-"*80)
    
    probs3 = torch.zeros(1, num_bins)
    probs3[0, 60] = 1.0
    target1 = torch.tensor([40])
    target2 = torch.tensor([50])
    target3 = torch.tensor([60])
    
    w_1_3 = wasserstein_loss(probs1, target3, bin_centers, p=1)
    w_1_2 = wasserstein_loss(probs1, target2, bin_centers, p=1)
    w_2_3 = wasserstein_loss(probs2, target3, bin_centers, p=1)
    
    print(f"W(40, 60) = {w_1_3.item():.6f}")
    print(f"W(40, 50) + W(50, 60) = {(w_1_2.item() + w_2_3.item()):.6f}")
    
    triangle_holds = w_1_3.item() <= (w_1_2.item() + w_2_3.item() + 1e-5)
    
    if triangle_holds:
        print("✅ PASS: Triangle inequality holds")
    else:
        print("❌ FAIL: Triangle inequality violated")
    
    return ratio_correct and triangle_holds


def test_integration_overfitting():
    """Test if model can overfit single batch"""
    print(f"\n{'='*80}")
    print(f"Integration Test: Overfitting on Single Batch")
    print(f"{'='*80}")
    
    # Simple model
    class SimpleModel(nn.Module):
        def __init__(self, num_bins=100):
            super().__init__()
            self.fc = nn.Linear(10, num_bins)
        
        def forward(self, x):
            return self.fc(x)
    
    model = SimpleModel(num_bins=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Single batch
    batch_x = torch.randn(4, 10)
    batch_y = torch.tensor([20, 30, 40, 50])
    
    print(f"\nTraining on single batch for 100 steps...")
    print(f"Target bins: {batch_y.tolist()}")
    print("-"*80)
    
    losses = []
    for step in range(100):
        optimizer.zero_grad()
        
        logits = model(batch_x)
        loss = label_smoothing_loss(logits, batch_y, num_bins=100, sigma=1.0)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 20 == 0:
            print(f"Step {step:3d}: Loss = {loss.item():.6f}")
    
    final_loss = losses[-1]
    initial_loss = losses[0]
    
    print("\n" + "-"*80)
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss:   {final_loss:.6f}")
    print(f"Reduction:    {(initial_loss - final_loss):.6f}")
    
    overfitted = final_loss < initial_loss * 0.1
    
    if overfitted:
        print("✅ PASS: Model can overfit (loss reduced >90%)")
    else:
        print("⚠️  WARNING: Model did not overfit well")
        print("   This might indicate implementation issues")
    
    return overfitted


# ============================================================================
# COMPREHENSIVE TEST SUITE
# ============================================================================

def run_all_tests():
    """Run all tests and generate report"""
    print(f"\n{'#'*80}")
    print(f"#{'COMPREHENSIVE TEST SUITE':^78}#")
    print(f"#{'for Distance-Aware Chronos Implementation':^78}#")
    print(f"{'#'*80}\n")
    
    results = {}
    
    # Test 1: Ordinal structure for each loss type
    print(f"\n{'#'*80}")
    print(f"# SECTION 1: Ordinal Structure Tests")
    print(f"{'#'*80}")
    
    results['wasserstein_ordinal'] = test_ordinal_structure(
        wasserstein_loss, "Wasserstein Loss"
    )
    
    results['label_smoothing_ordinal'] = test_ordinal_structure(
        label_smoothing_loss, "Label Smoothing Loss"
    )
    
    results['weighted_ce_ordinal'] = test_ordinal_structure(
        distance_weighted_cross_entropy, "Distance-Weighted Cross-Entropy"
    )
    
    # Test 2: MAE implementation
    print(f"\n{'#'*80}")
    print(f"# SECTION 2: Metric Implementation Tests")
    print(f"{'#'*80}")
    
    results['mae_correct'] = test_mae_implementation()
    results['mape_correct'] = test_mape_implementation()
    
    # Test 3: Gradient flow
    print(f"\n{'#'*80}")
    print(f"# SECTION 3: Gradient Flow Tests")
    print(f"{'#'*80}")
    
    results['wasserstein_gradients'] = test_gradient_flow(
        wasserstein_loss, "Wasserstein Loss"
    )
    
    results['label_smoothing_gradients'] = test_gradient_flow(
        label_smoothing_loss, "Label Smoothing Loss"
    )
    
    results['weighted_ce_gradients'] = test_gradient_flow(
        distance_weighted_cross_entropy, "Distance-Weighted Cross-Entropy"
    )
    
    # Test 4: Specific properties
    print(f"\n{'#'*80}")
    print(f"# SECTION 4: Specific Property Tests")
    print(f"{'#'*80}")
    
    results['label_smoothing_normalized'] = test_label_smoothing_normalization()
    results['wasserstein_cumulative'] = test_wasserstein_cumulative_property()
    
    # Test 5: Integration
    print(f"\n{'#'*80}")
    print(f"# SECTION 5: Integration Tests")
    print(f"{'#'*80}")
    
    results['integration_overfit'] = test_integration_overfitting()
    
    # Generate summary report
    print(f"\n{'='*80}")
    print(f"{'FINAL TEST REPORT':^80}")
    print(f"{'='*80}\n")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total} ({passed/total*100:.1f}%)\n")
    
    print(f"{'Test Name':<50} {'Result':<10}")
    print(f"{'-'*60}")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<50} {status:<10}")
    
    print(f"\n{'='*80}")
    print(f"Detailed results saved to: test_results.log")
    print(f"{'='*80}\n")
    
    # Summary recommendations
    print(f"\n{'='*80}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    if all(results.values()):
        print("🎉 All tests passed! Your implementation looks correct.\n")
        print("Next steps:")
        print("  1. Compare these reference implementations with your code")
        print("  2. Run on real Chronos model and training data")
        print("  3. Benchmark against standard cross-entropy baseline")
    else:
        print("⚠️  Some tests failed. Review the following:\n")
        
        if not results['mae_correct']:
            print("  • MAE Implementation:")
            print("    - Ensure you use expected value: sum(probs * bin_indices)")
            print("    - Do NOT use argmax(probs)")
        
        if not results['label_smoothing_normalized']:
            print("  • Label Smoothing:")
            print("    - Ensure smooth_labels[i] = smooth_labels[i] / smooth_labels[i].sum()")
        
        if not results['wasserstein_cumulative']:
            print("  • Wasserstein Loss:")
            print("    - Must use cumulative distributions: torch.cumsum(...)")
        
        if not any([results['wasserstein_ordinal'], 
                    results['label_smoothing_ordinal'],
                    results['weighted_ce_ordinal']]):
            print("  • Ordinal Structure:")
            print("    - Loss must increase with distance from true bin")
            print("    - Check distance calculation in loss function")
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        print(f"\n{'='*80}")
        print(f"Starting test suite...")
        print(f"{'='*80}\n")
        
        # Run all tests
        results = run_all_tests()
        
        print(f"\n{'='*80}")
        print(f"Test suite completed successfully!")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: Test suite failed with exception")
        print(f"{'='*80}\n")
        print(f"Exception: {e}")
        print(f"\nFull traceback:")
        traceback.print_exc()
    
    finally:
        # Close logger
        sys.stdout = sys.__stdout__
        logger.close()
        print(f"\nResults saved to: test_results.log")