# distance_aware_chronos_v2.py - Improved with soft labels and better distance awareness

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# For loading pretrained Chronos
from transformers import T5ForConditionalGeneration, T5Config
from chronos import ChronosPipeline

# ============================================================================
# PART 1: ENHANCED DISTANCE-AWARE LOSS FUNCTIONS
# ============================================================================

class ImprovedOrdinalLoss(nn.Module):
    """Enhanced distance-aware loss with soft labels and focal loss"""
    
    def __init__(self, num_bins: int = 4096, alpha: float = 2.0, beta: float = 0.02, gamma: float = 2.0):
        super().__init__()
        self.num_bins = num_bins
        self.alpha = alpha  # Weight for ordinal penalty (increased)
        self.beta = beta    # Temperature for soft labels (lower = more spread)
        self.gamma = gamma  # Focal loss gamma parameter
        
        # Precompute distance matrix (vectorized)
        self.register_buffer('distance_matrix', self._create_distance_matrix())
        
    def _create_distance_matrix(self):
        """Create matrix of distances between all bin pairs (vectorized)"""
        indices = torch.arange(self.num_bins).float()
        matrix = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
        matrix = matrix / self.num_bins  # Normalized distance [0, 1]
        return matrix
    
    def smooth_label_loss(self, logits, targets):
        """Enhanced label smoothing with Gaussian distribution"""
        batch_size = targets.size(0)
        device = logits.device
        
        # Create smooth labels with Gaussian distribution (vectorized)
        bin_indices = torch.arange(self.num_bins, device=device).float()
        targets_expanded = targets.unsqueeze(1).float()  # [batch, 1]
        
        # Gaussian kernel: exp(-beta * distance^2)
        distances = torch.abs(bin_indices - targets_expanded)  # [batch, num_bins]
        smooth_labels = torch.exp(-self.beta * distances ** 2)
        
        # Normalize to probability distribution
        smooth_labels = smooth_labels / smooth_labels.sum(dim=1, keepdim=True)
        
        # KL divergence with smooth labels
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(smooth_labels * log_probs, dim=-1)
        return loss.mean()
    
    def focal_loss(self, logits, targets):
        """Focal loss to focus on hard examples with distance awareness"""
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get probability of correct class
        target_probs = torch.gather(probs, 1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal loss term: (1 - p_t)^gamma
        focal_weight = (1 - target_probs) ** self.gamma
        
        # Standard cross-entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Apply focal weight
        focal_ce = focal_weight * ce_loss
        
        # Add distance penalty for incorrect predictions
        predictions = torch.argmax(logits, dim=-1)
        distances = torch.gather(
            self.distance_matrix[predictions],
            1,
            targets.unsqueeze(1)
        ).squeeze(1)
        
        # Combined focal + distance loss
        total_loss = focal_ce + self.alpha * distances ** 2
        return total_loss.mean()
    
    def ordinal_cross_entropy(self, logits, targets):
        """Cross-entropy with quadratic distance penalty"""
        # Standard cross-entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Get predicted bins
        predictions = torch.argmax(logits, dim=-1)
        
        # Calculate distance penalty (quadratic for stronger penalization)
        distances = torch.gather(
            self.distance_matrix[predictions], 
            1, 
            targets.unsqueeze(1)
        ).squeeze(1)
        
        # Quadratic distance penalty (penalizes large errors more)
        distance_penalty = (distances ** 2) * self.alpha
        
        total_loss = ce_loss + distance_penalty
        return total_loss.mean()
    
    def combined_loss(self, logits, targets):
        """Combine multiple loss components for robust training"""
        # Soft label loss (main component)
        soft_loss = self.smooth_label_loss(logits, targets)
        
        # Focal loss (hard examples)
        focal = self.focal_loss(logits, targets)
        
        # Ordinal loss (distance penalty)
        ordinal = self.ordinal_cross_entropy(logits, targets)
        
        # Weighted combination
        return 0.5 * soft_loss + 0.3 * focal + 0.2 * ordinal


# ============================================================================
# PART 2: ENHANCED DISTANCE-AWARE OUTPUT LAYER
# ============================================================================

class EnhancedDistanceAwareOutput(nn.Module):
    """Enhanced output layer with attention and distance awareness"""
    
    def __init__(self, d_model: int = 512, num_bins: int = 4096, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_bins = num_bins
        self.num_heads = num_heads
        
        # Self-attention for capturing temporal dependencies
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Enhanced multi-layer projection with skip connections
        self.fc1 = nn.Linear(d_model, d_model * 3)
        self.fc2 = nn.Linear(d_model * 3, d_model * 2)
        self.fc3 = nn.Linear(d_model * 2, d_model)
        self.fc_out = nn.Linear(d_model, num_bins)
        
        # Layer normalization
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model * 3)
        self.norm2 = nn.LayerNorm(d_model * 2)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.15)
        
        # Residual projection for skip connection
        self.residual_proj = nn.Linear(d_model, num_bins)
        
    def forward(self, hidden_states):
        # hidden_states: [batch, seq_len, d_model]
        batch_size, seq_len, _ = hidden_states.shape
        
        # Self-attention to capture temporal patterns
        attn_out, _ = self.self_attn(hidden_states, hidden_states, hidden_states)
        attn_out = self.dropout(attn_out)
        hidden_states = self.norm_attn(hidden_states + attn_out)  # Residual
        
        # Store input for skip connection
        residual = self.residual_proj(hidden_states)
        
        # Deep projection with residuals
        x = self.fc1(hidden_states)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.norm2(x)
        x = F.gelu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.norm3(x)
        x = F.gelu(x)
        x = self.dropout(x)
        
        # Final projection
        logits = self.fc_out(x)
        
        # Add skip connection from input
        logits = logits + 0.1 * residual  # Weighted residual
        
        return logits  # [batch, seq_len, num_bins]


# ============================================================================
# PART 3: IMPROVED DISTANCE-AWARE CHRONOS MODEL
# ============================================================================

class ImprovedDistanceAwareChronos(nn.Module):
    """Improved Chronos with enhanced distance-aware output"""
    
    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-small",
        num_bins: int = 4096,
        device: str = None,
        use_enhanced_output: bool = True
    ):
        super().__init__()
        
        self.num_bins = num_bins
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading pretrained model: {model_name}")
        
        # Load pretrained Chronos T5 model
        self.base_model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.config = self.base_model.config
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Enhanced distance-aware output layer
        if use_enhanced_output:
            self.distance_output = EnhancedDistanceAwareOutput(
                d_model=self.config.d_model,
                num_bins=num_bins,
                num_heads=8
            )
        else:
            # Fallback to simpler version
            self.distance_output = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_model * 2),
                nn.LayerNorm(self.config.d_model * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.config.d_model * 2, num_bins)
            )
        
        # Improved loss function
        self.loss_fn = ImprovedOrdinalLoss(
            num_bins=num_bins,
            alpha=2.0,  # Higher weight for distance penalty
            beta=0.02,  # Lower temp for wider smoothing
            gamma=2.0   # Focal loss parameter
        )
        
        # For prediction
        self.temperature = 1.0
        
        self.to(self.device)
    
    def forward(self, input_ids):
        """Forward pass through model"""
        # Get encoder outputs
        encoder_outputs = self.base_model.get_encoder()(input_ids)
        hidden_states = encoder_outputs.last_hidden_state
        
        # Pass through distance-aware output
        logits = self.distance_output(hidden_states)
        
        return {'logits': logits, 'hidden_states': hidden_states}
    
    def tokenize_time_series(self, series: np.ndarray) -> torch.Tensor:
        """Tokenize time series into bins"""
        series = np.array(series, dtype=np.float32)
        series = series[~np.isnan(series)]  # Remove NaN
        
        # Normalize
        scale = np.abs(series).mean() + 1e-10
        normalized = series / scale
        normalized = np.clip(normalized, -15, 15)
        
        # Discretize into bins
        bins = np.linspace(-15, 15, self.num_bins + 1)
        tokens = np.digitize(normalized, bins) - 1
        tokens = np.clip(tokens, 0, self.num_bins - 1)
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def predict(
        self, 
        context: np.ndarray, 
        horizon: int = 24,
        num_samples: int = 100
    ) -> np.ndarray:
        """Generate predictions with improved sampling"""
        self.eval()
        
        # Tokenize context
        scale = np.abs(context).mean() + 1e-10
        context_tokens = self.tokenize_time_series(context)
        context_tokens = context_tokens.unsqueeze(0).to(self.device)
        
        predictions = []
        
        with torch.no_grad():
            current_tokens = context_tokens
            
            for _ in range(horizon):
                # Get predictions
                outputs = self.forward(current_tokens)
                logits = outputs['logits']
                
                # Get probabilities for last position
                last_logits = logits[:, -1, :]
                probs = F.softmax(last_logits / self.temperature, dim=-1)
                
                # Sample multiple times for robustness
                step_samples = []
                for _ in range(max(1, num_samples // horizon)):
                    sampled_token = torch.multinomial(probs, 1)
                    step_samples.append(sampled_token.item())
                
                # Use mean of samples
                next_token = int(np.mean(step_samples))
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                current_tokens = torch.cat([current_tokens, next_token_tensor], dim=1)
                
                # Detokenize
                bins = np.linspace(-15, 15, self.num_bins)
                predicted_value = bins[next_token] * scale
                predictions.append(predicted_value)
        
        return np.array(predictions)


# ============================================================================
# PART 4: TRAINING FUNCTION
# ============================================================================

def train_improved_model(
    model,
    train_data: List[np.ndarray],
    val_data: List[np.ndarray],
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    checkpoint_dir: str = "./checkpoints_v2"
):
    """Train the improved distance-aware model"""
    import os
    from pathlib import Path
    from tqdm import tqdm
    
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup optimizer with warmup
    optimizer = torch.optim.AdamW(
        model.distance_output.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # Cosine annealing with warmup
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=epochs//2, T_mult=2)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        
        # Shuffle training data
        np.random.shuffle(train_data)
        
        pbar = tqdm(range(0, len(train_data), batch_size), desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx in pbar:
            batch_series = train_data[batch_idx:batch_idx + batch_size]
            
            # Prepare batch
            batch_inputs = []
            batch_targets = []
            
            for series in batch_series:
                if len(series) < 50:
                    continue
                
                # Random context length
                context_len = np.random.randint(32, min(512, len(series) - 10))
                start_idx = np.random.randint(0, len(series) - context_len - 1)
                
                context = series[start_idx:start_idx + context_len]
                target = series[start_idx + context_len]
                
                # Tokenize
                tokens = model.tokenize_time_series(context)
                target_token = model.tokenize_time_series(np.array([target]))[0]
                
                batch_inputs.append(tokens)
                batch_targets.append(target_token)
            
            if len(batch_inputs) == 0:
                continue
            
            # Pad sequences
            max_len = max(len(x) for x in batch_inputs)
            padded_inputs = torch.zeros(len(batch_inputs), max_len, dtype=torch.long)
            for i, inp in enumerate(batch_inputs):
                padded_inputs[i, :len(inp)] = inp
            
            padded_inputs = padded_inputs.to(model.device)
            targets = torch.tensor(batch_targets, dtype=torch.long).to(model.device)
            
            # Forward pass
            outputs = model.forward(padded_inputs)
            logits = outputs['logits'][:, -1, :]  # Last position
            
            # Compute combined loss
            loss = model.loss_fn.combined_loss(logits, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.distance_output.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        avg_train_loss = epoch_loss / max(len(pbar), 1)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for series in val_data[:100]:  # Validate on subset
                if len(series) < 50:
                    continue
                
                context_len = min(256, len(series) - 10)
                context = series[:context_len]
                target = series[context_len]
                
                tokens = model.tokenize_time_series(context).unsqueeze(0).to(model.device)
                target_token = model.tokenize_time_series(np.array([target]))[0].unsqueeze(0).to(model.device)
                
                outputs = model.forward(tokens)
                logits = outputs['logits'][:, -1, :]
                loss = model.loss_fn.combined_loss(logits, target_token)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / min(len(val_data), 100)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save checkpoint
        os.makedirs(checkpoint_dir, exist_ok=True)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.distance_output.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"  ✓ Saved best model (val_loss: {avg_val_loss:.4f})")
    
    return train_losses, val_losses
