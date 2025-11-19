# distance_aware_chronos.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from typing import Optional, Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# For loading pretrained Chronos
from transformers import T5ForConditionalGeneration, T5Config
from chronos import ChronosPipeline

# ============================================================================
# PART 1: DISTANCE-AWARE LOSS FUNCTIONS
# ============================================================================

class OrdinalLoss(nn.Module):
    """Collection of distance-aware loss functions"""
    
    def __init__(self, num_bins: int = 4096, alpha: float = 1.0, beta: float = 0.1):
        super().__init__()
        self.num_bins = num_bins
        self.alpha = alpha  # Weight for ordinal penalty
        self.beta = beta    # Temperature for soft labels
        
        # Precompute distance matrix
        self.register_buffer('distance_matrix', self._create_distance_matrix())
        
    def _create_distance_matrix(self):
        """Create matrix of distances between all bin pairs"""
        matrix = torch.zeros(self.num_bins, self.num_bins)
        for i in range(self.num_bins):
            for j in range(self.num_bins):
                matrix[i, j] = abs(i - j) / self.num_bins  # Normalized distance
        return matrix
    
    def ordinal_cross_entropy(self, logits, targets):
        """Cross-entropy with distance-based penalty"""
        batch_size = logits.size(0)
        
        # Standard cross-entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Get predicted bins
        predictions = torch.argmax(logits, dim=-1)
        
        # Calculate distance penalty
        distances = torch.gather(
            self.distance_matrix[predictions], 
            1, 
            targets.unsqueeze(1)
        ).squeeze(1)
        
        # Distance-weighted loss
        distance_penalty = torch.log(1 + distances) * self.alpha
        
        total_loss = ce_loss + distance_penalty
        return total_loss.mean()
    
    def smooth_label_loss(self, logits, targets):
        """Label smoothing based on bin proximity"""
        batch_size = targets.size(0)
        device = logits.device
        
        # Create smooth labels with Gaussian-like distribution
        smooth_labels = torch.zeros(batch_size, self.num_bins).to(device)
        
        # Vectorized smooth label creation
        bin_indices = torch.arange(self.num_bins).to(device)
        for i in range(batch_size):
            distances = torch.abs(bin_indices - targets[i]).float()
            smooth_labels[i] = torch.exp(-distances * self.beta)
            smooth_labels[i] = smooth_labels[i] / smooth_labels[i].sum()
        
        # KL divergence with smooth labels
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(smooth_labels * log_probs, dim=-1)
        
        return loss.mean()
    
    def earth_movers_distance_loss(self, logits, targets):
        """EMD/Wasserstein distance for ordinal regression"""
        probs = F.softmax(logits, dim=-1)
        
        # Create one-hot targets
        targets_one_hot = F.one_hot(targets, self.num_bins).float()
        
        # Compute CDFs
        cdf_pred = torch.cumsum(probs, dim=-1)
        cdf_target = torch.cumsum(targets_one_hot, dim=-1)
        
        # EMD is L1 distance between CDFs
        emd = torch.sum(torch.abs(cdf_pred - cdf_target), dim=-1)
        
        return emd.mean()

# ============================================================================
# PART 2: DISTANCE-AWARE OUTPUT LAYER
# ============================================================================

class DistanceAwareOutputLayer(nn.Module):
    """Output layer that considers bin relationships"""
    
    def __init__(self, hidden_size: int, num_bins: int):
        super().__init__()
        self.num_bins = num_bins
        self.hidden_size = hidden_size
        
        # Gaussian kernel parameters for soft binning
        self.gaussian_centers = nn.Parameter(
            torch.linspace(-15, 15, num_bins).unsqueeze(0)
        )
        self.gaussian_widths = nn.Parameter(torch.ones(num_bins) * 0.5)
        
        # Ordinal embedding with sinusoidal initialization
        self.ordinal_embed = nn.Embedding(num_bins, 64)
        self._init_ordinal_embeddings()
        
        # Projection layers
        self.value_projection = nn.Linear(hidden_size, 1)
        self.confidence_projection = nn.Linear(hidden_size, 1)
        
        # Final mixing layer
        self.mix_layer = nn.Sequential(
            nn.Linear(hidden_size + 65, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_bins)
        )
        
    def _init_ordinal_embeddings(self):
        """Initialize embeddings with ordinal structure"""
        with torch.no_grad():
            for i in range(self.num_bins):
                # Sinusoidal position encoding
                position = i / self.num_bins
                frequencies = torch.tensor([
                    np.sin(2 * np.pi * position * (j+1)) 
                    for j in range(32)
                ] + [
                    np.cos(2 * np.pi * position * (j+1)) 
                    for j in range(32)
                ])
                self.ordinal_embed.weight[i] = frequencies
    
    def forward(self, hidden_states, temperature=1.0):
        batch_size, seq_len, _ = hidden_states.size()
        
        # Predict continuous value
        predicted_value = self.value_projection(hidden_states)
        confidence = torch.sigmoid(self.confidence_projection(hidden_states))
        
        # Gaussian soft binning
        distances = (predicted_value - self.gaussian_centers) ** 2
        gaussian_logits = -distances / (2 * self.gaussian_widths ** 2)
        
        # Get ordinal features (average embedding)
        ordinal_features = self.ordinal_embed.weight.mean(dim=0)
        ordinal_features = ordinal_features.unsqueeze(0).unsqueeze(0)
        ordinal_features = ordinal_features.expand(batch_size, seq_len, -1)
        
        # Combine all features
        combined = torch.cat([
            hidden_states,
            predicted_value,
            ordinal_features
        ], dim=-1)
        
        # Final logits
        logits = self.mix_layer(combined)
        
        # Confidence-based mixing
        final_logits = (1 - confidence) * gaussian_logits + confidence * logits
        
        return final_logits / temperature

# ============================================================================
# PART 3: COMPLETE DISTANCE-AWARE CHRONOS MODEL
# ============================================================================

class DistanceAwareChronos(nn.Module):
    """Complete distance-aware Chronos implementation"""
    
    def __init__(
        self, 
        model_name: str = "amazon/chronos-t5-small",
        num_bins: int = 4096,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.device = device
        self.num_bins = num_bins
        
        # Load pretrained Chronos T5
        print(f"Loading pretrained model: {model_name}")
        self.chronos = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # FREEZE all T5 parameters
        for param in self.chronos.parameters():
            param.requires_grad = False
        
        # Get model configuration
        self.config = self.chronos.config
        self.hidden_size = self.config.d_model
        
        # Distance-aware components (trainable)
        self.distance_output = DistanceAwareOutputLayer(self.hidden_size, num_bins)
        self.loss_fn = OrdinalLoss(num_bins)
        
        # Learnable temperature
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Move to device
        self.to(device)
    
    @classmethod
    def from_pretrained(cls, repo_id: str, device: str = None):
        """Load model from HuggingFace Hub"""
        from huggingface_hub import hf_hub_download
        import tempfile
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Loading Distance-Aware Chronos from {repo_id}...")
        
        # Download config
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Initialize model
        model = cls(
            model_name=config['base_model'],
            num_bins=config['num_bins'],
            device=device
        )
        
        # Load trained distance output weights
        distance_output_path = hf_hub_download(repo_id=repo_id, filename="distance_output.pt")
        state_dict = torch.load(distance_output_path, map_location=device)
        model.distance_output.load_state_dict(state_dict)
        
        print(f"✓ Loaded model (epoch {config['training_epoch']}, val_loss: {config['val_loss']:.4f})")
        
        return model
        
    def tokenize_time_series(self, time_series: np.ndarray) -> torch.Tensor:
        """Convert time series to tokens"""
        # Mean scaling
        scale = np.abs(time_series).mean() + 1e-10
        scaled = time_series / scale
        
        # Clip to range
        scaled = np.clip(scaled, -15, 15)
        
        # Quantize to bins
        bins = np.linspace(-15, 15, self.num_bins)
        tokens = np.digitize(scaled, bins) - 1  # -1 to make 0-indexed
        tokens = np.clip(tokens, 0, self.num_bins - 1)
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def detokenize(self, tokens: torch.Tensor, scale: float) -> np.ndarray:
        """Convert tokens back to values"""
        bins = np.linspace(-15, 15, self.num_bins)
        values = bins[tokens.cpu().numpy()]
        return values * scale
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> Dict:
        """Forward pass"""
        # Get frozen representations
        with torch.no_grad():
            outputs = self.chronos(
                input_ids=input_ids,
                decoder_input_ids=input_ids,
                output_hidden_states=True
            )
            # Use last decoder hidden state
            hidden_states = outputs.decoder_hidden_states[-1]
        
        # Apply distance-aware output layer
        logits = self.distance_output(hidden_states, self.temperature)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Reshape for loss calculation
            logits_flat = logits.view(-1, self.num_bins)
            labels_flat = labels.view(-1)
            
            # Filter out padding tokens (marked as -100)
            mask = labels_flat != -100
            if mask.sum() == 0:
                return {'loss': None, 'logits': logits, 'hidden_states': hidden_states}
            
            logits_masked = logits_flat[mask]
            labels_masked = labels_flat[mask]
            
            # Combine multiple losses
            loss_ordinal = self.loss_fn.ordinal_cross_entropy(logits_masked, labels_masked)
            loss_smooth = self.loss_fn.smooth_label_loss(logits_masked, labels_masked)
            loss_emd = self.loss_fn.earth_movers_distance_loss(logits_masked, labels_masked)
            
            # Weighted combination
            loss = 0.5 * loss_ordinal + 0.3 * loss_smooth + 0.2 * loss_emd
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states
        }
    
    def predict(
        self, 
        context: np.ndarray, 
        horizon: int = 24,
        num_samples: int = 100
    ) -> np.ndarray:
        """Generate predictions using the trained model"""
        self.eval()
        
        # Tokenize context
        scale = np.abs(context).mean() + 1e-10
        context_tokens = self.tokenize_time_series(context)
        context_tokens = context_tokens.to(self.device)
        
        # Generate predictions autoregressively
        predictions = []
        
        with torch.no_grad():
            # Start with context
            input_tokens = context_tokens.unsqueeze(0)  # [1, seq_len]
            
            for step in range(horizon):
                # Forward pass
                outputs = self.forward(
                    input_ids=input_tokens,
                    labels=None
                )
                
                # Get logits for the last position
                logits = outputs['logits'][:, -1, :]  # [1, num_bins]
                probs = F.softmax(logits / self.temperature, dim=-1)
                
                # Sample from the distribution
                samples = []
                for _ in range(min(num_samples, 20)):  # Limit samples per step
                    sampled_bin = torch.multinomial(probs, 1).item()
                    samples.append(sampled_bin)
                
                # Use median of samples
                next_bin = int(np.median(samples))
                
                # Detokenize to get value
                bins = np.linspace(-15, 15, self.num_bins)
                value = bins[next_bin] * scale
                predictions.append(value)
                
                # Append to input for next step
                next_token = torch.tensor([[next_bin]], dtype=torch.long, device=self.device)
                input_tokens = torch.cat([input_tokens, next_token], dim=1)
                
                # Truncate if too long (keep last 512 tokens)
                if input_tokens.shape[1] > 512:
                    input_tokens = input_tokens[:, -512:]
        
        return np.array(predictions)

# ============================================================================
# PART 4: TRAINING AND EVALUATION
# ============================================================================

def train_distance_aware_model(
    model: DistanceAwareChronos,
    train_data: List[np.ndarray],
    val_data: List[np.ndarray],
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4
):
    """Train the distance-aware model"""
    
    # Only optimize new parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    print(f"Training {sum(p.numel() for p in trainable_params):,} parameters")
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            
            # Prepare batch
            batch_tokens = []
            batch_labels = []
            
            for series in batch:
                tokens = model.tokenize_time_series(series)
                # Use last 80% for context, 20% for labels
                split_idx = int(len(tokens) * 0.8)
                batch_tokens.append(tokens[:split_idx])
                batch_labels.append(tokens[split_idx:])
            
            # Pad sequences
            max_len_input = max(len(t) for t in batch_tokens)
            max_len_label = max(len(l) for l in batch_labels)
            
            input_ids = torch.zeros(len(batch), max_len_input, dtype=torch.long)
            labels = torch.zeros(len(batch), max_len_label, dtype=torch.long)
            
            for j, (tokens, label) in enumerate(zip(batch_tokens, batch_labels)):
                input_ids[j, :len(tokens)] = tokens
                labels[j, :len(label)] = label
            
            input_ids = input_ids.to(model.device)
            labels = labels.to(model.device)
            
            # Forward pass
            outputs = model(input_ids, labels)
            loss = outputs['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i+batch_size]
                
                # Prepare batch (same as training)
                batch_tokens = []
                batch_labels = []
                
                for series in batch:
                    tokens = model.tokenize_time_series(series)
                    split_idx = int(len(tokens) * 0.8)
                    batch_tokens.append(tokens[:split_idx])
                    batch_labels.append(tokens[split_idx:])
                
                max_len_input = max(len(t) for t in batch_tokens)
                max_len_label = max(len(l) for l in batch_labels)
                
                input_ids = torch.zeros(len(batch), max_len_input, dtype=torch.long)
                labels = torch.zeros(len(batch), max_len_label, dtype=torch.long)
                
                for j, (tokens, label) in enumerate(zip(batch_tokens, batch_labels)):
                    input_ids[j, :len(tokens)] = tokens
                    labels[j, :len(label)] = label
                
                input_ids = input_ids.to(model.device)
                labels = labels.to(model.device)
                
                outputs = model(input_ids, labels)
                epoch_val_loss += outputs['loss'].item()
        
        scheduler.step()
        
        avg_train_loss = epoch_train_loss / (len(train_data) // batch_size)
        avg_val_loss = epoch_val_loss / (len(val_data) // batch_size)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return train_losses, val_losses

# ============================================================================
# PART 5: COMPARISON METRICS
# ============================================================================

def calculate_metrics(predictions: np.ndarray, actuals: np.ndarray) -> Dict:
    """Calculate evaluation metrics"""
    
    mae = np.mean(np.abs(predictions - actuals))
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((predictions - actuals) / (actuals + 1e-10))) * 100
    
    # Directional accuracy
    if len(predictions) > 1:
        pred_direction = np.sign(np.diff(predictions))
        actual_direction = np.sign(np.diff(actuals))
        directional_accuracy = np.mean(pred_direction == actual_direction) * 100
    else:
        directional_accuracy = 0
    
    return {
        'MAE': mae,
        'MSE': mse, 
        'RMSE': rmse,
        'MAPE': mape,
        'Directional_Accuracy': directional_accuracy
    }

# Save to file
if __name__ == "__main__":
    print("Distance-Aware Chronos implementation loaded successfully!")