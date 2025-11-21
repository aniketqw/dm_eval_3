# train_distance_aware_chronos.py

import torch
import torch.nn as nn
import numpy as np
import pickle
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Set CUDA memory allocation config to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Import from the distance_aware_chronos module
from distance_aware_chronos import (
    DistanceAwareChronos,
    OrdinalLoss,
    DistanceAwareOutputLayer,
    calculate_metrics
)


class ChronosTrainer:
    """Trainer for Distance-Aware Chronos Model"""
    
    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-small",
        data_dir: str = "./chronos_data",
        output_dir: str = "./checkpoints",
        num_bins: int = 4096,
        device: str = None
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Initialize model
        print(f"Initializing Distance-Aware Chronos model...")
        self.model = DistanceAwareChronos(
            model_name=model_name,
            num_bins=num_bins,
            device=self.device
        )
        
        self.num_bins = num_bins
        
    def load_data(self):
        """Load processed training data"""
        print("\nLoading training data...")
        
        data_path = self.data_dir / "train" / "timeseries.pkl"
        meta_path = self.data_dir / "metadata" / "info.json"
        
        if not data_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {data_path}. "
                "Please run download_dataset.py first!"
            )
        
        with open(data_path, 'rb') as f:
            timeseries = pickle.load(f)
        
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"✓ Loaded {len(timeseries)} time series")
        print(f"✓ Total observations: {metadata['total_observations']:,}")
        
        return timeseries, metadata
    
    def prepare_training_data(self, timeseries, train_split=0.9):
        """Split data into train and validation"""
        print("\nPreparing train/validation split...")
        
        # Shuffle
        np.random.seed(42)
        indices = np.random.permutation(len(timeseries))
        
        split_idx = int(len(indices) * train_split)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_data = [timeseries[i] for i in train_indices]
        val_data = [timeseries[i] for i in val_indices]
        
        print(f"✓ Train: {len(train_data)} series")
        print(f"✓ Validation: {len(val_data)} series")
        
        return train_data, val_data
    
    def train(
        self,
        train_data,
        val_data,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        save_every: int = 2
    ):
        """Train the model"""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        print(f"\nTraining parameters:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'epoch': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*60}")
            
            # Training phase
            train_loss = self._train_epoch(
                train_data, optimizer, batch_size, epoch
            )
            
            # Validation phase
            val_loss = self._validate_epoch(val_data, batch_size)
            
            # Learning rate step
            scheduler.step()
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['epoch'].append(epoch + 1)
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0 or val_loss < best_val_loss:
                self._save_checkpoint(epoch + 1, val_loss, optimizer, history)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"  ✓ New best validation loss!")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        # Plot training curves
        self._plot_training_curves(history)
        
        return history
    
    def _train_epoch(self, train_data, optimizer, batch_size, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Shuffle data
        indices = np.random.permutation(len(train_data))
        
        pbar = tqdm(range(0, len(train_data), batch_size), desc="Training")
        
        for i in pbar:
            batch_indices = indices[i:i+batch_size]
            batch = [train_data[idx] for idx in batch_indices]
            
            # Prepare batch
            try:
                input_ids, labels = self._prepare_batch(batch)
                
                if input_ids is None:
                    continue
                
                # Forward pass
                outputs = self.model(input_ids, labels)
                loss = outputs['loss']
                
                if loss is None or torch.isnan(loss):
                    continue
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    1.0
                )
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Clear CUDA cache periodically to prevent fragmentation
                if i % 100 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"\nError in batch: {e}")
                # Clear cache on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def _validate_epoch(self, val_data, batch_size):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(range(0, len(val_data), batch_size), desc="Validation")
            
            for i in pbar:
                batch = val_data[i:i+batch_size]
                
                try:
                    input_ids, labels = self._prepare_batch(batch)
                    
                    if input_ids is None:
                        continue
                    
                    outputs = self.model(input_ids, labels)
                    loss = outputs['loss']
                    
                    if loss is None or torch.isnan(loss):
                        continue
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
                    # Clear CUDA cache periodically during validation
                    if i % 50 == 0:
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def _prepare_batch(self, batch):
        """Prepare a batch for training"""
        batch_tokens = []
        batch_labels = []
        
        for series in batch:
            # Limit series length to prevent memory issues
            if len(series) > 10000:
                series = series[:10000]
            
            if len(series) < 100:  # Skip very short series
                continue
            
            # Tokenize
            tokens = self.model.tokenize_time_series(series)
            
            if len(tokens) < 20:
                continue
            
            # Use last 80% for context, 20% for labels
            split_idx = int(len(tokens) * 0.8)
            
            if split_idx < 10 or (len(tokens) - split_idx) < 5:
                continue
            
            batch_tokens.append(tokens[:split_idx])
            batch_labels.append(tokens[split_idx:])
        
        if len(batch_tokens) == 0:
            return None, None
        
        # Combine input and labels into full sequences
        full_sequences = []
        context_lengths = []
        
        for tokens, label in zip(batch_tokens, batch_labels):
            # Limit context to reasonable size
            if len(tokens) > 400:
                tokens = tokens[-400:]  # Use last 400 tokens
            
            context_lengths.append(len(tokens))
            # Concatenate context + future
            full_seq = torch.cat([tokens, label])
            full_sequences.append(full_seq)
        
        # Pad to max length
        max_len = min(max(len(s) for s in full_sequences), 512)
        
        input_ids = torch.zeros(len(full_sequences), max_len, dtype=torch.long)
        labels = torch.full((len(full_sequences), max_len), -100, dtype=torch.long)
        
        for j, (full_seq, ctx_len) in enumerate(zip(full_sequences, context_lengths)):
            seq_len = min(len(full_seq), max_len)
            input_ids[j, :seq_len] = full_seq[:seq_len]
            
            # Labels: only compute loss on prediction part (after context)
            # Ensure context doesn't exceed max_len
            actual_ctx_len = min(ctx_len, max_len)
            pred_start = actual_ctx_len
            
            if pred_start < seq_len:
                labels[j, pred_start:seq_len] = full_seq[pred_start:seq_len]
        
        return input_ids.to(self.device), labels.to(self.device)
    
    def _save_checkpoint(self, epoch, val_loss, optimizer, history):
        """Save model checkpoint"""
        checkpoint_dir = self.output_dir / f"epoch_{epoch}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model state
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'history': history
        }, checkpoint_dir / "checkpoint.pt")
        
        # Save model components separately for easier loading
        torch.save(
            self.model.distance_output.state_dict(),
            checkpoint_dir / "distance_output.pt"
        )
        
        # Save config
        config = {
            'epoch': epoch,
            'val_loss': float(val_loss),
            'num_bins': self.num_bins,
            'device': self.device
        }
        
        with open(checkpoint_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"  ✓ Checkpoint saved to {checkpoint_dir}")
    
    def _plot_training_curves(self, history):
        """Plot training and validation curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(history['epoch'], history['train_loss'], label='Train Loss', marker='o')
        plt.plot(history['epoch'], history['val_loss'], label='Val Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = self.output_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Training curves saved to {plot_path}")


def main():
    """Main training execution"""
    print("="*60)
    print("DISTANCE-AWARE CHRONOS TRAINING")
    print("="*60)
    
    # Initialize trainer
    trainer = ChronosTrainer(
        model_name="amazon/chronos-t5-small",
        data_dir="./chronos_data",
        output_dir="./checkpoints",
        num_bins=4096
    )
    
    # Load data
    timeseries, metadata = trainer.load_data()
    
    # Prepare splits
    train_data, val_data = trainer.prepare_training_data(timeseries)
    
    # Train
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=10,
        batch_size=8,  # Reduced to prevent OOM errors
        learning_rate=1e-4,
        save_every=2
    )
    
    print("\n✓ Training completed successfully!")
    print(f"✓ Checkpoints saved to: ./checkpoints")


if __name__ == "__main__":
    main()