"""
Training Script for Cross-Domain Equivariant Neural Networks
============================================================
Implements cross-geometry and cross-physics pretraining strategies
for learning shared representations across molecular domains.

Features:
- Cross-domain pretraining
- Multitask learning
- Mixed precision training
- Gradient accumulation
- Early stopping
- Comprehensive evaluation metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
from pathlib import Path
import pickle
import json
from tqdm import tqdm
import time
from collections import defaultdict

from model import build_model
from download_datasets import DatasetDownloader


class Trainer:
    """Trainer for cross-domain equivariant neural networks."""
    
    def __init__(self, config):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup paths
        self.exp_dir = Path(config['exp_dir'])
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = build_model(config['model']).to(self.device)
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Metrics storage
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        
    def load_data(self, data_config):
        """
        Load and prepare datasets.
        
        Args:
            data_config: Data configuration dictionary
            
        Returns:
            Train and validation dataloaders
        """
        downloader = DatasetDownloader(root_dir=data_config['data_dir'])
        
        if data_config['mode'] == 'cross_domain':
            # Load cross-domain pretraining data
            train_data = downloader.load_dataset('cross_domain_train')
            val_data = downloader.load_dataset('qm9')['val']
        
        elif data_config['mode'] == 'single_dataset':
            # Load single dataset
            splits = downloader.load_dataset(data_config['dataset'])
            train_data = splits['train']
            val_data = splits['val']
        
        else:
            raise ValueError(f"Unknown data mode: {data_config['mode']}")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_data,
            batch_size=data_config['batch_size'],
            shuffle=True,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_data,
            batch_size=data_config['batch_size'] * 2,
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True
        )
        
        print(f"Loaded {len(train_data)} training samples")
        print(f"Loaded {len(val_data)} validation samples")
        
        return train_loader, val_loader
    
    def compute_loss(self, predictions, targets, task='energy'):
        """
        Compute task-specific loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            task: Task name
            
        Returns:
            Loss value
        """
        if task == 'energy':
            loss = F.mse_loss(predictions, targets)
        
        elif task == 'forces':
            loss = F.l1_loss(predictions, targets)
        
        elif task in ['homo_lumo_gap', 'dipole_moment']:
            loss = F.mse_loss(predictions, targets)
        
        else:
            loss = F.mse_loss(predictions, targets)
        
        return loss
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: Training dataloader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0
        task_losses = defaultdict(float)
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, data in enumerate(pbar):
            data = data.to(self.device)
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                if isinstance(self.model, MultitaskCrossDomainModel):
                    # Multitask learning
                    predictions = self.model(data)
                    
                    # Compute weighted loss across tasks
                    losses = {}
                    for task, pred in predictions.items():
                        if hasattr(data, task):
                            target = getattr(data, task)
                            losses[task] = self.compute_loss(pred, target, task)
                    
                    # Task weighting (gradient-based balancing)
                    weights = self.compute_task_weights(losses)
                    loss = sum(w * l for w, l in zip(weights.values(), losses.values()))
                    
                    # Track individual task losses
                    for task, task_loss in losses.items():
                        task_losses[task] += task_loss.item()
                
                else:
                    # Single task
                    output, _ = self.model(data)
                    loss = self.compute_loss(output, data.y[:, 0:1], 'energy')
            
            # Backward pass with gradient scaling
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Compute average metrics
        metrics = {
            'total_loss': total_loss / n_batches
        }
        
        if task_losses:
            for task, task_loss in task_losses.items():
                metrics[f'{task}_loss'] = task_loss / n_batches
        
        return metrics
    
    def validate(self, val_loader):
        """
        Validate model.
        
        Args:
            val_loader: Validation dataloader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        n_batches = 0
        
        with torch.no_grad():
            for data in tqdm(val_loader, desc="Validation"):
                data = data.to(self.device)
                
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    output, _ = self.model(data)
                    loss = self.compute_loss(output, data.y[:, 0:1], 'energy')
                
                total_loss += loss.item()
                all_predictions.append(output.cpu())
                all_targets.append(data.y[:, 0:1].cpu())
                n_batches += 1
        
        # Concatenate all predictions and targets
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        mae = torch.abs(predictions - targets).mean().item()
        rmse = torch.sqrt(torch.mean((predictions - targets) ** 2)).item()
        relative_l2 = (torch.norm(predictions - targets) / torch.norm(targets)).item()
        
        metrics = {
            'loss': total_loss / n_batches,
            'mae': mae,
            'rmse': rmse,
            'relative_l2': relative_l2
        }
        
        return metrics
    
    def compute_task_weights(self, losses):
        """
        Compute gradient-based task weights.
        
        Args:
            losses: Dictionary of task losses
            
        Returns:
            Dictionary of task weights
        """
        weights = {}
        total_grad_norm = 0
        
        for task, loss in losses.items():
            # Compute gradient norm for this task
            grad = torch.autograd.grad(
                loss, 
                self.model.parameters(), 
                retain_graph=True,
                create_graph=False
            )
            grad_norm = sum(g.norm() for g in grad)
            total_grad_norm += 1.0 / (grad_norm + 1e-8)
        
        # Normalize weights
        for task in losses.keys():
            weights[task] = 1.0 / (total_grad_norm + 1e-8)
        
        return weights
    
    def save_checkpoint(self, is_best=False):
        """
        Save training checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_metrics': dict(self.train_metrics),
            'val_metrics': dict(self.val_metrics),
            'config': self.config
        }
        
        # Save current checkpoint
        checkpoint_path = self.exp_dir / 'checkpoint.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.exp_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (val_loss: {self.best_val_loss:.4f})")
    
    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_metrics = defaultdict(list, checkpoint['train_metrics'])
        self.val_metrics = defaultdict(list, checkpoint['val_metrics'])
        
        print(f"✓ Loaded checkpoint from epoch {self.epoch}")
    
    def train(self, train_loader, val_loader):
        """
        Main training loop.
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
        """
        print("\n" + "="*50)
        print(f"Starting training on {self.device}")
        print("="*50)
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            self.epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Store metrics
            for key, value in train_metrics.items():
                self.train_metrics[key].append(value)
            for key, value in val_metrics.items():
                self.val_metrics[key].append(value)
            
            # Print metrics
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val MAE: {val_metrics['mae']:.4f}")
            print(f"  Val RMSE: {val_metrics['rmse']:.4f}")
            print(f"  LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Check for improvement
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1
                self.save_checkpoint(is_best=False)
            
            # Early stopping
            if self.patience_counter >= self.config['patience']:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final metrics
        metrics_path = self.exp_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump({
                'train': dict(self.train_metrics),
                'val': dict(self.val_metrics),
                'best_val_loss': self.best_val_loss,
                'total_time_hours': total_time / 3600
            }, f, indent=2)


def main():
    """Main training function."""
    
    # Configuration
    config = {
        'exp_dir': './experiments/cross_domain_pretraining',
        
        # Model configuration
        'model': {
            'multitask': True,
            'hidden_dim': 128,
            'vector_dim': 64,
            'n_layers': 5,
            'cutoff': 5.0,
            'task_dims': {
                'energy': 1,
                'homo_lumo_gap': 1
            }
        },
        
        # Training configuration
        'epochs': 100,
        'lr': 1e-3,
        'min_lr': 1e-6,
        'weight_decay': 1e-5,
        'patience': 10,
        'use_amp': True,
        
        # Data configuration
        'data': {
            'mode': 'cross_domain',
            'data_dir': './data',
            'batch_size': 16,
            'num_workers': 4
        }
    }
    
    # Save configuration
    exp_dir = Path(config['exp_dir'])
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Load data
    train_loader, val_loader = trainer.load_data(config['data'])
    
    # Train model
    trainer.train(train_loader, val_loader)
    
    print("\n✓ Training complete!")
    print(f"Results saved to {config['exp_dir']}")


if __name__ == '__main__':
    main()
