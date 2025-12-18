"""Visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_training_curves(train_losses, val_losses, save_path=None):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_predictions(predictions, targets, property_name="Energy", save_path=None):
    """
    Plot predicted vs true values.
    
    Args:
        predictions: Predicted values
        targets: True values
        property_name: Name of property being predicted
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 8))
    
    # Scatter plot
    plt.scatter(targets, predictions, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # MAE and R²
    mae = np.mean(np.abs(predictions - targets))
    r2 = 1 - np.sum((predictions - targets)**2) / np.sum((targets - np.mean(targets))**2)
    
    plt.xlabel(f'True {property_name}', fontsize=12)
    plt.ylabel(f'Predicted {property_name}', fontsize=12)
    plt.title(f'{property_name} Predictions\nMAE: {mae:.4f}, R²: {r2:.4f}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_error_distribution(predictions, targets, save_path=None):
    """
    Plot error distribution.
    
    Args:
        predictions: Predicted values
        targets: True values
        save_path: Path to save figure
    """
    errors = predictions - targets
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    plt.axvline(x=np.mean(errors), color='g', linestyle='--', linewidth=2, 
                label=f'Mean Error: {np.mean(errors):.4f}')
    
    plt.xlabel('Prediction Error', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Error Distribution', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_learning_curve(train_sizes, train_scores, val_scores, save_path=None):
    """
    Plot learning curve (performance vs training set size).
    
    Args:
        train_sizes: Training set sizes
        train_scores: Training scores
        val_scores: Validation scores
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_sizes, train_scores, 'b-o', label='Training Score', linewidth=2)
    plt.plot(train_sizes, val_scores, 'r-o', label='Validation Score', linewidth=2)
    
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Learning Curve', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
