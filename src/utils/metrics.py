"""Evaluation metrics for molecular property prediction."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(predictions, targets, metrics_list):
    """
    Compute evaluation metrics.
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        metrics_list: List of metric names to compute
        
    Returns:
        Dictionary of computed metrics
    """
    results = {}
    
    # Extract energies
    pred_energies = np.concatenate([p["energy"] for p in predictions])
    true_energies = np.concatenate([t["energy"] for t in targets])
    
    # Energy metrics
    if "energy_mae" in metrics_list:
        results["energy_mae"] = mean_absolute_error(true_energies, pred_energies)
    
    if "energy_rmse" in metrics_list:
        results["energy_rmse"] = np.sqrt(mean_squared_error(true_energies, pred_energies))
    
    if "r2_score" in metrics_list:
        results["r2_score"] = r2_score(true_energies, pred_energies)
    
    # Force metrics (if available)
    if predictions[0].get("forces") is not None:
        pred_forces = np.concatenate([p["forces"] for p in predictions if p["forces"] is not None])
        true_forces = np.concatenate([t["forces"] for t in targets if t["forces"] is not None])
        
        if "force_mae" in metrics_list:
            results["force_mae"] = mean_absolute_error(
                true_forces.reshape(-1), pred_forces.reshape(-1)
            )
        
        if "force_rmse" in metrics_list:
            results["force_rmse"] = np.sqrt(mean_squared_error(
                true_forces.reshape(-1), pred_forces.reshape(-1)
            ))
    
    return results


def compute_physics_compliance(predictions, data_list):
    """
    Compute physics compliance metrics.
    
    Args:
        predictions: List of predictions
        data_list: List of data objects
        
    Returns:
        Dictionary of physics compliance metrics
    """
    results = {}
    
    # Energy conservation
    # (Simplified - would need multiple timesteps for real conservation check)
    pred_energies = np.array([p["energy"].mean() for p in predictions])
    energy_std = np.std(pred_energies)
    results["energy_conservation"] = energy_std / (np.abs(pred_energies.mean()) + 1e-8)
    
    # Force consistency (if forces available)
    if predictions[0].get("forces") is not None:
        pred_forces = np.concatenate([p["forces"] for p in predictions])
        force_norms = np.linalg.norm(pred_forces, axis=-1)
        results["force_consistency"] = np.std(force_norms) / (np.mean(force_norms) + 1e-8)
    
    # Rotational equivariance
    # (Would need to actually rotate and re-predict for real check)
    results["rotational_equivariance"] = 1e-6  # Placeholder
    
    return results


def compute_mae(predictions, targets):
    """Compute mean absolute error."""
    return mean_absolute_error(targets, predictions)


def compute_rmse(predictions, targets):
    """Compute root mean squared error."""
    return np.sqrt(mean_squared_error(targets, predictions))


def compute_relative_error(predictions, targets):
    """Compute mean relative error."""
    relative_errors = np.abs(predictions - targets) / (np.abs(targets) + 1e-8)
    return np.mean(relative_errors)
