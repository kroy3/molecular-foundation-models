"""
Evaluation script for cross-domain molecular models.

Evaluates trained models on various benchmarks including GeoShift transfer tasks.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.model import build_model
from src.data.datasets import get_dataset
from src.utils.metrics import compute_metrics, compute_physics_compliance


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate molecular property prediction model")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--benchmark", type=str, default="geoshift",
                        choices=["geoshift", "qm9", "md17", "custom"],
                        help="Benchmark to evaluate on")
    parser.add_argument("--task", type=str, default=None,
                        help="Specific task (for geoshift: small_to_large, organic_to_inorganic, etc.)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to custom dataset")
    parser.add_argument("--metrics", type=str, nargs="+",
                        default=["energy_mae", "energy_rmse", "force_mae"],
                        help="Metrics to compute")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--output-dir", type=str, default="experiments/evaluation",
                        help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    
    return parser.parse_args()


def load_checkpoint(checkpoint_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load config
    config = checkpoint.get("config", None)
    if config is None:
        raise ValueError("Checkpoint does not contain config")
    
    # Build model
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'unknown'):.4f}")
    
    return model, config


def evaluate_model(model, dataloader, metrics, device):
    """Evaluate model on a dataset."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_data = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(device)
            
            # Forward pass
            output = model(batch)
            
            # Store predictions and targets
            all_predictions.append({
                "energy": output["energy"].cpu().numpy(),
                "forces": output.get("forces", None)
            })
            all_targets.append({
                "energy": batch.energy.cpu().numpy(),
                "forces": batch.get("forces", None)
            })
            all_data.append(batch)
    
    # Compute metrics
    results = compute_metrics(all_predictions, all_targets, metrics)
    
    # Compute physics compliance
    physics_compliance = compute_physics_compliance(all_predictions, all_data)
    results["physics_compliance"] = physics_compliance
    
    return results


def evaluate_geoshift(model, config, task, device, batch_size):
    """Evaluate on GeoShift benchmark."""
    print(f"\n{'='*60}")
    print(f"GeoShift Benchmark Evaluation")
    print(f"{'='*60}\n")
    
    # Define GeoShift tasks
    tasks = {
        "small_to_large": {
            "source": "qm9",
            "target": "qm9_large",
            "description": "Small → Large molecules"
        },
        "organic_to_inorganic": {
            "source": "qm9",
            "target": "qm9_inorganic",
            "description": "Organic → Inorganic"
        },
        "equilibrium_to_md": {
            "source": "qm9",
            "target": "md17_aspirin",
            "description": "Equilibrium → Non-equilibrium"
        },
        "gas_to_condensed": {
            "source": "qm9",
            "target": "ani1x_condensed",
            "description": "Gas phase → Condensed phase"
        },
        "rigid_to_flexible": {
            "source": "qm9",
            "target": "md17_flexible",
            "description": "Rigid → Flexible backbones"
        }
    }
    
    # Evaluate specific task or all tasks
    task_list = [task] if task else list(tasks.keys())
    
    all_results = {}
    for task_name in task_list:
        if task_name not in tasks:
            print(f"Warning: Unknown task {task_name}, skipping")
            continue
        
        task_info = tasks[task_name]
        print(f"\nTask: {task_info['description']}")
        print(f"Target dataset: {task_info['target']}")
        
        # Load target dataset
        try:
            test_dataset = get_dataset(task_info["target"], split="test")
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # Evaluate
            metrics = ["energy_mae", "energy_rmse", "force_mae"]
            results = evaluate_model(model, test_loader, metrics, device)
            
            # Print results
            print(f"\nResults:")
            for metric_name, value in results.items():
                if metric_name != "physics_compliance":
                    print(f"  {metric_name}: {value:.4f}")
            
            print(f"\nPhysics Compliance:")
            for check, value in results["physics_compliance"].items():
                print(f"  {check}: {value:.4f}")
            
            all_results[task_name] = results
            
        except Exception as e:
            print(f"Error evaluating task {task_name}: {e}")
            continue
    
    return all_results


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    model, config = load_checkpoint(args.checkpoint, args.device)
    
    # Evaluate on specified benchmark
    if args.benchmark == "geoshift":
        results = evaluate_geoshift(
            model, config, args.task, args.device, args.batch_size
        )
    
    elif args.benchmark == "custom":
        if not args.dataset:
            raise ValueError("--dataset required for custom benchmark")
        
        # Load custom dataset
        test_dataset = get_dataset(args.dataset, split="test")
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Evaluate
        results = evaluate_model(model, test_loader, args.metrics, args.device)
        
        # Print results
        print(f"\nResults on {args.dataset}:")
        for metric_name, value in results.items():
            if metric_name != "physics_compliance":
                print(f"  {metric_name}: {value:.4f}")
    
    else:
        # Evaluate on standard benchmark (QM9, MD17, etc.)
        test_dataset = get_dataset(args.benchmark, split="test")
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        results = evaluate_model(model, test_loader, args.metrics, args.device)
        
        print(f"\nResults on {args.benchmark}:")
        for metric_name, value in results.items():
            if metric_name != "physics_compliance":
                print(f"  {metric_name}: {value:.4f}")
    
    # Save results
    results_file = os.path.join(args.output_dir, f"{args.benchmark}_results.json")
    with open(results_file, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {k: float(v) if isinstance(v, np.ndarray) else v 
                                    for k, v in value.items()}
            elif isinstance(value, np.ndarray):
                json_results[key] = float(value)
            else:
                json_results[key] = value
        
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
