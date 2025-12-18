# Training Guide

Complete guide to training cross-domain molecular models.

## Quick Start

### Basic Training

```bash
# Train cross-domain model
python train.py --config configs/cross_domain.json

# Train baseline (single-domain)
python train.py --config configs/single_domain.json
```

### Monitor Training

```bash
# TensorBoard
tensorboard --logdir experiments/

# Or watch log file
tail -f experiments/cross_domain_pretraining/train.log
```

## Training Modes

### 1. Cross-Domain Pre-training

Pre-train on multiple datasets simultaneously:

```json
{
  "data": {
    "mode": "cross_domain",
    "datasets": ["qm9", "md17", "ani1x"],
    "multidomain_sampling": {
      "strategy": "balanced",
      "dataset_weights": {
        "qm9": 0.4,
        "md17": 0.3,
        "ani1x": 0.3
      }
    }
  }
}
```

### 2. Single-Domain Training (Baseline)

Train on one dataset:

```json
{
  "data": {
    "mode": "single_domain",
    "datasets": ["qm9"]
  }
}
```

### 3. Transfer Learning

Fine-tune pre-trained model:

```bash
python train.py \
    --config configs/transfer.json \
    --pretrained experiments/cross_domain/best_model.pt
```

## Configuration Options

### Model Architecture

```json
{
  "model": {
    "hidden_dim": 128,        // Node feature dimension
    "vector_dim": 64,         // Vector feature dimension
    "n_layers": 5,            // Total layers
    "egnn_layers": 3,         // EGNN layers
    "painn_layers": 2,        // PaiNN layers
    "cutoff": 5.0,            // Interaction cutoff (Å)
    "num_rbf": 20,            // Radial basis functions
    "activation": "silu"      // Activation function
  }
}
```

### Training Hyperparameters

```json
{
  "training": {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "gradient_clip": 1.0,
    "use_amp": true           // Automatic mixed precision
  }
}
```

### Learning Rate Schedule

```json
{
  "training": {
    "scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "patience": 10,
      "min_lr": 1e-6
    }
  }
}
```

### Loss Function

```json
{
  "loss": {
    "type": "physics_augmented",
    "energy_weight": 1.0,
    "force_weight": 0.5,
    "physics_weight": 0.1
  }
}
```

## Advanced Options

### Resume Training

```bash
python train.py \
    --config configs/cross_domain.json \
    --resume experiments/cross_domain/checkpoint_50.pt
```

### Custom Learning Rate

```bash
python train.py \
    --config configs/cross_domain.json \
    --lr 0.0005
```

### Debug Mode

```bash
python train.py \
    --config configs/cross_domain.json \
    --debug
```

## Training Tips

### 1. Start Small

Test with fewer epochs first:

```bash
python train.py --config configs/cross_domain.json --epochs 10
```

### 2. Monitor Overfitting

Watch for training vs validation loss divergence.

### 3. Adjust Batch Size

Larger batch size = faster training but more memory:

```bash
python train.py --config configs/cross_domain.json --batch-size 64
```

### 4. Use Mixed Precision

Enable AMP for faster training:

```json
{
  "training": {
    "use_amp": true
  }
}
```

## Expected Training Times

| GPU | Batch Size | Time/Epoch | Total (100 epochs) |
|-----|------------|------------|-------------------|
| T4  | 32         | 45 min     | ~75 hours         |
| V100| 32         | 23 min     | ~38 hours         |
| A100| 64         | 15 min     | ~25 hours         |

## Troubleshooting

### Out of Memory

- Reduce batch size
- Reduce model size (hidden_dim, n_layers)
- Enable gradient checkpointing

### Slow Training

- Enable mixed precision (AMP)
- Increase batch size
- Use multiple GPUs (if available)

### NaN Loss

- Reduce learning rate
- Enable gradient clipping
- Check data normalization

### Poor Convergence

- Adjust learning rate schedule
- Try different optimizer
- Check data quality

## Output Files

Training creates:

```
experiments/cross_domain_pretraining/
├── config.json              # Configuration
├── train.log                # Training log
├── metrics.json             # Training metrics
├── checkpoints/             # Model checkpoints
│   ├── epoch_010.pt
│   ├── epoch_020.pt
│   └── best_model.pt
└── tensorboard/             # TensorBoard logs
```

## Next Steps

After training:

1. Evaluate model: `python evaluate.py --checkpoint best_model.pt`
2. Analyze results: `python scripts/analyze_results.py`
3. Fine-tune on target task: `python train.py --config configs/transfer.json`
