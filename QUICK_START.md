# Quick Start Guide

## What's Inside

This archive contains a complete, research-ready GitHub repository for molecular foundation models.

## Directory Structure

```
molecular-foundation-models/
├── README.md                     # Main repository README
├── LICENSE                       # MIT License
├── CONTRIBUTING.md              # Contribution guidelines
├── .gitignore                   # Git ignore patterns
├── setup.py                     # Package installation
├── requirements.txt             # Python dependencies
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── model.py                 # EGNN-PaiNN model (2,100+ lines)
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   ├── data/                    # Dataset loaders
│   │   ├── __init__.py
│   │   ├── qm9.py
│   │   ├── md17.py
│   │   └── ani1x.py
│   ├── models/                  # Model components
│   │   ├── __init__.py
│   │   ├── egnn.py
│   │   ├── painn.py
│   │   └── heads.py
│   └── utils/                   # Utilities
│       ├── __init__.py
│       ├── metrics.py
│       └── visualization.py
│
├── configs/                     # Training configurations
│   ├── cross_domain.json
│   ├── single_domain.json
│   └── transfer.json
│
├── scripts/                     # Utility scripts
│   ├── download_datasets.py
│   └── analyze_results.py
│
├── tests/                       # Unit tests
│   └── test_model.py
│
├── docs/                        # Documentation
│   ├── TRAINING.md
│   ├── SETUP_REPOSITORY.md
│   ├── FILE_INVENTORY.md
│   └── images/                  # SVG diagrams
│       ├── architecture.svg
│       ├── model_architecture.svg
│       ├── training_curves.svg
│       └── sample_efficiency.svg
│
└── [empty directories for data]
    ├── experiments/.gitkeep
    ├── checkpoints/.gitkeep
    └── data/.gitkeep
```

## Setup (3 Steps)

### 1. Extract Archive

```bash
tar -xzf molecular-foundation-models.tar.gz
cd molecular-foundation-models
```

### 2. Install Dependencies

```bash
# Create conda environment
conda create -n molecular-models python=3.10
conda activate molecular-models

# Install package
pip install -e .
```

### 3. Verify Installation

```bash
python -c "from src.model import build_model; print('✓ Installation successful!')"
```

## Usage

### Train Model

```bash
# Download datasets first
python scripts/download_datasets.py

# Train cross-domain model
python src/train.py --config configs/cross_domain.json

# Train baseline
python src/train.py --config configs/single_domain.json
```

### Evaluate Model

```bash
python src/evaluate.py \
    --checkpoint experiments/cross_domain_pretraining/best_model.pt \
    --benchmark geoshift
```

### Run Tests

```bash
pytest tests/ -v
```

## Push to GitHub

See `docs/SETUP_REPOSITORY.md` for complete instructions.

Quick version:

```bash
# 1. Create repo on GitHub (don't initialize)
# 2. Update YOUR_USERNAME in README.md
# 3. Initialize git

git init
git add .
git commit -m "Initial commit: Cross-domain foundation models"
git remote add origin https://github.com/YOUR_USERNAME/molecular-foundation-models.git
git branch -M main
git push -u origin main
```

## File Count

- 17 Python files
- 3 JSON configs
- 4 SVG diagrams
- 5 Documentation files
- **Total: 29 files + directory structure**

## Features

✅ Complete EGNN-PaiNN implementation (2,100+ lines)
✅ Dataset loaders (QM9, MD17, ANI-1x)
✅ Training & evaluation scripts
✅ Beautiful SVG diagrams
✅ Comprehensive documentation
✅ Unit tests
✅ Ready for GitHub

## Next Steps

1. Read `README.md` for project overview
2. Read `docs/TRAINING.md` for training guide
3. Read `docs/SETUP_REPOSITORY.md` for GitHub setup
4. Start training your models!

## Questions?

- Check `docs/TRAINING.md` for training help
- Check `docs/FILE_INVENTORY.md` for file details
- Check `docs/SETUP_REPOSITORY.md` for GitHub setup

**Happy researching!** 🚀
