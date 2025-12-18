# GitHub Repository Setup Guide

Quick guide to setting up your molecular foundation models repository on GitHub.

## 📋 Prerequisites

- GitHub account
- Git installed locally
- All repository files downloaded

## 🚀 Quick Setup (5 Minutes)

### 1. Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `molecular-foundation-models`
3. Description: `Cross-domain foundation models for molecular electrostatics using equivariant neural networks`
4. Choose Public or Private
5. **DON'T** check "Initialize with README"
6. Click "Create repository"

### 2. Prepare Local Repository

```bash
# Create project directory
mkdir molecular-foundation-models
cd molecular-foundation-models

# Create directory structure
mkdir -p src/{data,models,layers,utils}
mkdir -p configs scripts notebooks tests docs/images
mkdir -p experiments checkpoints data

# Add placeholder files
touch experiments/.gitkeep checkpoints/.gitkeep data/.gitkeep
```

### 3. Copy Files

```bash
# Core files
cp README.md .
cp LICENSE .
cp CONTRIBUTING.md .
cp .gitignore .
cp setup.py .
cp requirements_clean.txt requirements.txt

# Source code
cp model.py train.py evaluate.py src/

# Scripts
cp download_datasets.py scripts/

# Config
cp cross_domain.json configs/

# SVG diagrams
mkdir -p docs/images
cp architecture.svg docs/images/
cp model_architecture.svg docs/images/
cp training_curves.svg docs/images/
cp sample_efficiency.svg docs/images/
```

### 4. Initialize Git

```bash
# Initialize repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Cross-domain foundation models"

# Add GitHub remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/molecular-foundation-models.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### 5. Configure Repository

On GitHub:

**Settings → General:**
- Description: Add your description
- Topics: `molecular-modeling`, `neural-operators`, `pytorch`, `equivariant-networks`, `deep-learning`

**Settings → Features:**
- ✅ Issues
- ✅ Discussions
- ✅ Wiki (optional)

**Settings → Branches:**
- Add branch protection rule for `main`
- ✅ Require pull request before merging

## 📝 Before Pushing

### Update Placeholders

In all files, replace:
- `YOUR_USERNAME` → your GitHub username
- `your.email@institution.edu` → your email
- `Your Name` → your actual name

### Quick find/replace:

```bash
# In README.md
sed -i 's/YOUR_USERNAME/your-username/g' README.md
sed -i 's/your.email@institution.edu/your@email.edu/g' README.md
sed -i 's/Your Name/FirstName LastName/g' README.md

# In LICENSE
sed -i 's/Your Name/FirstName LastName/g' LICENSE
```

## ✅ File Checklist

### Root Directory
- [ ] README.md
- [ ] LICENSE
- [ ] CONTRIBUTING.md
- [ ] .gitignore
- [ ] setup.py
- [ ] requirements.txt

### Source Code (`src/`)
- [ ] model.py
- [ ] train.py
- [ ] evaluate.py
- [ ] data/ (directory)
- [ ] models/ (directory)
- [ ] utils/ (directory)

### Configuration (`configs/`)
- [ ] cross_domain.json

### Scripts (`scripts/`)
- [ ] download_datasets.py

### Documentation (`docs/`)
- [ ] images/architecture.svg
- [ ] images/model_architecture.svg
- [ ] images/training_curves.svg
- [ ] images/sample_efficiency.svg

### Placeholders
- [ ] experiments/.gitkeep
- [ ] checkpoints/.gitkeep
- [ ] data/.gitkeep

## 🎨 Optional Enhancements

### Add Badges to README

At the top of README.md:

```markdown
![CI](https://github.com/YOUR_USERNAME/molecular-foundation-models/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
```

### Add Social Preview

1. Create banner image (1280x640 recommended)
2. Settings → Social preview → Upload image

### Create First Release

```bash
# Tag version
git tag -a v0.1.0 -m "Initial release"
git push origin v0.1.0

# On GitHub: Releases → Draft new release
```

## 📚 Repository Structure

Your final structure should look like:

```
molecular-foundation-models/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── .gitignore
├── setup.py
├── requirements.txt
│
├── src/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── data/
│   ├── models/
│   └── utils/
│
├── configs/
│   └── cross_domain.json
│
├── scripts/
│   └── download_datasets.py
│
├── notebooks/ (optional)
│
├── docs/
│   └── images/
│       ├── architecture.svg
│       ├── model_architecture.svg
│       ├── training_curves.svg
│       └── sample_efficiency.svg
│
├── tests/ (add later)
│
└── [generated, gitignored]/
    ├── experiments/
    ├── checkpoints/
    └── data/
```

## 🔧 Testing Locally

Before pushing:

```bash
# Create conda environment
conda create -n molecular-models python=3.10
conda activate molecular-models

# Install package
pip install -e .

# Verify installation
python -c "from src.model import build_model; print('✓ Import successful')"

# Run linting (optional)
black src/
isort src/
flake8 src/
```

## 🎯 Post-Setup

After pushing to GitHub:

1. ✅ Add description and topics
2. ✅ Enable issues and discussions
3. ✅ Set up branch protection
4. ✅ Add social preview image (optional)
5. ✅ Create first release tag
6. ✅ Share with community!

## 📞 Need Help?

- **Git Issues:** https://docs.github.com
- **Repository Setup:** https://docs.github.com/en/repositories

## 🎉 You're Done!

Your research repository is ready to share with the world!

```bash
# Share on Twitter/X
I just released my molecular foundation models!
Cross-domain neural operators for computational chemistry.

🔗 https://github.com/YOUR_USERNAME/molecular-foundation-models
⭐ Star if you find it useful!

#MachineLearning #Chemistry #PyTorch
```

---

**Happy coding!** 🚀
