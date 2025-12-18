# Contributing to Molecular Foundation Models

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## 🎯 How to Contribute

We welcome contributions in many forms:

- 🐛 **Bug reports** - Found a bug? Let us know!
- 💡 **Feature requests** - Have an idea? Share it!
- 📝 **Documentation** - Help improve our docs
- 🔧 **Code contributions** - Submit pull requests
- 🧪 **Testing** - Help us test on different systems
- 📊 **Benchmarks** - Run experiments and share results

## 🚀 Quick Start

1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub, then:
   git clone https://github.com/YOUR_USERNAME/molecular-foundation-models.git
   cd molecular-foundation-models
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write code
   - Add tests
   - Update documentation

4. **Test your changes**
   ```bash
   pytest tests/
   ```

5. **Commit and push**
   ```bash
   git add .
   git commit -m "Description of your changes"
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**
   - Go to your fork on GitHub
   - Click "New Pull Request"
   - Describe your changes

## 📋 Contribution Guidelines

### Code Style

We follow **PEP 8** for Python code:

```python
# Good
def calculate_energy(positions: torch.Tensor, 
                     charges: torch.Tensor) -> torch.Tensor:
    """Calculate electrostatic energy.
    
    Args:
        positions: Atomic positions (N, 3)
        charges: Atomic charges (N,)
        
    Returns:
        Total energy (scalar)
    """
    distances = torch.cdist(positions, positions)
    energy = torch.sum(charges[:, None] * charges[None, :] / distances)
    return energy
```

**Style guidelines:**
- Use 4 spaces for indentation
- Maximum line length: 88 characters
- Type hints for function arguments
- Docstrings for all public functions
- Descriptive variable names

**Formatting tools:**
```bash
# Install
pip install black isort flake8

# Format code
black src/
isort src/

# Check style
flake8 src/
```

### Commit Messages

Write clear, descriptive commit messages:

```bash
# Good
git commit -m "Add force prediction to EGNN layer"
git commit -m "Fix gradient computation in multi-task loss"
git commit -m "Update documentation for Vertex AI deployment"

# Bad
git commit -m "fix"
git commit -m "update"
git commit -m "changes"
```

**Format:**
```
<type>: <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

### Testing

All code contributions should include tests:

```python
# tests/test_model.py
import torch
from src.model import build_model

def test_model_forward():
    """Test model forward pass."""
    config = {
        "model": {
            "hidden_dim": 64,
            "n_layers": 3,
            ...
        }
    }
    model = build_model(config)
    
    # Create dummy input
    batch = {
        "pos": torch.randn(10, 3),
        "z": torch.randint(1, 10, (10,)),
        "batch": torch.zeros(10, dtype=torch.long)
    }
    
    # Test forward pass
    output = model(batch)
    assert "energy" in output
    assert output["energy"].shape == (1,)
```

**Run tests:**
```bash
# All tests
pytest tests/

# Specific test
pytest tests/test_model.py::test_model_forward

# With coverage
pytest --cov=src tests/
```

### Documentation

Update documentation when you:
- Add new features
- Change APIs
- Add configuration options
- Create new scripts

**Documentation locations:**
- Code: Docstrings in Python files
- Usage: `docs/` directory
- Examples: `notebooks/` directory
- API: `docs/API.md`

**Docstring format:**
```python
def my_function(arg1: str, arg2: int = 10) -> bool:
    """Short description.
    
    Longer description if needed. Can span
    multiple lines.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2 (default: 10)
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When arg2 is negative
        
    Example:
        >>> result = my_function("test", 5)
        >>> print(result)
        True
    """
    pass
```

## 🐛 Bug Reports

**Before submitting:**
1. Check existing issues
2. Try latest version
3. Verify it's reproducible

**Include in report:**
```markdown
### Description
Clear description of the bug

### Steps to Reproduce
1. Run command X
2. Open file Y
3. See error Z

### Expected Behavior
What should happen

### Actual Behavior
What actually happens

### Environment
- OS: Ubuntu 22.04
- Python: 3.10.8
- PyTorch: 2.0.1
- CUDA: 11.8

### Error Message
```
Full error traceback here
```

### Minimal Reproducible Example
```python
# Minimal code to reproduce
import torch
from src.model import build_model

model = build_model({...})
# Error occurs here
```
```

## 💡 Feature Requests

**Good feature requests include:**
- Clear description of feature
- Use case / motivation
- Examples of how it would work
- Alternatives considered

**Template:**
```markdown
### Feature Description
I would like to add support for...

### Motivation
This would be useful because...

### Proposed Solution
The feature would work by...

### Example Usage
```python
# How users would use this feature
model.new_feature(...)
```

### Alternatives
I also considered... but this approach is better because...
```

## 🎯 Areas for Contribution

### High Priority

- [ ] Additional datasets (Transition1x, SPICE, etc.)
- [ ] More architecture variants (SchNet, DimeNet++)
- [ ] Improved training strategies (curriculum learning)
- [ ] Deployment optimizations (TorchScript, ONNX)
- [ ] Better documentation (tutorials, examples)

### Medium Priority

- [ ] Distributed training support
- [ ] Mixed precision optimization
- [ ] Hyperparameter tuning utilities
- [ ] Visualization tools
- [ ] Performance benchmarks

### Good First Issues

Look for issues tagged `good-first-issue`:
- Documentation improvements
- Adding unit tests
- Code cleanup
- Example notebooks

## 📊 Benchmarking

If you run experiments, please share:

```markdown
### Hardware
- GPU: NVIDIA V100
- RAM: 32GB
- Storage: SSD

### Configuration
- Model: EGNN-PaiNN (128 hidden, 5 layers)
- Dataset: QM9
- Batch size: 32
- Epochs: 100

### Results
- Training time: 12 hours
- Best validation MAE: 0.042 eV
- GPU memory: 14.2 GB
- Throughput: 850 molecules/sec

### Comparison
Compared to baseline (training from scratch):
- Error reduction: 35%
- Sample efficiency: 5.2x
```

## 🔄 Review Process

**What to expect:**
1. **Automated checks** - CI runs tests and linting
2. **Maintainer review** - We'll review your code
3. **Feedback** - We may request changes
4. **Approval** - Once approved, we'll merge!

**Review criteria:**
- Code quality and style
- Test coverage
- Documentation
- Performance impact
- Breaking changes

**Timeline:**
- Initial response: Within 48 hours
- Full review: Within 1 week
- Merge: After approval and CI passes

## ❓ Questions?

- **General questions:** Open a [Discussion](https://github.com/YOUR_USERNAME/molecular-foundation-models/discussions)
- **Bug reports:** Open an [Issue](https://github.com/YOUR_USERNAME/molecular-foundation-models/issues)
- **Security issues:** Email security@domain.com

## 🙏 Recognition

Contributors will be:
- Listed in README.md
- Acknowledged in paper
- Invited to join author list (for significant contributions)

Thank you for contributing to advancing molecular modeling! 🚀
