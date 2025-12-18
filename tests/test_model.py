"""Unit tests for model."""

import pytest
import torch
from src.model import build_model
from torch_geometric.data import Data, Batch


def test_model_creation():
    """Test model can be created."""
    config = {
        "model": {
            "hidden_dim": 64,
            "vector_dim": 32,
            "n_layers": 3,
            "egnn_layers": 2,
            "painn_layers": 1,
            "cutoff": 5.0,
            "num_rbf": 10,
            "multitask": True
        }
    }
    
    model = build_model(config)
    assert model is not None
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0


def test_model_forward():
    """Test model forward pass."""
    config = {
        "model": {
            "hidden_dim": 64,
            "vector_dim": 32,
            "n_layers": 3,
            "egnn_layers": 2,
            "painn_layers": 1,
            "cutoff": 5.0,
            "num_rbf": 10,
            "multitask": True
        }
    }
    
    model = build_model(config)
    
    # Create dummy batch
    pos = torch.randn(10, 3)
    z = torch.randint(1, 10, (10,))
    batch = torch.zeros(10, dtype=torch.long)
    
    data = Data(pos=pos, z=z, batch=batch)
    batch_data = Batch.from_data_list([data])
    
    # Forward pass
    output = model(batch_data)
    
    assert "energy" in output
    assert output["energy"].shape == (1, 1)


def test_model_gradients():
    """Test gradients flow through model."""
    config = {
        "model": {
            "hidden_dim": 64,
            "vector_dim": 32,
            "n_layers": 3,
            "egnn_layers": 2,
            "painn_layers": 1,
            "cutoff": 5.0,
            "num_rbf": 10,
            "multitask": False
        }
    }
    
    model = build_model(config)
    
    # Create dummy batch
    pos = torch.randn(10, 3, requires_grad=True)
    z = torch.randint(1, 10, (10,))
    batch = torch.zeros(10, dtype=torch.long)
    
    data = Data(pos=pos, z=z, batch=batch)
    batch_data = Batch.from_data_list([data])
    
    # Forward and backward
    output = model(batch_data)
    loss = output["energy"].sum()
    loss.backward()
    
    # Check gradients
    assert pos.grad is not None
    assert not torch.isnan(pos.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
