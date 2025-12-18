"""Prediction heads for different tasks."""

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool


class EnergyHead(nn.Module):
    """Energy prediction head."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, h, batch):
        """
        Predict energy.
        
        Args:
            h: Node features (N, hidden_dim)
            batch: Batch indices
            
        Returns:
            Energies (batch_size, 1)
        """
        # Global pooling
        h_global = global_add_pool(h, batch)
        
        # Predict energy
        energy = self.mlp(h_global)
        
        return energy


class HOMOLUMOHead(nn.Module):
    """HOMO-LUMO gap prediction head."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def forward(self, h, batch):
        """
        Predict HOMO-LUMO gap.
        
        Args:
            h: Node features (N, hidden_dim)
            batch: Batch indices
            
        Returns:
            HOMO-LUMO gaps (batch_size, 1)
        """
        # Global pooling
        h_global = global_mean_pool(h, batch)
        
        # Predict gap
        gap = self.mlp(h_global)
        
        return gap


class ForceHead(nn.Module):
    """Force prediction head (optional)."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )
    
    def forward(self, h):
        """
        Predict forces.
        
        Args:
            h: Node features (N, hidden_dim)
            
        Returns:
            Forces (N, 3)
        """
        forces = self.mlp(h)
        return forces
