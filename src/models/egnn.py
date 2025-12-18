"""E(n) Equivariant Graph Neural Network layer."""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class EGNNLayer(MessagePassing):
    """
    E(n) Equivariant Graph Neural Network layer.
    
    Implements equivariant message passing that preserves SE(3) symmetry.
    """
    
    def __init__(self, hidden_dim, edge_dim=0, activation="silu"):
        super().__init__(aggr="add")
        
        self.hidden_dim = hidden_dim
        
        # Message MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim + 1, hidden_dim),
            nn.SiLU() if activation == "silu" else nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU() if activation == "silu" else nn.ReLU()
        )
        
        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU() if activation == "silu" else nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Coordinate update MLP
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU() if activation == "silu" else nn.ReLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
    
    def forward(self, h, pos, edge_index, edge_attr=None):
        """
        Forward pass.
        
        Args:
            h: Node features (N, hidden_dim)
            pos: Node positions (N, 3)
            edge_index: Edge indices (2, E)
            edge_attr: Edge attributes (E, edge_dim)
            
        Returns:
            Updated node features and positions
        """
        # Propagate messages
        h_out = self.propagate(edge_index, h=h, pos=pos, edge_attr=edge_attr)
        
        # Update node features
        h_out = h + self.node_mlp(torch.cat([h, h_out], dim=-1))
        
        return h_out, pos
    
    def message(self, h_i, h_j, pos_i, pos_j, edge_attr=None):
        """
        Compute messages.
        
        Args:
            h_i: Source node features
            h_j: Target node features
            pos_i: Source positions
            pos_j: Target positions
            edge_attr: Edge attributes
            
        Returns:
            Messages
        """
        # Compute squared distance
        rel_pos = pos_j - pos_i
        dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)
        
        # Message features
        if edge_attr is not None:
            edge_input = torch.cat([h_i, h_j, dist_sq, edge_attr], dim=-1)
        else:
            edge_input = torch.cat([h_i, h_j, dist_sq], dim=-1)
        
        # Compute message
        msg = self.edge_mlp(edge_input)
        
        # Coordinate update
        coord_weight = self.coord_mlp(msg)
        
        # Update positions (not returned in this version)
        # pos_update = coord_weight * rel_pos / (dist_sq.sqrt() + 1e-8)
        
        return msg
