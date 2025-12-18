"""Polarizable Atom Interaction Neural Network layer."""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, radius_graph


class PaiNNLayer(MessagePassing):
    """
    PaiNN layer for equivariant molecular modeling.
    
    Handles scalar and vector features with equivariant message passing.
    """
    
    def __init__(self, hidden_dim, vector_dim, cutoff=5.0, num_rbf=20):
        super().__init__(aggr="add")
        
        self.hidden_dim = hidden_dim
        self.vector_dim = vector_dim
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        
        # Message function
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3)
        )
        
        # Update function for scalars
        self.update_mlp_s = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3)
        )
        
        # Update function for vectors
        self.update_mlp_v = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2)
        )
        
        # RBF embedding
        self.rbf_layer = RBFLayer(num_rbf, cutoff)
    
    def forward(self, s, v, pos, batch):
        """
        Forward pass.
        
        Args:
            s: Scalar features (N, hidden_dim)
            v: Vector features (N, vector_dim, 3)
            pos: Positions (N, 3)
            batch: Batch indices
            
        Returns:
            Updated scalar and vector features
        """
        # Build radius graph
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        
        # Message passing
        s_msg, v_msg = self.propagate(edge_index, s=s, v=v, pos=pos)
        
        # Update
        s_out, v_out = self.update(s, v, s_msg, v_msg)
        
        return s_out, v_out
    
    def message(self, s_i, s_j, v_j, pos_i, pos_j):
        """
        Compute messages.
        
        Args:
            s_i, s_j: Scalar features
            v_j: Vector features
            pos_i, pos_j: Positions
            
        Returns:
            Scalar and vector messages
        """
        # Distance and direction
        rel_pos = pos_j - pos_i
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        direction = rel_pos / (dist + 1e-8)
        
        # RBF embedding
        rbf = self.rbf_layer(dist.squeeze(-1))
        
        # Message MLP
        msg_features = self.message_mlp(s_j)
        msg_s, msg_v1, msg_v2 = torch.split(
            msg_features, self.hidden_dim, dim=-1
        )
        
        # Scalar message
        s_msg = msg_s * rbf.sum(dim=-1, keepdim=True)
        
        # Vector message
        v_msg = msg_v1.unsqueeze(-1) * direction.unsqueeze(1) + \
                msg_v2.unsqueeze(-1) * v_j
        
        return s_msg, v_msg
    
    def update(self, s, v, s_msg, v_msg):
        """
        Update features.
        
        Args:
            s: Scalar features
            v: Vector features  
            s_msg: Scalar messages
            v_msg: Vector messages
            
        Returns:
            Updated features
        """
        # Scalar update
        v_norm = torch.norm(v, dim=-1)
        s_input = torch.cat([s, s_msg, v_norm], dim=-1)
        s_update = self.update_mlp_s(s_input)
        s_a, s_b, s_c = torch.split(s_update, self.hidden_dim, dim=-1)
        
        s_out = s_a * s + s_b * s_msg + s_c
        
        # Vector update
        v_input = torch.cat([s, v_norm], dim=-1)
        v_update = self.update_mlp_v(v_input)
        v_a, v_b = torch.split(v_update, self.hidden_dim, dim=-1)
        
        v_out = v_a.unsqueeze(-1) * v + v_b.unsqueeze(-1) * v_msg
        
        return s_out, v_out


class RBFLayer(nn.Module):
    """Radial basis function layer."""
    
    def __init__(self, num_rbf, cutoff):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        
        # Centers and widths
        self.register_buffer("centers", torch.linspace(0, cutoff, num_rbf))
        self.register_buffer("widths", torch.ones(num_rbf) * (cutoff / num_rbf))
    
    def forward(self, distances):
        """
        Compute RBF features.
        
        Args:
            distances: Edge distances (E,)
            
        Returns:
            RBF features (E, num_rbf)
        """
        distances = distances.unsqueeze(-1)
        rbf = torch.exp(-((distances - self.centers) / self.widths) ** 2)
        
        # Cutoff
        cutoff_values = 0.5 * (torch.cos(distances * 3.14159 / self.cutoff) + 1)
        cutoff_values = cutoff_values * (distances < self.cutoff).float()
        
        return rbf * cutoff_values
