"""
Cross-Domain Equivariant Neural Network Model
=============================================
Hybrid EGNN-PaiNN architecture for learning shared physics representations
across biological and electronic structure systems.

Architecture combines:
- E(n)-equivariant message passing (EGNN)
- Scalar and vector feature representations (PaiNN)
- Cross-domain pretraining capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_scatter import scatter
import math


class SiLU(nn.Module):
    """Sigmoid Linear Unit activation function."""
    def forward(self, x):
        return x * torch.sigmoid(x)


class MLP(nn.Module):
    """Multi-layer perceptron with customizable architecture."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, activation=SiLU):
        super().__init__()
        
        layers = []
        dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation on last layer
                layers.append(activation())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class EquivariantMessagePassingLayer(MessagePassing):
    """
    E(n)-equivariant message passing layer combining EGNN and PaiNN concepts.
    
    Updates both scalar features h and vector features V while maintaining
    E(n) equivariance (rotations, reflections, translations).
    """
    
    def __init__(self, hidden_dim, vector_dim, cutoff=5.0):
        super().__init__(aggr='add')
        
        self.hidden_dim = hidden_dim
        self.vector_dim = vector_dim
        self.cutoff = cutoff
        
        # Edge features MLP
        self.edge_mlp = MLP(
            input_dim=2 * hidden_dim + 1,  # h_i, h_j, distance
            hidden_dim=hidden_dim,
            output_dim=hidden_dim
        )
        
        # Node feature update MLP
        self.node_mlp = MLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim
        )
        
        # Vector feature gating
        self.vector_gate = MLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=vector_dim
        )
        
    def forward(self, h, pos, V, edge_index, batch):
        """
        Forward pass of equivariant message passing.
        
        Args:
            h: Node scalar features [N, hidden_dim]
            pos: Node positions [N, 3]
            V: Node vector features [N, 3, vector_dim]
            edge_index: Graph connectivity [2, E]
            batch: Batch assignment [N]
            
        Returns:
            Updated h and V
        """
        # Message passing
        h_update, V_update = self.propagate(
            edge_index,
            h=h,
            pos=pos,
            V=V,
            size=None
        )
        
        # Update scalar features with residual connection
        h = h + h_update
        
        # Update vector features with residual connection
        V = V + V_update
        
        return h, V
    
    def message(self, h_i, h_j, pos_i, pos_j):
        """
        Construct messages between nodes.
        
        Args:
            h_i, h_j: Source and target node features
            pos_i, pos_j: Source and target positions
            
        Returns:
            Messages and relative position vectors
        """
        # Compute relative position and distance
        rel_pos = pos_j - pos_i  # [E, 3]
        distance = torch.norm(rel_pos, dim=-1, keepdim=True)  # [E, 1]
        
        # Apply cutoff
        cutoff_weight = self.cutoff_function(distance)
        
        # Compute edge features
        edge_feat = torch.cat([h_i, h_j, distance], dim=-1)
        edge_message = self.edge_mlp(edge_feat) * cutoff_weight
        
        # Directional message (for vector features)
        direction = rel_pos / (distance + 1e-8)  # [E, 3]
        
        return edge_message, direction
    
    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        """Aggregate messages."""
        edge_message, direction = inputs
        
        # Aggregate scalar messages
        h_message = scatter(edge_message, index, dim=0, dim_size=dim_size, reduce='add')
        
        # Aggregate vector messages (directional)
        # Each edge contributes to vector features based on direction
        V_message = torch.zeros(h_message.size(0), 3, self.vector_dim, device=edge_message.device)
        
        for i in range(self.vector_dim):
            # Weight direction by edge message
            weighted_dir = direction * edge_message[:, i:i+1]
            V_message[:, :, i] = scatter(weighted_dir, index, dim=0, dim_size=dim_size, reduce='add')
        
        return h_message, V_message
    
    def update(self, aggr_out):
        """Update node features."""
        h_message, V_message = aggr_out
        
        # Update scalar features
        h_update = self.node_mlp(h_message)
        
        # Update vector features with gating
        gate = self.vector_gate(h_message).unsqueeze(1)  # [N, 1, vector_dim]
        V_update = V_message * torch.sigmoid(gate)
        
        return h_update, V_update
    
    def cutoff_function(self, distance):
        """Smooth cutoff function."""
        x = distance / self.cutoff
        x = torch.clamp(x, 0, 1)
        return 1 - 6*x**5 + 15*x**4 - 10*x**3


class ScalarVectorMixing(nn.Module):
    """
    Mix scalar and vector features (PaiNN-style).
    
    Enables information flow between scalar and vector representations
    while maintaining equivariance.
    """
    
    def __init__(self, hidden_dim, vector_dim):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.vector_dim = vector_dim
        
        # Vector norm to scalar
        self.norm_mlp = MLP(
            input_dim=hidden_dim + vector_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim
        )
        
        # Scalar to vector gating
        self.gate_mlp = MLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=vector_dim
        )
    
    def forward(self, h, V):
        """
        Mix scalar and vector features.
        
        Args:
            h: Scalar features [N, hidden_dim]
            V: Vector features [N, 3, vector_dim]
            
        Returns:
            Updated h and V
        """
        # Compute vector norms (invariant)
        V_norm = torch.norm(V, dim=1)  # [N, vector_dim]
        
        # Update scalars using vector information
        h_input = torch.cat([h, V_norm], dim=-1)
        h_update = self.norm_mlp(h_input)
        h = h + h_update
        
        # Gate vectors using scalar information
        gate = self.gate_mlp(h).unsqueeze(1)  # [N, 1, vector_dim]
        V = V * torch.sigmoid(gate)
        
        return h, V


class CrossDomainEquivariantNet(nn.Module):
    """
    Cross-domain equivariant neural network for molecular property prediction.
    
    Combines EGNN and PaiNN architectures for learning shared physics
    representations across biological and electronic structure domains.
    """
    
    def __init__(
        self,
        hidden_dim=128,
        vector_dim=64,
        n_layers=5,
        cutoff=5.0,
        n_outputs=1,
        max_atomic_num=100
    ):
        """
        Initialize model.
        
        Args:
            hidden_dim: Dimension of scalar features
            vector_dim: Dimension of vector features
            n_layers: Number of message passing layers
            cutoff: Interaction cutoff radius (Angstroms)
            n_outputs: Number of output properties
            max_atomic_num: Maximum atomic number for embedding
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.vector_dim = vector_dim
        self.n_layers = n_layers
        self.cutoff = cutoff
        
        # Atomic number embedding
        self.embedding = nn.Embedding(max_atomic_num, hidden_dim)
        
        # Message passing layers
        self.mp_layers = nn.ModuleList([
            EquivariantMessagePassingLayer(hidden_dim, vector_dim, cutoff)
            for _ in range(n_layers)
        ])
        
        # Scalar-vector mixing layers
        self.mixing_layers = nn.ModuleList([
            ScalarVectorMixing(hidden_dim, vector_dim)
            for _ in range(n_layers)
        ])
        
        # Output layers
        self.output_mlp = MLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=n_outputs
        )
        
    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data: PyG Data object with:
                - z: Atomic numbers [N]
                - pos: Atomic positions [N, 3]
                - edge_index: Graph connectivity [2, E]
                - batch: Batch assignment [N]
                
        Returns:
            Predictions [batch_size, n_outputs]
        """
        z, pos, edge_index, batch = data.z, data.pos, data.edge_index, data.batch
        
        # Initial embedding
        h = self.embedding(z)  # [N, hidden_dim]
        
        # Initialize vector features to zero
        V = torch.zeros(h.size(0), 3, self.vector_dim, device=h.device)
        
        # Message passing with mixing
        for mp_layer, mix_layer in zip(self.mp_layers, self.mixing_layers):
            # Equivariant message passing
            h, V = mp_layer(h, pos, V, edge_index, batch)
            
            # Scalar-vector mixing
            h, V = mix_layer(h, V)
        
        # Per-atom predictions
        atom_pred = self.output_mlp(h)  # [N, n_outputs]
        
        # Aggregate to molecule-level
        mol_pred = global_mean_pool(atom_pred, batch)  # [batch_size, n_outputs]
        
        return mol_pred, atom_pred
    
    def compute_forces(self, data):
        """
        Compute forces as negative gradient of energy.
        
        Args:
            data: PyG Data object
            
        Returns:
            Forces [N, 3]
        """
        # Enable gradient computation
        data.pos.requires_grad_(True)
        
        # Compute energy
        energy, _ = self.forward(data)
        energy = energy.sum()
        
        # Compute forces as -dE/dpos
        forces = -torch.autograd.grad(
            energy,
            data.pos,
            create_graph=True
        )[0]
        
        return forces


class MultitaskCrossDomainModel(nn.Module):
    """
    Multitask model for cross-physics learning.
    
    Shares encoder across tasks while using task-specific output heads.
    """
    
    def __init__(
        self,
        hidden_dim=128,
        vector_dim=64,
        n_layers=5,
        cutoff=5.0,
        task_dims=None
    ):
        """
        Initialize multitask model.
        
        Args:
            hidden_dim: Dimension of scalar features
            vector_dim: Dimension of vector features
            n_layers: Number of message passing layers
            cutoff: Interaction cutoff radius
            task_dims: Dictionary mapping task names to output dimensions
        """
        super().__init__()
        
        if task_dims is None:
            task_dims = {
                'energy': 1,
                'forces': 3,
                'homo_lumo_gap': 1,
                'dipole_moment': 1
            }
        
        self.task_dims = task_dims
        
        # Shared encoder
        self.encoder = CrossDomainEquivariantNet(
            hidden_dim=hidden_dim,
            vector_dim=vector_dim,
            n_layers=n_layers,
            cutoff=cutoff,
            n_outputs=hidden_dim  # Output features for task heads
        )
        
        # Task-specific output heads
        self.task_heads = nn.ModuleDict({
            task: MLP(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim // 2,
                output_dim=dim
            )
            for task, dim in task_dims.items()
        })
    
    def forward(self, data, tasks=None):
        """
        Forward pass for specified tasks.
        
        Args:
            data: PyG Data object
            tasks: List of task names to compute (None = all tasks)
            
        Returns:
            Dictionary mapping task names to predictions
        """
        if tasks is None:
            tasks = list(self.task_dims.keys())
        
        # Shared encoding
        features, _ = self.encoder(data)
        
        # Task-specific predictions
        predictions = {}
        for task in tasks:
            if task == 'forces':
                # Forces require gradient computation
                predictions[task] = self.encoder.compute_forces(data)
            else:
                predictions[task] = self.task_heads[task](features)
        
        return predictions


def build_model(config):
    """
    Build model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Model instance
    """
    if config.get('multitask', False):
        model = MultitaskCrossDomainModel(
            hidden_dim=config.get('hidden_dim', 128),
            vector_dim=config.get('vector_dim', 64),
            n_layers=config.get('n_layers', 5),
            cutoff=config.get('cutoff', 5.0),
            task_dims=config.get('task_dims', None)
        )
    else:
        model = CrossDomainEquivariantNet(
            hidden_dim=config.get('hidden_dim', 128),
            vector_dim=config.get('vector_dim', 64),
            n_layers=config.get('n_layers', 5),
            cutoff=config.get('cutoff', 5.0),
            n_outputs=config.get('n_outputs', 1)
        )
    
    return model


if __name__ == '__main__':
    # Test model
    print("Testing CrossDomainEquivariantNet...")
    
    # Create dummy data
    n_atoms = 10
    batch_size = 2
    
    data = Data(
        z=torch.randint(1, 10, (n_atoms,)),
        pos=torch.randn(n_atoms, 3),
        edge_index=torch.randint(0, n_atoms, (2, 20)),
        batch=torch.zeros(n_atoms, dtype=torch.long)
    )
    
    # Build model
    model = build_model({
        'hidden_dim': 128,
        'vector_dim': 64,
        'n_layers': 5,
        'n_outputs': 1
    })
    
    # Forward pass
    output, atom_output = model(data)
    print(f"Output shape: {output.shape}")
    print(f"Atom output shape: {atom_output.shape}")
    
    # Test forces
    forces = model.compute_forces(data)
    print(f"Forces shape: {forces.shape}")
    
    print("\n✓ Model test passed!")
