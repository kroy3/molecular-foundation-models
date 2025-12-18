"""MD17 dataset loader."""

import os
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data, download_url


class MD17Dataset(InMemoryDataset):
    """
    MD17 dataset wrapper.
    
    Molecular dynamics trajectories for 8 small molecules.
    Contains energies and forces from DFT calculations.
    """
    
    MOLECULES = [
        'aspirin', 'benzene', 'ethanol', 'malonaldehyde',
        'naphthalene', 'salicylic_acid', 'toluene', 'uracil'
    ]
    
    BASE_URL = 'http://www.quantum-machine.org/gdml/data/npz/'
    
    def __init__(self, root="./data/md17", molecule="aspirin", split="train", transform=None):
        """
        Args:
            root: Root directory for dataset
            molecule: Molecule name
            split: train, val, or test
            transform: Optional transform
        """
        if molecule not in self.MOLECULES:
            raise ValueError(f"Unknown molecule: {molecule}. Choose from {self.MOLECULES}")
        
        self.molecule = molecule
        self.split = split
        super().__init__(root, transform)
        
        # Load split
        if split == "train":
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif split == "val":
            self.data, self.slices = torch.load(self.processed_paths[1])
        else:  # test
            self.data, self.slices = torch.load(self.processed_paths[2])
    
    @property
    def raw_file_names(self):
        return [f'md17_{self.molecule}.npz']
    
    @property
    def processed_file_names(self):
        return [f'train_{self.molecule}.pt', f'val_{self.molecule}.pt', f'test_{self.molecule}.pt']
    
    def download(self):
        """Download MD17 dataset."""
        url = f"{self.BASE_URL}md17_{self.molecule}.npz"
        download_url(url, self.raw_dir)
    
    def process(self):
        """Process MD17 dataset."""
        # Load numpy file
        data_file = os.path.join(self.raw_dir, f'md17_{self.molecule}.npz')
        data = np.load(data_file)
        
        # Extract data
        positions = torch.from_numpy(data['R']).float()  # (N, n_atoms, 3)
        energies = torch.from_numpy(data['E']).float()   # (N,)
        forces = torch.from_numpy(data['F']).float()     # (N, n_atoms, 3)
        atomic_numbers = torch.from_numpy(data['z']).long()  # (n_atoms,)
        
        # Create PyG data objects
        data_list = []
        for i in range(len(positions)):
            data_obj = Data(
                pos=positions[i],
                z=atomic_numbers,
                energy=energies[i].unsqueeze(0),
                forces=forces[i]
            )
            data_list.append(data_obj)
        
        # Split: 80% train, 10% val, 10% test
        n = len(data_list)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        
        # Save splits
        torch.save(self.collate(data_list[:n_train]), self.processed_paths[0])
        torch.save(self.collate(data_list[n_train:n_train+n_val]), self.processed_paths[1])
        torch.save(self.collate(data_list[n_train+n_val:]), self.processed_paths[2])
