"""ANI-1x dataset loader."""

import os
import h5py
import torch
from torch_geometric.data import InMemoryDataset, Data


class ANI1xDataset(InMemoryDataset):
    """
    ANI-1x dataset wrapper.
    
    5M molecular conformations with DFT energies.
    Diverse set of small organic molecules.
    """
    
    def __init__(self, root="./data/ani1x", split="train", transform=None, max_samples=None):
        """
        Args:
            root: Root directory for dataset
            split: train, val, or test
            transform: Optional transform
            max_samples: Maximum samples to load (for testing)
        """
        self.split = split
        self.max_samples = max_samples
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
        return ['ani1x-release.h5']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']
    
    def download(self):
        """
        Download ANI-1x dataset.
        
        Note: Dataset is ~12GB. Download from:
        https://github.com/isayev/ANI1x_datasets
        """
        print("Please download ANI-1x dataset manually from:")
        print("https://github.com/isayev/ANI1x_datasets")
        print(f"Place the file 'ani1x-release.h5' in {self.raw_dir}")
        raise FileNotFoundError("ANI-1x dataset not found. Please download manually.")
    
    def process(self):
        """Process ANI-1x dataset."""
        # Load HDF5 file
        h5_file = os.path.join(self.raw_dir, 'ani1x-release.h5')
        
        if not os.path.exists(h5_file):
            self.download()
            return
        
        data_list = []
        
        with h5py.File(h5_file, 'r') as f:
            for mol_name in f.keys():
                mol_group = f[mol_name]
                
                # Extract data
                coordinates = torch.from_numpy(mol_group['coordinates'][:]).float()
                energies = torch.from_numpy(mol_group['energies'][:]).float()
                species = torch.from_numpy(mol_group['species'][:]).long()
                
                # Create data objects
                for i in range(len(coordinates)):
                    data_obj = Data(
                        pos=coordinates[i],
                        z=species[0],  # Same species for all conformations
                        energy=energies[i].unsqueeze(0)
                    )
                    data_list.append(data_obj)
                    
                    # Limit samples if specified
                    if self.max_samples and len(data_list) >= self.max_samples:
                        break
                
                if self.max_samples and len(data_list) >= self.max_samples:
                    break
        
        # Split: 80% train, 10% val, 10% test
        n = len(data_list)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        
        # Shuffle with fixed seed
        torch.manual_seed(42)
        perm = torch.randperm(n)
        
        train_data = [data_list[i] for i in perm[:n_train]]
        val_data = [data_list[i] for i in perm[n_train:n_train+n_val]]
        test_data = [data_list[i] for i in perm[n_train+n_val:]]
        
        # Save splits
        torch.save(self.collate(train_data), self.processed_paths[0])
        torch.save(self.collate(val_data), self.processed_paths[1])
        torch.save(self.collate(test_data), self.processed_paths[2])
