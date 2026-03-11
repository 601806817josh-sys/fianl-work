import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from pathlib import Path


class CTDataset(Dataset):
    def __init__(self, directory_path):
        self.directory_path = Path(directory_path)
        nii_files = list(self.directory_path.rglob("*.nii"))
        nii_gz_files = list(self.directory_path.rglob("*.nii.gz"))
        self.files = sorted(nii_files + nii_gz_files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = nib.load(str(self.files[index]))
        volume = image.get_fdata().astype(np.float32)

        volume_min = np.min(volume)
        volume_max = np.max(volume)

        if volume_max > volume_min:
            volume = (volume - volume_min) / (volume_max - volume_min)
        else:
            volume = np.zeros_like(volume, dtype=np.float32)

        tensor = torch.from_numpy(volume)
        tensor = tensor.unsqueeze(0)

        return tensor
