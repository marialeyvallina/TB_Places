from torch.utils.data import Dataset
import json
import h5py
from scipy.spatial.distance import squareform
import torch
import os
from PIL import Image

class TestDataSet(Dataset):
    def __init__(self, idx_file, transform=None):
        """
        Args:
            idx_file (string): Path to the idx file (.json)
            gt_file (string): Path to the GT file with pairwise similarity (.h5).
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(idx_file) as f:
            data = json.load(f)
            self.root_dir = data["im_prefix"]
            self.im_paths = data["im_paths"]
        self.transform = transform

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        return self.read_image(self.im_paths[idx])

    def read_image(self, impath):
        img_name = os.path.join(self.root_dir,
                                impath)
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image=self.transform(image)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        return image

