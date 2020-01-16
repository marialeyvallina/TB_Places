import sys
from dataset import TestDataSet
import torchvision.transforms as ttf
import torch
from models import BaseNet
from torchvision.models import densenet161, resnet152
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    use_cuda = True

    idx_file = sys.argv[1]
    weight_file = sys.argv[2]
    feature_length = int(sys.argv[3])
    savename = sys.argv[4]
    image_t = ttf.Compose([ttf.Resize(size=224),
                           ttf.ToTensor(),
                           ttf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                           ])

    weights = torch.load(weight_file)
    if "densenet" in weight_file:
        model = BaseNet(densenet161().features,"avg")
    elif "resnet" in weight_file:
        model = BaseNet(resnet152())
    else:
        raise NotImplementedError()
    model.load_state_dict(weights)
    model.eval()
    if use_cuda:
        model = model.cuda()
    ds = TestDataSet(idx_file, image_t)
    batch_size = 8
    dl = DataLoader(ds, batch_size=batch_size, num_workers=4)

    features = torch.zeros((len(ds), feature_length))
    for i, b in tqdm(enumerate(dl)):
        if use_cuda:
            b = b.cuda()
        features[i*batch_size:i*batch_size+batch_size, :] = model.forward(b).squeeze().detach().cpu()
    np.save(savename, features.numpy())