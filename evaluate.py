import sys
import numpy as np
import h5py
from scipy.spatial.distance import pdist
from sklearn.metrics import average_precision_score


def eval(gt_file, feats):
    with h5py.File(gt_file, "r") as f:
        gt_labels = f["sim"][:].flatten()

    sim = np.float16(pdist(feats))*(-1)
    ap = average_precision_score(gt_labels, sim)
    print("AP: "+str(ap))


if __name__ == "__main__":
    gt_file = sys.argv[1]
    feats_file = sys.argv[2]
    feats = np.load(feats_file)
    eval(gt_file, feats)