# TB-Places
TB-Places is data set of garden images for testing algorithms for visual place recognition. It contains images with ground truth camera pose recorded in two real gardens at different times, with a total of four different recording sessions, with varying light conditions. We also provide ground truth for all possible pairs of images, indicating whether they depict the same place or not. 

The poses of the images are included in the \[datasetname].json files (i.e. W17.json), and come in the format of a 7 feature vector (t_x, t_y, t_z, q_x, q_y, q_z, q_w), with t corresponding to the translation vector and q to the rotation expressed in form of a quaternion.

Here is an example on how to load the poses:

```python3
import json
import numpy as np

with open("W17.json", "r") as f:
    data = json.load(f)
im_paths = data["im_paths"] #list of image paths, length = n
poses = np.array(data["poses"]) #pose matrix of shape = (n, 7)

```

The similarity matrix is provided with the \[datasetname]\_similarity.h5 files (i.e. W17\_similarity.h5). It consists of a condensed binary similarity matrix in a vector form (see [here](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.squareform.html)), where 0 values indicate dissimiliar image pairs and 1 indicate similar scenes. 

Here is an example on how to load a similarity matrix:

```python3
import h5py
from scipy.spatial.distance import squareform

with h5py.File("W17_similarity.h5", "r") as f:
    gt_labels = f["sim"][:].flatten() #similarity labels in condensed form (shape=(1,n * (n-1) / 2))
    gt_labels = squareform(gt_labels) #similarity labels in matrix form (shape=(n, n))

```

| Garden     | Set | # imgs | # similar pairs | % similar pairs |
|------------|:---:|:------:|:---------------:|:---------------:|
| Wageningen | [W16](https://drive.google.com/drive/folders/1OhGArOsgo8T2idGGIWmxHTcML6lnj3jY?usp=sharing)|  40752 |      5.12M      |      0.6168     |
| Wageningen | [W17](https://drive.google.com/drive/folders/1kPHZgqD8akFQNpLgrCnQikOGpp4pSz5n?usp=sharing)|  10948 |       330K      |      0.5441     |
| Renningen  | R17|  7999  |       150K      |      0.4822     |
| Wageningen | [W18](https://drive.google.com/open?id=1OhGArOsgo8T2idGGIWmxHTcML6lnj3jY)|  23043 |       1.03M | 0.4822|


Please cite the following [IEEE Access](https://ieeexplore.ieee.org/document/8698240) or [Computer Analysis of Images and Patterns](https://link.springer.com/chapter/10.1007/978-3-030-29888-3_26) papers if you use the data.


```
 @article{leyvavallina2019tbplaces,
 title={TB-Places: A Data Set for Visual Place Recognition in Garden Environments}, 
 author={Leyva-Vallina, Maria and Strisciuglio, Nicola and López Antequera, Manuel and Tylecek, Radim and Blaich, Michael and Petkov, Nicolai}, 
   journal={IEEE Access}, 
   year={2019},
   publisher={IEEE}
 }
 
 @inproceedings{leyvavallina2019place,
  title={Place recognition in gardens by learning visual representations: data set and benchmark analysis},
  author={Leyva-Vallina, Mar{\'\i}a and Strisciuglio, Nicola and Petkov, Nicolai},
  booktitle={International Conference on Computer Analysis of Images and Patterns},
  pages={324--335},
  year={2019},
  organization={Springer}
}

```
