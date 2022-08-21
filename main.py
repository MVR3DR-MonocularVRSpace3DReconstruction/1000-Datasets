import os
import numpy as np
import hloc
import tqdm, tqdm.notebook
tqdm.tqdm = tqdm.notebook.tqdm  # notebook-friendly progress bars
from pathlib import Path

from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d

from rtvec2extrinsic import *


images = Path('datasets/redwood-livingroom/image/')
outputs_path = 'outputs/test/'
outputs = Path(outputs_path)
os.system("rm -rf {}".format(outputs_path))
sfm_pairs = outputs / 'pairs-sfm.txt'
loc_pairs = outputs / 'pairs-loc.txt'
sfm_dir = outputs / 'sfm'
features = outputs / 'features.h5'
matches = outputs / 'matches.h5'

feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['superglue']

print("=> Reference files")
references = sorted([p.relative_to(images).as_posix() for p in (images).iterdir()])
references = [references[p] for p in [0, 50, 100, 150, 200]]
print(len(references), "mapping images")
# plot_images([read_image(images / r) for r in references[:5]], dpi=50)

print("=> Match features")
extract_features.main(feature_conf, images, image_list=references, feature_path=features)
pairs_from_exhaustive.main(sfm_pairs, image_list=references)
match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

print("=> Generate model")
model = reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, image_list=references)

if model != None:

    result_list = sorted([i for i in model.images])
    for i in result_list:
        print("IMG: {}".format(model.images[i].name))
        # qw qx qy qz tx ty tz
        vec = np.append(model.images[i].qvec,model.images[i].tvec)
        print(vec)
        T = rtvec2matrix(vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6])
        print(T)