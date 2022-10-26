import os
from tkinter.tix import MAX
import numpy as np
import hloc
import tqdm, tqdm.notebook
import datetime
from pathlib import Path

from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d
from hloc.utils.read_write_model import read_cameras_binary, read_points3D_binary, read_images_binary

from rtvec2extrinsic import *


images = Path('inputs/redwood-livingroom/image/')

multiprocess = True
processTime = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
feature_conf = extract_features.confs['superpoint_inloc'] # ['superpoint_aachen']
matcher_conf = match_features.confs['superglue-fast'] #  'superglue-fast'   'superglue'  'NN-superpoint' 'NN-ratio'  'NN-mutual'

print("=> Reference files")
references = sorted([p.relative_to(images).as_posix() for p in (images).iterdir()])
n_references = len(references)
print("=> ",n_references, "mapping images")

def sfm_reconstruction(block_idx, sid, eid):
    ref_images = sorted([p.relative_to(images).as_posix() for p in (images).iterdir()])[sid:eid]
    
    outputs_path = 'outputs/{}/{}'.format(processTime, block_idx)
    outputs = Path(outputs_path)
    os.system("rm -rf {}".format(outputs_path))
    sfm_pairs = outputs / 'pairs-sfm.txt'
    # loc_pairs = outputs / 'pairs-loc.txt'
    sfm_dir = outputs / 'sfm'
    features = outputs / 'features.h5'
    matches = outputs / 'matches.h5'

    print("=> Match features")
    extract_features.main(feature_conf, images, image_list=ref_images, feature_path=features)
    pairs_from_exhaustive.main(sfm_pairs, image_list=ref_images)
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

    print("=> Generate model")
    reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, image_list=ref_images)

def batch_process(blocks):
    print("=>",blocks)
    for block_idx in blocks:
        if block_idx == -1: break
        sid = block_idx*block_size
        eid = min((block_idx+1)*block_size+extend_frames, n_references)
        sfm_reconstruction(block_idx, sid, eid)
        
block_size = 120
extend_frames = 30 # frames must >= 2        
if multiprocess:
    import math
    import multiprocessing
    
    MAX_THREAD = min(multiprocessing.cpu_count()-1, 7)
    n_blocks = math.ceil(n_references/block_size)
    print("=> MAX THREAD {}// Total {} blocks {} per block".format(MAX_THREAD,n_blocks,block_size))
    thread_distribution = np.array([idx for idx in range(n_blocks)] + \
        [ -1 for _ in range(math.ceil(n_blocks/MAX_THREAD)*MAX_THREAD-n_blocks)]).reshape(-1, MAX_THREAD).T
    print("=> Thread distribution:\n", thread_distribution)
    import torch.multiprocessing as mp
    # 注意：这是 "fork" 方法工作所必需的
    processes = []
    
    for idx in range(MAX_THREAD):
        # print(thread_distribution[idx])
        p = mp.Process(target=batch_process, args=(thread_distribution[idx], ))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
else:
    for idx in range(n_references//block_size):
        sid = idx*block_size
        eid = min((idx+1)*block_size+extend_frames, n_references)
        sfm_reconstruction(0, 0, n_references[sid:eid])

# if model != None:

#     image_source_list = ""
#     traj_list = ""

#     os.system("rm -rf {}traj.log".format(outputs_path))
#     os.system("rm -rf {}image_source.txt".format(outputs_path))

#     os.system("touch {}traj.log".format(outputs_path))
#     os.system("touch {}image_source.txt".format(outputs_path))

    
#     cam = read_cameras_binary(sfm_dir/'cameras.bin')
#     for c in cam:
#         print(cam[c])
#     print("="*50)
#     img = read_images_binary(sfm_dir/'images.bin')
#     for i in img:
#         print(img[i])
#     print("="*50)
#     points = read_points3D_binary(sfm_dir/'points3D.bin')
#     for p in points:
#         print(points[p])
#     print("="*50)
    
    
#     count = 1
#     result_list = sorted([i for i in model.images])
#     for i in result_list:
#         print("IMG: {}".format(model.images[i].name))
#         # qw qx qy qz tx ty tz
#         vec = np.append(model.images[i].qvec,model.images[i].tvec / 50)
#         print(vec)
#         T = rtvec2matrix(*vec)
#         traj = "{0} {0} {1}\n{2}\n".format(count-1,count,'\n'.join([' '.join([str(e) for e in row]) for row in T ]))
#         print(traj)
#         traj_list += traj

#         image_name = "{}\n".format(model.images[i].name)
#         image_source_list += image_name

#         count += 1

#     with open("{}traj.log".format(outputs_path), 'w') as f:
#         f.write(traj_list)

#     with open("{}image_source.txt".format(outputs_path), 'w') as f:
#         f.write(image_source_list)
    

    

