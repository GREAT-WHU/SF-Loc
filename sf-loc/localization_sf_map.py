import cv2
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import re
import geoFunc.trans as trans
import geoFunc.data_utils as data_utils
import math
import torch
import time
from scipy.spatial import KDTree
import bisect
import gtsam
import gtsam_unstable
from gtsam.symbol_shorthand import B, V, X, L

from sklearn.cluster import DBSCAN
from o3dvis import create_camera_actor,create_point_actor,gen_pcd,vis_disp,vis_setup, vis_run
import open3d as o3d
import tqdm
from scipy.spatial.transform import Rotation
from scipy.interpolate import RegularGridInterpolator

# vis = vis_setup()
# disp0: N * 1
# disp1: N * 1
def align_disp(disp0,disp1):
    scale = 1.0
    for iiter in range(10):
        r = disp0[None].T - disp1[None].T * scale
        # print(np.vstack([disp0,disp1]).T)
        H = disp1[None].T
        if iiter>1:
            mask = np.fabs(r)<0.1
            # print(r.shape)
            r = r[mask][None].T
            H = H[mask][None].T
        x = np.matmul(np.linalg.inv(np.matmul(H.T,H)), np.matmul(H.T,r))
        scale += x[0,0]
    # scale  =1.0
    return scale

def find_local_minima(sequence, N, thr):
    extended_sequence = np.pad(sequence, (N, N), mode='constant', constant_values=np.inf)
    is_minima = np.ones(len(sequence), dtype=bool)
    for offset in range(1, N + 1):
        is_minima &= (sequence < extended_sequence[N - offset:-(N + offset)])
        is_minima &= (sequence < extended_sequence[N + offset:-(N - offset)] if offset != N else sequence < extended_sequence[N + offset:])
    minima_indices = np.where(is_minima & (sequence < thr))[0]
    print(sequence[minima_indices])
    return minima_indices

ENABLE_BATCH = True

ENABLE_GLUE = False
ENABLE_VOTING = False

if ENABLE_GLUE:
    from lightglue import LightGlue, SuperPoint, DISK
    from lightglue.utils import load_image, rbd, load_image_undist, numpy_image_to_torch
    from lightglue import viz2d

    torch.set_grad_enabled(False);

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'mps', 'cpu'
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
    matcher = LightGlue(features='superpoint').eval().to(device)


#TODO