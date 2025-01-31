import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import pickle
import numpy as np
import cv2
import sys
sys.path.append('dbaf')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import tqdm
from scipy.spatial.transform import Rotation
from lietorch import SE3
import geom.projective_ops as pops
import geoFunc.dataset_utils as dataset_utils
import matplotlib
import matplotlib.cm as cm
import argparse
import bisect

torch.set_grad_enabled(False);

def format_indicies(ii, jj):
    """ to device, long, {-1} """

    if not isinstance(ii, torch.Tensor):
        ii = torch.as_tensor(ii)

    if not isinstance(jj, torch.Tensor):
        jj = torch.as_tensor(jj)

    ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
    jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

    return ii, jj

def gen_circle_line(p0,p1):
    pts = []
    center = (p0+p1)/2
    direction = p1 - p0
    r = np.linalg.norm(direction)/2.0
    start_theta = np.arctan2(direction[1],direction[0])
    for i in range(0,181,10):
        theta = start_theta+i/180.0*np.pi
        pts.append(center + r * np.array([np.cos(theta),np.sin(theta)]))
    return np.array(pts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory",default='/home/zhouyuxuan/data/1023_01/image/colmap/recon/cam0')
    parser.add_argument("--imagestamp", type=str, help="",default='/home/zhouyuxuan/data/1023_01/image/colmap/recon/stamp_rearrange_merged.txt')
    parser.add_argument("--depth_video",  type=str,default='results/depth_video.pkl')
    parser.add_argument("--poses_post",  type=str,default='results/result_post.txt')
    parser.add_argument("--map_indices",  type=str,default='results/map_indice_1023n.pkl')
    parser.add_argument("--map_indices_txt",  type=str,default='results/map_stamps.txt')
    parser.add_argument("--calib", type=str, help="",default='calib/1023n.txt')
    parser.add_argument("--threshold", type=float, default=0.4)
    args = parser.parse_args()

    MAX_RATIO_THRESHOLD = args.threshold
    MAX_DISTANCE_DISP = -0.01
    MAX_NEAREST_DISTANCE = 20.0
    ENABLE_PLOT = True
    calib = args.calib

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]
    h0, w0 = (384, 512)
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
    intrinsics = torch.as_tensor([fx, fy, cx, cy ],dtype=torch.float32)
    intrinsics[0::2] *= (w1 / w0)
    intrinsics[1::2] *= (h1 / h0)
    ht = h1
    wd = w1


    poses_post = np.loadtxt(args.poses_post)
    keyframe_path = args.depth_video
    image_data = dataset_utils.ImageDataset(args.imagedir,
                                            args.imagestamp,
                                            calib[:4],calib[4:8],None,[w1,h1])
    fp_out = open(args.map_indices_txt,'wt')
    pkl_out = open(args.map_indices,'wb')

    f = open(keyframe_path,'rb')
    dump_data= pickle.load(f)
    Tic = dump_data['Tic']

    disps = torch.tensor(np.array(list(dump_data['disps'].values())),device='cuda')
    ll = len(list(dump_data['stamps'].keys()))
    poses = torch.zeros(ll, 7, device="cpu", dtype=torch.float)

    valid_indices = []
    for i in sorted(dump_data['stamps'].keys()):
        p = dump_data['cameras'][i]
        idx_post = bisect.bisect(poses_post[:,0], dump_data['stamps'][i].item()-0.001)

        # use post-processing poses
        if poses_post[idx_post,0] == dump_data['stamps'][i].item():
            Twb = np.eye(4)
            Twb[0:3,3] = poses_post[idx_post,1:4]
            Twb[0:3,0:3] = Rotation.from_quat(np.array(poses_post[idx_post,4:8])).as_matrix()
            Twc = Twb @ Tic
            TTT =  torch.tensor(np.linalg.inv(Twc))
            q = torch.tensor(Rotation.from_matrix(TTT[:3, :3]).as_quat())
            t = TTT[:3,3]
            poses[i] = torch.cat([t,q])
            valid_indices.append(i)
    poses = poses.to('cuda')

    Gs = SE3(torch.tensor(poses).clone().detach())
    if ENABLE_PLOT:
        plt.figure('111',figsize=[8,8])
        plt.ion()
        plt.grid()
        pts_tri = np.array([[0.0,0,0,1.0],
                            [-1.0,0,1.0,1.0],
                            [ 1.0,0,1.0,1.0],
                            [0.0,0,0,1.0]])
        plt.tight_layout()

    map_indice = []
    map_time = []
    map_TTT = []

    for i in tqdm.tqdm(sorted(valid_indices)):

        # retrieve nearby map frames
        Gs_mat_this = np.linalg.inv(Gs[map_indice+[i]].cpu().matrix())
        near_dist = np.linalg.norm(Gs_mat_this[:-1,0:3,3] - Gs_mat_this[-1,0:3,3],axis= 1)
        near_idx = near_dist < MAX_NEAREST_DISTANCE
        near_idx = np.concatenate([near_idx,[True]])
        Gs_this = Gs[map_indice+[i]][near_idx]
        disps_this = disps[map_indice+[i]][near_idx]

        TTT = np.linalg.inv(Gs_this.cpu().matrix())
        if ENABLE_PLOT:
            pts = (TTT[-1] @ pts_tri.T).T

        # only this frame itself within the range
        if Gs_this.shape[0] ==1 : 
            map_indice.append(i)
            map_time.append(dump_data['stamps'][i])
            fp_out.writelines('%.5f\n'%dump_data['stamps'][i])
            fp_out.flush()
            map_TTT.append(TTT[-1])
            if ENABLE_PLOT:
                plt.plot(pts[:,0],pts[:,1],c='r',linewidth=5)
                plt.xlim([TTT[-1,0,3]-100,TTT[-1,0,3]+100])
                plt.ylim([TTT[-1,1,3]-100,TTT[-1,1,3]+100])
                if len(map_indice) % 20 == 0:
                    plt.pause(0.01)
            continue
        
        ii, jj = format_indicies(list(range(Gs_this.shape[0]-1)) +  [Gs_this.shape[0]-1]*(Gs_this.shape[0]-1),
                                  [Gs_this.shape[0]-1]*(Gs_this.shape[0]-1) + list(range(Gs_this.shape[0]-1)))

        coords1, valid_mask = pops.projective_transform(Gs_this[None], disps_this[None], 
                                               torch.tensor(np.tile(intrinsics/8,(1,Gs_this.shape[0],1)),device='cuda'), ii, jj)
        coords0 = pops.coords_grid(ht//8, wd//8, device='cuda')
        padding = 0
        if padding > 0:
            coords1 = coords1[:,:,padding:-padding,padding:-padding,:]
            valid_mask = valid_mask[:,:,padding:-padding,padding:-padding,:]
        max_ratio_dist = -1.0

        for idx in range(ii.shape[0]//2):
            # mask01 = torch.logical_and(torch.logical_and(torch.logical_and(
            #       torch.logical_and(coords1[0,idx,:,:,0]>0,coords1[0,idx,:,:,0]<wd//8-1),
            #       torch.logical_and(coords1[0,idx,:,:,1]>0,coords1[0,idx,:,:,1]<ht//8-1)
            #       ), valid_mask[0,idx,:,:,0]),disps_this[ii[idx]]>MAX_DISTANCE_DISP)
            # mask10 = torch.logical_and(torch.logical_and(torch.logical_and(
            #       torch.logical_and(coords1[0,idx+ii.shape[0]//2,:,:,0]>0,coords1[0,idx+ii.shape[0]//2,:,:,0]<wd//8-1),
            #       torch.logical_and(coords1[0,idx+ii.shape[0]//2,:,:,1]>0,coords1[0,idx+ii.shape[0]//2,:,:,1]<ht//8-1)
            #       ), valid_mask[0,idx+ii.shape[0]//2,:,:,0]),disps_this[ii[idx+ii.shape[0]//2]]>MAX_DISTANCE_DISP)
            mask01 = torch.logical_and(torch.logical_and(
                  torch.logical_and(coords1[0,idx,:,:,0]>0,coords1[0,idx,:,:,0]<wd//8-1),
                  torch.logical_and(coords1[0,idx,:,:,1]>0,coords1[0,idx,:,:,1]<ht//8-1)
                  ), valid_mask[0,idx,:,:,0])
            mask10 = torch.logical_and(torch.logical_and(
                  torch.logical_and(coords1[0,idx+ii.shape[0]//2,:,:,0]>0,coords1[0,idx+ii.shape[0]//2,:,:,0]<wd//8-1),
                  torch.logical_and(coords1[0,idx+ii.shape[0]//2,:,:,1]>0,coords1[0,idx+ii.shape[0]//2,:,:,1]<ht//8-1)
                  ), valid_mask[0,idx+ii.shape[0]//2,:,:,0])
            ratio0 = torch.sum(mask01).item()/mask01.shape[0]/mask01.shape[1]
            ratio1 = torch.sum(mask10).item()/mask01.shape[0]/mask01.shape[1]


            if idx == ii.shape[0]//2 - 1 and len(map_TTT)>5:# and np.linalg.norm(TTT[-1,0:3,3] - map_TTT[-1][0:3,3])>30.0:
                save_vis = False
                if save_vis:
                    mm0 = image_data.get_image(dump_data['stamps'][map_indice[-1]].item())
                    mm1 = image_data.get_image(dump_data['stamps'][i].item())
                    mm = np.vstack([mm0,mm1])
                    mask_stacked = np.vstack([mask01.cpu().numpy().astype(np.uint8)*255,mask10.cpu().numpy().astype(np.uint8)*255])
                    mask_stacked = cv2.resize(mask_stacked,[int(mask_stacked.shape[1]*8),int(mask_stacked.shape[0]*8)],interpolation=cv2.INTER_NEAREST)
                    mask_stacked = cv2.cvtColor(mask_stacked,cv2.COLOR_GRAY2BGR)
                    print(mask_stacked.shape)
                    print(mm.shape)
                    mm_out = cv2.addWeighted(mask_stacked,  0.5, mm, 1.0,0)
                    cv2.imshow('mm_out',mm_out)


                    mm_disp = np.vstack([dump_data['disps'][map_indice[-1]],dump_data['disps'][i]])
                    mm_disp = cv2.resize(mm_disp,
                                       [mask_stacked.shape[1],mask_stacked.shape[0]],interpolation=cv2.INTER_NEAREST)
                    normalizer = matplotlib.colors.Normalize(vmin=0.0, vmax=0.2)
                    mapper = cm.ScalarMappable(norm=normalizer,cmap='magma')
                    colormapped_im = (mapper.to_rgba(mm_disp)[:, :, :3] * 255).astype(np.uint8)
                    colormapped_im = cv2.cvtColor(colormapped_im,cv2.COLOR_RGB2BGR)
                    cv2.imshow('mm_disp',colormapped_im)
                    cv2.imwrite("mm_out.png",mm_out)
                    cv2.imwrite("mm_disp.png",colormapped_im)
                    cv2.waitKey(0)

                    # new frame
                    # mm = image_data.get_image(dump_data['stamps'][i].item())
                    # cv2.imshow('mask',np.vstack([mask01.cpu().numpy().astype(np.uint8)*255,mask10.cpu().numpy().astype(np.uint8)*255]))
                    # absflow = (coords1[0,0] - coords0).cpu().numpy()
                    # colormapped_im = mm
                    # for iii in range(0,absflow.shape[0],4):
                    #     for jjj in range(0,absflow.shape[1],4):
                    #         colormapped_im = cv2.line(colormapped_im,  (jjj * 8, iii * 8),
                    #                           (int(round((jjj - absflow[iii, jjj, 0]) * 8)),
                    #                            int(round((iii - absflow[iii, jjj, 1]) * 8))),
                    #                             (255,255,255), 1, cv2.LINE_AA)
                    # cv2.imshow('colormapped_im',colormapped_im)
                    # cv2.waitKey(1)

            if min(ratio0,ratio1) > max_ratio_dist:
                max_ratio_dist = min(ratio0,ratio1)
        # print('ratio: ',dump_data['stamps'][i],max_ratio_dist)
        if max_ratio_dist < MAX_RATIO_THRESHOLD:
            map_indice.append(i)
            map_time.append(dump_data['stamps'][i])
            fp_out.writelines('%.5f\n'%dump_data['stamps'][i])
            fp_out.flush()
            map_TTT.append(TTT[-1])
            if ENABLE_PLOT:
                plt.plot(pts[:,0],pts[:,1],c='r',linewidth=5)
                plt.xlim([TTT[-1,0,3]-100,TTT[-1,0,3]+100])
                plt.ylim([TTT[-1,1,3]-100,TTT[-1,1,3]+100])
                if len(map_indice) % 20 == 0:
                    plt.pause(0.01)
        else:
            if ENABLE_PLOT:
                plt.plot(pts[:,0],pts[:,1],c='r',alpha= 0.2)

    pickle.dump({'indices':map_indice,'time':map_time},pkl_out)
    print(map_indice)
    if ENABLE_PLOT:
        plt.ioff()
        plt.show()