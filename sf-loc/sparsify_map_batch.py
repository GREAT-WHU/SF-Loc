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
    parser.add_argument("--imagedir", type=str, nargs='+',help="path to image directory",default='/home/zhouyuxuan/data/1023_01/image/colmap/recon/cam0')
    parser.add_argument("--imagestamp", type=str,nargs='+', help="",default='/home/zhouyuxuan/data/1023_01/image/colmap/recon/stamp_rearrange_merged.txt')
    parser.add_argument("--depth_video",  type=str,nargs='+',default='results/depth_video.pkl')
    parser.add_argument("--poses_post",  type=str,nargs='+',default='results/result_post.txt')
    parser.add_argument("--map_indices",  type=str,default='results/map_indice_1023n.pkl')
    parser.add_argument("--map_indices_txt",  type=str,default='results/map_stamps.txt')
    parser.add_argument("--calib", type=str, help="",default='calib/1023n.txt')
    parser.add_argument("--threshold", type=float, default=0.4)
    args = parser.parse_args()

    assert(len(args.imagedir) == len(args.imagestamp))
    assert(len(args.imagedir) == len(args.depth_video))
    assert(len(args.imagedir) == len(args.poses_post))

    color_list = ['r','g','b']
    N_seq = len(args.poses_post)
    MAX_RATIO_THRESHOLD = args.threshold - 0.002
    MAX_DISTANCE_DISP = -0.01
    MAX_NEAREST_DISTANCE = 20.0
    ENABLE_PLOT = False
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

    fp_out = open(args.map_indices_txt,'wt')
    pkl_out = open(args.map_indices,'wb')

    ACCUMULATED = 0
    for i_seq in range(N_seq):
        poses_post = np.loadtxt(args.poses_post[i_seq])
        keyframe_path = args.depth_video[i_seq]
        image_data = dataset_utils.ImageDataset(args.imagedir[i_seq],
                                                args.imagestamp[i_seq],
                                                calib[:4],calib[4:8],None,[w1,h1])
        f = open(keyframe_path,'rb')
        dump_data= pickle.load(f)
        Tic = dump_data['Tic']

        # Map data
        if i_seq == 0:
            map_disps = torch.tensor(np.array(list(dump_data['disps'].values())),device='cuda')
            map_poses = torch.zeros(len(list(dump_data['stamps'].keys())), 7, device="cpu", dtype=torch.float)
            map_indice = []
            map_time = []
            map_TTT = []
        else:
            map_disps = torch.concat([map_disps,torch.tensor(np.array(list(dump_data['disps'].values())),device='cuda')])
            map_poses = torch.concat([map_poses.to('cpu'),torch.zeros(len(list(dump_data['stamps'].keys())), 7, device="cpu", dtype=torch.float)])

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
                map_poses[ACCUMULATED + i] = torch.cat([t,q])
                valid_indices.append(i)
        map_poses = map_poses.to('cuda')

        map_Gs = SE3(torch.tensor(map_poses).clone().detach())
        if ENABLE_PLOT:
            plt.figure('111',figsize=[8,8])
            plt.ion()
            plt.grid()
            pts_tri = np.array([[0.0,0,0,1.0],
                                [-1.0,0,1.0,1.0],
                                [ 1.0,0,1.0,1.0],
                                [0.0,0,0,1.0]])
            plt.tight_layout()

        for i in tqdm.tqdm(sorted(valid_indices)):

            # retrieve nearby map frames
            Gs_mat_this = np.linalg.inv(map_Gs[map_indice+[i+ACCUMULATED]].cpu().matrix())
            near_dist = np.linalg.norm(Gs_mat_this[:-1,0:3,3] - Gs_mat_this[-1,0:3,3],axis= 1)
            near_idx = near_dist < MAX_NEAREST_DISTANCE
            if len(map_indice) > 0:
                indice_this = np.array(map_indice)[near_idx]
            near_idx = np.concatenate([near_idx,[True]])
            Gs_this = map_Gs[map_indice+[i+ACCUMULATED]][near_idx]
            disps_this = map_disps[map_indice+[i+ACCUMULATED]][near_idx]

            TTT = np.linalg.inv(Gs_this.cpu().matrix())
            if ENABLE_PLOT:
                pts = (TTT[-1] @ pts_tri.T).T

            # only this frame itself within the range
            if Gs_this.shape[0] ==1 : 
                map_indice.append(i+ACCUMULATED)
                map_time.append(dump_data['stamps'][i])
                fp_out.writelines('%.5f\n'%dump_data['stamps'][i])
                fp_out.flush()
                map_TTT.append(TTT[-1])
                if ENABLE_PLOT:
                    plt.plot(pts[:,0],pts[:,1],c=color_list[i_seq],linewidth=1)
                    plt.xlim([TTT[-1,0,3]-100,TTT[-1,0,3]+100])
                    plt.ylim([TTT[-1,1,3]-100,TTT[-1,1,3]+100])
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
            max_idx = -1
            max_ratio_dist_list = []
            max_idx_list = []
            for idx in range(ii.shape[0]//2):
                mask01 = torch.logical_and(torch.logical_and(
                      torch.logical_and(coords1[0,idx,:,:,0]>0,coords1[0,idx,:,:,0]<wd//8-1),
                      torch.logical_and(coords1[0,idx,:,:,1]>0,coords1[0,idx,:,:,1]<ht//8-1)
                      ), valid_mask[0,idx,:,:,0])
                mask10 = torch.logical_and(torch.logical_and(
                      torch.logical_and(coords1[0,idx+ii.shape[0]//2,:,:,0]>0,coords1[0,idx+ii.shape[0]//2,:,:,0]<wd//8-1),
                      torch.logical_and(coords1[0,idx+ii.shape[0]//2,:,:,1]>0,coords1[0,idx+ii.shape[0]//2,:,:,1]<ht//8-1)
                      ), valid_mask[0,idx+ii.shape[0]//2,:,:,0])
                ratio0 = torch.sum(mask01).item()/(mask01.shape[0]+padding*2)/(mask01.shape[1]+padding*2)
                ratio1 = torch.sum(mask10).item()/(mask01.shape[0]+padding*2)/(mask01.shape[1]+padding*2)

                max_ratio_dist_list.append(min(ratio0,ratio1))
                max_idx_list.append(idx)

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

                if min(ratio0,ratio1) > max_ratio_dist:
                    max_ratio_dist = min(ratio0,ratio1)
                    max_idx = ii[idx]
            
            if max_ratio_dist < MAX_RATIO_THRESHOLD:
                map_indice.append(i+ACCUMULATED)
                map_time.append(dump_data['stamps'][i])
                fp_out.writelines('%.5f %.5f\n'%(dump_data['stamps'][i],max_ratio_dist))
                fp_out.flush()
                map_TTT.append(TTT[-1])
                if ENABLE_PLOT:
                    plt.plot(pts[:,0],pts[:,1],c=color_list[i_seq],linewidth=1)
                    plt.xlim([TTT[-1,0,3]-100,TTT[-1,0,3]+100])
                    plt.ylim([TTT[-1,1,3]-100,TTT[-1,1,3]+100])
                    plt.pause(0.01)
            else:
                # freshness first

                idx_sort = np.argsort(max_ratio_dist_list)
                del_num = 0

                discard = False
                for ii in idx_sort:
                    ratio = max_ratio_dist_list[ii]
                    if max_ratio_dist_list[ii] > MAX_RATIO_THRESHOLD:
                        max_idx = max_idx_list[ii]
                        if  indice_this[max_idx] < ACCUMULATED:
                            123
                        else:
                            discard = True
                if discard: continue

                for ii in idx_sort:
                    ratio = max_ratio_dist_list[ii]
                    if max_ratio_dist_list[ii] > MAX_RATIO_THRESHOLD:
                        max_idx = max_idx_list[ii]
                        if  indice_this[max_idx] < ACCUMULATED:
                            idx_to_delete = np.where(np.array(map_indice) == indice_this[max_idx])[0][0]
                            dT = np.linalg.inv(map_TTT[idx_to_delete]) @ TTT[-1]
                            att  = Rotation.from_matrix(dT[0:3,0:3]).as_euler('zyx')
                            print(att * 57.3)
                            if np.linalg.norm(att)*57.3 > 60.0 : 
                                pass
                            else:
                                fp_out.writelines('-%.5f\n'%map_time[idx_to_delete])
                                if ENABLE_PLOT:
                                    plt.plot(pts[:,0],pts[:,1],c=color_list[i_seq],linewidth=1)
                                    plt.scatter(map_TTT[idx_to_delete][0,3],map_TTT[idx_to_delete][1,3],marker='x',color='black')
                                    plt.pause(0.01)
                                del map_indice[idx_to_delete]
                                del map_time[idx_to_delete]
                                del map_TTT[idx_to_delete]
                                del_num += 1
                                break
                if del_num>0:
                    map_indice.append(i+ACCUMULATED)
                    map_time.append(dump_data['stamps'][i])
                    fp_out.writelines('%.5f\n'%dump_data['stamps'][i])
                    fp_out.flush()
                    map_TTT.append(TTT[-1])
                    if ENABLE_PLOT:
                        plt.plot(pts[:,0],pts[:,1],c=color_list[i_seq],linewidth=1)
                        plt.xlim([TTT[-1,0,3]-100,TTT[-1,0,3]+100])
                        plt.ylim([TTT[-1,1,3]-100,TTT[-1,1,3]+100])
                        plt.pause(0.01)
                            # if False:
                            #     plt.plot(pts[:,0],pts[:,1],c=color_list[i_seq],alpha= 0.2)
        ACCUMULATED = map_disps.shape[0]

    pickle.dump({'indices':map_indice,'time':map_time},pkl_out)
    print(map_indice)
    if ENABLE_PLOT:
        plt.ioff()
        plt.show()