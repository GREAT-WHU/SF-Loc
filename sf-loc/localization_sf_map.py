import sys
sys.path.append('sf-loc/VPR-methods-evaluation')
sys.path.append('sf-loc/VPR-methods-evaluation/third_party/deep-image-retrieval')
import cv2
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import geoFunc.trans as trans
import geoFunc.data_utils as data_utils
import torch
import time
from scipy.spatial import KDTree
import bisect
import gtsam
import gtsam_unstable
from gtsam.symbol_shorthand import B, V, X, L
from sklearn.cluster import DBSCAN
from o3dvis import create_camera_actor,create_point_actor,gen_pcd,vis_disp,vis_setup, vis_run
import tqdm
from scipy.spatial.transform import Rotation
import vpr_models
import torchvision.transforms as transforms
from PIL import Image
from lietorch import SE3
import argparse

def find_local_minima(sequence, N, thr):
    extended_sequence = np.pad(sequence, (N, N), mode='constant', constant_values=np.inf)
    is_minima = np.ones(len(sequence), dtype=bool)
    for offset in range(1, N + 1):
        is_minima &= (sequence < extended_sequence[N - offset:-(N + offset)])
        is_minima &= (sequence < extended_sequence[N + offset:-(N - offset)] if offset != N else sequence < extended_sequence[N + offset:])
    minima_indices = np.where(is_minima & (sequence < thr))[0]
    print(sequence[minima_indices])
    return minima_indices

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory",default='/mnt/e/WHU1023/WHU0412/image_undist/cam0')
    parser.add_argument("--calib", type=str, help="",default='calib/0412.txt')
    parser.add_argument("--map_file", type=str, help="",default='sf_map_NE50_384_384_0_4.pkl')
    parser.add_argument("--enable_fine_localization", type=bool, help="", default=True)
    parser.add_argument("--enable_user_gt", type=bool, help="", default=True)
    parser.add_argument("--enable_map_gt", type=bool, help="", default=True)
    parser.add_argument("--force_integer_sec", type=bool, help="", default=True)
    parser.add_argument("--multiframe_vpr_mode", type=str, help="",default='SSS',choices=["SSS", "VOTING"])
    parser.add_argument("--vpr_extra_framenum", type=str, help="",default='10',choices=["0", "5", "10"])
    parser.add_argument("--fine_localization_mode", type=str, help="",default='FGO',choices=["FGO", "PNP"])
    parser.add_argument("--fine_localization_windowsize", type=int, default=10, help="Only valid when using FGO.")
    parser.add_argument("--show_vpr_heat", action="store_true", help="", default=False)
    parser.add_argument("--show_snapshot", action="store_true", help="", default=False)
    parser.add_argument("--map_gt_file", type=str, help="",default='/mnt/e/WHU1023/WHU1023/gt.txt')
    parser.add_argument("--user_gt_file", type=str, help="",default='/mnt/e/WHU1023/WHU0412/gt.txt')
    parser.add_argument("--user_odo_file", type=str, help="",default='/mnt/e/WHU1023/WHU0412/odo.txt')
    parser.add_argument("--map_extrinsic", type=str, help="",default='calib/1023.yaml')
    parser.add_argument("--user_extrinsic", type=str, help="",default='calib/0412.yaml')
    parser.add_argument("--result_coarse", type=str, help="",default='results/result_coarse.txt')
    parser.add_argument("--result_fine", type=str, help="",default='results/result_fine.txt')
    parser.add_argument("--use_upper_mask", type=bool, help="",default=True)
    args = parser.parse_args()

    ENABLE_FINE_LOCALIZATION = args.enable_fine_localization
    ENABLE_MAP_GT = args.enable_map_gt     # If True, use the ground-truth map frame poses *WHEN* outputting the localization results.
                                           # The ground-truth file is needed.
    ENABLE_USER_GT = args.enable_user_gt
    SHOW_VPR_HEAT = args.show_vpr_heat
    SHOW_SNAPSHOT = args.show_snapshot
    MULTIFRAME_VPR_MODE = args.multiframe_vpr_mode
    VPR_EXTRA_FRAMENUM = args.vpr_extra_framenum
    FINE_LOCALIZATION_WINDOWSIZE = args.fine_localization_windowsize
    FINE_LOCALIZATION_MODE = args.fine_localization_mode
    MAP_GT_FILE =   args.map_gt_file
    USER_GT_FILE =  args.user_gt_file
    USER_ODO_FILE = args.user_odo_file

    print('Loading dataset... ',time.time())

    #! load map
    all_dd = pickle.load(open(args.map_file,'rb'))
    all_T = SE3(torch.tensor(all_dd['poses'])).matrix().numpy()
    all_tt = np.array(all_dd['tstamps'])

    #! build KDTree
    kdtree = KDTree(all_T[:,0:3,3])
    
    #! dataset (map)
    fs = cv2.FileStorage(args.map_extrinsic, cv2.FILE_STORAGE_READ)
    image_dataset1 = data_utils.ImageDataset(None,None,all_dd['calib'][:4],all_dd['calib'][4:],Tic = fs.getNode('i_T_c').mat())
    for i in range(len(all_tt)):
        image_dataset1.all_data_global[all_tt[i]] = {'T':all_T[i]}
    image_dataset1.all_data_global_keys = np.array(sorted(image_dataset1.all_data_global.keys()))
    image_dataset1.ref_xyz = all_dd['xyz_ref']


    #! dataset (query)
    fs = cv2.FileStorage(args.user_extrinsic, cv2.FILE_STORAGE_READ)
    calib = np.loadtxt(args.calib)
    image_dataset0 = data_utils.ImageDataset(args.imagedir,None,calib[:4],calib[4:],Tic = fs.getNode('i_T_c').mat())
    image_dataset0.load_odo(USER_ODO_FILE,np.eye(4,4))


    # During localization, the poses in the SF map are used.
    # If ENABLE_MAP_GT is turned on, the outputted pose results will be 
    # ``the ground-truth map pose + user-to-map relative pose'', thus to
    # evaluate the localization results in a relative way.
    if ENABLE_MAP_GT:
        image_dataset1.load_gt(MAP_GT_FILE,fs.getNode('ref_T_i').mat(),image_dataset1.ref_xyz)

    # This is used for pose error evaluation during processing.
    if ENABLE_USER_GT:
        image_dataset0.load_gt(USER_GT_FILE,fs.getNode('ref_T_i').mat(),image_dataset1.ref_xyz)

    print('Loaded dataset. ',time.time())

    #! preparing models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'mps', 'cpu'

    model = vpr_models.get_model(all_dd['vpr_model']['method'], all_dd['vpr_model']['backbone'], all_dd['vpr_model']['descriptors_dimension'])
    model = model.eval().to(device)
    transformations = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(size=all_dd['vpr_model']['image_size'], antialias=True)
    ]
    custom_transform = transforms.Compose(transformations)
    if ENABLE_FINE_LOCALIZATION:
        from lightglue import LightGlue, SuperPoint, DISK
        from lightglue.utils import load_image, rbd, load_image_undist, numpy_image_to_torch
        from lightglue import viz2d
        torch.set_grad_enabled(False);
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
        matcher = LightGlue(features='superpoint').eval().to(device)

    if SHOW_VPR_HEAT:
        plt.ion()

    #! for results
    fp_fine = open(args.result_fine,'wt')
    fp_coarse = open(args.result_coarse,'wt')

    if VPR_EXTRA_FRAMENUM == '5':
        multi_frame_win = [-1,-2,-3,-4,-5]
    elif VPR_EXTRA_FRAMENUM == '10':
        multi_frame_win = [-2,-4,-6,-8,-10]

    #! history states
    # These states should have the same sizes.
    history_Twk        = []
    history_distance   = []
    history_time_query = []
    history_time_db    = []
    history_x_series = []
    history_y_series = []
    history_c_series = []
    history_best_index      = []
    history_disps  = []
    history_pts0 = []; history_image0 = []
    history_pts1 = []; history_image1 = []

    is_first = True
    query_filelist = sorted(os.listdir(args.imagedir))
    with torch.inference_mode():
        for ii in tqdm.tqdm(range(len(query_filelist))):
            tt_query = float(query_filelist[ii].split('.')[0])/1e9
            if args.force_integer_sec:
                if int(round(tt_query*10))%10 != 0: continue
            
            #! VPR descriptor & similarity computation
            mm = image_dataset0.get_image(tt_query)
            mmrgb = cv2.cvtColor(mm, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(mmrgb)
            normalized_img = custom_transform(pil_img)[None]
            query_desc = model(normalized_img.to(device)).cpu().numpy()
            distance_this = np.linalg.norm(all_dd['descriptor']-query_desc,axis=1)

            t_series=[]; x_series=[]; y_series=[]; c_series=[]
            Twk = image_dataset0.get_pose_odo(float(query_filelist[ii].split('.')[0])/1e9)

            if len(history_Twk) > 0:
                Tkk_1 = np.linalg.inv(Twk) @ history_Twk[-1]

                #! Skip frames
                if np.linalg.norm(Tkk_1[0:3,3])< 2.0:  # small motion
                    history_Twk.pop(); history_distance.pop(); 
                    history_time_query.pop(); history_time_db.pop()
                    history_x_series.pop(); history_y_series.pop(); history_c_series.pop()
                    history_best_index.pop()
                    if ENABLE_FINE_LOCALIZATION:
                        history_pts0.pop(); history_image0.pop()
                        history_pts1.pop(); history_image1.pop()
                        history_disps.pop()
                
                #! Multi-frame VPR (SSS)
                if MULTIFRAME_VPR_MODE == 'SSS' and VPR_EXTRA_FRAMENUM != '0' and len(history_Twk) > 10:
                    particles = [Twk, Twk @ trans.att2m_4x4([0,0,30.0/57.3]),Twk @ trans.att2m_4x4([0,0,-30.0/57.3])]
                    distance_before_list = [[] for _ in range(len(particles))]
                    for jjj in range(len(particles)):
                        distance_before_list[jjj].append(distance_this)
                        for jj in multi_frame_win:
                            Tkk_1 = np.linalg.inv(particles[jjj]) @ history_Twk[jj]
                            all_Tk_1 = np.matmul(all_T,Tkk_1)
                            KNN_K = 3
                            kd_dist, kd_idx = kdtree.query(all_Tk_1[:,0:3,3],k=KNN_K)
                            r0 = Rotation.from_matrix(all_Tk_1[:,:3,:3]).as_rotvec()
                            r1 = Rotation.from_matrix(all_T[kd_idx,:3,:3].reshape(-1,3,3)).as_rotvec().reshape(-1,KNN_K,3)
                            r0 = np.repeat(r0,KNN_K,axis=0).reshape(-1,KNN_K,3)
                            rotation_mask = np.fabs(np.fmod((r1-r0)[:,:,2]*57.3 + 540,360)-180)>30.0
                            kd_dist[rotation_mask] = 1000.0
                            ss = np.argsort(kd_dist,axis=1)
                            kd_idx = np.take_along_axis(kd_idx,ss,axis=1)
                            distance_KNN = history_distance[jj][kd_idx]
                            distance_before_list[jjj].append(distance_KNN[:,0])
                        distance_before_list[jjj] = np.array(distance_before_list[jjj])
                    distance_before_list = np.array(distance_before_list)
                    distance = np.min(np.linalg.norm(distance_before_list,axis=1),axis=0)
                    arg_distance = np.argmin(np.linalg.norm(distance_before_list,axis=1),axis=0)
                else:
                    distance = distance_this
            else:
                distance = distance_this
            c_series = distance
            t_series = np.array(all_tt)
            x_series = np.array(all_T[:,0,3])
            y_series = np.array(all_T[:,1,3])
            c_series = np.array(c_series)
            best_index = np.argmin(c_series)
            tt_map = t_series[best_index]

            #! Multi-frame VPR (voting)
            if MULTIFRAME_VPR_MODE == 'VOTING' and VPR_EXTRA_FRAMENUM != '0' and len(history_Twk) > 10:
                TOP_N = 10
                x_all = []
                y_all = []
                sort_idx = np.argsort(c_series)
                x = x_series[sort_idx[:TOP_N]];x_all.append(x)
                y = y_series[sort_idx[:TOP_N]];y_all.append(y)
                for jj in multi_frame_win:
                    sort_idx = np.argsort(history_c_series[jj])
                    x = history_x_series[jj][sort_idx[:TOP_N]];x_all.append(x)
                    y = history_y_series[jj][sort_idx[:TOP_N]];y_all.append(y)
                x_all = np.concatenate(x_all)
                y_all = np.concatenate(y_all)
                xy_all =np.vstack([x_all,y_all]).T
                y_pred = DBSCAN(eps = 30.0).fit_predict(xy_all)
                try:
                    mmax = np.max(y_pred)
                    mcount = np.zeros(mmax+1)
                    for iiiii in range(mmax):
                        mcount[iiiii] = np.sum(y_pred==iiiii)
                    cccc = np.argmax(mcount)
                    # best_indice = np.argsort(c_series)[np.where(y_pred[:TOP_N]==cccc)[0]]
                    sort_idx = np.argsort(c_series)
                    best_index = sort_idx[np.where(y_pred[:TOP_N]==cccc)[0][0]]
                    tt_map = t_series[best_index]
                except:
                    best_index = np.argmin(c_series)
                    tt_map = t_series[best_index]

            history_Twk.append(Twk)
            history_distance.append(distance_this)
            history_x_series.append(x_series)
            history_y_series.append(y_series)
            history_c_series.append(c_series)
            history_best_index.append(best_index)
            history_time_query.append(tt_query)
            history_time_db.append(tt_map)
            if ENABLE_FINE_LOCALIZATION:
                #! Feature matching
                # C, H, W
                image0 = numpy_image_to_torch(image_dataset0.get_image(tt_query)[:,:,::-1])
                image1 = numpy_image_to_torch(cv2.imdecode(all_dd['image'][best_index], cv2.IMREAD_COLOR)[:,:,::-1])
                feats0 = extractor.extract(image0.to(device))
                feats1 = extractor.extract(image1.to(device))
                matches01 = matcher({'image0': feats0, 'image1': feats1})
                feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
                kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']
                m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

                #! 3D information
                pts0 = m_kpts0.cpu().numpy()
                pts1 = m_kpts1.cpu().numpy()
                mm = cv2.imdecode(all_dd['disps'][best_index], -1)
                W1 = image0.shape[2]
                H1 = image0.shape[1]
                disp = cv2.resize(1/(mm/100.0),[W1,H1]) 

                if args.use_upper_mask:
                    mask_up = pts0[:,1] < H1 // 2.0
                    pts0 = pts0[mask_up]
                    pts1 = pts1[mask_up]

                #! Notice that we use the camera model of the user
                pts1_raw = np.copy(pts1)
                pts1 = ((image_dataset0.K @ np.linalg.inv(image_dataset1.K) @ np.hstack([pts1,np.ones([pts1.shape[0],1])]).T).T)[:,:2]
                pts1_int = pts1_raw.astype(np.int32)
                # pts1_int = pts1.astype(np.int32)
                disp_values = disp[pts1_int[:, 1]+4, pts1_int[:, 0]+4]

                history_pts0.append(pts0)
                history_pts1.append(pts1)
                history_disps.append(disp_values)
                history_image0.append(image0)
                history_image1.append(image1)
                
                if FINE_LOCALIZATION_MODE == 'PNP':
                    x = (pts1[:,0]-image_dataset0.intrinsics[2])/image_dataset0.intrinsics[0]
                    y = (pts1[:,1]-image_dataset0.intrinsics[3])/image_dataset0.intrinsics[1]
                    XX = x / disp_values
                    YY = y / disp_values
                    ZZ = 1.0 / disp_values
                    XYZ = np.vstack([XX,YY,ZZ]).T
                    try:
                        _, rr, tt, inliers= cv2.solvePnPRansac(XYZ,pts0.astype(np.float32),image_dataset0.K,np.zeros(4))
                    except:
                        rr = np.zeros(3)
                        tt = np.zeros(3)
                    RR, _ = cv2.Rodrigues(rr)
                    TT = np.eye(4,4);TT[0:3,0:3] = RR; TT[0:3,3] = (tt.T)[0]
                    TT = np.linalg.inv(TT)
                    T_cv = TT; R = TT[0:3,0:3]; t = TT[0:3,3][None].T
                elif FINE_LOCALIZATION_MODE == 'FGO':
                    t0 = time.time()
                    initial = gtsam.Values()
                    gcam = gtsam.Cal3_S2(image_dataset0.intrinsics[0],image_dataset0.intrinsics[1],0,
                                         image_dataset0.intrinsics[2],image_dataset0.intrinsics[3])
                    graph = gtsam.NonlinearFactorGraph()
                    iwin = len(history_pts0) - 1
                    for iwin in range(max(len(history_pts0) - FINE_LOCALIZATION_WINDOWSIZE,0),len(history_pts0)):
                        Twc1 = image_dataset1.get_camera_pose_global(history_time_db[iwin])
                        initial.insert(X(iwin*100000 + 0), gtsam.Pose3(Twc1))
                        fixpose_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.000001)
                        f_p = gtsam.PriorFactorPose3(X(iwin*100000 + 0), gtsam.Pose3(Twc1), fixpose_noise)
                        graph.push_back(f_p)
                        initial.insert(X(iwin*100000 + 1), gtsam.Pose3(Twc1))
                        for iii in range(history_pts1[iwin].shape[0]):
                            noise = gtsam.noiseModel.Robust.Create(\
                                                      gtsam.noiseModel.mEstimator.Cauchy(1.0),\
                                          gtsam.noiseModel.Diagonal.Sigmas(np.array([2.0,2.0])))
                            f = gtsam_unstable.AnchoredFixedInvDepthFactor(history_pts1[iwin][iii].astype(np.float64),
                                                                      history_pts0[iwin][iii].astype(np.float64),
                                                                      history_disps[iwin][iii],
                                                                      noise,
                                                                      X(iwin*100000 + 0),
                                                                      X(iwin*100000 + 1),gcam)
                            graph.push_back(f)
                    for iwin in range(max(len(history_pts0) - FINE_LOCALIZATION_WINDOWSIZE,0),len(history_pts0)):
                        if iwin < len(history_pts0)-1:
                            Twck_1 = image_dataset0.get_camera_pose_odo(history_time_query[iwin])
                            Twck = image_dataset0.get_camera_pose_odo(history_time_query[iwin+1])
                            dT11 = np.linalg.inv(Twck_1) @  Twck
                            fixpose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.00001,0.00001,0.00001,0.001,0.05,0.05]))
                            f_odo = gtsam.BetweenFactorPose3(X(iwin*100000 + 1), X((iwin+1)*100000 + 1), gtsam.Pose3(dT11), fixpose_noise)
                            graph.push_back(f_odo)
                    try:
                        params = gtsam.LevenbergMarquardtParams()
                        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
                        params.setVerbosityLM("SUMMARY")
                        cur_result = optimizer.optimize()
                    except Exception as e:
                        cur_result = initial
                    # print('Factor graph error: %.10f => %.10f' % (graph.error(initial),graph.error(cur_result)))
                    iwin = len(history_pts0) - 1
                    T1 = cur_result.atPose3(X(iwin*100000 + 0)).matrix()
                    T0 = cur_result.atPose3(X(iwin*100000 + 1)).matrix()
                    t1 = time.time()
                    TT = np.linalg.inv(T1) @ T0 # T^db_query
                    T_cv = TT; R = TT[0:3,0:3]; t = TT[0:3,3][None].T

                #! Output results
                if ENABLE_MAP_GT:
                    Twc0_est = image_dataset1.get_camera_pose_gt(tt_map) @ T_cv
                    Twc1 = image_dataset1.get_camera_pose_gt(tt_map)
                else:
                    Twc0_est = image_dataset1.get_camera_pose_global(tt_map) @ T_cv
                    Twc1 = image_dataset1.get_camera_pose_global(tt_map)

                 #! Evaluate localization error
                if ENABLE_USER_GT:
                    Twc0 = image_dataset0.get_camera_pose_gt(tt_query)
                    Tc1c0 = np.linalg.inv(Twc1) @ Twc0
                    Ti1i0 = image_dataset1.Tic @ Tc1c0 @ np.linalg.inv(image_dataset0.Tic)
                    Ti1i0_cv = image_dataset1.Tic @ T_cv @ np.linalg.inv(image_dataset0.Tic)
                    print("ground-truth: ", Ti1i0[0:3,3])
                    print("estimated: ", Ti1i0_cv[0:3,3])
                    Terr = np.linalg.inv(Ti1i0) @ Ti1i0_cv
                    Terr = image_dataset0.Tic @ np.linalg.inv(Twc0) @ Twc0_est @ np.linalg.inv(image_dataset0.Tic)
                    err_att = np.array(trans.m2att(Terr[0:3,0:3])) * 57.3
                    # fp.writelines('%.3f %.3f %.3f %.3f %.3f %.3f %.3f\n'%(tt_query,Terr[0,3],Terr[1,3],Terr[2,3],err_att[0],err_att[1],err_att[2]))

                T_est = Twc0_est @ np.linalg.inv(image_dataset0.Tic)
                xyz = T_est[0:3,3]
                att = np.array(trans.m2att(T_est[0:3,0:3])) * 57.3
                xyz_ecef = image_dataset1.ref_xyz + np.array(trans.enu2cart(image_dataset1.ref_xyz,xyz))
                fp_fine.writelines('%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n'%(tt_query,xyz[0],xyz[1],xyz[2],
                                                                                              att[0],att[1],att[2],
                                                                                              xyz_ecef[0],xyz_ecef[1],xyz_ecef[2]))
                fp_fine.flush()

            if ENABLE_USER_GT:
                x_query = image_dataset0.get_pose_gt(tt_query)[0,3]
                y_query = image_dataset0.get_pose_gt(tt_query)[1,3]
                query_distance =np.linalg.norm(np.array([x_query,y_query]) - np.array([x_series[best_index],y_series[best_index]]))
                print("%f Query distance: %.5f" %(tt_query, query_distance))

            
            xyz = all_T[best_index][0:3,3]
            xyz_ecef = image_dataset1.ref_xyz + np.array(trans.enu2cart(image_dataset1.ref_xyz,xyz))
            fp_coarse.writelines('%.5f %.10f %.10f %.10f %d %.5f\n'%(tt_query,xyz_ecef[0],xyz_ecef[1],xyz_ecef[2],best_index,t_series[best_index]))
            fp_coarse.flush()

            #! show vpr heat results
            if SHOW_VPR_HEAT:
                plt.cla()
                plt.scatter(x_series[best_index],y_series[best_index],c='none',s=200.0,edgecolors='r',marker='o')
                plt.scatter(x_query,y_query,c='none',s=200.0,edgecolors='b',marker='d')
                plt.scatter(x_series,y_series,c=c_series,cmap='jet',s=1.0)
                c_min = np.min(c_series); c_max = np.max(c_series)
                plt.clim(c_min-0.1*(c_max-c_min),c_max+0.1*(c_max-c_min))
                minima_indices = find_local_minima(c_series,10,c_min+0.15*(c_max-c_min))
                plt.scatter(x_series[minima_indices],y_series[minima_indices],c=c_series[minima_indices],cmap='jet',s=20.0)
                plt.clim(c_min-0.1*(c_max-c_min),c_max+0.1*(c_max-c_min))
                if is_first:
                    plt.colorbar()
                    is_first = False
                plt.pause(0.01)
            #! show point-wise snapshot
            if SHOW_SNAPSHOT and ENABLE_FINE_LOCALIZATION:
                if False:
                    plt.figure('snapshot')
                    ax = plt.gca()
                    ax.set_aspect(1)
                    pts_tri = np.array([[0,0,0,1],
                                        [1,0,1.3,1],
                                        [-1,0,1.3,1],
                                        [0,0,0,1]])
                    ref_T1_inv = None
                    x_ref = None
                    y_ref = None
                    for iwin in range(max(len(history_pts0) - FINE_LOCALIZATION_WINDOWSIZE,0),len(history_pts0)):
                        T1 = cur_result.atPose3(X(iwin*100000 + 0)).matrix()
                        T0 = cur_result.atPose3(X(iwin*100000 + 1)).matrix()
                        if ref_T1_inv is None:
                            ref_T1_inv = np.linalg.inv(T1 @ np.linalg.inv(image_dataset1.Tic))
                        Twc1 = image_dataset1.get_camera_pose(history_time_db[iwin])
                        Twc0 = image_dataset0.get_camera_pose(history_time_query[iwin])
                        Tc1c0 = np.linalg.inv(Twc1) @ Twc0
                        T0_gt = T1 @ Tc1c0
                        for iii in range(history_pts1[iwin].shape[0]):
                            f = gtsam_unstable.AnchoredInvDepthFactor(history_pts1[iwin][iii].astype(np.float64),
                                                                      history_pts0[iwin][iii].astype(np.float64),
                                                                      noise,
                                                                      X(iwin*100000 + 0),
                                                                      X(iwin*100000 + 1),L(iwin*100000 + iii),gcam)
                            err = f.evaluateError(cur_result.atPose3(X(iwin*100000 + 0)),cur_result.atPose3(X(iwin*100000 + 1)),history_disps[iwin][iii])
                            pt = np.hstack([gcam.calibrate(history_pts1[iwin][iii].astype(np.float64)),[1.0]]) / history_disps[iwin][iii]
                            pt = np.hstack([pt,[1.0]])
                            plt.plot([(ref_T1_inv@T1@pt)[1],(ref_T1_inv@T1)[1,3]],[(ref_T1_inv@T1@pt)[0],(ref_T1_inv@T1)[0,3]],c=[0.7,0.7,0.7],linewidth = 0.5,alpha = 0.2)
                            plt.plot([(ref_T1_inv@T1@pt)[1],(ref_T1_inv@T0)[1,3]],[(ref_T1_inv@T1@pt)[0],(ref_T1_inv@T0)[0,3]],c=[0.7,0.7,0.7],linewidth = 0.5,alpha = 0.2)
                            if np.linalg.norm(err)<5:
                                plt.scatter((ref_T1_inv@T1@pt)[1],(ref_T1_inv@T1@pt)[0],  c='green')
                            else:
                                plt.scatter((ref_T1_inv@T1@pt)[1],(ref_T1_inv@T1@pt)[0],  c='red')
                        plt.plot(((ref_T1_inv@T0@pts_tri.T).T)[:,1]     ,((ref_T1_inv@T0@pts_tri.T).T)[:,0]     ,  c='orange',linewidth= 1,zorder=10000)
                        plt.plot(((ref_T1_inv@T0_gt@pts_tri.T).T)[:,1]  ,((ref_T1_inv@T0_gt@pts_tri.T).T)[:,0]  ,  c='r',linewidth= 1,zorder=10000)
                        plt.plot(((ref_T1_inv@T1@pts_tri.T).T)[:,1]  ,((ref_T1_inv@T1@pts_tri.T).T)[:,0]  ,  c='b',linewidth= 1,zorder=10000)
                        x_ref = ((ref_T1_inv@T0_gt@pts_tri.T).T)[0,1]
                        y_ref = ((ref_T1_inv@T0_gt@pts_tri.T).T)[0,0]
                    plt.gca().invert_yaxis()
                    plt.ylim([30+y_ref,-30+y_ref])
                    plt.xlim([-20+x_ref, 80+x_ref])
                    ax = plt.gca()
                    ax.spines['left'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    plt.tick_params(left = False, right = False , labelleft = False , 
                        labelbottom = False, bottom = False) 
                    plt.tight_layout()
                    plt.savefig('temp/%d_%d_bev_match.png'%(int(round(tt_query*1e6)*1e3),FINE_LOCALIZATION_WINDOWSIZE),dpi=600)
                    for iwin in range(max(len(history_pts0) - FINE_LOCALIZATION_WINDOWSIZE,0),len(history_pts0)):
                        m_kpts0 = history_pts0[iwin].astype(np.float64)
                        m_kpts1 = history_pts1[iwin].astype(np.float64)
                        image0 = history_image0[iwin]
                        image1 = history_image1[iwin]
                        if image0 is None:
                            image0 = numpy_image_to_torch(image_dataset0.get_image(history_time_query[iwin])[:,:,::-1])
                            image1 = numpy_image_to_torch(cv2.imdecode(all_dd['image'][history_best_index[iwin]], cv2.IMREAD_COLOR)[:,:,::-1])
                        idx = bisect.bisect(stamp_list,history_time_db[iwin]-0.01)
                        disp = cv2.resize(dump_data['disps'][idx],[W1,H1]) # Attention!!!!!
                        disp = 1/(np.round((1/disp)*100)/100)
                        disp_show = vis_disp(disp,1,False,str(iwin),True)
                        axes = viz2d.plot_images([disp_show[:,:,::-1], image0])
                        viz2d.plot_matches(m_kpts1, m_kpts0, color='lime', lw=0.2)
                        fig = plt.gcf()
                        ax = fig.axes
                        ax0, ax1 = ax[1], ax[0]
                        for iii in range(history_pts1[iwin].shape[0]):
                            f = gtsam_unstable.AnchoredInvDepthFactor(history_pts1[iwin][iii].astype(np.float64),
                                                                      history_pts0[iwin][iii].astype(np.float64),
                                                                      noise,
                                                                      X(iwin*100000 + 0),
                                                                      X(iwin*100000 + 1),L(iwin*100000 + iii),gcam)
                            err = f.evaluateError(cur_result.atPose3(X(iwin*100000 + 0)),cur_result.atPose3(X(iwin*100000 + 1)),history_disps[iwin][iii])
                            if np.linalg.norm(err)<5:
                                ax0.scatter(m_kpts0[iii, 0], m_kpts0[iii, 1], s=50,marker='o',c='none',edgecolors='yellow')
                                ax1.scatter(m_kpts1[iii, 0], m_kpts1[iii, 1], s=50,marker='o',c='none',edgecolors='yellow')
                        plt.savefig('temp/%d_%d_iwin_%d_match.png'%(int(round(tt_query*1e6)*1e3),FINE_LOCALIZATION_WINDOWSIZE,iwin),dpi=600)
                        axes = viz2d.plot_images([image1, image0])
                        viz2d.plot_matches(m_kpts1, m_kpts0, color='lime', lw=0.2)
                        fig = plt.gcf()
                        ax = fig.axes
                        ax0, ax1 = ax[1], ax[0]
                        for iii in range(history_pts1[iwin].shape[0]):
                            f = gtsam_unstable.AnchoredInvDepthFactor(history_pts1[iwin][iii].astype(np.float64),
                                                                      history_pts0[iwin][iii].astype(np.float64),
                                                                      noise,
                                                                      X(iwin*100000 + 0),
                                                                      X(iwin*100000 + 1),L(iwin*100000 + iii),gcam)
                            err = f.evaluateError(cur_result.atPose3(X(iwin*100000 + 0)),cur_result.atPose3(X(iwin*100000 + 1)),history_disps[iwin][iii])
                            if np.linalg.norm(err)<5:
                                ax0.scatter(m_kpts0[iii, 0], m_kpts0[iii, 1], s=50,marker='o',c='none',edgecolors='yellow')
                                ax1.scatter(m_kpts1[iii, 0], m_kpts1[iii, 1], s=50,marker='o',c='none',edgecolors='yellow')
                        plt.savefig('temp/%d_%d_iwin_%d_match_.png'%(int(round(tt_query*1e6)*1e3),FINE_LOCALIZATION_WINDOWSIZE,iwin),dpi=600)
                    cv2.waitKey(1)
                    plt.pause(0.001)
                    plt.close('all')
        plt.pause(0.01)
        fp_fine.close()
        fp_coarse.close()