import torch
import lietorch
import numpy as np

import matplotlib.pyplot as plt
from lietorch import SE3
from modules.corr import CorrBlock, AltCorrBlock
import geom.projective_ops as pops
import flow_vis
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
import time
from side_factor_graph import SideFactorGraph
from depth_video import DepthVideo
import matplotlib.cm as cm
import matplotlib

import bisect

class FactorGraph:
    def __init__(self, video: DepthVideo, update_op, device="cuda:0", corr_impl="volume", max_factors=-1, upsample=False, far_threshold=0.01, inac_range = 3, oppo = False):
        self.video = video
        self.update_op = update_op
        self.device = device
        self.max_factors = max_factors
        self.corr_impl = corr_impl
        self.upsample = upsample

        # operator at 1/8 resolution
        self.ht = ht = video.ht // 8
        self.wd = wd = video.wd // 8

        self.coords0 = pops.coords_grid(ht, wd, device=device)
        self.ii = torch.as_tensor([], dtype=torch.long, device=device) # 单向的边
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)
        self.age = torch.as_tensor([], dtype=torch.long, device=device)

        self.corr, self.net, self.inp = None, None, None
        self.damping = 1e-6 * torch.ones_like(self.video.disps)

        self.target = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

        # inactive factors
        self.ii_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.ii_bad = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_bad = torch.as_tensor([], dtype=torch.long, device=device)

        self.target_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

        self.far_threshold = far_threshold
        self.inac_range = inac_range
        self.mask_threshold = 0.0
        self.img_count = 0

        self.oppo = oppo

    def __filter_repeated_edges(self, ii, jj):
        """ remove duplicate edges """

        keep = torch.zeros(ii.shape[0], dtype=torch.bool, device=ii.device)
        eset = set(
            [(i.item(), j.item()) for i, j in zip(self.ii, self.jj)] +
            [(i.item(), j.item()) for i, j in zip(self.ii_inac, self.jj_inac)])

        for k, (i, j) in enumerate(zip(ii, jj)):
            keep[k] = (i.item(), j.item()) not in eset

        return ii[keep], jj[keep]

    def print_edges(self):
        ii = self.ii.cpu().numpy()
        jj = self.jj.cpu().numpy()

        ix = np.argsort(ii)
        ii = ii[ix]
        jj = jj[ix]

        w = torch.mean(self.weight, dim=[0,2,3,4]).cpu().numpy()
        w = w[ix]
        for e in zip(ii, jj, w):
            print(e)
        print()

    def filter_edges(self):
        """ remove bad edges """
        conf = torch.mean(self.weight, dim=[0,2,3,4])
        mask = (torch.abs(self.ii-self.jj) > 2) & (conf < 0.001)

        self.ii_bad = torch.cat([self.ii_bad, self.ii[mask]])
        self.jj_bad = torch.cat([self.jj_bad, self.jj[mask]])
        self.rm_factors(mask, store=False)

    def clear_edges(self):
        self.rm_factors(self.ii >= 0)
        self.net = None
        self.inp = None

    @torch.cuda.amp.autocast(enabled=True)
    def add_factors(self, ii, jj, remove=False):
        """ add edges to factor graph """
        # print('add_factor: ',ii,jj)
        # print(self.ii,self.jj)
        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii, dtype=torch.long, device=self.device)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj, dtype=torch.long, device=self.device)

        # remove duplicate edges
        ii, jj = self.__filter_repeated_edges(ii, jj)

        if ii.shape[0] == 0:
            return

        # place limit on number of factors
        if self.max_factors > 0 and self.ii.shape[0] + ii.shape[0] > self.max_factors \
                and self.corr is not None and remove:
            
            ix = torch.arange(len(self.age))[torch.argsort(self.age).cpu()]
            self.rm_factors(ix >= self.max_factors - ii.shape[0], store=True)

        net = self.video.nets[ii].to(self.device).unsqueeze(0)

        # correlation volume for new edges
        if self.corr_impl == "volume":
            c = (ii == jj).long()
            fmap1 = self.video.fmaps[ii,0].to(self.device).unsqueeze(0)
            fmap2 = self.video.fmaps[jj,c].to(self.device).unsqueeze(0)
            corr = CorrBlock(fmap1, fmap2)
            self.corr = corr if self.corr is None else self.corr.cat(corr)

            inp = self.video.inps[ii].to(self.device).unsqueeze(0)
            self.inp = inp if self.inp is None else torch.cat([self.inp, inp], 1)

        with torch.cuda.amp.autocast(enabled=False):
            target, _ = self.video.reproject(ii, jj) # 根据深度和位姿获取伪光流
            weight = torch.zeros_like(target)

        self.ii = torch.cat([self.ii, ii], 0)
        self.jj = torch.cat([self.jj, jj], 0)
        self.age = torch.cat([self.age, torch.zeros_like(ii)], 0)

        # reprojection factors
        self.net = net if self.net is None else torch.cat([self.net, net], 1)

        self.target = torch.cat([self.target, target], 1)
        self.weight = torch.cat([self.weight, weight], 1)

    @torch.cuda.amp.autocast(enabled=True)
    def rm_factors(self, mask, store=False):
        """ drop edges from factor graph """

        # store estimated factors
        if store:
            self.ii_inac = torch.cat([self.ii_inac, self.ii[mask]], 0)
            self.jj_inac = torch.cat([self.jj_inac, self.jj[mask]], 0)
            self.target_inac = torch.cat([self.target_inac, self.target[:,mask]], 1)
            self.weight_inac = torch.cat([self.weight_inac, self.weight[:,mask]], 1)

        self.ii = self.ii[~mask]
        self.jj = self.jj[~mask]
        self.age = self.age[~mask]
        
        if self.corr_impl == "volume":
            self.corr = self.corr[~mask]

        if self.net is not None:
            self.net = self.net[:,~mask]

        if self.inp is not None:
            self.inp = self.inp[:,~mask]

        self.target = self.target[:,~mask]
        self.weight = self.weight[:,~mask]


    @torch.cuda.amp.autocast(enabled=True)
    def rm_keyframe(self, ix):
        """ drop edges from factor graph """


        with self.video.get_lock():
            self.video.images[ix] = self.video.images[ix+1]
            self.video.poses[ix] = self.video.poses[ix+1]
            self.video.disps[ix] = self.video.disps[ix+1]
            self.video.disps_sens[ix] = self.video.disps_sens[ix+1]
            self.video.disps_meas[ix] = self.video.disps_meas[ix+1]
            self.video.intrinsics[ix] = self.video.intrinsics[ix+1]

            self.video.nets[ix] = self.video.nets[ix+1]
            self.video.inps[ix] = self.video.inps[ix+1]
            self.video.fmaps[ix] = self.video.fmaps[ix+1]

            self.video.tstamp[ix] = self.video.tstamp[ix+1] # BUG

        m = (self.ii_inac == ix) | (self.jj_inac == ix)
        self.ii_inac[self.ii_inac >= ix] -= 1
        self.jj_inac[self.jj_inac >= ix] -= 1

        if torch.any(m):
            self.ii_inac = self.ii_inac[~m]
            self.jj_inac = self.jj_inac[~m]
            self.target_inac = self.target_inac[:,~m]
            self.weight_inac = self.weight_inac[:,~m]

        m = (self.ii == ix) | (self.jj == ix)

        self.ii[self.ii >= ix] -= 1
        self.jj[self.jj >= ix] -= 1
        self.rm_factors(m, store=False)

    # def est_velocity_flow(self, disp_i, disp_j, pose_i, pose_j, target_ij):
        
    #     coords, valid_mask = \
    #         pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)
    
    @torch.cuda.amp.autocast(enabled=True)
    def update_lc(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, motion_only=False, marg = False, hessian = False):
        """ run update operator on factor graph """

        self.video.logger.info('update')
        # motion features
        # 利用深度计算伪光流
        with torch.cuda.amp.autocast(enabled=False):
            coords1, mask = self.video.reproject(self.ii, self.jj)
            motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
            motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0) # 1,2,4,48,96

        corr = self.corr(coords1) # 这里的corr已经是所有关键帧的堆叠了
        self.net, delta, weight = \
            self.update_op(self.net, self.inp, corr, motn, self.ii, self.jj)

        if t0 is None:
            t0 = max(1, self.ii.min().item()+1)
            
        self.video.logger.info('ba')

        with torch.cuda.amp.autocast(enabled=False):
            self.target = coords1 + delta.to(dtype=torch.float)
            self.weight = weight.to(dtype=torch.float)
 
            # comprehensive
            rgb = self.video.images[torch.max(self.ii)].cpu().numpy().transpose(1,2,0)
            new_flow_id = 10
            weight_cpu = weight[0,10].cpu().detach().numpy().astype(np.float32)

            weight_cpu = np.linalg.norm(weight_cpu,axis=2)
            normalizer = matplotlib.colors.Normalize(vmin=-0.0, vmax=1.5)
            mapper = cm.ScalarMappable(norm=normalizer,cmap='jet')
            colormapped_im = (mapper.to_rgba(weight_cpu)[:, :, :3] * 255).astype(np.uint8)
            colormapped_im = cv2.cvtColor(colormapped_im,cv2.COLOR_RGB2BGR)
            colormapped_im = cv2.resize(colormapped_im,[rgb.shape[1],rgb.shape[0]])
            colormapped_im = cv2.addWeighted(rgb,0.5,colormapped_im,0.5,0)
            absflow = (self.target[0,new_flow_id] - self.coords0).cpu().detach().numpy()
            # for iii in range(0,absflow.shape[0],4):
            #     for jjj in range(0,absflow.shape[1],4):
            #         colormapped_im = cv2.line(colormapped_im, (jjj * 8,iii * 8),(int(round((jjj-absflow[iii,jjj,0])* 8)) ,int(round((iii-absflow[iii,jjj,1]) * 8))),(255,255,255),1,cv2.LINE_AA)

            cv2.imshow('weight_cpu',colormapped_im)
            self.img_count += 1
            cv2.waitKey(1)
            
            ht, wd = self.coords0.shape[0:2]

            if use_inactive:
                m = (self.ii_inac >= t0 - self.inac_range) & (self.jj_inac >= t0 - self.inac_range)
                ii = torch.cat([self.ii_inac[m], self.ii], 0)
                jj = torch.cat([self.jj_inac[m], self.jj], 0)
                target = torch.cat([self.target_inac[:,m], self.target], 1)
                weight = torch.cat([self.weight_inac[:,m], self.weight], 1)
                if self.video.imu_enabled:
                    i0 = min(ii)
                    i1 = max(ii)
                    ppp = SE3(self.video.poses[i0:(i1+1)]).inv().matrix()[:,0:3,3].cpu().numpy()
                    # [:,:3].cpu().numpy()
                    scale = max(max(ppp[:,0]) - min(ppp[:,0]),max(ppp[:,1]) - min(ppp[:,1]))
                    ppp[:,0] -= np.mean(ppp[:,0])
                    ppp[:,1] = -(ppp[:,1]- np.mean(ppp[:,1]))
                    ppp *= max(round(1/scale * 200 / 50)*50,50)
                    ppp += 500
                    mmm = np.zeros([1000,1000],dtype=np.uint8)
                    for iii in range(0,i1+1-i0):
                        mmm = cv2.circle(mmm,(int(round(ppp[iii,0])),int(round(ppp[iii,1]))),4,255,0)
                    for iii in range(self.ii_inac[m].shape[0]):
                        iiii = self.ii_inac[m][iii]-i0
                        jjjj = self.jj_inac[m][iii]-i0
                        mmm = cv2.line(mmm,(int(round(ppp[iiii,0])),int(round(ppp[iiii,1]))),(int(round(ppp[jjjj,0])),int(round(ppp[jjjj,1]))),128,1)
                    for iii in range(self.ii.shape[0]):
                        iiii = self.ii[iii]-i0
                        jjjj = self.jj[iii]-i0
                        mmm = cv2.line(mmm,(int(round(ppp[iiii,0])),int(round(ppp[iiii,1]))),(int(round(ppp[jjjj,0])),int(round(ppp[jjjj,1]))),255,1)
                    cv2.imshow('window',mmm)
            else:
                ii, jj, target, weight = self.ii, self.jj, self.target, self.weight

            damping = .2 * self.damping[torch.unique(ii)].contiguous() + EP

            target = target.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
            weight = weight.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()

            # dense bundle adjustment
            if hessian:
                t0 = 0
                H, v = self.video.ba_local(target, weight, damping, ii, jj, t0, t1, 
                    itrs=itrs, lm=1e-4, ep=0.1, motion_only=False)
                return H,v
            else:
                t0 = max(self.ii)
                self.video.ba_local(target, weight, damping, ii, jj, t0, t1, 
                    itrs=itrs, lm=1e-4, ep=0.1, motion_only=False)
        self.age += 1


    @torch.cuda.amp.autocast(enabled=True)
    def update(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, motion_only=False, marg = False):
        """ run update operator on factor graph """

        self.video.logger.info('update')
        # motion features
        # 利用深度计算伪光流
        with torch.cuda.amp.autocast(enabled=False):
            coords1, mask = self.video.reproject(self.ii, self.jj)
            motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
            motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0) # 1,2,4,48,96
            # flow_color = flow_vis.flow_to_color(flow_uv, convert_to_bgr=False)

        # a = motn[0,0].permute(1,2,0).cpu().numpy()[:,:,0:2]
        # flow_motn = flow_vis.flow_to_color(a, convert_to_bgr=False)
        # flow_motn = cv2.resize(flow_motn,[flow_motn.shape[1]*8,flow_motn.shape[0]*8],interpolation =  cv2.INTER_NEAREST)
        # cv2.imshow('flow_motn',flow_motn)
        # cv2.waitKey(1)

        # correlation features
        # 利用伪光流计算相关性
        corr = self.corr(coords1) # 这里的corr已经是所有关键帧的堆叠了
        # 核心中的核心，基于伪光流先验计算光流的更新量
        # self.net, delta, weight, damping, upmask = \
        self.net, delta, weight = \
            self.update_op(self.net, self.inp, corr, motn, self.ii, self.jj)


        # idea: 由delta, depth0, depth1, pose0, pose1获取一个velocity map
        #       考虑到BA当中
        #       两种考虑：
        #       1）补偿掉光流
        #       2）在BA中考虑速度场，即p = R * p + t + dt
        print(t0)
        if t0 is None:
            t0 = max(1, self.ii.min().item()+1)
            
        self.video.logger.info('ba')

        with torch.cuda.amp.autocast(enabled=False):
            self.target = coords1 + delta.to(dtype=torch.float)
            self.weight = weight.to(dtype=torch.float)

            # Gs = lietorch.SE3(self.video.poses[None])
            # self.est_velocity_flow(self.disps[ii[0]],self.disps[jj[0]],Gs[ii[0]],Gs[jj[0]],self.target)

            # good visualization!!!!!!!!!!!!!!!!!!!!!!
            # disp_show_front = self.video.disps[self.ii[0]].cpu().numpy()
            # disp_show_front = cv2.resize(disp_show_front,[disp_show_front.shape[1]*8,disp_show_front.shape[0]*8],interpolation =  cv2.INTER_NEAREST)
            # disp_show_front= disp_show_front.astype(np.float32)
            self.pred_depth = False
            if self.pred_depth:
                # normalizer = matplotlib.colors.Normalize(vmin=-0.2, vmax=1.0)
                # mapper = cm.ScalarMappable(norm=normalizer,cmap='magma')
                # colormapped_im = (mapper.to_rgba(disp_show_front)[:, :, :3] * 255).astype(np.uint8)
                # colormapped_im = cv2.cvtColor(colormapped_im,cv2.COLOR_RGB2BGR)
                # cv2.imshow('qqq',colormapped_im)
                # cv2.waitKey(1)

                disp_show_front = self.video.disps[self.ii[0]].cpu().numpy()
                disp_show_front = cv2.resize(disp_show_front,[disp_show_front.shape[1]*8,disp_show_front.shape[0]*8],interpolation =  cv2.INTER_NEAREST)
                disp_show_front= disp_show_front.astype(np.float32)

                normalizer = matplotlib.colors.Normalize(vmin=-0.2, vmax=1.0)
                mapper = cm.ScalarMappable(norm=normalizer,cmap='magma')
                colormapped_im = (mapper.to_rgba(disp_show_front)[:, :, :3] * 255).astype(np.uint8)
                colormapped_im = cv2.cvtColor(colormapped_im,cv2.COLOR_RGB2BGR)
                cv2.imshow('colormapped_im',colormapped_im)
                cv2.waitKey(1)

                # self.video.disps_sens[:,:,:] = 0.0

                disp_info_idx = np.array([])
                disp_info = np.array([])
                try:
                    disp_info_idx = torch.unique(self.video.cur_ii).cpu().numpy()
                    disp_info = self.video.cur_disp_info
                except:
                    123
                for iii in range(torch.min(self.ii),torch.max(self.ii)+1):
                    disp_show_front = self.video.disps[iii].cpu().numpy()
                    x   = self.video.disps_meas[iii].numpy().reshape([-1,1])
                    ab = np.array([[0.0],[0.01]])
                    idx = np.where(disp_info_idx==iii)[0]
                    disp_info_this = np.zeros_like(x) + 1.0
                    if idx.shape[0] > 0:
                        disp_info_this = disp_info[idx[0].item()]
                    disp_info_this = disp_info_this.reshape([-1,1])
                    try:
                        for i in range(6):
                            lx = x*ab[0] + ab[1]
                            y   = disp_show_front.reshape([-1,1])
                            x1  = np.zeros_like(x) + 1.0
                            xx1 = np.hstack([x,x1])
                            mmask = disp_info_this > 0.05
                            if i == 0:
                                xx1 = xx1[mmask[:,0],:]
                                lx  =  lx[mmask[:,0],:]
                                y   =   y[mmask[:,0],:]
                            if i >= 1:
                                mmask = np.logical_and(mmask,np.logical_and(np.logical_and(lx>0.10, y>0.10), np.abs(lx - y)< 0.10))
                                if np.sum(mmask) < 250: raise Exception()
                                xx1 = xx1[mmask[:,0],:]
                                lx  =  lx[mmask[:,0],:]
                                y   =   y[mmask[:,0],:]
                            N   = np.matmul(xx1.T,xx1) + np.array([[0,0],[0,1]]) * 10.0**2
                            l   = np.matmul(xx1.T,y - lx) + np.matmul(np.array([[0,0],[0,1]]),np.array([[0],[-ab[1,0]+0.01]])) * 10.0**2
                            ab  += np.matmul(np.linalg.inv(N),l)
                    except:
                        ab = np.zeros([2,1],dtype=np.float32)
                    mm_d_corr = (self.video.disps_meas[iii]).numpy()*ab[0] + ab[1]
                    mm_d_corr[np.logical_or(mm_d_corr < 0.10,np.abs(mm_d_corr - disp_show_front) > 0.10*10000) ] = 0.0

                    if self.video.imu_enabled:
                        self.video.disps_sens[iii] = torch.tensor((mm_d_corr).astype(np.float32)).cuda()

                    if iii == torch.max(self.ii):
                        normalizer      = matplotlib.colors.Normalize(vmin=-0.2, vmax=1.0)
                        mapper          = cm.ScalarMappable(norm=normalizer,cmap='magma')
                        colormapped_im  = (mapper.to_rgba(mm_d_corr)[:, :, :3] * 255).astype(np.uint8)
                        colormapped_im  = cv2.cvtColor(colormapped_im,cv2.COLOR_RGB2BGR)
                        colormapped_im  = cv2.resize(colormapped_im,[colormapped_im.shape[1]*8,colormapped_im.shape[0]*8],interpolation =  cv2.INTER_NEAREST)
                        cv2.imshow('disp%d' % (torch.max(self.ii) - iii),colormapped_im)
                        disp_show_front = self.video.disps[self.ii[0]].cpu().numpy()
                        disp_show_front = cv2.resize(disp_show_front,[disp_show_front.shape[1]*8,disp_show_front.shape[0]*8],interpolation =  cv2.INTER_NEAREST)
                        disp_show_front= disp_show_front.astype(np.float32)

                        normalizer = matplotlib.colors.Normalize(vmin=-0.2, vmax=1.0)
                        mapper = cm.ScalarMappable(norm=normalizer,cmap='magma')
                        colormapped_im = (mapper.to_rgba(disp_show_front)[:, :, :3] * 255).astype(np.uint8)
                        colormapped_im = cv2.cvtColor(colormapped_im,cv2.COLOR_RGB2BGR)
                        cv2.imshow('colormapped_im',colormapped_im)
                        cv2.waitKey(1)


                        disp_show_front = self.video.disps_sens[self.ii[0]].cpu().numpy()
                        disp_show_front = cv2.resize(disp_show_front,[disp_show_front.shape[1]*8,disp_show_front.shape[0]*8],interpolation =  cv2.INTER_NEAREST)
                        disp_show_front= disp_show_front.astype(np.float32)
                        normalizer = matplotlib.colors.Normalize(vmin=-0.2, vmax=1.0)
                        mapper = cm.ScalarMappable(norm=normalizer,cmap='magma')
                        colormapped_im = (mapper.to_rgba(disp_show_front)[:, :, :3] * 255).astype(np.uint8)
                        colormapped_im = cv2.cvtColor(colormapped_im,cv2.COLOR_RGB2BGR)
                        cv2.imshow('qqq',colormapped_im)
                        cv2.waitKey(1)
                        # # absflow = (self.target[0] - self.coords0).cpu().numpy()
                        # # flow_data_front = absflow[0]
                        # # flow_data_front = flow_data_front.astype(np.float32)
                        # # flow_color_front = flow_vis.flow_to_color(flow_data_front, convert_to_bgr=False)
                        # # flow_color_front = cv2.resize(flow_color_front,[flow_color_front.shape[1]*8,flow_color_front.shape[0]*8],interpolation =  cv2.INTER_NEAREST)

                        # # for iii in range(0,absflow.shape[1],2):
                        # #     for jjj in range(0,absflow.shape[2],2):
                        # #         flow_color_front = cv2.line(flow_color_front, (jjj * 8,iii * 8),(int(round((jjj-absflow[0,iii,jjj,0])* 8)) ,int(round((iii-absflow[0,iii,jjj,1]) * 8))),(123,123,123),1)
                        # #         # print((iii * 8,jjj * 8),((iii+int(round(absflow[0,iii,jjj,0]/10))) * 8,(jjj+int(round(absflow[0,iii,jjj,1]/10))) * 8))



            # # comprehensive
            # rgb = self.video.images[torch.max(self.ii)].cpu().numpy().transpose(1,2,0)
            # new_flow_id = torch.where(torch.logical_and(self.ii==torch.max(self.ii),self.jj==torch.max(self.ii)-1))[0][0].item()
            # weight_cpu = weight[0,new_flow_id].cpu().numpy().astype(np.float32)
            # weight_cpu = np.linalg.norm(weight_cpu,axis=2)
            # normalizer = matplotlib.colors.Normalize(vmin=-0.0, vmax=1.5)
            # mapper = cm.ScalarMappable(norm=normalizer,cmap='jet')
            # colormapped_im = (mapper.to_rgba(weight_cpu)[:, :, :3] * 255).astype(np.uint8)
            # colormapped_im = cv2.cvtColor(colormapped_im,cv2.COLOR_RGB2BGR)
            # colormapped_im = cv2.resize(colormapped_im,[rgb.shape[1],rgb.shape[0]])
            # colormapped_im = cv2.addWeighted(rgb,0.5,colormapped_im,0.5,0)
            # absflow = (self.target[0,new_flow_id] - self.coords0).cpu().numpy()
            # for iii in range(0,absflow.shape[0],4):
            #     for jjj in range(0,absflow.shape[1],4):
            #         colormapped_im = cv2.line(colormapped_im, (jjj * 8,iii * 8),(int(round((jjj-absflow[iii,jjj,0])* 8)) ,int(round((iii-absflow[iii,jjj,1]) * 8))),(255,255,255),1,cv2.LINE_AA)

            # cv2.imshow('weight_cpu',colormapped_im)
            # # cv2.imwrite('img_flow/%010d.png'%self.img_count,colormapped_im)
            # self.img_count += 1
            # # cv2.imshow('disp_show_front',colormapped_im)
            # # # cv2.imshow('flow_color_front',flow_color_front)
            # cv2.waitKey(1)
            
            ht, wd = self.coords0.shape[0:2]
            # self.damping[torch.unique(self.ii)] = damping

            if use_inactive:
                m = (self.ii_inac >= t0 - self.inac_range) & (self.jj_inac >= t0 - self.inac_range)
                ii = torch.cat([self.ii_inac[m], self.ii], 0)
                jj = torch.cat([self.jj_inac[m], self.jj], 0)
                target = torch.cat([self.target_inac[:,m], self.target], 1)
                weight = torch.cat([self.weight_inac[:,m], self.weight], 1)
                # if self.video.imu_enabled:
                #     i0 = min(ii)
                #     i1 = max(ii)
                #     ppp = SE3(self.video.poses[i0:(i1+1)]).inv().matrix()[:,0:3,3].cpu().numpy()
                #     # [:,:3].cpu().numpy()
                #     scale = max(max(ppp[:,0]) - min(ppp[:,0]),max(ppp[:,1]) - min(ppp[:,1]))
                #     ppp[:,0] -= np.mean(ppp[:,0])
                #     ppp[:,1] = -(ppp[:,1]- np.mean(ppp[:,1]))
                #     ppp *= max(round(1/scale * 200 / 50)*50,50)
                #     ppp += 500
                #     mmm = np.zeros([1000,1000],dtype=np.uint8)
                #     for iii in range(0,i1+1-i0):
                #         mmm = cv2.circle(mmm,(int(round(ppp[iii,0])),int(round(ppp[iii,1]))),4,255,0)
                #     for iii in range(self.ii_inac[m].shape[0]):
                #         iiii = self.ii_inac[m][iii]-i0
                #         jjjj = self.jj_inac[m][iii]-i0
                #         mmm = cv2.line(mmm,(int(round(ppp[iiii,0])),int(round(ppp[iiii,1]))),(int(round(ppp[jjjj,0])),int(round(ppp[jjjj,1]))),128,1)
                #     for iii in range(self.ii.shape[0]):
                #         iiii = self.ii[iii]-i0
                #         jjjj = self.jj[iii]-i0
                #         mmm = cv2.line(mmm,(int(round(ppp[iiii,0])),int(round(ppp[iiii,1]))),(int(round(ppp[jjjj,0])),int(round(ppp[jjjj,1]))),255,1)
                #     cv2.imshow('window',mmm)
            else:
                ii, jj, target, weight = self.ii, self.jj, self.target, self.weight
            if self.far_threshold > 0 and self.video.imu_enabled:
                disp_mask = (self.video.disps < self.far_threshold)
                mask = disp_mask[ii, :, :]
                # weight[:, mask] = 0.0001
                # weight[:, mask] = 0.0
                weight[:, mask] /= 1000.0
            
            if self.mask_threshold > 0 and self.video.imu_enabled:
                pose0 = SE3(self.video.poses[ii])
                pose1 = SE3(self.video.poses[jj])
                pose01 = pose0*pose1.inv()
                mask = torch.norm(pose01.translation()[:,:3],dim=1) < self.mask_threshold
                weight[:,mask,:,:,:] /= 1000.0
                

            # weight_cpu = weight.cpu().numpy()
            # weight_front = weight_cpu[0,1]
            # weight_front = flow_vis.flow_to_color(weight_front, convert_to_bgr=False)
            # weight_front = cv2.resize(weight_front,[weight_front.shape[1]*8,weight_front.shape[0]*8],interpolation =  cv2.INTER_NEAREST)
            # cv2.imshow('weight_front',weight_front)
            # cv2.waitKey(1)

            downweight_newframe = True
            if downweight_newframe:
                weight[:,ii==max(ii)] /= 10.0
                weight[:,jj==max(jj)] /= 4.0

            damping = .2 * self.damping[torch.unique(ii)].contiguous() + EP

            target = target.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
            weight = weight.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()

            # dense bundle adjustment
            self.video.ba(target, weight, damping, ii, jj, t0, t1, 
                itrs=itrs, lm=1e-4, ep=0.1, motion_only=motion_only)
            print(t0)
            print(ii)
            print(jj)
        
            if self.upsample:
                self.video.upsample(torch.unique(self.ii), upmask)

            # self.video.upsample(torch.unique(self.ii), upmask)
            # disp_show_front = self.video.disps_up[self.ii[0]].cpu().numpy()
            # disp_show_front= disp_show_front.astype(np.float32)
            # normalizer = matplotlib.colors.Normalize(vmin=-0.2, vmax=1.0)
            # mapper = cm.ScalarMappable(norm=normalizer,cmap='magma')
            # colormapped_im = (mapper.to_rgba(disp_show_front)[:, :, :3] * 255).astype(np.uint8)
            # colormapped_im = cv2.cvtColor(colormapped_im,cv2.COLOR_RGB2BGR)
            # cv2.imshow('disp_show_front',colormapped_im)
            # cv2.waitKey(1)

        self.age += 1

    @torch.cuda.amp.autocast(enabled=False)
    def update_lowmem(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, steps=8):
        """ run update operator on factor graph - reduced memory implementation """

        # alternate corr implementation
        t = self.video.counter.value

        num, rig, ch, ht, wd = self.video.fmaps.shape
        corr_op = AltCorrBlock(self.video.fmaps.view(1, num*rig, ch, ht, wd))

        for step in range(steps):
            print("Global BA Iteration #{}".format(step+1))
            with torch.cuda.amp.autocast(enabled=False):
                coords1, mask = self.video.reproject(self.ii, self.jj)
                motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
                motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)

            s = 8
            for i in range(0, self.jj.max()+1, s):
                v = (self.ii >= i) & (self.ii < i + s)
                iis = self.ii[v]
                jjs = self.jj[v]

                ht, wd = self.coords0.shape[0:2]

                corr1 = corr_op(coords1[:,v], rig * iis, rig * jjs + (iis == jjs).long())
                with torch.cuda.amp.autocast(enabled=True):
                 
                    net, delta, weight, damping, upmask = \
                        self.update_op(self.net[:,v], self.video.inps[None,iis], corr1, motn[:,v], iis, jjs)

                    if self.upsample:
                        self.video.upsample(torch.unique(iis), upmask)

                self.net[:,v] = net
                self.target[:,v] = coords1[:,v] + delta.float()
                self.weight[:,v] = weight.float()
                self.damping[torch.unique(iis)] = damping

            damping = .2 * self.damping[torch.unique(self.ii)].contiguous() + EP
            target = self.target.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
            weight = self.weight.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()

            # dense bundle adjustment
            self.video.ba(target, weight, damping, self.ii, self.jj, 1, t, 
                itrs=itrs, lm=1e-5, ep=1e-2, motion_only=False)

            self.video.dirty[:t] = True

    def add_neighborhood_factors(self, t0, t1, r=3):
        """ add edges between neighboring frames within radius r """

        ii, jj = torch.meshgrid(torch.arange(t0,t1), torch.arange(t0,t1))
        ii = ii.reshape(-1).to(dtype=torch.long, device=self.device)
        jj = jj.reshape(-1).to(dtype=torch.long, device=self.device)

        c = 1 if self.video.stereo else 0

        keep = ((ii - jj).abs() > c) & ((ii - jj).abs() <= r)
        self.add_factors(ii[keep], jj[keep])

    
    def add_proximity_factors(self, t0=0, t1=0, rad=2, nms=2, beta=0.25, thresh=16.0, remove=False):
        """ add edges to the factor graph based on distance """

        t = self.video.counter.value
        ix = torch.arange(t0, t)
        jx = torch.arange(t1, t)

        ii, jj = torch.meshgrid(ix, jx)
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)
        
        cc = ii.shape[0]
        # Opportunistic
        if self.oppo:
            if torch.max(ii) - torch.min(ii) == 4:
                jj = torch.cat([jj,torch.max(torch.min(ii)-5,torch.tensor(0))[None],torch.max(torch.min(ii)-4,torch.tensor(0))[None],torch.max(torch.min(ii)-6,torch.tensor(0))[None]])
                ii = torch.cat([ii,torch.max(ii)[None],torch.max(ii)[None],torch.max(ii)[None]])

        d = self.video.distance(ii, jj, beta=beta)
        d[ii - rad < jj] = np.inf
        d[d > 100] = np.inf

        ii1 = torch.cat([self.ii, self.ii_bad, self.ii_inac], 0)
        jj1 = torch.cat([self.jj, self.jj_bad, self.jj_inac], 0)
        for i, j in zip(ii1.cpu().numpy(), jj1.cpu().numpy()):
            for di in range(-nms, nms+1):
                for dj in range(-nms, nms+1):
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf

        es = []
        for i in range(t0, t):
            if self.video.stereo:
                es.append((i, i))
                d[(i-t0)*(t-t1) + (i-t1)] = np.inf

            for j in range(max(i-rad-1,0), i):
                es.append((i,j))
                es.append((j,i))
                if (i-t0)*(t-t1) + (j-t1) >=0:
                    d[(i-t0)*(t-t1) + (j-t1)] = np.inf

        ix = torch.argsort(d)
        for k in ix:
            if k >= cc:
                continue

            if d[k].item() > thresh:
                continue

            if len(es) > self.max_factors:
                break

            i = ii[k]
            j = jj[k]
            
            # bidirectional
            es.append((i, j))
            es.append((j, i))

            for di in range(-nms, nms+1):
                for dj in range(-nms, nms+1):
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf
        
        if ii.shape[0] > cc:
            ix = torch.argsort(d[cc:ii.shape[0]])
            if d[cc + ix[0]] < thresh and  d[cc + ix[0]]  > 0:
                es.append((ii[cc+ix[0]],jj[cc+ix[0]]))
                es.append((jj[cc+ix[0]],ii[cc+ix[0]]))
                # print('append oppo edge: ',d[cc + ix[0]])
        # for k in range(cc,ii.shape[0]):
        #     if d[k] < thresh * 5:
        #         es.append(())

        ii, jj = torch.as_tensor(es, device=self.device).unbind(dim=-1)
        self.add_factors(ii, jj, remove)
