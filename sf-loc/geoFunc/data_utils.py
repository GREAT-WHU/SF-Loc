import numpy as np
import re
import geoFunc.trans as trans
import math
import bisect
import cv2
import os
from scipy.spatial.transform import Rotation

class ImageDataset:
    def __init__(self, image_dir = None, image_stamps = None, intrinsics = None, distortion = None, intrinsics_new = None, resize = None, Tic = None):
        self.image_dir = image_dir
        if not image_stamps is None:
            dd = np.loadtxt(image_stamps,str,delimiter=',')
            self.image_stamps = dd[:,0].astype(np.float32)
            self.image_names = dd[:,1]
        else:
            if not image_dir is None:
                self.image_stamps = np.array(sorted(map(lambda s: float(s.split('.')[0])/1e9,os.listdir(image_dir))))
                self.image_names = np.array(sorted(os.listdir(image_dir),key = lambda s: float(s.split('.')[0])/1e9))
        self.intrinsics = intrinsics
        self.intrinsics_new = intrinsics_new
        if self.intrinsics_new is None:
            self.intrinsics_new = np.copy(self.intrinsics)
        self.distortion = distortion
        self.resize = resize
        if self.resize is not None:
            self.intrinsics = self.intrinsics * self.resize
            self.intrinsics_new = self.intrinsics_new * self.resize
        self.Tic = Tic
        self.all_data_gt = {}
        self.all_data_gt_keys = []
        self.all_data_odo = {}
        self.all_data_odo_keys = []
        self.all_data_global = {}
        self.all_data_global_keys = []
        self.K = np.array([[self.intrinsics_new[0],0,self.intrinsics_new[2]],
                          [0,self.intrinsics_new[1],self.intrinsics_new[3]],
                          [0,0,1]])
        self.ref_xyz = None

    def load_gt(self, path : str,  Ti0i1 : np.ndarray, ref_xyz : np.ndarray = None):
        xyz = loadGTCustom(path,self.all_data_gt,Ti0i1,ref_xyz)
        self.all_data_gt_keys = np.array(sorted(self.all_data_gt.keys()))
        self.ref_xyz = xyz
        return xyz

    def get_pose_gt(self, tt :float):
        idx = bisect.bisect(self.all_data_gt_keys,tt-0.001)
        if idx < self.all_data_gt_keys.shape[0] and np.fabs(self.all_data_gt_keys[idx] - tt)<0.01:
            return self.all_data_gt[self.all_data_gt_keys[idx]]['T']
        else:
            raise Exception()

    def load_global(self, path : str,  Ti0i1 : np.ndarray, ref_xyz : np.ndarray = None):
        xyz = loadGlobal(path,self.all_data_global,Ti0i1,ref_xyz)
        self.all_data_global_keys = np.array(sorted(self.all_data_global.keys()))
        self.ref_xyz = xyz
        return xyz

    def get_pose_global(self, tt :float):
        idx = bisect.bisect(self.all_data_global_keys,tt-0.001)
        if idx < self.all_data_global_keys.shape[0] and np.fabs(self.all_data_global_keys[idx] - tt)<0.01:
            return self.all_data_global[self.all_data_global_keys[idx]]['T']
        else:
            raise Exception()

    def load_odo(self,path:str, Ti0i1 : np.ndarray):
        dd = np.loadtxt(path)
        for i in range(dd.shape[0]):
            self.all_data_odo[dd[i,0]] = {}
            TTT = np.eye(4,4)
            TTT[0:3,3] = dd[i,1:4]
            TTT[0:3,0:3] = Rotation.from_quat(dd[i,4:8]).as_matrix()
            self.all_data_odo[dd[i,0]]['T'] = TTT @ Ti0i1
        self.all_data_odo_keys = np.array(sorted(self.all_data_odo.keys()))

    def get_pose_odo(self, tt:float):
        idx = bisect.bisect(self.all_data_odo_keys,tt-0.001)
        if idx < self.all_data_odo_keys.shape[0] and np.fabs(self.all_data_odo_keys[idx] - tt)<0.01:
            return self.all_data_odo[self.all_data_odo_keys[idx]]['T']
        else:
            print(self.all_data_odo_keys[idx],tt)
            raise Exception()
    
    def get_camera_pose_odo(self,tt:float):
        return self.get_pose_odo(tt) @ self.Tic

    def get_camera_pose_gt(self, tt :float):
        return self.get_pose_gt(tt) @ self.Tic
    
    def get_camera_pose_global(self, tt :float):
        return self.get_pose_global(tt) @ self.Tic
    
    def get_image(self, tt : float, grayscale = False, compress = False):
        idx = bisect.bisect(self.image_stamps,tt-0.001)
        if idx < self.image_stamps.shape[0] and np.fabs(self.image_stamps[idx] - tt)<0.01:
            if grayscale:
                mm = cv2.imread(os.path.join(self.image_dir,self.image_names[idx]),0)
                mm = cv2.cvtColor(mm,cv2.COLOR_GRAY2BGR)
            else:
                mm = cv2.imread(os.path.join(self.image_dir,self.image_names[idx]))
                if compress:
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
                    _, buffer = cv2.imencode('.jpg', mm, encode_param)
                    # cv2.imwrite('temp/%.3f.jpg'%tt,mm, encode_param)
                    image_bytes = buffer.tobytes()
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    mm = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            K = np.array([[self.intrinsics[0],0,self.intrinsics[2]],
                          [0,self.intrinsics[1],self.intrinsics[3]],
                          [0,0,1]])
            if not self.intrinsics_new is None:
                Knew = np.array([[self.intrinsics_new[0],0,self.intrinsics_new[2]],
                              [0,self.intrinsics_new[1],self.intrinsics_new[3]],
                              [0,0,1]])
            else:
                Knew = K
            if not (self.intrinsics is None or self.distortion is None):
                if not self.resize is None:
                    K_temp = np.copy(K); K_temp[0:2,:] /= self.resize
                    Knew_temp = np.copy(Knew); Knew_temp[0:2,:] /= self.resize
                else:
                    K_temp = np.copy(K)
                    Knew_temp = np.copy(Knew)
                mm = cv2.undistort(mm,K_temp,self.distortion,Knew_temp)
            if not self.resize is None:
                mm = cv2.resize(mm,[int(mm.shape[1]*self.resize),int(mm.shape[0]*self.resize)])
            return mm
        else:
            raise Exception()
    
def loadKAIST(path : str, all_data : dict, Ti0i1 : np.ndarray, ref_xyz : np.ndarray = None):
    fp = open(path,'rt')
    while True:
        line = fp.readline().strip()
        if line == '': break
        if line[0] == '#': continue
        line = re.sub('\s\s+',' ',line)
        elem = line.split(',')
        all_data[float(elem[0])/1e9] = {'T':np.array(elem[1:17]+['0.0','0.0','0.0','1.0']).astype(np.float64).reshape([4,4])}


def loadGTCustom(path : str, all_data : dict, Ti0i1 : np.ndarray, ref_xyz : np.ndarray = None):
    fp = open(path,'rt')

    if ref_xyz is None:
        is_ref_set  = False
    else:
        is_ref_set = True
        Ten0 = np.eye(4,4)
        Ten0[0:3,0:3] = trans.Cen(ref_xyz)
        Ten0[0:3,3] = ref_xyz

    while True:
        line = fp.readline().strip()
        if line == '':break
        if line[0] == '#' :continue
        line = re.sub('\s\s+',' ',line)
        elem = line.split(' ')
        sod = float(elem[0])
        if sod not in all_data.keys():
            all_data[sod] ={}
        all_data[sod]['X0']   = float(elem[1])
        all_data[sod]['Y0']   = float(elem[2])
        all_data[sod]['Z0']   = float(elem[3])
        all_data[sod]['VX0']  = float(elem[4])
        all_data[sod]['VY0']  = float(elem[5])
        all_data[sod]['VZ0']  = float(elem[6])
        Ren = trans.Cen([all_data[sod]['X0'],all_data[sod]['Y0'],all_data[sod]['Z0']])
        Rni0 = Rotation.from_quat(np.array([float(elem[7]),float(elem[8]),float(elem[9]),float(elem[10])])).as_matrix()
        Rni1 = np.matmul(Rni0,Ti0i1[0:3,0:3])
        Rei1 = np.matmul(Ren,Rni1)
        tei0 = np.array([all_data[sod]['X0'],all_data[sod]['Y0'],all_data[sod]['Z0']])
        tei1 = tei0 + Ren @ Rni0 @ Ti0i1[0:3,3]
        Tei1 = np.eye(4,4)
        Tei1[0:3,0:3] = Rei1
        Tei1[0:3,3] = tei1
        if not is_ref_set:
            is_ref_set = True
            Ten0 = np.eye(4,4)
            Ten0[0:3,0:3] = trans.Cen(tei1)
            Ten0[0:3,3] = tei1
        Tn0i = np.matmul(np.linalg.inv(Ten0),Tei1)
        all_data[sod]['T'] = Tn0i
    fp.close()
    return Ten0[0:3,3]

def loadGlobal(path : str, all_data: dict, Ti0i1 : np.ndarray, ref_xyz : np.ndarray = None):
    fp = open(path,'rt')
    if ref_xyz is None:
        is_ref_set  = False
    else:
        is_ref_set = True
        Ten0 = np.eye(4,4)
        Ten0[0:3,0:3] = trans.Cen(ref_xyz)
        Ten0[0:3,3] = ref_xyz
    while True:
        line = fp.readline().strip()
        if line == '':break
        if line[0] == '#' :continue
        line = re.sub('\s\s+',' ',line)
        elem = line.split(' ')
        sod = float(elem[0])
        if sod not in all_data.keys():
            all_data[sod] ={}
        all_data[sod]['X0']   = float(elem[-3])
        all_data[sod]['Y0']   = float(elem[-2])
        all_data[sod]['Z0']   = float(elem[-1])
        Ren = trans.Cen([all_data[sod]['X0'],all_data[sod]['Y0'],all_data[sod]['Z0']])
        Rni0 = Rotation.from_quat(np.array([float(elem[4]),float(elem[5]),float(elem[6]),float(elem[7])])).as_matrix()
        Rni1 = np.matmul(Rni0,Ti0i1[0:3,0:3])
        Rei1 = np.matmul(Ren,Rni1)
        tei0 = np.array([all_data[sod]['X0'],all_data[sod]['Y0'],all_data[sod]['Z0']])
        tei1 = tei0 + Ren @ Rni0 @ Ti0i1[0:3,3]
        Tei1 = np.eye(4,4)
        Tei1[0:3,0:3] = Rei1
        Tei1[0:3,3] = tei1
        if not is_ref_set:
            is_ref_set = True
            Ten0 = np.eye(4,4)
            Ten0[0:3,0:3] = trans.Cen(tei1)
            Ten0[0:3,3] = tei1
        Tn0i = np.matmul(np.linalg.inv(Ten0),Tei1)
        all_data[sod]['T'] = Tn0i
    fp.close()
    return Ten0[0:3,3]
        