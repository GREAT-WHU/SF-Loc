import numpy as np
import cv2 
import bisect
import os

class ImageDataset:
    def __init__(self, image_dir, image_stamps, intrinsics = None, distortion = None, resize = None, resize_shape = None):
        self.image_dir = image_dir
        dd = np.loadtxt(image_stamps,str,delimiter=',')
        self.image_stamps = dd[:,0].astype(np.float32)
        self.image_names = dd[:,1]
        self.intrinsics = intrinsics
        self.distortion = distortion
        self.resize = resize
        self.resize_shape = resize_shape
    
    def get_image(self, tt : float):
        idx = bisect.bisect(self.image_stamps,tt-0.001)
        if idx < self.image_stamps.shape[0] and np.fabs(self.image_stamps[idx] - tt)<0.01:
            mm = cv2.imread(os.path.join(self.image_dir,self.image_names[idx]))
            K = np.array([[self.intrinsics[0],0,self.intrinsics[2]],
                          [0,self.intrinsics[1],self.intrinsics[3]],
                          [0,0,1]])
            if not (self.intrinsics is None or self.distortion is None):
                mm = cv2.undistort(mm,K,self.distortion)
            if not self.resize is None:
                mm = cv2.resize(mm,[mm.shape[1]//self.resize,mm.shape[0]//self.resize])
            if not self.resize_shape is None:
                mm = cv2.resize(mm,self.resize_shape)
            return mm
        else:
            raise Exception()
        
