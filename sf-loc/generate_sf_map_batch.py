import sys
sys.path.append('sf-loc/VPR-methods-evaluation')
sys.path.append('sf-loc/VPR-methods-evaluation/third_party/deep-image-retrieval')

import pickle
import numpy as np
import geoFunc.dataset_utils as dataset_utils
import tqdm
import cv2
import argparse

import torch.utils.data as data
import torchvision.transforms as transforms
import vpr_models
import torch
from PIL import Image
import bisect

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, nargs='+', help="path to image directory",default='/home/zhouyuxuan/data/1023_01/image/colmap/recon/cam0')
    parser.add_argument("--imagestamp", type=str, nargs='+', help="",default='/home/zhouyuxuan/data/1023_01/image/colmap/recon/stamp_rearrange_merged.txt')
    parser.add_argument("--depth_video",  type=str, nargs='+',default='results/depth_video.pkl')
    parser.add_argument("--poses_post",  type=str, nargs='+',default='results/poses_post.txt')
    parser.add_argument("--map_indices",  type=str,default='results/map_indice_1023n.pkl')
    parser.add_argument("--map_file",  type=str,default='results/sf_map.pkl')
    parser.add_argument("--calib", type=str, help="",default='calib/1023n.txt')
    parser.add_argument("--method", type=str, help="",default='eigenplaces')
    parser.add_argument("--backbone", type=str, help="",default=None)
    parser.add_argument("--descriptors_dimension", type=int, help="",default=512)
    parser.add_argument("--image_size", type=int, default=[384,512], nargs="+")
    parser.add_argument("--jpeg_quality", type=int, default=60)
    args = parser.parse_args()

    device = 'cuda'
    
    all_map_data = {'descriptor':[],'image':[],'disps':[],\
                    'poses':[],'xyz_ref':[],'tstamps':[],\
                        'vpr_model':{'method':args.method,'backbone':args.backbone,
                                     'descriptors_dimension':args.descriptors_dimension,'image_size':args.image_size}}
    
    map_indices_time = pickle.load(open(args.map_indices,'rb'))
    calib = np.loadtxt(args.calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]
    
    assert(len(args.imagedir) == len(args.imagestamp))
    assert(len(args.imagedir) == len(args.depth_video))
    assert(len(args.imagedir) == len(args.poses_post))
    N_seq = len(args.poses_post)

    accum_list = []
    image_data_list = []
    for i_seq in range(N_seq):
        if i_seq == 0:
            poses_post =  np.loadtxt(args.poses_post[i_seq])
            dump_data  =  pickle.load(open(args.depth_video[i_seq],'rb'))
            disps      =  np.array(list(dump_data['disps'].values()))
            image_data =  dataset_utils.ImageDataset(args.imagedir[i_seq],
                                                args.imagestamp[i_seq],
                                                calib[:4],calib[4:8],None)
            image_data_list.append(image_data)
            tstamps    =  list(dump_data['stamps'].values())
            accum_list.append(len(tstamps))
        else:
            poses_post = np.concatenate([poses_post,np.loadtxt(args.poses_post[i_seq])])
            dump_data  =  pickle.load(open(args.depth_video[i_seq],'rb'))
            disps      =  np.concatenate([disps, np.array(list(dump_data['disps'].values()))])
            image_data =  dataset_utils.ImageDataset(args.imagedir[i_seq],
                                                args.imagestamp[i_seq],
                                                calib[:4],calib[4:8],None)
            image_data_list.append(image_data)
            tstamps    = tstamps + list(dump_data['stamps'].values())
            accum_list.append(len(tstamps))
        continue
            

    model = vpr_models.get_model(args.method, args.backbone, args.descriptors_dimension)
    model = model.eval().to(device)
    transformations = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(size=args.image_size, antialias=True)
    ]
    custom_transform = transforms.Compose(transformations)

    with torch.inference_mode():
        for i in tqdm.tqdm(range(len(map_indices_time['indices']))):
            pose_idx = bisect.bisect(poses_post[:,0],tstamps[map_indices_time['indices'][i]]-0.01)
            if poses_post[pose_idx,0] != tstamps[map_indices_time['indices'][i]]:
                continue
            all_map_data['tstamps'].append(tstamps[map_indices_time['indices'][i]])
            all_map_data['poses'].append(poses_post[pose_idx,1:8])

            ''' Pack RGB images and disparities. '''
            idx = bisect.bisect(accum_list,map_indices_time['indices'][i])
            tttt = tstamps[map_indices_time['indices'][i]]
            if tttt == 7153.6:
                mm = image_data_list[idx].get_image(7158.0)
            else:
                mm = image_data_list[idx].get_image(tstamps[map_indices_time['indices'][i]])
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
            _, buffer = cv2.imencode('.jpg', mm, encode_param)
            image_bytes = buffer.tobytes()
            nparr = np.frombuffer(image_bytes, np.uint8)
            all_map_data['image'].append(nparr)

            disp = np.round((1/disps[map_indices_time['indices'][i]])*100).astype(np.uint16)
            _, buffer = cv2.imencode('.png', disp)
            image_bytes = buffer.tobytes()
            nparr = np.frombuffer(image_bytes, np.uint8)
            all_map_data['disps'].append(nparr)

            ''' Generate VPR descriptors. '''
            mmrgb = cv2.cvtColor(mm, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(mmrgb)
            normalized_img = custom_transform(pil_img)[None]

            # This could be faster with bigger batchsize. 
            # See VPR-methods-evaluation/main.py for reference.
            descriptors = model(normalized_img.to(device))
            all_map_data['descriptor'].append(descriptors.cpu().numpy())

    all_map_data['descriptor'] = np.concatenate(all_map_data['descriptor'])
    all_map_data['poses'] = np.array(all_map_data['poses'])
    all_map_data['xyz_ref'] = dump_data['xyz_ref']
    all_map_data['calib'] = calib
    pickle.dump(all_map_data,open(args.map_file,'wb'))

    
