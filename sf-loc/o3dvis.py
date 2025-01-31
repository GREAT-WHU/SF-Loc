import open3d as o3d
import re
import numpy as np
import cv2
import numpy as np
import matplotlib
import matplotlib.cm as cm
import math

def m2att(R):
    att=[0,0,0]

    att[0] = math.asin(R[2, 1])
    att[1] = math.atan2(-R[2, 0], R[2, 2])
    att[2] = math.atan2(-R[0, 1], R[1, 1])

    return np.array(att)

CAM_POINTS = np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5]])

CAM_LINES = np.array([
    [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])

def create_camera_actor(g, scale=0.05):
    """ build open3d camera polydata """
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))

    color = (g * 1.0, 0.5 * (1-g), 0.9 * (1-g))
    camera_actor.paint_uniform_color(color)
    return camera_actor

def create_point_actor(points, colors = None):
    """ open3d point cloud from numpy array """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    if colors is  None:
        pass
    else:
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud

def str2array(ss):
    elem = re.sub('\s\s+',' ',ss).split(' ')
    num=[]
    for e in elem:
        num.append(float(e))
    return np.array(num)

def gen_pcd(K,disp,color=None):
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    h1 = disp.shape[0]
    w1 = disp.shape[1]
    p = np.zeros((h1,w1,3),np.float64)
    i, j= np.indices((h1,w1))
    p[...,0] = (j-cx )/ fx 
    p[...,1] = (i-cy )/ fy 
    p[...,2] = 1
    p_list = p * (1.0/disp)[:,:,np.newaxis]
    p_list = p_list.reshape(-1,3)

    color = cv2.cvtColor(color,cv2.COLOR_BGR2RGB)
    
    # source_depth = np.reciprocal(disp.squeeze())
    # rgb = color.squeeze()
    # rgb = np.rollaxis(rgb,0,3)
    # print(rgb.shape,source_depth.shape)
    # rgb = cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
    # p = np.zeros((384,512,3),np.float64)
    # p[...,0] = (j-cx) / fx
    # p[...,1] = (i-cy) / fy
    # p[...,2] = 1

    # p_list = p *source_depth[:,:,np.newaxis]
    # p_list = p_list.reshape(-1,3)
    color_list = color.reshape(-1,3)/255.0

    mask = p_list[:,2] < 500.0
    mask1 = np.logical_not(np.logical_and(p_list[:,1]<-1.0,color_list[:,2]>0.8))
    mask = np.logical_and(mask,mask1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p_list[mask,:])
    pcd.colors = o3d.utility.Vector3dVector(color_list[mask,:])
    return pcd

def vis_disp(disp,resize = 8,wait = True, name = 'im',ret = False):
    normalizer = matplotlib.colors.Normalize(vmin=-0.0, vmax=0.1)
    mapper = cm.ScalarMappable(norm=normalizer,cmap='magma')
    colormapped_im = (mapper.to_rgba(disp)[:, :, :3] * 255).astype(np.uint8)
    colormapped_im = cv2.cvtColor(colormapped_im,cv2.COLOR_RGB2BGR)
    colormapped_im = cv2.resize(colormapped_im,[colormapped_im.shape[1]*resize,colormapped_im.shape[0]*resize])
    if ret:
        return colormapped_im
    cv2.imshow(name,colormapped_im)
    # cv2.imwrite(name+'.png',colormapped_im)
    if wait:
        cv2.waitKey(0)

def vis_setup():
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='123')
    vis.get_render_option().point_size = 2
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1,1,1])

    gridline_x = np.arange(-100,100,2).astype(np.float32)
    p0 = np.vstack([gridline_x,np.zeros_like(gridline_x)+1,np.zeros_like(gridline_x)+100]).T
    p1 = np.vstack([gridline_x,np.zeros_like(gridline_x)+1,np.zeros_like(gridline_x)-100]).T
    p01 = np.vstack([p0,p1])
    gridline_x_lines= np.vstack([np.arange(0,gridline_x.shape[0],1),np.arange(0,gridline_x.shape[0],1)+gridline_x.shape[0]]).T
    grid_actor =  o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(p01),
        lines=o3d.utility.Vector2iVector(gridline_x_lines.astype(np.int32)))
    color = (0.7,0.7,0.7)
    grid_actor.paint_uniform_color(color)
    # vis.add_geometry(grid_actor)

    gridline_x = np.arange(-100,100,2).astype(np.float32)
    p0 = np.vstack([np.zeros_like(gridline_x)+100,np.zeros_like(gridline_x)+1,gridline_x]).T
    p1 = np.vstack([np.zeros_like(gridline_x)-100,np.zeros_like(gridline_x)+1,gridline_x]).T
    p01 = np.vstack([p0,p1])
    gridline_x_lines= np.vstack([np.arange(0,gridline_x.shape[0],1),np.arange(0,gridline_x.shape[0],1)+gridline_x.shape[0]]).T
    grid_actor =  o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(p01),
        lines=o3d.utility.Vector2iVector(gridline_x_lines.astype(np.int32)))
    color = (0.7,0.7,0.7)
    grid_actor.paint_uniform_color(color)
    # vis.add_geometry(grid_actor)
    
    return vis

def vis_run(vis):
    ctr = vis.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters() 
    camera_params.extrinsic = np.array([[1.0,0.0,0.0,0.0],
                                        [0.0,1.0,0.0,0.0],
                                        [0.0,0.0,1.0,30.0],
                                        [0.0,0.0,0.0,1.0]])

    ctr.convert_from_pinhole_camera_parameters(camera_params)

    vis.run()
    vis.capture_screen_image("1.PNG")
    vis.destroy_window()