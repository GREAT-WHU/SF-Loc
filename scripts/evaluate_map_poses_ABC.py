import sys
sys.path.append('sf-loc')
import geoFunc.trans as trans
import matplotlib.pyplot as plt
import numpy as np
import re
import math
import matplotlib
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import os
import copy
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
matplotlib.rcParams['mathtext.bf'] = 'Arial:bold'
matplotlib.rcParams['font.family'] = 'Arial'

font1={'family':'Arial',
     'style':'normal',
    'weight':'bold',
      'color':'black',
      'size':7
}
t_series_list = []
horiz_err_list = []
def seconds_to_hhmmss(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
plt.close('all')
all_data={}
x_ref_1=[-2267776.6894,  5009356.7938 , 3220981.5589]
denu=[0,0,0]
enu_ref = [0,0,0]
Ri0i1=np.matrix([[ 1,  0,  0],
                 [ 0,  1,  0],
                 [ 0,  0,  1]])
ti0i0i1 = np.matrix([[ 0],
                    [-0.465],
                    [0.359]]) - np.matrix([[ -0.0125],
                    [-0.30],
                    [0.2091]])
ti0i0g = np.matrix([[ 0],
                    [-0.465],
                    [0.359]])

last_dd=[]

all_data ={}
Ri0i1=trans.att2m([0.0/180*math.pi,0.0/180*math.pi,0.0/180*math.pi])
Ten0 = None
is_ref_set  = False
fp = open('/mnt/e/dataset/WHU1023m/gt.txt','rt')
e_series = []
n_series = []
u_series = []
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
    Rni0 = Rotation.from_quat(np.array([float(elem[7]),float(elem[8]),float(elem[9]),float(elem[10])])).as_matrix()
    Ren = trans.Cen([all_data[sod]['X0'],all_data[sod]['Y0'],all_data[sod]['Z0']])
    Rei0 = np.matmul(Ren,Rni0)
    Rei1 = Rei0 @ Ri0i1
    xyz = np.array([all_data[sod]['X0'],all_data[sod]['Y0'],all_data[sod]['Z0']])
    dxyz = np.matmul(Rei0,ti0i0i1)
    dxyz_g = np.matmul(Rei0,ti0i0g)

    all_data[sod]['X0'] = float(xyz[0] + dxyz[0,0])
    all_data[sod]['Y0'] = float(xyz[1] + dxyz[1,0])
    all_data[sod]['Z0'] = float(xyz[2] + dxyz[2,0])
    all_data[sod]['X0G'] = float(xyz[0] + dxyz_g[0,0])
    all_data[sod]['Y0G'] = float(xyz[1] + dxyz_g[1,0])
    all_data[sod]['Z0G'] = float(xyz[2] + dxyz_g[2,0])
    tei1 = np.array([all_data[sod]['X0'],all_data[sod]['Y0'],all_data[sod]['Z0']])
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
    e_series.append(Tn0i[0,3])
    n_series.append(Tn0i[1,3])
    u_series.append(Tn0i[2,3])
all_data_old =copy.deepcopy(all_data)



ax_3d = None
for seq in ['A','B','C']:
    ### 1) MS-DBA (realtime)
    last_dd=[]
    first_sod=0
    last_sod=0
    fp = open(r'results/poses_%s_realtime.txt'%seq,'rt')
    while True:
        line = fp.readline().strip()
        if line == '':
            break
        if line[0]=='%' or line[0]=='#':
            continue
        line = re.sub('\s\s+',' ',line)
        elem = line.split(' ')
        if len(elem)<10:continue
        sod= int(float(elem[0]))
        if math.fabs(float(elem[0])-round(float(elem[0])))>0.01:continue
        if sod not in all_data.keys():
            all_data[sod]={}
        if first_sod == 0:
            first_sod =sod
        if len(elem) < 9 : continue
        if 'X1' in all_data[sod].keys():
            continue
        all_data[sod]['X1']   = float(elem[8])
        all_data[sod]['Y1']   = float(elem[9])
        all_data[sod]['Z1']   = float(elem[10])
        last_sod = sod
    fp.close()

    t_series=[]
    x_series=[];y_series=[];z_series=[]
    e_series=[];n_series=[]

    dist = 0.0

    for sod in sorted(all_data.keys()):
        dd = all_data[sod]
        if 'X0' in dd.keys() and 'X1' in dd.keys():
            t_series.append(sod)
            xyz = np.array([dd['X0'],dd['Y0'],dd['Z0']])
            dxyz = np.array([dd['X0']-dd['X1'],dd['Y0']-dd['Y1'],dd['Z0']-dd['Z1']])
            denu = trans.cart2enu(xyz,dxyz)
            x_series.append(denu[0])
            y_series.append(denu[1])
            z_series.append(denu[2])
            enu = trans.cart2enu(xyz,np.array([dd['X0']-x_ref_1[0],dd['Y0']-x_ref_1[1],dd['Z0']-x_ref_1[2]]))
            if len(e_series)>0:
                dist += np.sqrt((enu[0] - e_series[-1])**2 + (enu[1] - n_series[-1])**2)
            e_series.append(enu[0])
            n_series.append(enu[1])
    x_series = np.array(x_series)
    y_series = np.array(y_series)
    z_series = np.array(z_series)
    fig = plt.figure('bev',figsize=[10,10])
    if ax_3d is None: ax_3d = fig.add_subplot(111,projection='3d')
    if seq == 'A': ax_3d.plot(e_series,n_series,np.zeros_like(e_series),color=[1,0.5,0.5])
    if seq == 'B': ax_3d.plot(e_series,n_series,np.zeros_like(e_series)+5,color=[0.5,1.0,0.5])
    if seq == 'C': ax_3d.plot(e_series,n_series,np.zeros_like(e_series)+10,color=[0.5,0.5,1.0])
    ax_3d.set_zlim([-50,300])
    ax_3d.set_proj_type('ortho')
    ax_3d.patch.set_alpha(0.0)
    fig.patch.set_alpha(0.0)
    ax_3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_3d.set_zticklabels([])
    ax_3d.grid(False)
    ax_3d.zaxis.line.set_visible(False)
    
    plt.figure('mapping pose error',figsize=[3.8,5])
    if seq == 'A': plt.subplot(3,1,1)
    if seq == 'B': plt.subplot(3,1,2)
    if seq == 'C': plt.subplot(3,1,3)
    horiz_err = np.sqrt((np.power(x_series,2)+np.power(y_series,2))/1.0)
    rmse = np.linalg.norm(horiz_err)/math.sqrt(horiz_err.shape[0])
    plt.plot(t_series,horiz_err,c='r',linewidth=1,zorder = 1000,label='DBA (Real-time) $\mathbf{%.3f}$'%rmse)
    print('RMSE: ',np.linalg.norm(horiz_err)/math.sqrt(horiz_err.shape[0]))
    t_series_list.append(t_series)
    horiz_err_list.append(horiz_err)

    t_temp_list = []
    x_temp_list = []
    y_temp_list = []
    for iiii in range(len(t_series)):
        if t_series[iiii] > 8500 -60  and t_series[iiii] < 8500 + 70:
            t_temp_list.append(t_series[iiii])
            x_temp_list.append(x_series[iiii])
            y_temp_list.append(y_series[iiii])
        
    ### 2) MS-DBA (post-processing)
    all_data = copy.deepcopy(all_data_old)
    fp = open(r'results/poses_%s_post.txt'%seq,'rt')
    while True:
        line = fp.readline().strip()
        if line == '':
            break
        if line[0]=='%' or line[0]=='#':
            continue
        line = re.sub('\s\s+',' ',line)
        elem = line.split(' ')
        if len(elem)<10:continue
        sod= int(float(elem[0]))
        if math.fabs(float(elem[0])-round(float(elem[0])))>0.01:continue
        if sod not in all_data.keys():
            all_data[sod]={}
        if len(elem) < 9 : continue
        if 'X1' in all_data[sod].keys():
            continue
        all_data[sod]['X1']   = float(elem[8])
        all_data[sod]['Y1']   = float(elem[9])
        all_data[sod]['Z1']   = float(elem[10])
    fp.close()

    t_series=[]
    x_series=[]
    y_series=[]
    z_series=[]
    e_series=[]
    n_series=[]

    for sod in sorted(all_data.keys()):
        dd = all_data[sod]
        if sod < first_sod: continue
        if sod > last_sod: continue
        if 'X0' in dd.keys() and 'X1' in dd.keys():
            t_series.append(sod)
            xyz = np.array([dd['X0'],dd['Y0'],dd['Z0']])
            dxyz = np.array([dd['X0']-dd['X1'],dd['Y0']-dd['Y1'],dd['Z0']-dd['Z1']])
            denu = trans.cart2enu(xyz,dxyz)
            x_series.append(denu[0])
            y_series.append(denu[1])
            z_series.append(denu[2])
            enu = trans.cart2enu(xyz,np.array([dd['X0']-x_ref_1[0],dd['Y0']-x_ref_1[1],dd['Z0']-x_ref_1[2]]))
            e_series.append(enu[0])
            n_series.append(enu[1])
    horiz_err = np.sqrt((np.power(x_series,2)+np.power(y_series,2))/1.0)
    rmse = np.linalg.norm(horiz_err)/math.sqrt(horiz_err.shape[0])
    plt.plot(t_series,horiz_err,c='b',linewidth=1,zorder = 2000,label='DBA (Global) $\mathbf{%.3f}$'%rmse)
    print('RMSE: ',np.linalg.norm(horiz_err)/math.sqrt(horiz_err.shape[0]))
    t_series_list.append(t_series)
    horiz_err_list.append(horiz_err)


    t_temp_list = []
    x_temp_list = []
    y_temp_list = []
    for iiii in range(len(t_series)):
        if t_series[iiii] > 8500 -60  and t_series[iiii] < 8500 + 70:
            t_temp_list.append(t_series[iiii])
            x_temp_list.append(x_series[iiii])
            y_temp_list.append(y_series[iiii])

        
    ### 3) RTK 
    all_data = copy.deepcopy(all_data_old)
    fp = open(r'/mnt/e/dataset/WHU1023/rtk.txt','rt')
    while True:
        line = fp.readline().strip()
        if line == '':
            break
        if line[0]=='%' or line[0]=='#':
            continue
        line = re.sub('\s\s+',' ',line)
        elem = line.split(' ')
        if len(elem)<10:continue
        sod= int(float(elem[0]))
        if sod < first_sod: continue
        if sod > last_sod: continue
        print(sod,first_sod,last_sod)
        if math.fabs(float(elem[0])-round(float(elem[0])))>0.01:continue
        if sod not in all_data.keys():
            all_data[sod]={}
        all_data[sod]['X2']   = float(elem[1])
        all_data[sod]['Y2']   = float(elem[2])
        all_data[sod]['Z2']   = float(elem[3])
    fp.close()

    t_series=[]
    x_series=[]
    y_series=[]
    z_series=[]

    for sod in sorted(all_data.keys()):
        dd = all_data[sod]
        if 'X0' in dd.keys() and 'X2' in dd.keys():
            t_series.append(sod)
            xyz = np.array([dd['X0'],dd['Y0'],dd['Z0']])
            dxyz = np.array([dd['X0G']-dd['X2'],dd['Y0G']-dd['Y2'],dd['Z0G']-dd['Z2']])
            denu = trans.cart2enu(xyz,dxyz)
            x_series.append(denu[0])
            y_series.append(denu[1])
            z_series.append(denu[2])

    horiz_err = np.sqrt((np.power(x_series,2)+np.power(y_series,2))/1.0)
    horiz_err_rmse = horiz_err[horiz_err<100]
    rmse = np.linalg.norm(horiz_err_rmse)/math.sqrt(horiz_err_rmse.shape[0])
    plt.plot(t_series,horiz_err,c=[0.7,0.7,0.7],linewidth=1,zorder = 200,label='RTK $\mathbf{%.3f}$'%rmse)
    # plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: seconds_to_hhmmss(x+8*3600)))
    def get_10_minute_ticks(start, end):
        start_minutes = (start // 60) // 10 * 10 * 60  
        end_minutes = ((end // 60) // 10 + 1) * 10 * 60
        return range(start_minutes, end_minutes + 1, 10 * 60)


    # plt.gca().xaxis.set_major_locator(ticker.FixedLocator(get_10_minute_ticks(0, 86400)))
    t_series_list.append(t_series)
    horiz_err_list.append(horiz_err)



    plt.ylim([-0.2,8.2])
    plt.ylabel('Horizontal Error [m]')
    plt.plot([7350,7600,7600,7350,7350],[0,0,2,2,0],c='black',zorder=10000000,linewidth=0.8)
    plt.tight_layout(pad=0.1)
    plt.xlim([first_sod-30,last_sod+30])


    lg = plt.legend(loc='upper left',markerscale=3,fontsize=9,framealpha=1,ncol=2,columnspacing=0.3,handletextpad=0.3,edgecolor='black',fancybox=False)
    lg.get_frame().set_linewidth(0.8)
    lg.set_zorder(3000)

    plt.tight_layout(pad=0.1)
plt.show()