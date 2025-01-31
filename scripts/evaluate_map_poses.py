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
fp = open('/mnt/e/WHU1023/gt.txt','rt')
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


### 1) MS-DBA (realtime)
all_data_old =copy.deepcopy(all_data)
last_dd=[]
first_sod=0
last_sod=0
fp = open(r'results/poses_realtime.txt','rt')
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
fp.close()

t_series=[]
x_series=[]
y_series=[]
z_series=[]
e_series=[]
n_series=[]

dist = 0.0

for sod in sorted(all_data.keys()):
    dd = all_data[sod]
    if sod < 6360: continue
    if sod > 11760: continue
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
plt.figure('bev',figsize=[6,6.5])


from matplotlib.collections import LineCollection

points = np.array([e_series, n_series]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, linewidths=5, alpha=0.2, color='blue')
plt.gca().add_collection(lc)
plt.scatter(e_series[0],n_series[0],marker='^',s= 60,zorder= 1000)
plt.scatter(e_series[-1],n_series[-1],marker='^',s= 60,zorder= 1000)


x_series = np.array(x_series)
y_series = np.array(y_series)
z_series = np.array(z_series)
plt.figure()
plt.subplot(3,1,1)
plt.plot(t_series,x_series,c='r',linewidth=1,marker='*',markersize=4.0)
plt.ylim([-5,5])
plt.subplot(3,1,2)
plt.plot(t_series,y_series,c='g',linewidth=1,marker='*',markersize=4.0)
plt.ylim([-5,5])
plt.subplot(3,1,3)
plt.plot(t_series,z_series,c='b',linewidth=1,marker='*',markersize=4.0)
plt.ylim([-5,5])

plt.figure('mapping pose error',figsize=[5,3.2])
horiz_err = np.sqrt((np.power(x_series,2)+np.power(y_series,2))/1.0)
rmse = np.linalg.norm(horiz_err)/math.sqrt(horiz_err.shape[0])
plt.plot(t_series,horiz_err,c='r',linewidth=1,zorder = 1000,label='DBA (Real-time) $\mathbf{%.3f}$'%rmse)
print('RMSE: ',np.linalg.norm(horiz_err)/math.sqrt(horiz_err.shape[0]))
t_series_list.append(t_series)
horiz_err_list.append(horiz_err)


### 2) MS-DBA (post-processing)
all_data = copy.deepcopy(all_data_old)
fp = open(r'results/poses_post.txt','rt')
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
fp.close()

t_series=[]
x_series=[]
y_series=[]
z_series=[]
e_series=[]
n_series=[]

for sod in sorted(all_data.keys()):
    dd = all_data[sod]
    if sod < 6360: continue
    if sod > 11760: continue
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


### 3) RTK 
all_data = copy.deepcopy(all_data_old)
fp = open(r'/mnt/e/WHU1023/rtk.txt','rt')
while True:
    line = fp.readline().strip()
    if line == '':
        break
    if line[0]=='%' or line[0]=='#':
        continue
    line = re.sub('\s\s+',' ',line)
    elem = line.split(' ')
    if len(elem)<10:continue
    # if 'Fixed' not in line: continue
    # if int(elem[-7])<10: continue
    sod= int(float(elem[0]))
    if sod < 6360: continue
    if sod > 11760: continue
    if math.fabs(float(elem[0])-round(float(elem[0])))>0.01:continue
    if sod not in all_data.keys():
        all_data[sod]={}
    if first_sod == 0:
        first_sod =sod
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
t_series_list.append(t_series)
horiz_err_list.append(horiz_err)


lg = plt.legend(loc='upper left',markerscale=3,fontsize=10,framealpha=1,ncol=1,columnspacing=0.3,handletextpad=0.3,edgecolor='black',fancybox=False)
lg.get_frame().set_linewidth(0.8)
lg.set_zorder(3000)
plt.ylim([-0.2,10])
plt.xlabel('Time [s]')
plt.ylabel('Horizontal Error [m]')
plt.plot([7350,7600,7600,7350,7350],[0,0,2,2,0],c='black',zorder=10000000,linewidth=0.8)
plt.tight_layout(pad=0.1)
plt.xlim([6360-100,11760+100])


ax = plt.gca()
axin1 = ax.inset_axes([0.6, 0.6,  
                       0.35, 0.3],zorder=10000) 
axin1.plot(t_series_list[0],horiz_err_list[0],linewidth=1.0,c='r',zorder=10002)
axin1.plot(t_series_list[1],horiz_err_list[1],linewidth=1.0,c='b',zorder=10003)
axin1.plot(t_series_list[2],horiz_err_list[2],linewidth=1.0,c=[0.7,0.7,0.7],zorder=10001,marker='x',fillstyle='none')
axin1.set_xlim([7350,7600])
axin1.set_ylim([-0.0,2])

plt.tight_layout(pad=0.1)
plt.savefig('mapping_error.svg')

plt.show()