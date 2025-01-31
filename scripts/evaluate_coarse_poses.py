import matplotlib.pyplot as plt
import numpy as np
import re
import sys
sys.path.append('sf-loc')
import geoFunc.trans as trans
import geoFunc.data_utils as data_utils
import math
import matplotlib
from scipy.spatial.transform import Rotation

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
matplotlib.rcParams['mathtext.bf'] = 'Arial:bold'
matplotlib.rcParams['font.family'] = 'Arial'

font1={'family':'Sans',
     'style':'normal',
    'weight':'bold',
      'color':'black',
      'size':7
}
props = dict(boxstyle='square', facecolor='white', alpha=1)

all_data ={}
Ri0i1=trans.att2m([-0.1/180*math.pi,-0.1/180*math.pi,-1.0/180*math.pi])
ti0i0i1 = np.array([0.0,0.47,-0.04])
Ti0i1 = np.eye(4,4); Ti0i1[0:3,0:3] = Ri0i1; Ti0i1[0:3,3] = np.array([0.0,0.47,-0.04])
Ten0 = None
is_ref_set  = False
fp = open('/mnt/e/WHU1023/WHU0412/gt.txt','rt')
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
    all_data[sod]['X0'] = float(xyz[0] + dxyz[0])
    all_data[sod]['Y0'] = float(xyz[1] + dxyz[1])
    all_data[sod]['Z0'] = float(xyz[2] + dxyz[2])
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

dd = np.loadtxt('results/result_coarse.txt')
t_series = []
err_series = []
y_series = []
for i in range(dd.shape[0]):
    tt = dd[i,0]
    t_series.append(tt)
    dxyz = dd[i,1:4]
    Rni = all_data[tt]['T'][0:3,0:3]
    dxyz = [dxyz[0]-all_data[tt]['X0'],
            dxyz[1]-all_data[tt]['Y0'],
            dxyz[2]-all_data[tt]['Z0']]
    denu = trans.cart2enu([all_data[tt]['X0'],all_data[tt]['Y0'],all_data[tt]['Z0']],dxyz)
    err_series.append(np.linalg.norm(denu[:2]))
error_coarse = np.array([t_series,err_series]).T

error_in = error_coarse[error_coarse[:,1]<100,1]
print('& \makecell[c]{%.2f\\%%} & \makecell[c]{%.2f\\%%} & \makecell[c]{%.2f\\%%} & \makecell[c]{%.2f} \\\\'%(\
np.sum(np.array(error_coarse[:,1])<5.0) /error_coarse.shape[0]*100,\
np.sum(np.array(error_coarse[:,1])<10.0)/error_coarse.shape[0]*100,\
np.sum(np.array(error_coarse[:,1])<20.0)/error_coarse.shape[0]*100,\
np.linalg.norm(error_in)/np.sqrt(error_in.shape[0])))
plt.plot(error_coarse[:,0],error_coarse[:,1],marker='*')

plt.show()

