import pickle
import numpy as np
import argparse
import gtsam
from gtsam.symbol_shorthand import B, V, X
import time
import geoFunc.trans as trans
from scipy.spatial.transform import Rotation
import re


def id2pos(id):
    if id>=B(0) and id<B(1000000):
        return np.array([id-B(0),2])
    if id>=V(0) and id<V(1000000):
        return np.array([id-V(0),1])
    if id>=X(0) and id<X(1000000):
        return np.array([id-X(0),0])
def id2type(id):
    if id>=B(0) and id<B(1000000): return 'B'
    if id>=V(0) and id<V(1000000): return 'V'
    if id>=X(0) and id<X(1000000): return 'X'
def id2str(id):
    if id>=B(0) and id<B(1000000): return 'B%d' % (id-B(0))
    if id>=V(0) and id<V(1000000): return 'V%d' % (id-V(0))
    if id>=X(0) and id<X(1000000): return 'X%d' % (id-X(0))
def ids2str(ids):
    strr = ''
    for id in ids:
        strr += id2str(id)
    return strr
def id2num(id):
    if id>=B(0) and id<B(1000000): return id - B(0)
    if id>=V(0) and id<V(1000000): return id - V(0)
    if id>=X(0) and id<X(1000000): return id - X(0)

def optimize_all(graph_path:str, result_file:str):
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    null_count = 0

    ten0 = None
    all_stamps={}
    fp = open(graph_path,'rb')
    cc = 0
    ccount = 0
    print('Loading graph...', time.time())
    while True:
        try:
            dd = pickle.load(fp)
        except:
            break
        str = ''
        ccount+=1
        if ccount == 15000: break
        for i in range(len(dd['factors'])):
            rollup = dd['rollup']
            str += dd['factors'][i]['type'] + ' '
            if 'tstamps' in dd.keys():
                for ii in dd['tstamps'].keys():
                    if ii+rollup in all_stamps.keys():
                        if (all_stamps[ii+rollup] != dd['tstamps'][ii]):
                            raise Exception()
                    all_stamps[ii+rollup] = dd['tstamps'][ii]
            else:
                pass
            if dd['factors'][i]['type']=='prior':
                keys = dd['factors'][i]['factor'].keys()
                kk = dd['factors'][i]['factor'].keys()
                pp = id2pos(kk[0])
                graph.push_back(dd['factors'][i]['factor'].rekey((np.array(keys)+rollup).tolist()))
            if dd['factors'][i]['type']=='imu':
                keys = dd['factors'][i]['factor'].keys()
                id = dd['factors'][i]['factor'].keys()[0] -X(0) + rollup
                ff = dd['factors'][i]['factor']
                ff_new = ff.rekey_explicit(keys,(np.array(keys)+rollup).tolist())
                graph.push_back(ff_new)
            if dd['factors'][i]['type']=='vis':
                keys = dd['factors'][i]['factor'].keys()
                kk = np.array(dd['factors'][i]['factor'].keys())-X(0) +rollup
                graph.push_back(dd['factors'][i]['factor'].rekey((np.array(keys)+rollup).tolist()))
            if dd['factors'][i]['type']=='gnss':
                gnss_factor = gtsam.GPSFactor(dd['factors'][i]['symbol']+rollup, dd['factors'][i]['pos'],\
                    gtsam.noiseModel.Robust.Create(\
                    gtsam.noiseModel.mEstimator.Cauchy(0.1),\
                    gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5,0.5,1.0]))))
                graph.push_back(gnss_factor)
            if 'ten0' in dd.keys():
                ten0 = dd['ten0']
        values_set = set(values.keys())
        if 'states' not in dd.keys(): continue
        for k in dd['states'].keys():
            if id2type(k) == 'B' : 
                if k + rollup in values_set: values.update(k + rollup,dd['states'].atConstantBias(k))
                else                       : values.insert(k + rollup,dd['states'].atConstantBias(k))
            if id2type(k) == 'X' : 
                if k + rollup in values_set: values.update(k + rollup,dd['states'].atPose3(k))
                else                       : values.insert(k + rollup,dd['states'].atPose3(k))
            if id2type(k) == 'V' : 
                if k + rollup in values_set: values.erase(k + rollup); values.insert(k + rollup,dd['states'].atVector(k))
                else                       : values.insert(k + rollup,dd['states'].atVector(k))
    print('Loading finished.', time.time())

    values_set = set(values.keys())
    keyVector_set = set(graph.keyVector())
    for k in values.keys():
        if k not in keyVector_set:
            values.erase(k)

    print('Optimizing...', time.time())
    params = gtsam.LevenbergMarquardtParams();params.setMaxIterations(100)
    params.setVerbosityLM("SUMMARY")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
    result = optimizer.optimize()
    values = result

    fp_result_file = open(result_file,'wt')
    for i in result.keys():
        if id2type(i) == 'X':
            tt = all_stamps[i-X(0)]
            TTT = result.atPose3(i).matrix()
            ppp = TTT[0:3,3]
            qqq = Rotation.from_matrix(TTT[:3, :3]).as_quat()
            line = '%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f'%(tt,ppp[0],ppp[1],ppp[2]\
                                        ,qqq[0],qqq[1],qqq[2],qqq[3])
            p = ten0 + np.matmul(trans.Cen(ten0), ppp)
            line += ' %.6f %.6f %.6f'% (p[0],p[1],p[2]) 
            fp_result_file.writelines(line+'\n')
            fp_result_file.flush()
    fp_result_file.close()
    print('Optimization finished.', time.time())
    del graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_path", type=str, help="path to the generated graph file (.pkl)",default='results/graph.pkl')
    parser.add_argument("--result_file", type=str, help="",default='results/result_post.txt')
    args = parser.parse_args()
    optimize_all(args.graph_path,args.result_file)
