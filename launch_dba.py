import os
import subprocess

# VIO + GNSS
for seq in ['A','B','C']:
    p = subprocess.Popen("python demo_vio_WHU1023.py" +\
            " --imagedir=/mnt/e/WHU1023/WHU1023m/image/cam0" +\
            " --imagestamp=/mnt/e/WHU1023/WHU1023m/stamp_%s.txt"%seq +\
            " --imupath=/mnt/e/WHU1023/WHU1023m/imu.txt" +\
            " --gtpath=/mnt/e/WHU1023/WHU1023m/gt.txt" +\
            " --resultpath=results/poses_%s_realtime.txt"%seq +\
            " --calib=calib/1023.txt" +\
            " --stride=2" +\
            " --max_factors=48" +\
            " --active_window=12" +\
            " --frontend_window=5" +\
            " --frontend_radius=2" +\
            " --frontend_nms=1" +\
            " --inac_range=3" +\
            " --visual_only=0" +\
            " --far_threshold=-1" +\
            " --translation_threshold=0.0" +\
            " --mask_threshold=0.0" +\
            " --skip_edge=[]" +\
            " --save_pkl" +\
            " --use_gnss" +\
            " --gnsspath=/mnt/e/WHU1023/WHU1023m/rtk.txt" +\
            " --pklpath=results/depth_video_%s.pkl" %seq +\
            " --graphpath=results/graph_%s.pkl"%seq +\
            " --show_plot" +\
            # " --enable_h5"
            "",
     shell=True)
    p.wait()