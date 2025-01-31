import os
import subprocess


# VIO + GNSS
p = subprocess.Popen("python demo_vio_WHU1023.py" +\
        " --imagedir=/mnt/e/WHU1023/image_undist/cam0" +\
        " --imagestamp=/mnt/e/WHU1023/stamp.txt" +\
        " --imupath=/mnt/f/1023_01/smallimu_simple.txt" +\
        " --gtpath=/mnt/e/WHU1023/gt.txt" +\
        " --resultpath=results/poses_realtime.txt" +\
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
        " --gnsspath=/mnt/e/WHU1023/rtk.txt" +\
        " --pklpath=results/depth_video.pkl" +\
        " --graphpath=results/graph.pkl" +\
        " --show_plot" +\
        # " --enable_h5"
        "",
 shell=True)
p.wait()