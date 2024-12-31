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

# i = '1023n'
# p = subprocess.Popen("python demo_vio_1023n.py\
#  --imagedir=/home/zhouyuxuan/data/1023_01/image/colmap/recon/cam0\
#  --imagestamp=/home/zhouyuxuan/data/1023_01/image/colmap/recon/stamp_rearrange_merged.txt\
#  --imupath=/mnt/f/1023_01/smallimu_simple.txt\
#  --gtpath=/mnt/f/1023_01/IE16.txt\
#  --enable_h5\
#  --h5path=/home/zhouyuxuan/DROID-SLAM/%s.h5\
#  --resultpath=vins_temp_1023n.txt\
#  --calib=calib/1023n.txt\
#  --stride=2\
#  --active_window=12\
#  --frontend_window=5\
#  --frontend_radius=2\
#  --frontend_nms=1\
#  --inac_range=3\
#  --pure_visual=0\
#  --pklpath=%s\
#  --use_gnss\
#  --far_threshold=-1\
#  --translation_threshold=0.0\
#  --mask_threshold=0.0\
#  --disable_vis" % (i,i),shell=True)
# p.wait()