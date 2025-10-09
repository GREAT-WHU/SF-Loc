python sf-loc/generate_sf_map_batch.py --imagedir    /mnt/e/WHU1023/WHU1023m/image/cam0 /mnt/e/WHU1023/WHU1023m/image/cam0 /mnt/e/WHU1023/WHU1023m/image/cam0\
                                 --imagestamp  /mnt/e/WHU1023/WHU1023m/stamp_A.txt       /mnt/e/WHU1023/WHU1023m/stamp_B.txt       /mnt/e/WHU1023/WHU1023m/stamp_C.txt\
                                 --depth_video results/depth_video_A.pkl                results/depth_video_B.pkl                results/depth_video_C.pkl\
                                 --poses_post  results/poses_A_post.txt                 results/poses_B_post.txt                 results/poses_C_post.txt\
                                 --calib calib/1023.txt --map_indices results/map_indices.pkl --map_file sf_map.pkl --method=eigenplaces --backbone=ResNet50 \
                                 --descriptors_dimension=512 
