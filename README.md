# SF-Loc

>SF-Loc: A Visual Mapping and Geo-Localization System based on Sparse Visual Structure  Frames


<div align=center>
<img alt="" src="./assets/abstract.svg" width='600px' />
</div>

[[Arxiv](https://arxiv.org/abs/2412.01500)]

**An important typo**: In the current arxiv version of the manuscript, there is a description ``to verify the effectiveness of the proposed SSS metric, we also implement a naive multi-frame VPR model by clustering the top-K candidates of multiple queries''. Here, "top-K" should be "top-10". Sorry for this typo.

## What is this? 

**SF-Loc** is vision-centered mapping and localization system, based on the map representation of visual structure frames. We use **multi-sensor dense bundle adjustment (MS-DBA)** to generate the visual structure frames and sparsify them through co-visibility checking, leading to lightweight map storage. On this basis, multi-frame information is utilized to achieve high-recall, accurate user-side **map-based localization**.

## Update log
- [x] Mapping Pipeline (deadline: 2024/12)
- [x] Localization Pipeline (deadline: 2025/01)
- [x] Multi-session Setup (TMech manuscript version, 2025/10/09)

## Installation
The pipeline of the work is based on python, and the computation part is mainly based on Pytorch (with CUDA) and GTSAM.

Use the following command to set up the python environment.

```Bash
conda create -n sfloc python=3.11.9
conda activate sfloc

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

pip install \
    numpy==1.26.4 \
    numpy-quaternion==2024.0.10 \
    opencv-python==4.10.0.84 \
    scipy==1.16.0 \
    matplotlib==3.10.3 \
    h5py==3.14.0 \
    pyparsing==3.2.3 \
    tqdm==4.67.1 \
    gdown==5.2.0
```

As for GTSAM, we make some modifications to it to extend the python wrapper APIs, clone it from the following repository and install it under your python environment.

```Bash
git clone https://github.com/yuxuanzhou97/gtsam.git
cd gtsam
mkdir build
cd build
cmake .. -DGTSAM_BUILD_PYTHON=1 -DGTSAM_PYTHON_VERSION=3.11.9
make python-install
```

Finally, run the following command to compile the project.

```Bash
git clone --recurse-submodules https://github.com/GREAT-WHU/SF-Loc.git
cd SF-Loc
python setup.py install
```


## Run SF-Loc
We use the weights of [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM?tab=readme-ov-file) trained on TartanAir for optical flow estimation and DBA. Download [droid.pth](https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view?usp=sharing) (2021/9/1) and put it in this project. 

The provided dataset contains three sequences (2022/10/23) for multi-session mapping and one sequence (2023/04/12) for user-side localization, as shown below.

<div align=center>
<img alt="" src="./assets/ms-mapping.svg" width='400px' />
</div>

### 1. Mapping Phase (Multi-Session)

In the mapping phase, multi-sensor data are used for dense bundle adjustment (DBA) to recover image depths and poses. Based on the global optimization results, the lightweight structure frame map is generated. 

1.1 Download the [WHU1023m](TODO) data sequence.

1.2  Specify the data paths in [launch_dba.py](launch_dba.py), then run the following command 
```Bash
python launch_dba.py  # This would trigger demo_vio_WHU1023.py automatically. Three sessions (A/B/C) would be processed sequentially.
```
This will launch a task for online multi-sensor DBA. Generally, **1x** real-time performance is expected on a **4080 laptop**, which would cost around **90 minutes** to process the provided data sequences (A/B/C). After finished, the following files would be generated.

* **poses_{A/B/C}_realtime.txt** &emsp; IMU poses (both in world frame and ECEF frame) estimated by online multi-sensor DBA.
* **graph_{A/B/C}.pkl** &emsp; Serialized GTSAM factors that store the multi-sensor DBA information.
* **depth_video_{A/B/C}.pkl** &emsp; Dense depths estimated by DBA.

1.3 
Run the following command for global factor optimization (post-processing). This wouldn't cost a long time.
```Bash
sh post_optimization.sh
```
The following file would be generated.
* **poses_{A/B/C}_post.txt** &emsp; Estimated IMU poses after global optimization.

1.4 
Run the following command to sparsify the keyframe map.
```Bash
sh sparsify.sh # by default, eigenplaces (ResNet50) (accessed: 2025/10/09) would be employed.
```

The following file would be generated.
* **map_indices.pkl** &emsp; Map frame indices (and timestamps), indicating a subset of all DBA keyframes.

1.5
Run the following command to (finally) generate the lightweight structure frame map.
```Bash
sh generate.sh # by default, superpoint + lightglue (accessed: 2025/10/09) would be employed.
```

The following file would be generated.
* **sf_map.pkl**: The structure frame map, which is all you need for re-localization.

In this step, the scripts provided by [VPR-methods-evaluation](https://github.com/gmberton/VPR-methods-evaluation) would be called. Thanks for Gmberton's great work, which provides convenient interface for different VPR methods.

🎇So far, a lightweight map file (**≈ 50MB**) of the region is generated. To evaluate the global pose estimation performance, run the following command
```Bash
python scripts/evaluate_map_poses_ABC.py # for multi-session setup
```


### 2. Localization Phase

In the localization phase, [LightGlue](https://github.com/cvg/LightGlue) is needed for fine association. Please install it under the current environment first.

2.1 Download the [WHU0412m](TODO) data sequence.

2.1
Run the following command to perform the localization.
```Bash
sh localize.sh
```
The following file would be generated.
* **result_coarse.txt** &emsp; Coarse user localization results (position and map indice) based on VPR.
* **result_fine.txt** &emsp; Fine user localization results (local and global poses).

To evaluate the coarse/fine map-based localization performance, run the following commands
```Bash
python scripts/evaluate_coarse_poses.py
python scripts/evaluate_fine_poses.py
```




## Acknowledgement
SF-Loc is developed by [GREAT](http://igmas.users.sgg.whu.edu.cn/group) (GNSS+ REsearch, Application and Teaching) Group, School of Geodesy and Geomatics, Wuhan University. 

<br/>
<div align=center>
<img alt="" src="./assets/GREAT.png" width='300px' />
</div>
<br/>
<div align=center>
<img alt="" src="./assets/whu.png" width='300px' />
</div>
<br/>

This work is based on [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM), [VPR-methods-evaluation](https://github.com/gmberton/VPR-methods-evaluation) and [GTSAM](https://github.com/borglab/gtsam). 

All pretrained models are used under their respective licenses.