<!-- PROJECT LOGO -->
<h1 align="center">Benchmark for Supervised Surround-view Depth Estimation</h1>

###  [Paper](https://arxiv.org/abs/2303.07759)
> Benchmark for Supervised Surround-view Depth Estimation     
> [Xianda Guo](https://scholar.google.com.hk/citations?hl=zh-CN&user=jPvOqgYAAAAJ) , [Wenjie Yuan](https://scholar.google.com.hk/citations?user=3TjQ1soAAAAJ&hl=zh-CN), Yunpeng Zhang, Tian Yang, Chenming Zhang, [Zheng Zhu](http://www.zhengzhu.net/), [Long Chen](https://scholar.google.com/citations?user=jzvXnkcAAAAJ&hl=en)


## Introduction
Depth estimation has been widely studied and serves as the fundamental step of 3D perception for autonomous driving. Though significant progress has been made in monocular depth estimation in the past decades, these attempts are mainly conducted on the KITTI benchmark with only front-view cameras, which ignores the correlations across surround-view cameras. In this paper, we propose S3Depth, a Simple Baseline for Supervised Surround-view Depth Estimation, to jointly predict the depth maps across multiple surrounding cameras. Specifically, we employ a global-to-local feature extraction module that combines CNN with transformer layers for enriched representations. Further, the Adjacent-view Attention mechanism is proposed to enable the intra-view and inter-view feature propagation. The former is achieved by the self-attention module within each view, while the latter is realized by the adjacent attention module, which computes the attention across multi-cameras to exchange the multi-scale representations across surround-view feature maps. Extensive experiments show that our method achieves superior performance over existing state-of-the-art methods on both DDAD and nuScenes datasets.

## Model Zoo

| type     | dataset | Abs Rel | Sq Rel | RMSE | RMSE log | delta < 1.25 | delta < 1.25^2 | delta < 1.25^3 | pretrain|
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| pred_depth_median | DDAD | 0.153 | 2.417 | 10.664 | 0.254 | 0.818 | 0.927 | 0.963 | [model](https://pan.baidu.com/s/1kiEzMI8oFD6m4ikEj9h6rA?pwd=j1s2)|
| scale-aware | DDAD | 0.160  | 2.527 | 10.803 | 0.263 | 0.799 | 0.922 | 0.960 | [model](https://pan.baidu.com/s/1kiEzMI8oFD6m4ikEj9h6rA?pwd=j1s2) |
| pred_depth_median | nuScenes | 0.073 | 0.664 | 2.491 | 0.144 | 0.948 | 0.971 | 0.982 |  [model](https://pan.baidu.com/s/1ZfjIPVHPiBn8yC7yy30ahA?pwd=rfru) |
| scale-aware | nuScenes | 0.067  | 0.673 | 2.457 | 0.144 | 0.951 | 0.970 | 0.981 | [model](https://pan.baidu.com/s/1ZfjIPVHPiBn8yC7yy30ahA?pwd=rfru) |

## Install
* python 3.7.11, PyTorch 1.9.0, CUDA 11.1, RTX 3090
```bash
git clone https://github.com/XiandaGuo/SSDepth.git
conda create -n s3depth python=3.8
conda activate s3depth
pip install -r requirements.txt
```
Since we use [dgp codebase](https://github.com/TRI-ML/dgp) to generate ground-truth depth, you should also install it. 

## Data Preparation
Datasets are assumed to be downloaded under `data/<dataset-name>`.

### DDAD
* Please download the official [DDAD dataset](https://tri-ml-public.s3.amazonaws.com/github/DDAD/datasets/DDAD.tar) and place them under `data/ddad/raw_data`. You may refer to official [DDAD repository](https://github.com/TRI-ML/DDAD) for more info and instructions.
* Please download [metadata](https://cloud.tsinghua.edu.cn/f/50cb1ea5b1344db8b51c/?dl=1) of DDAD and place these pkl files in `datasets/ddad`.
* We provide annotated self-occlusion masks for each sequence. Please download [masks](https://cloud.tsinghua.edu.cn/f/c654cd272a6a42c885f9/?dl=1) and place them in `data/ddad/mask`.
* Export depth maps for evaluation 
```bash
cd tools
python export_gt_depth_ddad.py train
python export_gt_depth_ddad.py val
```

* The final data structure should be:
```
SurroundDepth
├── data
│   ├── ddad
│   │   │── raw_data
│   │   │   │── 000000
|   |   |   |── ...
|   |   |── ddad_depth_train
│   │   │   │── 000000
|   |   |   |── ...
|   |   |── ddad_depth_val
│   │   │   │── 000000
|   |   |   |── ...
```

### nuScenes
* Please download the official [nuScenes dataset](https://www.nuscenes.org/download) to `data/nuscenes/raw_data`
* Export depth maps for evaluation 
```bash
cd tools
python export_gt_depth_nusc.py train
python export_gt_depth_nusc.py val
```
* Gets a token that has a front and back frame ('pre' and 'next' attributes), which is given in the code.
```bash
python create_nusc_token.py
```
* The final data structure should be:
```
SurroundDepth
├── data
│   ├── nuscenes
│   │   │── raw_data
│   │   │   │── samples
|   |   |   |── sweeps
|   |   |   |── maps
|   |   |   |── v1.0-trainval
|   |   |── depth
│   │   │   │── samples
|   |   |── all_depth_train
│   │   │   │── samples
├── datasets
│   ├── nusc
│   │   │── depth_train_prenext.txt
│   │   │── depth_val.txt
```

## Training
Take nuScenes dataset as an example. 
```bash
python -m torch.distributed.launch --nproc_per_node 8 run.py  --model_name mpvit_selfadj  --config configs/nusc_mpvit_selfadj.txt
```

## Evaluation
```bash
python -m torch.distributed.launch --nproc_per_node ${NUM_GPU}  run.py  --model_name test  --config configs/${TYPE}.txt --models_to_load depth encoder   --load_weights_folder=${PATH}  --eval_only 
```


## Acknowledgement

[SurroundDepth](https://github.com/weiyithu/SurroundDepth).

## Citation

If you find this project useful in your research, please consider citing:
```
@article{guo2023simple,
  title={A simple baseline for supervised surround-view depth estimation},
  author={Guo, Xianda and Yuan, Wenjie and Zhang, Yunpeng and Yang, Tian and Zhang, Chenming and Zhu, Zheng and Chen, Long},
  journal={arXiv preprint arXiv:2303.07759},
  year={2023}
}
```


