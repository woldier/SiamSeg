

<div align="center">

<h2 style="border-bottom: 1px solid lightgray;">SiamSeg: Self-Training with Contrastive Learning for Unsupervised Domain Adaptation Semantic Segmentation in Remote Sensing</h2>

<div style="display: flex; align-items: center; justify-content: center;">

<p align="center">
  <a href="#">
  <br align="center">
    <a href='https://arxiv.org/abs/2410.13471'>
        <img src='http://img.shields.io/badge/Paper-arxiv.2410.13471-B31B1B.svg?logo=arXiv&logoColor=B31B1B'>
    </a>
    <img alt="Static Badge" src="https://img.shields.io/badge/python-v3.8-green?logo=python">
    <img alt="Static Badge" src="https://img.shields.io/badge/torch-v1.10.2-B31B1B?logo=pytorch">
    <img alt="Static Badge" src="https://img.shields.io/badge/mmcv-v1.5.0-blue">
    <img alt="Static Badge" src="https://img.shields.io/badge/torchvision-v0.11.3-B31B1B?logo=pytorch">
    </br>
    <img alt="GitHub Issues or Pull Requests" src="https://img.shields.io/github/issues/woldier/SiamSeg">
    <img alt="GitHub Issues or Pull Requests" src="https://img.shields.io/github/issues-closed/woldier/SiamSeg?color=ab7df8">
    <img alt="GitHub forks" src="https://img.shields.io/github/forks/woldier/SiamSeg?style=flat&color=red">
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/woldier/SiamSeg?style=flat&color=af2626">
  </p>
</p>

</div>

<br/>

<img src="figs/visual_res1.png" alt="Framework" style="max-width: 100%; height: auto;"/>
<div style="display: flex; align-items: center; justify-content: center;"> Prediction results of our proposed method. </div>
<br/>
<br/>


<img src="figs/overview.png" alt="SiamSeg" style="max-width: 100%; height: auto;"/>
<div style="display: flex; align-items: center; justify-content: center;">
    <img src="figs/contrastive_learning.png" alt="Contrastive Learning" style="width:64%; height: auto;"/>
<img src="figs/different.png" alt="Cross-domain Different" style="width:35%; height: auto;"/>
</div>
<div style="display: flex; align-items: center; justify-content: center;"> Network Architectural and Cross-domain Image Different. </div>


</div>


## News:

---
[//]: # (- [2024/09/26]  Our paper is accepted to **NeurIPS 2024**.)

[//]: # (- [2024/09/25] ✨✨ We have updated the [arxiv]&#40;https://arxiv.org/abs/2403.07721&#41; paper.)

[//]: # (- [2024/08/01] Update scripts for training and inference in different tasks.)

[//]: # (- [2024/05/19] Update the dataset loading scripts.)
- [2024/10/18] ✨✨This work was submitted  to **IEEE TGRS**.
- [2024/10/17] ✨✨The [arxiv](https://arxiv.org/abs/2410.13471) paper is available.
- [2024/10/13] 🔥🔥Update the code and scripts.




## 1. Creating Virtual Environment

---
Install the necessary dependencies:

torch=1.10.2,torchvision=0.11.3 and mmcv-full=1.5.0.

Recommended use of conda virtual environments
```shell
conda create -n SiamSeg python==3.8 -y
conda activate SiamSeg 
````
pip command to install torch && torchvision && mmcv-full
```shell
pip install torch==1.10.2+cu111 torchvision==0.11.3+cu111 -f https://download.pytorch.org/whl/torch_stable.html 
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
pip install kornia matplotlib prettytable timm yapf==0.40.1
```
for CN user:
```shell
pip install torch==1.10.2+cu111 -f https://mirror.sjtu.edu.cn/pytorch-wheels/cu111/?mirror_intel_list
pip install torchvision==0.11.3+cu111 -f https://download.pytorch.org/whl/torch_stable.html 
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
pip install kornia matplotlib prettytable timm yapf==0.40.1
```
Installation of the reference document refer:

Torch and torchvision versions relationship.

[![Official Repo](https://img.shields.io/badge/Pytorch-vision_refer-EE4C2C?logo=pytorch)](https://github.com/pytorch/vision#installation)
[![CSDN](https://img.shields.io/badge/CSDN-vision_refer-FC5531?logo=csdn)](https://blog.csdn.net/shiwanghualuo/article/details/122860521)

Version relationship of mmcv and torch.

[![MMCV](https://img.shields.io/badge/mmcv-vision_refer-blue)](https://mmcv.readthedocs.io/zh-cn/v1.5.0/get_started/installation.html)


## 2.Preparation of data sets

---
We selected Postsdam, Vaihingen and LoveDA as benchmark datasets and created train, val, test lists for researchers.

### 2.1 Download of datasets

#### ISPRS Potsdam
The [Potsdam](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-potsdam/)
dataset is for urban semantic segmentation used in the 2D Semantic Labeling Contest - Potsdam.

The dataset can be requested at the challenge [homepage](https://www2.isprs.org/commissions/comm2/wg4/benchmark/data-request-form/).
The '2_Ortho_RGB.zip', '3_Ortho_IRRG.zip' and '5_Labels_all_noBoundary.zip' are required.

#### ISPRS Vaihingen

The [Vaihingen](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-vaihingen/)
dataset is for urban semantic segmentation used in the 2D Semantic Labeling Contest - Vaihingen.

The dataset can be requested at the challenge [homepage](https://www2.isprs.org/commissions/comm2/wg4/benchmark/data-request-form/).
The 'ISPRS_semantic_labeling_Vaihingen.zip' and 'ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE.zip' are required.

#### LoveDA

The data could be downloaded from Google Drive [here](https://drive.google.com/drive/folders/1ibYV0qwn4yuuh068Rnc-w4tPi0U0c-ti?usp=sharing).

Or it can be downloaded from [zenodo](https://zenodo.org/record/5706578#.YZvN7SYRXdF), you should run the following command:

```shell

cd /{your_project_base_path}/SiamSeg/data/LoveDA

# Download Train.zip
wget https://zenodo.org/record/5706578/files/Train.zip
# Download Val.zip
wget https://zenodo.org/record/5706578/files/Val.zip
# Download Test.zip
wget https://zenodo.org/record/5706578/files/Test.zip
```



### 2.2 Data set preprocessing
Place the downloaded file in the corresponding path
The format is as follows
```text
SiamSeg/
├── data/
│   ├── LoveDA/
│   │   ├── Test.zip
│   │   ├── Train.zip
│   │   └── Val.zip
├── ├── Potsdam_IRRG_DA/
│   │   ├── 3_Ortho_IRRG.zip
│   │   └── 5_Labels_all_noBoundary.zip
├── ├── Potsdam_RGB_DA/
│   │   ├── 2_Ortho_RGB.zip
│   │   └── 5_Labels_all_noBoundary.zip
├── ├── Potsdam_IRRG_DA/
│   │   ├── ISPRS_semantic_labeling_Vaihingen.zip
│   │   └── ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE.zip


```
- Potsdam
```shell
python tools/convert_datasets/potsdam.py data/Potsdam_IRRG/ --clip_size 512 --stride_size 512
python tools/convert_datasets/potsdam.py data/Potsdam_RGB/ --clip_size 512 --stride_size 512
```
- Vaihingen
```shell
python tools/convert_datasets/vaihingen.py data/Vaihingen_IRRG/ --clip_size 512 --stride_size 256
```
- LoveDA
```shell
cd data/LoveDA
unzip Train.zip, Val.zip, Test.zip
```

## 3.Training 

---
### 3.1 Preparation of pre-trained models

mit_b5.pth :
We provide a script [`mit2mmseg.py`](./tools/model_converters/mit2mmseg.py) in the tools directory to convert the key of models from [the official repo](https://github.com/NVlabs/SegFormer) to MMSegmentation style.
```shell
python tools/model_converters/mit2mmseg.py ${PRETRAIN_PATH} ./pretrained
```
Or you can download it from [google drive](https://drive.google.com/drive/folders/1cmKZgU8Ktg-v-jiwldEc6IghxVSNcFqk?usp=sharing).

The structure of the file is as follows
```text
SiamSeg/
├── pretrained/
│   ├── mit_b5.pth (needed)
│   └── ohter.pth  (option)
```


### 3.2 Potsdam IRRG to Vaihingen IRRG
> tips
> 
> When using distributed training scripts under linux, you need to set the permissions of the training scripts due to the file permissions.
> ```shell
> cd SiamSeg
> chmod 777 ./tools/dist_train.sh
> chmod 777 ./tools/dist_test.sh
> ```


```shell
# Potsdam IRRG to Vaihingen IRRG
# CUDA_VISIBLE_DEVICES Visible GPU ids are 0-3 Total four GPU processors
# PORT Sets the communication port of the master for distributed training.
# The last 4 indicates the total number of GPUs used
CUDA_VISIBLE_DEVICES=0,1,2,3  PORT=10985 \
 ./tools/dist_train.sh \
 configs/SiamSeg/siamseg_daformer_sepaspp_mitb5_512x512_40k_PotsdamIRRG_2_VaihingenIRRG.py 4
```


### 3.3 Potsdam RGB to Vaihingen IRRG

```shell
# Potsdam RGB to Vaihingen IRRG
# CUDA_VISIBLE_DEVICES Visible GPU ids are 0-3 Total four GPU processors
# PORT Sets the communication port of the master for distributed training.
# The last 4 indicates the total number of GPUs used
CUDA_VISIBLE_DEVICES=0,1,2,3  PORT=10985 \
 ./tools/dist_train.sh \
 configs/SiamSeg/siamseg_daformer_sepaspp_mitb5_512x512_40k_PotsdamRGB_2_VaihingenIRRG.py 4
```
---
### 3.4 Vaihingen IRRG to Potsdam IRRG
```shell
# Potsdam IRRG to Vaihingen IRRG
# CUDA_VISIBLE_DEVICES Visible GPU ids are 0-3 Total four GPU processors
# PORT Sets the communication port of the master for distributed training.
# The last 4 indicates the total number of GPUs used
CUDA_VISIBLE_DEVICES=0,1,2,3  PORT=10985 \
 ./tools/dist_train.sh \
 configs/SiamSeg/siamseg_daformer_sepaspp_mitb5_512x512_40k_VaihingenIRRG_2_PotsdamIRRG.py 4
```

### 3.5 Vaihingen IRRG to Potsdam RGB
```shell
# Potsdam IRRG to Vaihingen IRRG
# CUDA_VISIBLE_DEVICES Visible GPU ids are 0-3 Total four GPU processors
# PORT Sets the communication port of the master for distributed training.
# The last 4 indicates the total number of GPUs used
CUDA_VISIBLE_DEVICES=0,1,2,3  PORT=10985 \
 ./tools/dist_train.sh \
 configs/SiamSeg/siamseg_daformer_sepaspp_mitb5_512x512_40k_VaihingenIRRG_2_PotsdamRGB.py 4
```

### 3.6 LoveDA Rural to Urban
```shell
# Potsdam IRRG to Vaihingen IRRG
# CUDA_VISIBLE_DEVICES Visible GPU ids are 0-3 Total four GPU processors
# PORT Sets the communication port of the master for distributed training.
# The last 4 indicates the total number of GPUs used
CUDA_VISIBLE_DEVICES=0,1,2,3  PORT=10985 \
 ./tools/dist_train.sh \
 configs/SiamSeg/siamseg_daformer_sepaspp_mitb5_512x512_40k_Rural_2_Urban.py 4
```


## 4.Testing

### 4.1 Potsdam IRRG to Vaihingen IRRG
```shell
# for dist test
CUDA_VISIBLE_DEVICES=4,5,6,7  PORT=10985 \
  sh tools/dist_test.sh \
  configs/SiamSeg/siamseg_daformer_sepaspp_mitb5_512x512_40k_PotsdamRGB_2_VaihingenIRRG.py  \
  {beast_model_path}  4 --eval mIoU mFscore

# for predict label save  
# launcher must set to 'none'
# opacity between 0 and 1
PYTHONPATH=$(pwd):$PYTHONPATH  CUDA_VISIBLE_DEVICES=0 \
  python tools/test.py \
  configs/SiamSeg/siamseg_daformer_sepaspp_mitb5_512x512_40k_PotsdamRGB_2_VaihingenIRRG.py  \
  {beast_model_path} --eval mIoU mFscore --launcher none  --opacity 1.0
```


### 4.2 Potsdam RGB to Vaihingen IRRG
```shell
# for dist test
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=10985 \
  sh tools/dist_test.sh \
  configs/SiamSeg/siamseg_daformer_sepaspp_mitb5_512x512_40k_PotsdamRGB_2_VaihingenIRRG.py  \
  {beast_model_path}  4 --eval mIoU mFscore

# for predict label save  
# launcher must set to 'none'
# opacity between 0 and 1
PYTHONPATH=$(pwd):$PYTHONPATH  CUDA_VISIBLE_DEVICES=0 \
  python tools/test.py \
  configs/SiamSeg/siamseg_daformer_sepaspp_mitb5_512x512_40k_PotsdamRGB_2_VaihingenIRRG.py  \
  {beast_model_path} --eval mIoU mFscore --launcher none  --opacity 1.0
```


### 4.3 Vaihingen IRRG to Potsdam IRRG
```shell
# for dist test
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=10985 \
  sh tools/dist_test.sh \
  configs/SiamSeg/siamseg_daformer_sepaspp_mitb5_512x512_40k_VaihingenIRRG_2_PotsdamIRRG.py  \
  {beast_model_path}  4 --eval mIoU mFscore
  
# for predict label save  
# launcher must set to 'none'
# opacity between 0 and 1
PYTHONPATH=$(pwd):$PYTHONPATH  CUDA_VISIBLE_DEVICES=0 \
  python tools/test.py \
  configs/SiamSeg/siamseg_daformer_sepaspp_mitb5_512x512_40k_VaihingenIRRG_2_PotsdamIRRG.py  \
  {beast_model_path} --eval mIoU mFscore --launcher none  --opacity 1.0
```


### 4.4 Vaihingen IRRG to Potsdam RGB
```shell
# for dist test
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=10985 \
  sh tools/dist_test.sh \
  configs/SiamSeg/siamseg_daformer_sepaspp_mitb5_512x512_40k_VaihingenIRRG_2_PotsdamRGB.py  \
  {beast_model_path}  4 --eval mIoU mFscore
# for predict label save

# for predict label save  
# launcher must set to 'none'
# opacity between 0 and 1
PYTHONPATH=$(pwd):$PYTHONPATH  CUDA_VISIBLE_DEVICES=0 \
  python tools/test.py \
  configs/SiamSeg/siamseg_daformer_sepaspp_mitb5_512x512_40k_VaihingenIRRG_2_PotsdamRGB.py  \
  {beast_model_path} --eval mIoU mFscore --launcher none  --opacity 1.0
```


### 4.5 LoveDA Rural to Urban
```shell
# for dist test
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=10985 \
  sh tools/dist_test.sh \
  configs/SiamSeg/siamseg_daformer_sepaspp_mitb5_512x512_40k_Rural_2_Urban.py  \
  {beast_model_path}  4 --eval mIoU mFscore

# for predict label save  
# launcher must set to 'none'
# opacity between 0 and 1
PYTHONPATH=$(pwd):$PYTHONPATH  CUDA_VISIBLE_DEVICES=0 \
  python tools/test.py \
  configs/SiamSeg/siamseg_daformer_sepaspp_mitb5_512x512_40k_Rural_2_Urban.py  \
  {beast_model_path} --eval mIoU mFscore --launcher none  --opacity 1.0
```


# References

---
Many thanks to their excellent works
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [DACS](https://github.com/vikolss/DACS)
* [DAFormer](https://github.com/lhoyer/DAFormer)
