## Awesome Infrared Small Target Detection

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/yongxianLiu/Awesome-IRSTD)

-----
Note: 
- ![](https://img.shields.io/badge/Code-PyTorch-orange) represents offical code.
- ![](https://img.shields.io/badge/Code-PyTorch-green) represents reproduced code.

## Contents

- [Single-Frame](#Single-Frame)
	- [1) Variants of U-Net](#Variants-of-U-Net)
 	- [2) Variants of Generative Adversarial Network (GAN)](#Variants-of-Generative-Adversarial-Network)
  	- [3) Variants of Transformer](#Variants-of-Transformer)
  	- [4) Dual-branch](#Dual-branch)
  	- [5) Variants of Mamba](#Variants-of-Mamba)
  	- [6) Variants of Graph Neural Network (GNN)](#Variants-of-Graph-Neural-Network)
  	- [7) Variants of Diffusion Model](#Variants-of-Diffusion-Model)
  	- [8) Variants of Pyramid](#Variants-of-Pyramid)
  	- [9) Frequency](#Frequency)
- [Multi-Frame](#Multi-Frame)
- [Weak Supervision](#Weak-Supervision)
- [Lightweight](#Lightweight)
	- [1) Network](#Network)
 	- [2) Quantization](#Quantization)
  	- [3) Distillation](#Distillation)
  	- [4) Prune](#Prune)
- [Large Model](#Large-Model)
- [Loss](#Loss)
- [Self-Supervised Learning](#Self-Supervised-Learning)
- [Box](#Box)
- [Surveys](#Surveys)
- [Datasets](#Datasets)
- [Challenges](#Challenges)


## [Single-Frame](#Contents)


### [Variants of U-Net](#Contents)

- **MRF3Net**, MRF3Net: An Infrared Small Target Detection Network Using Multireceptive Field Perception and Effective Feature Fusion
  + X. Zhang, X. Zhang, S. -Y. Cao, B. Yu, C. Zhang and H. -L. Shen. **TGRS 2024**
  + [![](https://img.shields.io/badge/Link-Paper-blue)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10562332) [![](https://img.shields.io/badge/Code-PyTorch-orange)](https://github.com/Temperature-ai/MRF3Net)

- **TENet**, Target-Focused Enhancement Network for Distant Infrared Dim and Small Target Detection
  + Y. Tong, Y. Leng, H. Yang and Z. Wang. **TGRS 2024**
  + [![](https://img.shields.io/badge/Link-Paper-blue)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10697465)

- **SeRankDet**, Pick of the Bunch: Detecting Infrared Small Targets Beyond Hit-Miss Trade-Offs via Selective Rank-Aware Attention
  + Y. Dai et al. **TGRS 2024**
  + [![](https://img.shields.io/badge/Link-Paper-blue)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10677425) [![](https://img.shields.io/badge/Code-PyTorch-orange)](https://github.com/GrokCV/SeRankDet)

- Learning Contrast-Enhanced Shape-Biased Representations for Infrared Small Target Detection
  + Lin F, Bao K, Li Y, et al. **TIP 2024**
  + [![](https://img.shields.io/badge/Link-Paper-blue)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10508299)

- **CMNet**, Cross-Layer Feature Guided Multiscale Infrared Small Target Detection
  + B. Li et al. **GRSL 2024**
  + [![](https://img.shields.io/badge/Link-Paper-blue)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10415029)

- **AMFU-net**, Lightweight Infrared Small Target Detection Network Using Full-Scale Skip Connection U-Net 
  + W. Y. Chung, I. H. Lee and C. G. Park. **GRSL 2023**
  + [![](https://img.shields.io/badge/Link-Paper-blue)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10124752) [![](https://img.shields.io/badge/Code-PyTorch-orange)](https://github.com/cwon789/AMFU-net)

- **Dim2Clear**, Dim2Clear Network for Infrared Small Target Detection
  + M. Zhang, R. Zhang, J. Zhang, J. Guo, Y. Li and X. Gao. **TGRS 2023**
  + [![](https://img.shields.io/badge/Link-Paper-blue)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10091167)

- **ABC**, ABC: Attention with Bilinear Correlation for Infrared Small Target Detection
  + P. Pan, H. Wang, C. Wang and C. Nie. **ICME 2023**
  + [![](https://img.shields.io/badge/Link-Paper-blue)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10219645) [![](https://img.shields.io/badge/Code-PyTorch-orange)](https://github.com/PANPEIWEN/ABC)

- **UIU-Net**, UIU-Net: U-Net in U-Net for Infrared Small Object Detection
  + Wu X, Hong D, Chanussot J. **TIP 2022**
  + [![](https://img.shields.io/badge/Link-Paper-blue)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9989433) [![](https://img.shields.io/badge/Code-PyTorch-orange)](https://github.com/danfenghong/IEEE_TIP_UIU-Net) [![](https://img.shields.io/badge/Code-PyTorch-green)](https://github.com/XinyiYing/BasicIRSTD)

- **DNA-Net**, Dense Nested Attention Network for Infrared Small Target Detection
  + Li B, Xiao C, Wang L, et al. **TIP 2022**
  + [![](https://img.shields.io/badge/Link-Paper-blue)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9864119) [![](https://img.shields.io/badge/Code-PyTorch-orange)](https://github.com/YeRen123455/Infrared-Small-Target-Detection) [![](https://img.shields.io/badge/Code-PyTorch-green)](https://github.com/XinyiYing/BasicIRSTD)

- **MTUNet**, A Multi-Task Framework for Infrared Small Target Detection and Segmentation
  + Y. Chen, L. Li, X. Liu and X. Su. **TGRS 2022**
  + [![](https://img.shields.io/badge/Link-Paper-blue)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9847264) [![](https://img.shields.io/badge/Code-PyTorch-orange)](https://github.com/Chenastron/MTUNet)

- **SPSCNet**, Detecting Dim Small Target in Infrared Images via Subpixel Sampling Cuneate Network
  +  He X, Ling Q, Zhang Y, et al. **GRSL 2022**
  +  [![](https://img.shields.io/badge/Link-Paper-blue)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9817112)

- **ISTDU-Net**, ISTDU-Net: Infrared Small-Target Detection U-Net
  + Q. Hou, L. Zhang, F. Tan, Y. Xi, H. Zheng and N. Li. **GRSL 2022**
  + [![](https://img.shields.io/badge/Link-Paper-blue)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9674870) [![](https://img.shields.io/badge/Code-PyTorch-orange)](https://github.com/zhanglw882/ISTDU-Net) [![](https://img.shields.io/badge/Code-PyTorch-green)](https://github.com/XinyiYing/BasicIRSTD)

- **ALCNet**, Attentional Local Contrast Networks for Infrared Small Target Detection
  + Dai Y, Wu Y, Zhou F, et al. **TGRS 2021**
  + [![](https://img.shields.io/badge/Link-Paper-blue)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9314219) [![](https://img.shields.io/badge/Code-MXNet-orange)](https://github.com/YimianDai/open-alcnet) [![](https://img.shields.io/badge/Code-PyTorch-green)](https://github.com/XinyiYing/BasicIRSTD)

- **ACM**, Asymmetric Contextual Modulation for Infrared Small Target Detection
  + Dai Y, Wu Y, Zhou F, et al. **WACV 2021**
  + [![](https://img.shields.io/badge/Link-Paper-blue)](https://openaccess.thecvf.com/content/WACV2021/papers/Dai_Asymmetric_Contextual_Modulation_for_Infrared_Small_Target_Detection_WACV_2021_paper.pdf) [![](https://img.shields.io/badge/Code-MXNet-orange)](https://github.com/YimianDai/open-acm) [![](https://img.shields.io/badge/Code-PyTorch-green)](https://github.com/Tianfang-Zhang/acm-pytorch) [![](https://img.shields.io/badge/Code-PyTorch-green)](https://github.com/XinyiYing/BasicIRSTD)



### [Variants of Generative Adversarial Network](#Contents)

- **DOCI-GAN**, Generative data augmentation by conditional inpainting for multi-class object detection in infrared images
  + Peng Wang, Zhe Ma, Bo Dong, Xiuhua Liu, Jishiyu Ding, Kewu Sun, Ying Chen, **PR 2024**
  + [![](https://img.shields.io/badge/Link-Paper-blue)](https://pdf.sciencedirectassets.com/272206/1-s2.0-S0031320324X00059/1-s2.0-S0031320324002528/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEKb%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQC0xOo%2B47CHgSzjfecw6oBK3RaOFFx0ovRZkKb7dar6zgIgLQGlpobmMmmjzylh9G9Ts2LyAgA1XC4EMRKgWFWC6oUqvAUI7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDM9s3rFj%2BJRoZHWmuCqQBbWn5qXPVOvfVqseP6VqFpldvQwuC2YqtJHXu6PbGaCTy6mZmMGcrxdQlmzDDBPvY1vhQ9A%2FzD5mJ5y7omM9s4Dbzsf5cSouI4hXYtl3QYuINgpHfL48%2BTypWKmDO%2BLuMrMljclJB5vQai8d9ZgYZYz8XJ%2Fhjbe4UcCFX0Eq7jGmr1wa0x94EPX6t4MynDDPUNm%2BjSviCrAZ2evr8OBnyWyLSnYqQ1KQvgP%2FXEKA9vjMP%2BOTLczL%2Fcen8%2B6xoosdJ1LMSSNd1nMnNo6bYkyWqsVr72DtawiJRGO5hvsJgeHEAitZcO19b1KJWkwyE%2BgQmmT2QBHgb6Yw2wvpjgrQHUgyhJ4E5k%2F3xOuDoTboHLJGKQWArfiZrudX00sSUYwv4FG9aEZ9WrOoD6nJ9d4TD4lW6HrmF3U3KSPFleKey7edQNoIThqtBydibFRYd0mq8YYdWk%2F%2F%2FsOL%2F1BqyP8yPPRI2acAj1JWkLXsSlpYFzCpkaylGQscfyOA44WCEqO0SV7cMvqnluI5T2vmBjyxHuMq6D0Nh%2Fd4MYZl6VQmBnjLRUiwO%2BQ03bbzhyF88sQTcpLqoKdyBhldaYuiRLr3q0H%2FR7qJX%2F%2F78kLWBBmH27gUZg23W7U528YG6OIZvpFEyaa0%2FwM99vmcsZ%2FG7gC%2FJRdL2oEbNUM7hdEklz%2BE7u8CW%2BwHebGsI9YldUKbZyMP3PIVPLwBD2f5KNK2y4P6sPym43A7lofo%2Bfd86lPC6p7mKOc4jnFhbdfP9il5XzFld1MFMQ9%2FPmSFOI5Ztnl5Va6XzpjWLJCgYqIaNiiy%2Fm9ETVnYzCtz5DAs%2FIKb27uBhTVyGLCn2JjRxc9hDFJoo6ki5p7nR7TBM%2BU%2FAuLhH%2BZxMNKAhbgGOrEB9vvUDE%2BcREpjCsvGNCcA1cs%2Bm1BNO1KTYxBosaaW2aDJRDS2KIY1S%2Bvrz2Blo6zwXQaxmllvyDKFSxifsxjOffnU3k5kU8Vy0uoCZ2BhotPr2rTX2sXH1g6Fv2M4JcUKZ7uIVzz3C3%2FtXjnGyfkzwdSdjeMot%2BPR8rsUCPGENNltqAd6B40l9hb4OaycS8Dj91guymeGjCLyKy9nFfm4aTk%2BNAEJyCRtLyhPGK%2FFwrZr&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20241005T135532Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYXV6GWBGC%2F20241005%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=9b5fe58148b506c67f01e6966c72ab6eeb23e9d818575feffa9d227f6f31be15&hash=5d4508d7b2cae41eaaff95854dd07c4797c3c78affaae70d1f9382afeeae865f&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0031320324002528&tid=spdf-d7217271-516a-4813-874a-cb37a77fc002&sid=ab9d21a230c801488008c5f33cc949289fbagxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=15165b070601055e07&rr=8cdde8ead9c55f4e&cc=us&kca=eyJrZXkiOiJVcXZPbjRZL3NxTS9YcU9HWG9LSzl3Sm5zekgzdGxjdVcvSnpnbldXZEhFeHZrd3o0VWJKem8xMlRzSEx2U2FYMlRwcFJsL3AvZGRscEdhM05MR3FxZ3FMMnRubmpCSVY0ckhqVEpURnIrWS9PUmRaa1VJZXd5NnZnU09JTWlmM1VhaStyYmRleE9pTWtmM2VudksyRUwwTDh6Y0JWVDNwZnhNa1NRNVJ2UEluTlU2eSIsIml2IjoiYzZmMzcxNzBhYzM3ZmFlZDVlZDZiNjEyNzEzZTc4MGQifQ==_1728136544113)

- **MDvsFA-cGAN**, Miss Detection vs. False Alarm Adversarial Learning for Small Object Segmentation in Infrared Images
  + Wang H, Zhou L, Wang L. **ICCV 2019**
  + [![](https://img.shields.io/badge/Link-Paper-blue)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Miss_Detection_vs._False_Alarm_Adversarial_Learning_for_Small_Object_ICCV_2019_paper.pdf) [![](https://img.shields.io/badge/Code-PyTorch-orange)](https://github.com/wanghuanphd/MDvsFA_cGAN)


### [Variants of Transformer](#Contents)
- **IRSTFormer**, IRSTFormer: A Hierarchical Vision Transformer for Infrared Small Target Detection
  + Chen G, Wang W, Tan S. **Remote Sensing 2022**
  + [![](https://img.shields.io/badge/Link-Paper-blue)](https://www.mdpi.com/2072-4292/14/14/3258)

### [Dual-branch](#Contents)

### [Variants of Mamba](#Content)

### [Variants of Graph Neural Network](#Content)
### [Variants of Diffusion Model](#Content)
### [Variants of Pyramid](#Content)
### [Frequency](#Content)


## [Multi-Frame](#Contents)


## [Weak Supervision](#Content)

## [Lightweight](#Content)

### [Network](#Content)
### [Quantization](#Content)
### [Distillation](#Content)
### [Prune](#Content)


## [Large Model](#Content)


## [Loss](#Content)


## [Self-Supervised Learning](#Content)

## [Box](#Content)

## [Surveys](#Content)


## [Datasets](#Contents)





## [Challenges](#Contents)





