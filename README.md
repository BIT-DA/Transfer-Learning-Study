# 迁移学习paper list与学习上手指南
> 推荐大家经常使用知乎和google搜索你看不懂的问题~自己主动学习十分重要！<br>
> Best代表马上阅读<br>
> Oral代表以后必须读<br>
*******************
## 目录
- [迁移学习paper list与学习上手指南](#迁移学习paper-list与学习上手指南)
  - [目录](#目录)
  - [0. 热点方向<br>](#0-热点方向)
    - [3D point cloud<br>](#3d-point-cloud)
    - [Continuous Learning<br>](#continuous-learning)
    - [Few-shot/ semi-supervised/ long-tailed classification/segmentaiton/detection<br>](#few-shot-semi-supervised-long-tailed-classificationsegmentaitondetection)
    - [Vision-Language/ prompt learning/ Multimodal<br>](#vision-language-prompt-learning-multimodal)
    - [Diffusion model<br>](#diffusion-model)
    - [OOD detection/generalization](#ood-detectiongeneralization)
    - [大模型/ pre-training/ self-supervised/ representation learning](#大模型-pre-training-self-supervised-representation-learning)
    - [BEV](#bev)
    - [edge computing transfer/ federate learning](#edge-computing-transfer-federate-learning)
    - [transfer reinforcement learning](#transfer-reinforcement-learning)
  - [1.迁移学习的背景介绍<br>](#1迁移学习的背景介绍)
  - [2.样本加权方法](#2样本加权方法)
  - [3.特征学习方法及其扩展](#3特征学习方法及其扩展)
  - [4.纯深度学习网络结构研究](#4纯深度学习网络结构研究)
  - [5.生成对抗网络（GAN）](#5生成对抗网络gan)
  - [6.深度迁移学习](#6深度迁移学习)
  - [7.Partial Domain Adaptation （PDA）](#7partial-domain-adaptation-pda)
  - [8.迁移学习在Semantic Segmentation中的应用](#8迁移学习在semantic-segmentation中的应用)
  - [9.迁移学习在Object Detection中的应用](#9迁移学习在object-detection中的应用)
  - [10. 迁移学习在Video Classification中的应用](#10-迁移学习在video-classification中的应用)
  - [11.迁移学习在Person Re-identification中的应用](#11迁移学习在person-re-identification中的应用)
  - [12.迁移学习理论文章](#12迁移学习理论文章)
  - [其他应该看论文](#其他应该看论文)
  - [顶会接收论文网站](#顶会接收论文网站)
  - [迁移学习资源汇总网站](#迁移学习资源汇总网站)
  - [大牛主页](#大牛主页)
*******************

## 0. 热点方向<br>

### 3D point cloud<br>
| paper | 来源 | Novelty | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
*******************

### Continuous Learning<br>
| paper | 来源 | Novelty | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
*******************

### Few-shot/ semi-supervised/ long-tailed classification/segmentaiton/detection<br>
| paper | 来源 | Novelty | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
*******************

### Vision-Language/ prompt learning/ Multimodal<br>
| paper | 来源 | Novelty | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
Learning transferable visual models from natural language supervision | ICML 2021 | Best | 需要 | CLIP | 
StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery | ICCV 2021 | Best | | StyleCLIP |
Learning to Prompt for Vision-Language Models | IJCV 2022 | Oral | | CoOp |
Open-Vocabulary Object Detection via Vision and Language Knowledge Distillation | ICLR 2022 | Oral | |  VILD |
OPEN-VOCABULARY OBJECT DETECTION VIA VISION AND LANGUAGE KNOWLEDGE DISTILLATION | ICLR 2022 | Oral | | LSeg | 
PIX2SEQ: A LANGUAGE MODELING FRAMEWORK FOR OBJECT DETECTION | ICLR 2022 | Oral | | Pix2Seq | 
DenseCLIP: Language-Guided Dense Prediction with Context-Aware Prompting | CVPR 2022 | Oral | | DenseCLIP|
*******************

### Diffusion model<br>
| paper | 来源 | Novelty | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
*******************

### OOD detection/generalization
| paper | 来源 | Novelty | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
*******************

### 大模型/ pre-training/ self-supervised/ representation learning
| paper | 来源 | Novelty | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
*******************

### BEV 
| paper | 来源 | Novelty | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
*******************

### edge computing transfer/ federate learning
| paper | 来源 | Novelty | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
*******************

### transfer reinforcement learning
| paper | 来源 | Novelty | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
*******************



## 1.迁移学习的背景介绍<br>
> 这些文章无需讲解，最先阅读，对迁移学习有个浅显了解

| paper | 来源 | Novelty | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
|迁移学习简明手册			|	这是中文版pdf	|Best|||
|A Survey on Transfer Learning |	IEEE TKDE	| Best |	 |  |
|Deep Visual Domain Adaptation A Survey		|	|	Oral		 |	 |  |
|Transfer Adaptation Learning：A Decade Survey||Oral||  |

*******************
## 2.样本加权方法
> 这类方法已是很早期的方法了，但是需要了解

| paper | 来源 | Novelty | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
|Prediction Reweighting for Domain Adaptation		|		IEEE TNNLS |	Best	|	PRDA |
|Correcting Sample Selection Bias by Unlabeled Data	|			NIPS	|Oral	|	KMM|
|Unsupervised Domain Adaptation with Distribution Matching Machines		|		AAAI 2018	| Oral	|

*******************
## 3.特征学习方法及其扩展
> 主要是浅层优化算法


| paper | 来源 | Novelty | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
|Domain Adaptation via Transfer Component Analysis|		IEEE TNNLS|	Best||TCA|
|Transfer Feature Learning with Joint Distribution Adaptation|CVPR|Best|需要|JDA|
|Domain Invariant and Class Discriminative Feature Learning for Visual Domain Adapation|IEEE TIP|Best|需要|DICD|
|Transfer Joint Matching for Unsupervised Domain Adaptation|CVPR|Oral||TJM|
|Unsupervised Domain Adaptation With Label and Structural Consistency|IEEE TIP|Oral||			
|Joint Geometrical and Statistical Alignment for Visual Domain Adaptation|CVPR 2018|Oral||JGSA|
|Adaptation Regularization: A General Framework for Transfer Learning|IEEE TKDE|Oral|		

*******************
## 4.纯深度学习网络结构研究
> 基础中的基础！推荐先看Stanford CS231n课程，百度搜索即可，可以快速了解深度学习

| paper | 来源 | Novelty | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
|
|ImageNet Classiﬁcation with Deep Convolutional Neural Networks			|NIPS|Best|需要|AlexNet|
|Very Deep Convolutional Networks for Large-Scale Image Recognition			||Best|需要|VGG|
|Deep Residual Learning for Image Recognition			|CVPR 2016 Best paper|Best|需要|ResNet|
|Densely Connected Convolutional Networks			|CVPR 2017 Best paper|Best|需要|DenseNet|
|Squeeze-and-Excitation Networks			|CVPR|Best||SENet|
|Batch Normalization Accelerating Deep Network Training by Reducing Internal Covariate Shift		|ICML|Best ||BN|
|Deep Networks with Stochastic Depth |ECCV 2016 spotlight|Oral|||
|Going deeper with convolutions			|NIPS|Oral||InceptionV1/GoogLeNet|
|Rethinking the Inception Architecture for Computer Vision			|CVPR|Oral||InceptionV2/V3|
|Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning			|CVPR|Oral||InceptionV4/Inception-ResNet|
|Aggregated Residual Transformations for Deep Neural Networks			|CVPR|Oral||ResNext|

*******************
## 5.生成对抗网络（GAN）
| paper | 来源 | Novelty | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
|Generative Adversarial Nets				|NIPS	|Best	|需要	|GAN|
|Wasserstein GAN	|  |Best | |WGAN|
|Conditional Generative Adversarial Nets|	|Oral|	|	CGAN|
|Least Squares Generative Adversarial Networks|	ICCV|	Oral| | LSGAN|

*******************
## 6.深度迁移学习
> Deep Domain Adaptation，针对分类问题，**研究重点！！**

| paper | 来源 | Novelty | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
|How transferable are features in deep neural networks|				NIPS 2014|	Best|||		
|Learning Transferable Features with Deep Adaptation Networks|				ICML 2015|	Best|	需要|	DAN|
|Unsupervised Domain Adaptation by Backpropagation|				ICML 2015|	Best|需要|	DANN/RevGrad|
|Maximum Classifier Discrepancy for Unsupervised Domain Adaptation|				CVPR 2018|	Best|	需要|	MCD|
|Unsupervised Domain Adaptation with Residual Transfer Networks|NIPS 2015|	Oral||		RTN|
|CyCADA: Cycle-Consistent Adversarial Domain Adaptation|				ICML	|Oral		||CyCADA|
|Multi-Adversarial Domain Adaptation|	AAAI 2018|	Oral|	|	MADA|
|Sliced Wasserstein Discrepancy for Unsupervised Domain Adaptation|				CVPR 2019|	Oral| |	SWD|
|Moment Matching for Multi-Source Domain Adaptation| ICML 2019|	Oral|||		
|Bridging Theory and Algorithm for Domain Adaptation| ICML 2019|	Oral||		MDD|
|Conditional Adversarial Domain Adaptation|	NIPS 2019|	Best||		CDAN|
|Contrastive Adaptation Network for Unsupervised Domain Adaptation|	CVPR 2018	|Oral| |CAN|
*******************
## 7.Partial Domain Adaptation （PDA）

| paper | 来源 | Novelty | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
|Partial Transfer Learning with Selective Adversarial Networks|				CVPR 2018|	Best|	需要|	SAN|
|Importance Weighted Adversarial Nets for Partial Domain Adaptation|				CVPR 2018|	Best|	需要|	IWAN|
|Partial Adversarial Domain Adaptation|				ECCV 2018|	Oral |  |	PADA|
|Learning to Transfer Examples for Partial Domain Adaptation|CVPR 2019|	Oral|	|	ETN|
Universal Domain Adaptation| CVPR 2019|	Oral|||

*******************
## 8.迁移学习在Semantic Segmentation中的应用

| paper | 来源 | Novelty | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
|Fully Convolutional Networks for Semantic Segmentation|				CVPR 2015|	Best|	|	FCN|		
|SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers | NeurIPS 2021 | 必读 1 | 需要 | SegFormer |
|Learning to Adapt Structured Output Space for Semantic Segmentation | CVPR 2018 | 必读 1 | 需要 | AdaptSegNet |
|DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation | CVPR 2022 | 必读 1 | 需要 | DAFormer |
|CyCADA - Cycle-Consistent Adversarial Domain Adaptation|				ICML|	Best|	|	CyCADA|		
|Taking A Closer Look at Domain Shift: Category-level Adversaries for Semantics Consistent Domain Adaptation|				CVPR 2019|	Oral|| CLAN |
|ADVENT: Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation|	CVPR 2019|	Oral|	|ADVENT|
|FDA: Fourier Domain Adaptation for Semantic Segmentation | CVPR 2020 | Oral |  | FDA |
|Confidence Regularized Self-Training | ICCV 2019 | Oral |  | CRST |
| Prototypical Pseudo Label Denoising and Target Structure Learning for Domain Adaptive Semantic Segmentation | CVPR 2021 | Oral |  | ProDA |
*******************
## 9.迁移学习在Object Detection中的应用

| paper | 来源 | Novelty | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
|Domain Adaptive Faster R-CNN for Object Detection in the Wild|				CVPR 2018|	Oral|||
|Multi-adversarial Faster-RCNN for Unrestricted Object Detection|				ICCV 2019|	Oral|||

*******************
## 10. 迁移学习在Video Classification中的应用

| paper | 来源 | Novelty | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
|Temporal Attentive Alignment for Large-Scale Video Domain Adaptation|				ICCV 2019	|Best|||

*******************
## 11.迁移学习在Person Re-identification中的应用

| paper | 来源 | Novelty | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
|Image-Image Domain Adaptation with Preserved Self-Similarity and Domain-Dissimilarity for Person Re-identification|				CVPR 2018|	Best|||
|Self-similarity Grouping: A Simple Unsupervised Cross Domain Adaptation Approach for Person Re-identification|				ICCV 2019|	Best|||

*******************
## 12.迁移学习理论文章

| paper | 来源 | Novelty | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
|Analysis of Representations for Domain Adaptation|				NIPS|	Best|||		
|A Theory of Learning from Different Domains|				|	Best|		||
|A Kernel Method for the Two Sample Problem|				NIPS|	Best|		|MMD的全面分析|

*******************
## 其他应该看论文

| paper | 来源 | Novelty | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
|Distance Metric Learning for Large Margin Nearest Neighbor Classification|				NIPS|	Best|	|	LMNN|
|FaceNet - A Unified Embedding for Face Recognition and Clustering|					Best|	|	Triplet Loss|

*******************
## 顶会接收论文网站

* [CVPR/ICCV/ECCV](http://openaccess.thecvf.com/CVPR2018.py)
* [ICML 2019](https://icml.cc/Conferences/2019/ScheduleMultitrack)
* [AAAI](https://icml.cc/Conferences/2019/ScheduleMultitrack)
* [ICLR 2019](https://openreview.net/group?id=ICLR.cc/2019/Conference#accepted-oral-papers)
* [NIPS 2018](http://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)

## 迁移学习资源汇总网站

* [awsome DA](https://github.com/zhaoxin94/awsome-domain-adaptation) _貌似只有论文_
* [迁移学习](http://transferlearning.xyz/)

## 大牛主页
* [清华 黄高老师](http://www.gaohuang.net/)
* [清华 龙明盛老师](http://ise.thss.tsinghua.edu.cn/~mlong/)
* [日本 原田研究所（MCD作者等）](https://www.mi.t.u-tokyo.ac.jp/publication/)	
