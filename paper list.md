# 迁移学习paper list与学习上手指南
> 推荐大家经常使用知乎和google搜索你看不懂的问题~自己主动学习十分重要！<br>
> 必读1代表马上阅读<br>
> 必读2代表以后必须读<br>
*******************
## 目录
* [背景介绍](#1-迁移学习的背景介绍-)
* [样本加权](#2-样本加权方法-)
* [特征学习](#3-特征学习方法及其扩展-)
* [深度学习](#4-纯深度学习网络结构研究-)
* [生成对抗网络](#5-生成对抗网络（GAN）-)
* [深度迁移学习](#6-深度迁移学习-)
* [Partial-Domain-Adaptation](#7-Partial Domain Adaptation-)
* [Semantic Segmentation](#8-迁移学习在Semantic Segmentation中的应用-)
* [Object Detection](#9-迁移学习在Object Detection中的应用-)
*******************
## 1. 迁移学习的背景介绍 <br>
> 这些文章无需讲解，最先阅读，对迁移学习有个浅显了解

| paper | 来源 | 备注 | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
|迁移学习简明手册			|	这是中文版pdf	|必读1|||
|A Survey on Transfer Learning |	IEEE TKDE	| 必读2 |	 |  |
|Deep Visual Domain Adaptation A Survey		|	|	必读2		 |	 |  |
|Transfer Adaptation Learning：A Decade Survey||必读2||  |

*******************
## 2. 样本加权方法
> 这类方法已是很早期的方法了，但是需要了解

| paper | 来源 | 备注 | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
|Correcting Sample Selection Bias by Unlabeled Data	|			NIPS	|必读2	|	KMM|
|Prediction Reweighting for Domain Adaptation		|		IEEE TNNLS |	必读1	|	PRDA |
|Unsupervised Domain Adaptation with Distribution Matching Machines		|		AAAI 2018	| 必读2	|

*******************
## 3.特征学习方法及其扩展
> 主要是浅层优化算法


| paper | 来源 | 备注 | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
|Domain Adaptation via Transfer Component Analysis|		IEEE TNNLS|	必读2||TCA|
|Transfer Feature Learning with Joint Distribution Adaptation|CVPR|必读1|需要|JDA|
|Transfer Joint Matching for Unsupervised Domain Adaptation|CVPR|必读2||TJM|
|Unsupervised Domain Adaptation With Label and Structural Consistency|IEEE TIP|必读1||		
|Domain Invariant and Class Discriminative Feature Learning for Visual Domain Adapation|IEEE TIP|必读1|需要|DICD|
|Discriminative and Geometry Aware Unsupervised Domain Adaptation||必读2||		
|Joint Geometrical and Statistical Alignment for Visual Domain Adaptation|CVPR 2018|必读1||JGSA|
|Visual Domain Adaptation with Manifold Embedded Distribution Alignment|ACMMM 2018|必读2||EDA|
|Adaptation Regularization: A General Framework for Transfer Learning|IEEE TKDE|必读1|		

*******************
## 4. 纯深度学习网络结构研究
> 基础中的基础！推荐先看Stanford CS231n课程，百度搜索即可，可以快速了解深度学习

| paper | 来源 | 备注 | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
|Deep Residual Learning for Image Recognition			|CVPR 2016 Best paper|必读1|需要|ResNet|
|Densely Connected Convolutional Networks			|CVPR 2017 Best paper|必读1|需要|DenseNet|
|Deep Networks with Stochastic Depth			|ECCV 2016 spotlight|必读2|||
|ImageNet Classiﬁcation with Deep Convolutional Neural Networks			|NIPS|必读1||AlexNet|
|Very Deep Convolutional Networks for Large-Scale Image Recognition			||必读1||VGG|
|Squeeze-and-Excitation Networks			|CVPR|必读1||SENet|
|Going deeper with convolutions			|NIPS|必读2||InceptionV1/GoogLeNet|
|Rethinking the Inception Architecture for Computer Vision			|CVPR|必读2||InceptionV2/V3|
|Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning			|CVPR|必读2||InceptionV4/Inception-ResNet|
|Aggregated Residual Transformations for Deep Neural Networks			|CVPR|必读1||ResNext|
|Batch Normalization Accelerating Deep Network Training by Reducing Internal Covariate Shift		|ICML|必读1 ||BN|

*******************
## 5. 生成对抗网络（GAN）
| paper | 来源 | 备注 | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
|Generative Adversarial Nets				|NIPS	|必读1	|需要	|GAN|
|Conditional Generative Adversarial Nets|					|必读1|	需要|	CGAN|
|Self-Attention Generative Adversarial Networks				|CVPR	|必读2		||
|Wasserstein GAN					||必读1		||WGAN|
|Least Squares Generative Adversarial Networks|				ICCV|	必读2||		LSGAN|

*******************
## 6. 深度迁移学习
> Deep Domain Adaptation，针对分类问题，**研究重点！！**

| paper | 来源 | 备注 | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
|How transferable are features in deep neural networks|				NIPS 2014|	必读1|||		
|Learning Transferable Features with Deep Adaptation Networks|				ICML 2015|	必读1|	需要|	DAN|
|Unsupervised Domain Adaptation by Backpropagation|				ICML 2015|	必读1|需要|	DANN/RevGrad|
|Unsupervised Domain Adaptation with Residual Transfer Networks|NIPS 2015|	必读2||		RTN|
|CyCADA: Cycle-Consistent Adversarial Domain Adaptation|				ICML	|必读2		||CyCADA|
|Maximum Classifier Discrepancy for Unsupervised Domain Adaptation|				CVPR 2018|	必读1|	需要|	MCD|
|Multi-Adversarial Domain Adaptation|				AAAI 2018|	必读2|	|	MADA|
|Sliced Wasserstein Discrepancy for Unsupervised Domain Adaptation|				CVPR 2019|	必读1|需要|	SWD|
|Moment Matching for Multi-Source Domain Adaptation|				ICML 2019|	必读2|||		
|Bridging Theory and Algorithm for Domain Adaptation|				ICML 2019|	必读1||		MDD|
|Conditional Adversarial Domain Adaptation|				NIPS 2019|	必读1||		CDAN|
|Contrastive Adaptation Network for Unsupervised Domain Adaptation|				CVPR 2018	|必读1||		CAN|

*******************
## 7. Partial Domain Adaptation （PDA）

| paper | 来源 | 备注 | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
|Partial Transfer Learning with Selective Adversarial Networks|				CVPR 2018|	必读1|	需要|	SAN|
|Importance Weighted Adversarial Nets for Partial Domain Adaptation|				CVPR 2018|	必读1|	需要|	IWAN|
|Partial Adversarial Domain Adaptation|				ECCV 2018|	必读1|需要|	PADA|
|Learning to Transfer Examples for Partial Domain Adaptation|CVPR 2019|	必读1|	需要|	ETN|
Universal Domain Adaptation|				CVPR 2019|	必读2|||

*******************
## 8.  迁移学习在Semantic Segmentation中的应用

| paper | 来源 | 备注 | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
|Fully Convolutional Networks for Semantic Segmentation|				CVPR 2015|	必读1|	|	FCN|
|Structured Knowledge Distillation for Semantic Segmentation|				ICCV 2019|	必读2|||		
|ACFNet：Attentional Class Feature Network for Semantic Segmentation|				ICCV 2019|	必读1|||		
|CyCADA - Cycle-Consistent Adversarial Domain Adaptation|				ICML|	必读1|	|	CyCADA|
|Fully Convolutional Adaptation Networks for Semantic Segmentation|				CVPR 2018|	必读1|||		
|Domain Adaptation for Structured Output via Discriminative Patch Representations|				ICCV 2019	|必读1|||		
|Taking A Closer Look at Domain Shift: Category-level Adversaries for Semantics Consistent Domain Adaptation|				CVPR 2019|	必读1|||
|ADVENT: Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation|				CVPR 2019|	必读1|		|ADVENT|

*******************
## 9. 迁移学习在Object Detection中的应用

| paper | 来源 | 备注 | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
|Domain Adaptive Faster R-CNN for Object Detection in the Wild|				CVPR 2018|	必读2|||
|Multi-adversarial Faster-RCNN for Unrestricted Object Detection|				ICCV 2019|	必读2|||

*******************
## 10. 迁移学习在Video Classification中的应用

| paper | 来源 | 备注 | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
|Temporal Attentive Alignment for Large-Scale Video Domain Adaptation|				ICCV 2019	|必读1|||

*******************
## 11. 迁移学习在Person Re-identification中的应用

| paper | 来源 | 备注 | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
|Image-Image Domain Adaptation with Preserved Self-Similarity and Domain-Dissimilarity for Person Re-identification|				CVPR 2018|	必读1|||
|Self-similarity Grouping: A Simple Unsupervised Cross Domain Adaptation Approach for Person Re-identification|				ICCV 2019|	必读1|||

*******************
## 12. 迁移学习理论文章

| paper | 来源 | 备注 | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
|Analysis of Representations for Domain Adaptation|				NIPS|	必读1|||		
|A Theory of Learning from Different Domains|				|	必读1|		||
|A Kernel Method for the Two Sample Problem|				NIPS|	必读1|		|MMD的全面分析|

*******************
## 其他应该看论文

| paper | 来源 | 备注 | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
|Distance Metric Learning for Large Margin Nearest Neighbor Classification|				NIPS|	必读1|	|	LMNN|
|FaceNet - A Unified Embedding for Face Recognition and Clustering|					必读1|	|	Triplet Loss|

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
