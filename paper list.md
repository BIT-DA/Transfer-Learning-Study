# 迁移学习paper list与学习上手指南
> 推荐大家经常使用知乎和google搜索你看不懂的问题~自己主动学习十分重要！<br>
> 必读1代表马上阅读<br>
> 必读2代表以后必须读<br>
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

## 3.特征学习方法及其扩展
> 主要是浅层优化算法

| paper | 来源 | 备注 | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
|Domain Adaptation via Transfer Component Analysis|		IEEE TNNLS|	必读2||TCA|
|Transfer Feature Learning with Joint Distribution Adaptation|CVPR|必读1|需复现看代码|JDA|
|Transfer Joint Matching for Unsupervised Domain Adaptation|CVPR|必读2||TJM|
|Unsupervised Domain Adaptation With Label and Structural Consistency|IEEE TIP|必读1||		
|Domain Invariant and Class Discriminative Feature Learning for Visual Domain Adapation|IEEE TIP|必读1|需复现看代码|DICD|
|Discriminative and Geometry Aware Unsupervised Domain Adaptation||必读2||		
|Joint Geometrical and Statistical Alignment for Visual Domain Adaptation|CVPR 2018|必读1||JGSA|
|Visual Domain Adaptation with Manifold Embedded Distribution Alignment|ACMMM 2018|必读2||EDA|
|Adaptation Regularization: A General Framework for Transfer Learning|IEEE TKDE|必读1|		

## 4. 纯深度学习网络结构研究
> 基础中的基础！推荐先看Stanford CS231n课程，百度搜索即可，可以快速了解深度学习

| paper | 来源 | 备注 | 代码复现 | 简称 |
| :----: | :----:  | :----: | :----:  | :----: |
|Deep Residual Learning for Image Recognition			|CVPR 2016 Best paper|必读1|需要|ResNet|
|Densely Connected Convolutional Networks			|CVPR 2017 Best paper|必读1|需要|DenseNet|
|Deep Networks with Stochastic Depth			|ECCV 2016 spotlight|必读2|
|ImageNet Classiﬁcation with Deep Convolutional Neural Networks			|NIPS|必读1|AlexNet|
|Very Deep Convolutional Networks for Large-Scale Image Recognition			||必读1|VGG|
|Squeeze-and-Excitation Networks			|CVPR|必读1|SENet|
|Going deeper with convolutions			|NIPS|必读2|InceptionV1/GoogLeNet|
|Rethinking the Inception Architecture for Computer Vision			|CVPR|必读2|InceptionV2/V3|
|Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning			|CVPR|必读2|InceptionV4/Inception-ResNet|
|Aggregated Residual Transformations for Deep Neural Networks			|CVPR|必读1|ResNext|
|Batch Normalization Accelerating Deep Network Training by Reducing Internal Covariate Shift		|ICML|必读1|BN|

