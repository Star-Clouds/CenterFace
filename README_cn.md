## CenterFace: Face as Point

### 介绍 

实用的边缘设备无锚人脸检测与对齐算法Centerface, 模型大小7.3M。
**CenterFace-small** 性能达到centerface的同时模型大小仅为2.3M。

 ![image](results/bl4.jpg)   


### 更新

- 添加ncnn和opencv的工程

### 准确性

- WIDER FACE val集结果:

Model Version|Easy Set|Medium Set|Hard Set
------|--------|----------|--------
FaceBoxes|0.840 |0.766 |0.395
FaceBoxes3.2×|0.798|0.802|0.715
RetinaFace-mnet|0.887|0.870|0.792
LFFD-v1|0.910|0.881|0.780
LFFD-v2|0.837|0.835|0.729
CenterFace|0.935|0.924|0.875
CenterFace-small|0.931|0.924|0.870

- WIDER FACE test集结果:

Model Version|Easy Set|Medium Set|Hard Set
------|--------|----------|--------
FaceBoxes|0.839 |0.763 |0.396
FaceBoxes3.2×|0.791|0.794|0.715
LFFD-v1|0.910|0.881|0.780
LFFD-v2|0.837|0.835|0.729
CenterFace|0.932|0.921|0.873

> - 模型的训练数据仅包含：WIDER FACE train set
> - **RetinaFace-mnet** (RetinaFace-MobileNet-0.25)，来自于非常好的工作[insightface](https://github.com/deepinsight/insightface)。
> - **LFFD-v1** 也是很好的工作[LFFD](https://github.com/YonghaoHe/A-Light-and-Fast-Face-Detector-for-Edge-Devices)。
> - CenterFace/CenterFace-small的测试方法是MULTI-SCALE，因为训练图像和测试图像尺度的不一致性，多尺度测试才能反应centerface的真实性能。
   不过，对于SIO(原图单次推理)，CenterFace在val集上也可以达到：92.2% (Easy), 91.1% (Medium) and 78.2%，
   而RetinaFace-mnet在val集上是：89.6% (Easy), 87.1% (Medium) and 68.1% 
   
> - 关于Evaluation的一些思考:[人脸检测小江湖](evaluation.md)。

- FDDB的结果:

Model Version|Disc ROC curves score
------|--------
RetinaFace-mnet|96.0@1000
LFFD-v1|97.3@1000
LFFD-v2|97.2@1000
CenterFace|98.0@1000
CenterFace-small|98.1@1000

### 推理速度

- NVIDIA RTX 2080TI推理耗时:

Resolution->|640×480|1280×720(704)|1920×1080(1056)
------------|-------|--------|---------
RetinaFace-mnet|5.40ms|6.31ms|10.26ms
LFFD-v1|7.24ms|14.58ms|28.36ms
CenterFace|5.5ms|6.4ms|8.7ms
CenterFace-small|4.4ms|5.7ms|7.3ms
 
#### Results
   
 ![image](results/box_lm.jpg)  
 
 ![image](results/bl3.jpg)    
 
 ![image](results/bl1.jpg)   


### Discussion

  欢迎加入 **QQ Group(912759877)** 交流讨论, 包括但不限：人脸检测、稠密对齐、活体、3D重建等。


### 贡献者：
 - [ywlife](https://github.com/ywlife)
 - [SyGoing](https://github.com/SyGoing)
 - [MirrorYuChen](https://github.com/MirrorYuChen)