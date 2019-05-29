<center><b>Fast R-CNN</b></center>

论文:[Fast R-CNN](resource/FastRCNN/FastRCNN.pdf)
Caffe代码:https://github.com/rbgirshick/fast-rcnn

Fast RCNN 属于目标检测领域，是对RCNN和SPPNet的改进。
[RCNN](RCNN.md)主要有如下问题:
（1）训练分多阶段。预训练->调优训练->SVM分类->bbox回归
（2）训练需要大量的时间和空间。从候选区域提取的特征需要使用很长时间，并且特征需要保存到磁盘上，需要占据大量磁盘空间。
（3）检测速度慢。在测试阶段每个候选区域都要被检测一遍。


RCNN之所以慢主要是因为它对每一个候选区域做卷积运算，没有共享计算。空间金字塔池化网络（[SPPnets](SPPNet.md)）[11]通过共享计算来加速RCNN。SPPnet计算整个输入图像的卷积特征图，然后使用从共享特征图中提取的特征向量对每个物体进行分类。SPPnet通过将输入图像池化为三个固定尺寸的特征图来处理不同大小的图像输入。如下图所示：
![SPP](resource/FastRCNN/spp.dib)
SPPnet的测试时间比RCNN提升了10到100倍，训练时间比RCNN提升了3倍。但SPPnet和RCNN一样是多阶段的，需要预训练、调优训练、训练SVM分类器、bbox回归，特征也要保存到磁盘。SPPnet有一个独有的缺点就是模型调优时不能更新金字塔池化层前面的卷积层，这限制了比较深的网络的准确率。
Fast RCNN改进了RCNN和SPPnet的以上缺点，同时提升了速度和准确率。主要优点有：
（1）检测质量（mAP）比RCNN、SPPnet更高
（2）训练是单阶段的，使用多任务loss
（3）训练时可以更新所有的网络层
（4）特征不需要存储到磁盘

<span id="architecture">
<b>2. Fast R-CNN architecture and training</b>
</span>

![figure1](resource/FastRCNN/figure1.png)

Figure 1展示了Fast RCNN的结构。Fast RCNN以整张图像和一组候选区域作为输入。网络首先使用几个卷积层和最大池化层对输入图像提取特征图。然后对每个候选区域，由ROI(region of interest) 池化层从特征图中提取出固定大小的特征向量。每个特征向量都被送入一系列全卷积层，最后分成两个输出：一个是K类物体加上一个背景的softmax分类概率，另一个是每个物体的bbox位置（四个值）。

<span id="roi-pooling">
<b>2.1. The RoI pooling layer</b>
</span>

RoI池化层使用最大池化将感兴趣的有效区域内的特征转化为固定大小HxW（比如，7x7）的小特征图,其中 H 和 W 是层超参数，和 RoI无关。在本篇论文中，RoI是一个卷积特征图上的矩形窗口。每个RoI通过一个四元tuple (r,c,h,w) 定义，指定左上角(r,c)和 宽高(h,w)。

RoI最大池化将 h x w 的 RoI 窗口分成 H x W 的网格，每个网络尺寸大约为 h/H x w/W , 然后取每个网格的最大值。每个特征通道的池化就像最大池化那样是独立进行的。RoI池化是SPPnet中空间金字塔池化的特例，这里只有一层金字塔。


<span id="pre-train">
<b>2.2. Initializing from pre-trained networks</b>
</span>

我们实验了三个ImageNet[4]预训练模型，每个都有 5 个最大池化层和 5 到 13 个卷积层（详见4.1章）。使用预训练网络初始化Fast RCNN，有三个修改。
第一、最后一个最大池化层被一个RoI池化层代替，通过设置H和W以和后面的第一个全连接层兼容（比如，对于VGG16，H=W=7）。
第二、网络的最后一个全连接层和softmax层（ImageNet1000分类）被替换为两个并列层（一个全连接层用于K+1分类，另一个做指定类别的bbox回归）。
第三、网络修改为两个输入：一组图像和一组图像对应的候选区域。

<span id="fine-tuning">
<b>2.3. Fine-tuning for detection</b>
</span>

使用反向传播训练所有网络权重是Fast RCNN的 重要能力。受限，我们阐述为什么SPPnet不能更新空间金字塔池化下的权重。

根本原因是当每个训练样例（比如 RoI）都来自不同的图像时，通过SPP层的反向传播非常低效，这正是RCNN和SPPnet的训练方式。效率低下的原因在于每个RoI可能有非常大的感受域，通常跨越整个输入图像。由于前向传播必须处理整个感受域，所以训练输入会很大（通常是整个图像）。

我们提出了一个更有效的训练方法，利用训练期间的特征共享。在Fast RCNN训练阶段，随机梯度下降（SGD）mini-batches 被分层抽样。首先抽样N个图片然后从每张图片中抽样 R/N 个RoIs。重要的是，同一张图片的RoIs在前向和反向传播中共享计算和存储。减小N就可以减小mini-batch的计算量。比如，当N=2,R=128,所提出的训练方案大约比从128个不同图片中各抽样一个RoI快大约64倍（这就是RCNN和SPPnet的策略）。

对这种策略的一个担忧是它可能导致训练收敛慢，因为来自同一张图片的RoIs是相关的。这个问题似乎不是一个实际问题，我们使N=2,R=128,通过更少的SGD迭代次数得到比RCNN更好的结果。

除了分层抽样，Fast RCNN使用简化的训练过程，一个调优阶段联合优化softmax分类器和bbox回归器，而不用分三个阶段训练[9,11]softmax分类器、SVM、和回归器。该步骤的组成部分（损失函数、mini-batch抽样策略、RoI池化层的反向传播、和SGD超参数）在下面描述。



整体框架

训练过程

测试过程

数据处理



问题：
1、为什么在Fast RCNN中可以使用卷积+全连接层对候选区域分类?