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

使用反向传播训练所有网络权重是Fast RCNN的 重要能力。首先，我们阐述为什么SPPnet不能更新空间金字塔池化下的权重。

根本原因是当每个训练样例（比如 RoI）都来自不同的图像时，通过SPP层的反向传播非常低效，这正是RCNN和SPPnet的训练方式。效率低下的原因在于每个RoI可能有非常大的感受域，通常跨越整个输入图像。由于前向传播必须处理整个感受域，所以训练输入会很大（通常是整个图像）。

我们提出了一个更有效的训练方法，利用训练期间的特征共享。在Fast RCNN训练阶段，随机梯度下降（SGD）mini-batches 被分层抽样。首先抽样N个图片然后从每张图片中抽样 R/N 个RoIs。重要的是，同一张图片的RoIs在前向和反向传播中共享计算和存储。减小N就可以减小mini-batch的计算量。比如，当N=2,R=128,所提出的训练方案大约比从128个不同图片中各抽样一个RoI快大约64倍（这就是RCNN和SPPnet的策略）。

对这种策略的一个担忧是它可能导致训练收敛慢，因为来自同一张图片的RoIs是相关的。这个问题似乎不是一个实际问题，我们使N=2,R=128,通过更少的SGD迭代次数得到比RCNN更好的结果。

除了分层抽样，Fast RCNN使用简化的训练过程，一个调优阶段联合优化softmax分类器和bbox回归器，而不用分三个阶段训练[9,11]softmax分类器、SVM、和回归器。该步骤的组成部分（损失函数、mini-batch抽样策略、RoI池化层的反向传播、和SGD超参数）在下面描述。


<span id="multi-task-loss">
<b>Multi-task loss</b>
</span>

Fast RCNN 网络有两个并列输出层。一个输出 K+1 个类别的离散概率分布（每个RoI），$p=(p_0,...,p_k)$。通常，$p$由一个全连接层的 K+1 个输出计算得出。另一个输出bbox回归偏移，对 K 个物体类别的每一个类别k有 $t^{k}=\left(t_{x}^{k}, t_{y}^{k}, t_{\mathrm{w}}^{k}, t_{\mathrm{h}}^{k}\right)$。我们使用[9]中给出的$t^k$的参数化，其中$t^k$指定相对于候选区域的平移不变转换和对数空间高/宽平移。

每一个训练RoI都有一个类别$u$和一个bbox标签$v$。我们使用多任务损失函数$L$联合训练分类和bbox回归:
$L\left(p, u, t^{u}, v\right)=L_{\mathrm{cls}}(p, u)+\lambda[u \geq 1] L_{\mathrm{loc}}\left(t^{u}, v\right)$,(1)
其中$L_{\mathrm{cls}}(p, u)=-\log p_{u}$是真实类别$u$的对数损失。

第二项损失，$L_{\mathrm{loc}}$，定义在类别$u$的bbox回归目标$v=\left(v_{\mathrm{x}}, v_{\mathrm{y}}, v_{\mathrm{w}}, v_{\mathrm{h}}\right)$,和预测tuple $t^{u}=\left(t_{\mathrm{x}}^{u}, t_{\mathrm{y}}^{u}, t_{\mathrm{w}}^{u}, t_{\mathrm{h}}^{u}\right)$之间。当$u \geq 1$时，$[u \geq 1]$为1，否则为0.根据习惯背景类的标签为$u=0$。背景RoI没有bbox的概念，因此$L_{\mathrm{loc}}$被忽略。对bbox回归，损失函数为：
$$
L_{\mathrm{loc}}\left(t^{u}, v\right)=\sum_{i \in\{\mathrm{x}, \mathrm{y}, \mathrm{w}, \mathrm{h}\}} \operatorname{smooth}_{L_{1}}\left(t_{i}^{u}-v_{i}\right)
$$   ,（2）

其中，
$$
\operatorname{smooth}_{L_{1}}(x)=\left\{\begin{array}{ll}{0.5 x^{2}} & {\text { if }|x|<1} \\ {|x|-0.5} & {\text { otherwise }}\end{array}\right.
$$   , (3)

是一个很鲁邦的$L_1$损失，在RCNN和SPPnet中对异常值的敏感程度要比$L_2$损失低。

方程（1）中的超参数$\lambda$用来平衡两个loss。将bbox回归的目标（标签）$v_i$归一化为零均值和单位方差。所有的实验中$\lambda$都设置为1。

我们注意到[6]使用相关的损失函数训练一个类别无关(class-agnostic)的物体候选区域网络(We note that [6] uses a related loss to train a classagnostic object proposal network.)。和我们的方法不同，[6]提出双网络系统将定位和分类分开。OverFeat[19]，R-CNN[9]，和SPPnet[11]训练分类器和回归定位器，然而这些方法都是分阶段训练，不如Fast R-CNN好（第5.1节）。

<span id="mini-batch-sampling">
<b>Mini-batch sampling</b>
</span>

调优期间，每个SGD mini-batch由 N=2 个图像构成，随机选择（和通常的做法一样，我们实际上只是在数据集的排列上迭代）。每个 mini-batch R=128，从每张图片中抽样64个 RoI。和[9]一样，我们从候选区域中取 25% 的RoI，它们与ground-truth bbox的IoU至少为0.5。这些RoI为前景物体类别，即 u>=1。剩下的RoI从与ground-truth的IoU值为[0.1,0.5)的区间抽取，和[11]一样。这些是背景样例，标签u=0。阈值低于0.1的作为难例挖掘的启发式算法[8]。训练期间，图片以0.5的概率水平翻转。没有使用其它的数据增强手段。

<span id="back-propagation-roi">
<b>Back-propagation through RoI pooling layers.</b>
</span>

RoI 池化层的反向传播推导。声明，我们假定每个mini-batch只有一张图片(N=1)，将其推广为 N>1 是直接的，因为前向过程独立对待每张图片。
令$x_{i} \in \mathbb{R}$为RoI池化层的第 i 个激活输入，令$y_{r j}$为这一层的第 r 个 RoI 的第 j 个输出。

整体框架

训练过程

测试过程

数据处理



问题：
1、为什么在Fast RCNN中可以使用卷积+全连接层对候选区域分类?