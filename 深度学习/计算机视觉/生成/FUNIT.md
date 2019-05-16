<center><b>Few-Shot Unsupervised Image-to-Image Translation</b></center>

这是NVIDIA出的小样本无监督图像转换。论文[Few-Shot Unsupervised Image-to-Image Translation](resource/FUNIT/Few-Shot-Unsupervised-Image-to-Image-Translation.pdf)。

####摘要
无监督图像转换方法学习在不受限的图像数据集上绘画，将一张图像映射成相似的不同种类的图像。虽然有了显著的成果，但是现在的方法在训练时需要大量的源图像和目标图像。我们认为这严重影响了这些方法的应用。从人类可以从极少数样本中提取本质并泛化的能力得到启发，我们寻求一种小样本，无监督图像到图像转换算法。**这种算法适用于测试时仅由少量样本图像指定的以前没见过的目标类。我们的模型通过将对抗训练与新颖的网络设计相结合，实现了小样本泛化能力。** 通过广泛的实验验证和与基准数据集的几种baseline方法的比较，我们验证了所提出框架的有效性。
代码：https://github.com/NVlabs/FUNIT


>下文只是随读随记，全部读完之后会做一个总结整理。
这个框架或许可以用来做换脸。

![图1](resource/FUNIT/figure1.png)

#####3. Few-shot Unsupervised Image Translation
训练数据：
source classes: a set of object classes (e.g. images of various animal
species)
不假定任何两个类别之间存在成对的图像（比如，没有两个不同类别的动物有完全相同的姿势）
个人理解：作者只是没有做这个假定，不能确定是否有，或者说，即使有，也不去使用这个特性。
用源数据训练一个多类无监督图像转换模型。测试时，从一个新颖的物体类给模型提供少量的图片，这叫做target class。

模型不得不利用少量的目标图片将任意的源类别图像转换为目标类别的相似图像。
当我们更换不同的新物体类别给这个模型的时候，它也需要将任意的源类别图像转换成新的目标类别的相似图像。

这个框架包括一个**条件图像生成器G**和一个**多任务对抗判别器D**。和已有的无监督图像转换框架[54,29]中的传统图像生成器不同,我们的生成器G同时以内容图像(content image) x 和 一个K类图像集合${\{y_1,...,y_k\}}$作为输入，产生输出图像$\bar{x}$:
![output](resource/FUNIT/x.png)

我们假定 content image属于类别 $c_x$, K类图像中的每一个图像都属于类别$c_y$。
通常K很小并且$c_x$和$c_y$不同。我们会让G作为小样本图像转换器。
如图1所示，G将一个输入content image $x$映射为一个输出图片$\bar{x}$。$\bar{x}$外观上属于类别$c_y$，结构上(姿态上)却和$x$相近。
用$\mathbb{S}$和$\mathbb{T}$表示源类集合和目标类集合。在训练时，G学习转换从源类中随机抽样出来的两个类别$c_x,c_y\in \mathbb{S}$并且$ c_x\neq c_y$。在测试时，G从从未见过的目标类别$c\in \mathbb{T}$中取少量图像，将任意一个源类别图像映射为目标类别的相似图像。
个人理解：
训练时先从source class中随机抽出两个类别$c_x,c_y$,然后训练G时，$c_y$又作为K个不同类别去训练。这样K类的每个类别图像就相对$x$少了很多，符合测试时目标图像少的情况。

####网络的设计和训练
#####3.1. Few-shot Image Translator

小样本图像转换器G包含了一个内容编码器$E_x$，一个类别编码器$E_y$和一个解码器$F_x$。

内容编码器由几个残差块和2D卷积层组成。它将输入content image x 映射为 content 潜在编码 $z_x$(空间特征图)。

类别编码器由几个2D卷积层和一个样本轴上的均值操作组成。它首先将K个不同类别的图像${y_1,...,y_k}$映射为一个中间潜在向量，然后计算潜在向量的均值得到最后的类别潜在编码$z_y$。

解码器包含几个自适应实例标准化(instance normalization)（AdaIN[18]）残差块[19]和一对上采样(upscale)卷积层。AdaIN残差块就是使用AdaIN作为normalization layer的残差块。对每一个样例，AdaIN首先将样例在每个通道的激活值标准化为零均值和单位方差，然后使用学习到的仿射变换缩放激活值。这个仿射变换由scalars和biases集合组成。注意这个仿射变换具有空间不变性因此只能用于获取全局表观信息(global appearance information)。仿射变换参数使用$z_y$通过一个两层全连接网络自适应计算得到。代入$E_x,E_y,F_x$,(1)式可以分解为:
![output](resource/FUNIT/xbar2.png)
