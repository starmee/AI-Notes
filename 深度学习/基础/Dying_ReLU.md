<center><b>Dying ReLU</b></center>

The "Dying ReLU" refers to neuron which outputs 0 for your data in training set. This happens because sum of weight * inputs in a neuron (also called activation) becomes <= 0 for all input patterns. This causes ReLU to output 0. As derivative of ReLU is 0 in this case, no weight updates are made and neuron is stuck at outputting 0.
Things to note:
1.Dying ReLU doesn't mean that neuron's output will remain zero at the test time as well. Depending on distribution differences this may or may not be the case.
2.Dying ReLU is not permanent dead. If you add new training data or use pre-trained model for new training, these neurons might kick back!
3.Technically Dying ReLU doesn't have to output 0 for ALL training data. It may happen that it does output non-zero for some data but number of epochs are not enough to move weights significantly.


有这样一种情形：
假设有一个神经网络，它的输入X遵循某种分布。我们关注其中一个ReLU单元R。当参数固定时，X的分布指示着一个从输入到R的分布。为了更清晰，假设R的输入遵循一个低方差，中心在 +0.1 的高斯分布。
在这种情况下：
* R的大多数输入都是正的，因此
* 大多数输入会使 ReLU 门打开，因此
* 大多数输入能够使梯度在R处反向传播，因此
* R的输入通常能通过SGD反向传播更新。

现在假设在一次特殊的反向传播过程中有一个很大的梯度反向传播经过R。因为R是打开的，它会使这个大梯度反向传播到它的输入。这会导致计算R的输入的函数发生很大的变化。这意味着R的输入的分布改变了。我们假设R的输入现在的是一个低方差，中心在 -0.1 的高斯分布。

现在：
* R的大多数输入都是负值，因此
* 大多数输入会导致ReLU门关闭，因此
* 大多数输入会导致梯度不能反向传播经过R，因此
* R的输入通常不能再通过SGD反向传播更新。

发生了什么？R输入分布的一个相对小的变化（平均-0.2）导致了R的行为发生了质的变化。我们越过了0边界，R几乎永远关闭了。问题在于一个关闭的ReLU不能再更新它的输入参数，所以 “a dead (dead=always closed) ReLU stays dead”。

从数学上说，这是因为ReLU的数学公式导致的：
$r(x) = max(x,0)$
导数：
$\nabla_x r(x)=1 (x>0)$
所以当且仅当ReLU在前向传播时关闭了，那么它在反向传播时也会关闭。

关闭的ReLU是有机会被重新打开的。有许多上游会影响R输入分布的参数仍然在通过图中的其它路径更新。比如，R“活着的”的“兄妹”，可以更新R的上有参数，这样可能使R得输入分布回到正值区域。










**思考**
在 ReLU之前加上BatchNorm 应该会重新打开一些关闭了的ReLU。






**参考**
https://www.quora.com/What-is-the-dying-ReLU-problem-in-neural-networks
https://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks