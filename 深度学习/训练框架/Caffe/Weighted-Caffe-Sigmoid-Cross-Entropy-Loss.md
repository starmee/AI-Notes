<center><b>自定义Caffe Python 加权 sigmoid 交叉熵损失函数</b></center>  


#### 一、导数推导  

下面参考上述博客推到加权交叉熵损失的导数  
将权重$w$加在类别1上面，类别0的权重为1，则损失函数为：  
$$L=wtln(P) + (1-t)ln(1-P)$$  
其中$t$表示target或label, P表示Sigmoid 概率，$P=\frac{1}{1+e^{-x}}$  
化简后  
$$L=(t-1)x + (-wt+t-1)ln(1+e^{-x}) $$            (1)式  
 求导,得  
$$\frac{\partial L}{\partial x} = wt - (wt-t+1)P$$  
可以看出，当权重为1时就是不加权的Loss。  
#### 二、实现Python SigmoidCrossEntropyWeightLossLayer  

```
import caffe
import numpy as np

class SigmoidCrossEntropyWeightLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check for all inputs
        params = eval(self.param_str)
        self.cls_weight = float(params["cls_weight"])
        if len(bottom) != 2:
            raise Exception("Need two inputs (scores and labels) to compute sigmoid crossentropy loss.")

    def reshape(self, bottom, top):
        # check input dimensions match between the scores and labels
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference would be the same shape as any input
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # layer output would be an averaged scalar loss
        top[0].reshape(1)

    def forward(self, bottom, top):
        score=bottom[0].data
        label=bottom[1].data

        first_term = -(label-1)*score
        second_term = -((1-self.cls_weight)*label - 1)*np.log(1+np.exp(-score))

        top[0].data[...] = np.sum(first_term + second_term)

        sig = 1.0/(1.0+np.exp(-score))
        self.diff = ((self.cls_weight-1)*label+1)*sig - self.cls_weight*label
        if np.isnan(top[0].data):
                exit()

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...]=self.diff
```
在 prototxt中使用:  
```
layer{
    name: "loss"
    type: "Python"
    bottom: "conv5"
    bottom: "label"
    top: "loss"
    python_param{
        module: "SigmoidCrossEntropyWeightLossLayer"
        layer: "SigmoidCrossEntropyWeightLossLayer"
        param_str: "{\"cls_weight\":100}"
    }
    include {
        phase: TRAIN
    }
    # set loss weight so Caffe knows this is a loss layer.
    # since PythonLayer inherits directly from Layer, this isn't automatically
    # known to Caffe
    loss_weight: 1
}
```  

**注意使用自定义的Python 损失层时一定要加上参数 loss_weight，否则Caffe不知道这层时Loss层。**


**参考**:
[1]  [Caffe Loss 层 - SigmoidCrossEntropyLoss 推导与Python实现](https://blog.csdn.net/zziahgf/article/details/79259010)  

**相关**：
 [Caffe custom sigmoid cross entropy loss layer](http://vast.uccs.edu/~adhamija/blog/Caffe%20Custom%20Layer.html "Permalink to Caffe custom sigmoid cross entropy loss layer")  


