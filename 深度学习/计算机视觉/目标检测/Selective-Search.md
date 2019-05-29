<center id="Selective-Search"> <b>Selective Search for Object Recognition </b> </center>

选择性搜索算法是为了给物体识别生成候选区域。该论文发表于2012年。
[Selective Search for Object Recognition](resource/Selective-Search/Selective-Search-for-Object-Recognition.pdf)

之前的做法主要是基于穷举搜索:选择一个窗口大小扫描整张图像，改变窗口的大小，继续扫描整张图像。这种方法的计算量很大，非常耗时。
选择性搜索结合了穷举搜索和分割，旨在找到一些可能的目标位置集合。该算法采取组合策略保证搜索的多样性，其结果达到平均最好重合率为0.879，能够大幅度降低搜索空间，提高程序效率。


这篇博客翻译的不错 [论文笔记《Selective Search for object recognition》](https://blog.csdn.net/niaolianjiulin/article/details/52950797)














**参考**
[1] https://blog.csdn.net/niaolianjiulin/article/details/52950797