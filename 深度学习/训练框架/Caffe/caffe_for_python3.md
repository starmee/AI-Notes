本文将如何在CentOS上编译安装Python3的Caffe。  
**注意: 不要用Anaconda中的python环境，否则你会哭的。也不要尝试用Python2了，scipy已经不支持了。** 
使用Anaconda中的python环境可能会导致protobuf编译失败或者编译链接版本冲突 。
建议：能用Ubuntu就别用苦逼的CentOS。  

1、安装 python3 的 boost  
```
sudo yum install boost-python36-devel.x86_64
```

2、安装C++ protobuf库（我用的是protobuf-3.0.0-alpah3，Caffe官网https://caffe.berkeleyvision.org/installation.html说要用protobuf 3.0 alpha ）  

这个要下载源码编译安装 。  
安装完之后记得设置环境变量。  

3、安装python3 的 protobuf库，要和C++的版本一样  
```
pip3 install protobuf==3.0.0a3
```  

4、编译Caffe，具体参考[Build Caffe from Source on CentOS7](https://www.jianshu.com/p/a34f1066e89f)   ，这里有配置好的Caffe源码 [点击下载](resource/caffe/caffe-master-configured.zip)，里面已经添加了深度可分离卷积层和ShuffleNet层。  
这两个层来自：  
https://github.com/yonghenglh6/DepthwiseConvolution  
https://github.com/farmingyard/ShuffleNet  

如果使用这里配置好的源码编译安装,解压进入代码目录，执行以下命令：  
```
make all -j8
make pycaffe
make distribute
```
然后将distribute目录copy到指定位置，设置好环境变量即可使用。环境变量示例:  
在 `.bashrc`中添加以下内容
```
export PATH=$PATH:$HOME/tools/distribute/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/tools/distribute/lib
export C_INCLUDE_PATH=$C_INCLUDE_PATH:$HOME/tools/distribute/include
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$HOME/tools/distribute/include
export PYTHONPATH=$PYTHONPATH:$HOME/tools/distribute/python
```

Caffe配置起来这么麻烦，可我还在用，这就是爱吧!  
Caffe这么优秀，用的人却越来越少了，可惜。。。 可我还在用，这就是不离不弃吧!  
o(╥﹏╥)o  

**其他**
1、根据依赖包版本的不同，运行caffe.bin时可能报类似下面的错误：
```
Warning! HDF5 library version mismatched error
The HDF5 header files used to compile this application do not match
the version used by the HDF5 library to which this application is linked.
Data corruption or segmentation faults may occur if the application continues.
This can happen when an application was compiled by one version of HDF5 but
linked with a different version of static or shared HDF5 library.
You should recompile the application or check your shared library related
settings such as 'LD_LIBRARY_PATH'.
You can, at your own risk, disable this warning by setting the environment
variable 'HDF5_DISABLE_VERSION_CHECK' to a value of '1'.
Setting it to 2 or higher will suppress the warning messages totally.
Headers are 1.10.1, library is 1.10.2
SUMMARY OF THE HDF5 CONFIGURATION
```

这是因为编译caffe时用的`libhdf5.so,libhdf5_hl.so`版本和python中安装的 `libhdf5-5773eb11.so.103.0.0m, libhdf5_hl-db841637.so.100.1.1`版本不一致。后面两个so文件的名字可能不是这样的，但也是这样的格式，这两个文件是在python中安装h5py时产生的。  
这时候需要将两个版本修改一致，一般情况下python包的版本会跟多，修改python包也不容易对系统产生其他影响，所以这里修改python中的h5py版本。报错信息中说的Headers的版本就是python中h5py包对应的版本，如果Header版本高了，就重新安装低版本的python包。