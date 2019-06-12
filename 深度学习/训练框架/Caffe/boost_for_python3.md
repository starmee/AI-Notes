本文讲解编译python3的boost库，**也可以直接安装python3的boost包**。  
以下是在CentOS上面的编译过程。  
先下载源码，这里的版本是1.56.0，进入源码目录。  


1、修改 bootstrap.sh， 里面有 python2的代码，需要修改一下。
将  
```
PYTHON_ROOT=`$PYTHON -c "import sys; print sys.prefix"
```
修改为  
```
PYTHON_ROOT=`$PYTHON -c "import sys; print (sys.prefix)"
```

2、先确保已经装上了python3，然后执行命令  
```
./bootstrap.sh --with-python=python3.6 --prefix="your install path"
```

3、修改 project-config.jam , 添加 include和链接路径  
将  
```
# Python configuration
using python : 3.6 : /usr

path-constant ICU_PATH : /usr ;
```
修改为  
```
# Python configuration
using python : 3.6 : /usr/bin/python3 : /usr/include/python3.6m : /usr/lib;

#path-constant ICU_PATH : /usr ;

```



后面就可以进行编译了  

4、执行命令 `./b2` 编译  

5、执行命令 `./b2 install` 安装  


编译安装完成，记得设置环境变量。

**参考**
[1] https://gist.github.com/melvincabatuan/a5a4a10b15ef31a5a481