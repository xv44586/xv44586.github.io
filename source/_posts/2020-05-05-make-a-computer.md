---
author:
  nick: 小蛋子
  link: 'https://github.com/xv44586'
title: 装机指北
date: 2020-05-05 09:20:56
categories: Life
tags: 
  - tensorflow-gpu
  - 装机
  - CUDA
cover: /2020/05/05/make-a-computer/boxes.jpeg
---
<!-- toc -->

  今年回来，第一件事就是要打造一个自己的实验环境，而这也是我第一次自己从硬件开始，所以前前后后差不多折腾了大半个月，总算是搞定了。为了纪念这次从0开始打造自己的深度学习实验环境，所以写了这篇指北。

# 硬件篇
  关于硬件，最主要的就是显卡、CPU、主板和散热了，接下来一个一个介绍经验。
  
  ## 显卡
  显卡的选择主要参考两个维度：1.显存大小；2.浮点计算能力。
  显卡主要分AMD与Nvidia系列。因为Nvidia有CUDA加持，加速计算，自然是Nvidia系列了，Nvidia显卡目前针对PC有三个系列：Quadro、GeForce和Tesla。其中Quadro系列是专业绘图，GeoForce是专业游戏显卡（可绘图可计算），而Tesla是专业计算卡。综合考虑价格与 state-of-the-art 的模型对硬件的最低要求，最终选择了2080Ti。关于GPU的选型，可以参考[深度学习GPU对比](https://zhuanlan.zhihu.com/p/61411536)
  确定了型号，就是出品厂家选择了，主要区别就是公版非公版。公版就是Nvidia自己设计自己出的，而非公就是第三方厂商出的，包括evga，技嘉，微星等等。有一篇2080ti进行深度学习时的显卡性能测评的文章，找不到地址了，结论是evga > Nvidia > others 。evga与Nvidia的主要区别有两个：1.价格上evga略高 10% 左右；2. Nvidia 采用双风扇风冷，evga 采用单风扇风冷-水冷混合。最终选择了公版。
  
  ## CPU
  CPU主要有两个系列：Intel 和 AMD 。也是因为主要用来做计算，所以肯定首选 AMD。目前ADM系列顶配是 3990X，但是价格大概在三万左右，太感人了。第二的是 3700X ，京东上一千三上下，那就是他了。
  
  ## 主板
  CPU确定了，主板型号基本就定了。 主要参考如下图：
  ![cpu-主板对应型号参考](/2020/05/05/make-a-computer/board.png)
  看了一些测评，最终选择了微星[B450](https://item.jd.com/8259910.html)
  
  ## 散热
  散热目前两个方式：风冷和水冷。两者的区别主要参考[风冷与水冷区别](https://www.zhihu.com/question/57695465/answer/440467918)
  水冷在散热上还是要强一些的（240以上），所以打算试水一款水冷。主要推荐两款：[乔思伯光影240](https://item.m.jd.com/product/100003859323.html?wxa_abtest=o&utm_user=plusmember&ad_od=share&utm_source=androidapp&utm_medium=appshare&utm_campaign=t_335139774&utm_term=Wxfriends&from=singlemessage&isappinstalled=0) [九州风神水元素240T](https://item.m.jd.com/product/6454809.html?wxa_abtest=o&utm_user=plusmember&ad_od=share&utm_source=androidapp&utm_medium=appshare&utm_campaign=t_335139774&utm_term=Wxfriends&from=singlemessage&isappinstalled=0)。最后选择了九州风神水元素，因为买那天乔思伯涨价了～

  ## 存储
  存储上打算采用 32G + 1T ssd。内存自然上[金士顿骇客神条](https://item.jd.com/8391349.html), ssd主要参考性价比，最终选择[三星1T SSD](https://item.jd.com/100002580230.html)

  ## 电源与机箱
  由于目前暂时只插一张显卡，所以电源在 600W 以上即可。选一个品牌比较好的，那就是[安钛克750](https://item.jd.com/6828141.html) 了
  至于机箱，主要参考能不能够合理安放显卡、主板以及后续可能的散热。对于我当前的配置，只要是中塔的基本都够插显卡。买个安静低调的，那就他了：[爱国者M2](https://item.jd.com/100004999668.html)
  最终的配置清单如下：
  ![list](/2020/05/05/make-a-computer/list.png)
# 安装
  安装前，强烈建议多看几期安装教程视频，我主要看的是 B站 的[跟装机猿搞装机](https://www.bilibili.com/video/BV1vx41187cm) 系列。最困难的可能是水冷散热的安装了。由于平台不同，安装方式不同，推荐看对应水冷厂商给的安装视频。我主要参考[水元素安装教程](https://www.bilibili.com/video/BV1EE411h7R1)
  安装过程大致总结一下：
  1. 拿出主板和CPU，按照说明书将CPU装到主板上。
  2. 将水冷拿出，拿出硅脂，涂抹在CPU上，将冷头上的塑料膜撕掉，安装好支撑后，将水冷头贴在CPU上，安装固定。
  3. 拆开机箱侧板，将主板装上，并将水冷的风扇固定在机箱上。注意：风扇上没有挡板的是出风口，一般原则是从机箱内往外吹，注意风向。
  4. 插入内存条，固定固态硬盘
  5. 将线按说明一个一个插上
  6. 寻找机箱上显卡位置，可能需要扣掉挡板铁片。将显卡插入主板，同时固定在机箱上
  7. 放入电源，插好对应线
  主要注意的主要是：1、水冷头上的膜一定要撕去，否则CPU上的热不能很好的传导出来。2、装之前大概看一下机箱各个位置，有些时候因为安装顺序会导致没地方下手，需要卸了重组。
  
# 软件安装

  ## 系统安装
  系统选Linux的稳定版，所以装了CentOS 7。
  1.首先下载centos的镜像文件，外网可能会慢，选择[阿里源](http://mirrors.aliyun.com/centos/7/isos/x86_64/)
  2.下载相应的刻录工具，插入u盘，将镜像刻录进u盘（刻录时会先格式化u盘，如果是windows系统，u盘默认是FAT32,这种格式下是没法放大于4G的单文件，所以需要进行格式转换：cmd下执行 convert e: /fs:ntfs)
  3.插入u盘，开机，引导开机进入u盘（大部分是自动进来），选择 test & install
  4.一般安装时，对应的命令行中的目录是错误的，执行会报错： 
  ```bash
    ERROR，could not insert 'floppy' 
```
  而无法进入安装界面
  此时需要查找u盘对应名字，修改命令：
  ```bash
  $ cd:/dev & ls
```
  找到一个 <code>s##数字 </code>的串，我的是sdb4 ,然后重启，进入后先按 e 进入编辑模式，修改命令行 
  ```bash
  linuxefi/images/pxeboot/vmlinuz inst.stage2=hd:LABEL=CentOS\x207\x20x86_64 xdriver=vesa nomodeset quiet
  linuxefi/images/pxeboot/vmlinuz inst.stage2=hd:/dev/sdb4 xdriver=vesa nomodeset quiet
  ```
  由于我的显卡是2080Ti,系统自带的nouveau驱动不匹配，需要禁用暂时，所以在后面加上 <code> nouveau.modeset=0 </code>
  最终修改后完整的命令行为：
  ```bash
  linuxefi/images/pxeboot/vmlinuz inst.stage2=hd:/dev/sdb4 xdriver=vesa nomodeset quiet nouveau.modeset=0
```
  此时保存后退出，选择test & install 即可进入安装界面
  进入后安装，安装时，注意软件选择中选择带网络的，剩下的就是按提示一步一步来即可。
  
  ## 安装Nvidia驱动
  1.检查显卡是否正常
  ```bash
  $ lspci | grep -i nvidia
  ```
  
  2.检查驱动版本
  添加EIRepo源
  ```bash
  $ rpm --import https://www.elrepo.org/RPM-GPG-KEY-elrepo.org
  $ rpm -Uvh http://www.elrepo.org/elrepo-release-7.6-5.el7.elrepo.noarch.rpm
  ``` 
  
  3.安装显卡驱动检查包
  ```bash
  $ yum install nvidia-detect
  ```
  检查驱动版本
  ```bash
  $ nvidia-detect -v
  ```
  此时会得到对应的版本信息，注意那个数字
  4.安装编译环境
```bash
  $ yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r) dkms
  $ yum -y update //注意这是升级系统
  $ yum -y install gcc kernel-devel kernel-headers dkms
```
  5.禁用vouveau
  ```bash
  $ vim /etc/modprobe.d/blacklist-nouveau.conf
      blacklist nouveau
      options nouveau modeset=0
```
  6.重新建立initramfs image文件
```bash
  $ mv /boot/initramfs-$(uname -r).img /boot/initramfs-$(uname -r).img.bak
  $ dracut /boot/initramfs-$(uname -r).img $(uname -r)
```
  7.reboot

  ## CUDA的安装与卸载
  
  ### 卸载cuda
  ```bash
  $ sudo yum remove "*cublas*" "cuda*"
```
  
参考：[removing-cuda](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#removing-cuda-tk-and-driver)

  ### 安装cuda
  ```bash
  wget path/to/your-version/install.rpm
  sudo rpm -i cuda-repo-rhel7-10-1-local-10.1.105-418.39-1.0-1.x86_64.rpm
  sudo yum clean all
  sudo yum install cuda
```


参考：[cuda-download](https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Linux&target_arch=x86_64&target_distro=CentOS&target_version=7&target_type=rpmlocal)
注意：一定不要通过浏览器去下载，因为会非常非常非常慢，但是wget大概几分钟就搞定了

  ## 安装cudnn
  1.下载相应的包 libcudnn*.rpm
  2. 安装包
  ```bash
  rpm -ivh libcudnn7-*.x86_64.rpm
  rpm -ivh libcudnn7-devel-*.x86_64.rpm
  rpm -ivh libcudnn7-doc-*.x86_64.rpm
```
  3.验证
  ```bash
  cd  $HOME/cudnn_samples_v7/mnistCUDNN
  $make clean && make
  $ ./mnistCUDNN
 ```
 if: Test passed! 则验证通过
参考：[cudnn-install](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-linux)

  ## Python环境
  python环境采用Anaconda + jupyter notebook
  ### conda配置
```bash
  conda create -n py3 python=3
  activate py3
  conda install ipykernel -n py3
  python -m ipykernel install --user --name py3 --display-name 'py3'
```
  删除环境
```bash jupyter kernelspec remove py3 ```
  pip 安装 与修改源
```bash 
  conda install pip -n py3
  vim ~/.pip/pip.conf

  [global]
  index-url=https://pypi.tuna.tsinghua.edu.cn/simple
```

  ### 安装jupyter
  ```bash pip install jupyter```
  jupyter 作为后台服务器
  1. 添加密码
```bash 
  ipython
  from jupyter.auth import passwd
  passwd()
```
  此时会让你输入两次密码，输入后得到一串hash码，保存下来， 回到bash, 添加配置文件
  ```bash
  $ jupyter notebook --generate-config
  $ vim ~/.jupyter/jupyter-notebook-config.py

  # edit

  c.NotebookApp.ip='*'                                  # * 代表所有iP都能访问 ，也可以指定ip
  c.NotebookApp.password = u'sha1:ce...'       # 刚才复制的那个密文
  c.NotebookApp.open_browser = False       # 禁止自动打开浏览器
  c.NotebookApp.port =8888                         #指定一个端口
     
  c.NotebookApp.notebook_dir = '/home/user/user1'  #指定工作空间
  c.PAMAuthenticator.encoding = 'utf8'         #指定utf-8编码，解决读取中文路径或者文件乱码问题
```

  后台运行
  ```bash
   $ nohup jupyter notebook --allow-root > jupyter.log 2>&1 &
```
  
  关闭后台运行
  ```bash
   # ps -axu | grep jupyter
  # kill -9 pid
  ```
  ## 安装tensorflow-gpu
  tensorflow-gpu对cuda有版本要求，所以在安装cuda前需要提前查看，确定自己版本。参考官网[install](https://www.tensorflow.org/install/source)
  ![tensorflow-gpu vs cuda](/2020/05/05/make-a-computer/gpu.png)
  还记得cuda安装时，给出的卸载方法吗？就是因为装tensorflow-gpu后发现gpu不带动的，后来才发现是cuda版本太高了～
```bash 
  $ pip install tensorflow-gpu==2.1
```
  验证
```bash
  $ ipython
    import tensorflow as tf
    tf.config.list_physical_devices('GPU')
```
  没有报错则正常

# 总结
  以上就是从零开始搭建搭建一套自己的深度学习实验平台的个人经验，整个过程从硬件到最终的软件适配都走了很多坑，希望能给你一些借鉴。

# 关于头图
  部分硬件盒子