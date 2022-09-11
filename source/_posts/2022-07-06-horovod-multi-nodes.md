---
author:
  nick: 小蛋子
  link: 'https://github.com/xv44586'
title: 训练加速篇（3）horovod之多机多卡
date: 2022-07-06 11:20:31
categories: NLP
tags:
  - horovod
  - speed-up
cover: /2022/07/06/horovod-multi-nodes/cat.JPG
---
<!-- toc -->

# horovod 多机多卡
[上一篇](https://xv44586.github.io/2022/05/25/horovod/) 中介绍了如何在单机多卡下使用horovod 进行训练，本篇介绍如何在多机多卡下使用horovod 进行训练。
这篇中的测试GPU 为V100， 上篇A100 中遇到的环境问题在V100 中全都没有了，所以整个环境的搭建就异常简单了。
## 环境搭建
拉取最新的ngc 中的image，加载镜像并在container 中配置互相免密登陆，注意docker 启动时需要加入<code>privileged</code>权限，以便docker能够访问RDMA网口
```bash
docker run -itd --rm --gpus all --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 --net=host --privileged -v /data:/data --name horovod tensorflow:22.06-tf1-py3
```
容器内互相免密
```bash
# 允许root 使用ssh
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# 修改容器内 ssh 默认端口为2222，防止与 host 所使用的22端口冲突
sed -i 's/#Port 22/Port 2222/' /etc/ssh/sshd_config

# 重启ssh 服务
service ssh restart && netstat -tulpn

# 设置 root 密码
passwd root

# SSH Key
ssh-keygen

# 创建 ~/.ssh/config，并添加以下内容后，保存并退出，完成 host alias 配置。
# ！注意：
# 如果是CVM机型，则ip是两台机器`ifconfig eth0`显示的ip
# 如果是黑石RDMA机型，则ip是两台机器`ifconfig bond0`显示的ip
Host gpu1
 hostname 172.0.0.1
 port 2222
Host gpu2
 hostname 172.0.0.2
 port 2222
```

## RDMA
上面提到了RDMA，这里简单介绍一下。
```
在数据中心领域，远程直接内存访问（英语：remote direct memory access，RDMA）是一种绕过远程主机操作系统内核访问其内存中数据的技术，由于不经过操作系统，不仅节省了大量CPU资源，同样也提高了系统吞吐量、降低了系统的网络通信延迟，尤其适合在大规模并行计算机集群中有广泛应用。
```
这段话引自wiki，通过使用RDMA技术，可以进一步提高分布式系统的整体性能。而我们使用的NCCL 进行通信，NCCL 是支持RDMA的。此外，我们使用的ngc 中是包含了RDMA 驱动的，如果image 内未安装，参考[容器安装用户态 RDMA 驱动](https://cloud.tencent.com/document/product/1573/74101)

## 启动训练
启动训练时，需要根据节点信息和通信方案调整参数，如在支持RDMA 下的启动命令：
```bash
mpirun -np 16 -H gpu1:8,gpu2:8 --allow-run-as-root -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x NCCL_IB_DISABLE=0 -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_GID_INDEX=3 -x NCCL_NET_GDR_LEVEL=0 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl_tcp_if_include bond0 -mca btl ^openib python3 train.py
```
其中 ：
`-H` 后面指定节点及显卡数；
`-np` 需要根据 `-H` 调整为其总计worker 数量；
`btl_tcp_if_include` RDMA 下为bond0，普通网络则为eth0；
`NCCL_SOCKET_IFNAME` 为网络接口，RDMA 下为bond0，普通网络则需切换为 eth0
`NCCL_NET_GDR_LEVEL` 为GDR 相关，GDR的概念是运行GPU 与RDMA 直接通信，开启后进一步降低通信延迟。不过GDR需要配合RDMA 一起使用；

NCCL 相关变量含义可以参考[Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)


# 性能对比
在bert-base 规模的模型上进行测试，其结果如下：
|  GPU | batch size per GPU |   net  |  node           | speed
| ------------- |:-------------:|:-------------:|:-------------:|:-------------:|
| A100-40g    | 16 | vpc | single | 430 me/step |
| V100        | 8  | vpc | single | 485 ms/step |
| V100        | 8  | vpc | multi  | 617 ms/step |
| V100        | 8  | rdma | single | 485 ms/step |
| V100        | 8  | rdma | multi | 510 ms/step |

可以看到，通过RDMA 进一步降低网络延迟后，多机多卡的加速效果接近线性加速了。如果开启GDR 网络延迟能进一步降低，加速效果应该会更解决线性加速。

# ref
[RDMA技术详解（一）：RDMA概述](https://zhuanlan.zhihu.com/p/55142557)

[远程直接内存访问](https://zh.m.wikipedia.org/zh-hans/%E8%BF%9C%E7%A8%8B%E7%9B%B4%E6%8E%A5%E5%86%85%E5%AD%98%E8%AE%BF%E9%97%AE)

[配置容器 SSH 免密访问](https://cloud.tencent.com/document/product/1573/74100)

# 关于头图
可爱小猫