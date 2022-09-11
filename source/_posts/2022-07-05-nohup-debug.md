---
author:
  nick: 小蛋子
  link: 'https://github.com/xv44586'
title: nohup踩坑记
date: 2022-07-05 10:07:22
categories: Programming
tags:
  - horovod
  - nohup
  - debug
cover: /2022/07/05/nohup-debug/dl.jpeg
---
<!-- toc -->

# 背景
最近在测试[horovod](https://xv44586.github.io/2022/05/25/horovod/)下的单机多卡/多机多卡的性能，结果遇到一个神奇的现象：训练总是会在一段时间后意外停止，停止后从系统到程序，没有任何异常，没有任何错误日志。而训练停止这个现象总是能复现，但是复现时又不完全一样，即同样的训练脚本，其每次训练意外停止时完成的更新次数还不一样。
为了解决这个神奇的bug，我们尝试从物理机、系统资源、驱动、软件环境、训练脚本、训练数据等多个环节进行检查，均未发现任何问题，最终无意间发现使用xshell （之前使用的WSL2)后训练停止的现象不在发生。最终发现竟然是因为对nohup 的坑，特此记录一下。

# nohup
```
nohup is a POSIX command which means "no hang up". Its purpose is to execute a command such that it ignores the HUP (hangup) signal and therefore does not stop when the user logs out.
```
上面这段是引自[wikipedia](https://en.wikipedia.org/wiki/Nohup#Overcoming_hanging)，即nohup 的作用是忽略HUP 信号，从而让用户log out 时，程序依然保持执行而不中断。

然而有些场景下，nohup 依然会失效，即使用了nohup 后程序依然可能被中断，如[hy-process-killed-with-nohup](https://unix.stackexchange.com/questions/420594/why-process-killed-with-nohup)提到的两个场景，此外，通常我们使用nohup 的场景时ssh 下，而ssh 为了避免丢失数据，会拒绝log out。 参考wiki中提到的：
```
Note that nohupping backgrounded jobs is typically used to avoid terminating them when logging off from a remote SSH session. A different issue that often arises in this situation is that ssh is refusing to log off ("hangs"), since it refuses to lose any data from/to the background job(s).[6][7] This problem can also be overcome by redirecting all three I/O streams:

$ nohup ./myprogram > foo.out 2> foo.err < /dev/null &
```

我的训练脚本使用时，对stdout/stderr 进行了重定向，未对stdin 进行重定向，即：
```bash
nohup ./myprogram > foo.out 2>&1 &
```
最终导致训练在进行一段时间后意外中断而没有任何异常信息。而这个像幽灵一样无声无息的“小问题”让我们花了接近三周时间debug，最终在猜测中定位到nohup 。

# 替代
由上面可以看出，nohup 并不是一个"好" 的后台执行解决方案，而对应的替代品，推荐使用[tmux](https://github.com/tmux/tmux/wiki)
```
tmux is a terminal multiplexer. It lets you switch easily between several programs in one terminal, detach them (they keep running in the background) and reattach them to a different terminal.
```
tmux稳定可靠，利用tmux，将session 与 terminal 进行detach，从而让程序在后台执行。

# 关于头图
深度debug