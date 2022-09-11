---
author:
  nick: 小蛋子
  link: 'https://github.com/xv44586'
title: optimizer of bert
date: 2020-08-01 20:48:48
categories: NLP
tags:  
  - Optimizer
  - BERT
cover: /2020/08/01/optimizer-in-bert/head.jpeg
---
<!-- toc -->

最近尝试实现了 bert ,在最后 pretraining 时发现 bert 中的优化方法比较有趣，所以记录一下自己的理解。

# 整体优化方案
bert中的优化方案可以总结为：线性分段学习率 + weight decay Adam

# Adam in bert
首先简单回忆一下 Adam Optimizer：
整体框架：
$$
g_{t}=\bigtriangledown f(w_{t})
$$
$$
m_{t}=\Phi (g_{1},g_{2},...,g_{t})
$$
$$
v_{t}=\Psi (g_{1},g_{2},...,g_{t})
$$
$$
\eta =\alpha \cdot m_{t}/\sqrt{V_{t}}
$$
$$
\omega_{t+1}=\omega_{t}-\eta_{t}
$$

其中一阶动量 m 与二阶动量 v 的计算方式：
$$
m_{t}=\beta_{1}m_{t-1} + (1-\beta_{1})\cdot g_{t}
$$
$$
v_{t}=\beta_{2}v_{t-1} + (1-\beta_{2})\cdot g_{t}^{2}
$$
参数一般取值：ß1=0.9，ß2=0.999
而也是这个原因，初期对一阶动量与二阶动量v的估算都偏小，会导致优化方向朝着 0 走，所以，一般会进行一个修正（bias correct），方式是：
$$
\hat{m_{t}}=m_{t}/1-{\beta_{1}}^{t}
$$
$$
\hat{v_{t}}=v_{t}/1-{\beta_{2}}^{t}
$$
而 bert 中实现的 Adam 却没有进行这个修正，至于原因，放在下面一起说。


## weight decay
在 bert 中对 Adam 进行了weight decay，具体代码上是这一段：
```python
# Just adding the square of the weights to the loss function is *not*
# the correct way of using L2 regularization/weight decay with Adam,
# since that will interact with the m and v parameters in strange ways.
#
# Instead we want ot decay the weights in a manner that doesn't interact
# with the m/v parameters. This is equivalent to adding the square
# of the weights to the loss with plain (non-momentum) SGD.
if self._do_use_weight_decay(param_name):
      update += self.weight_decay_rate * param
```
这里讲到<code>直接将权重的平方加入到loss 上进行L2 regularization 在 Adam 上是一种错误到方式</code>
### weight decay
Weight decay是在每次更新的梯度基础上减去一个梯度

$$\theta_{t+1}=(1-\lambda )\theta_{t} -\alpha \bigtriangledown f_{t}(\theta_{t})$$

### L2 regularization
L2 regularrization是在参数上加上L2惩罚

$$ f_{t}^{reg}(\theta)=f_{t}(\theta)+\frac{ {\lambda }'}{2}\left \\| \theta\right \\| \_{2}^{2}$$

可以看出，在标准SGD下，两者是等价的
但是，在Adam下，两者却不是。我们将Adam下的梯度更新完整公式写出来：

$$ \\theta_{t}\leftarrow \theta_{t-1} -\alpha \frac{\beta_{1}m_{t-1}+(1-\beta_{1})(\bigtriangledown f_{t}+\lambda \theta_{t-1})}{\sqrt{\hat{v_{t}}} + \varepsilon  }$$

而与参数有关的是右上角的部分：<code>$\frac{\lambda \theta_{t-1}}{\sqrt{v_{t}}}$</code> 而这一项表明，在梯度变化越大的方向上，v的值也越大，但对应的权重约束却越小，这显然是不合理的，此外，L2 与 weight decay 都是各个方向同性的，
所以针对这一问题，一种调整方式是将梯度更新与weight decay 解偶，
![](/2020/08/01/optimizer-in-bert/de.png)
具体参考[DECOUPLED WEIGHT DECAY REGULARIZATION](https://arxiv.org/pdf/1711.05101.pdf)
而 bert 中也是使用了这种weight decay 方式，来达到与L2正则等效

# Learning rate
### Learning rate decay
通常，为了让模型在后期避免震荡，更加稳定，都会随着训练的进行，将learning rate 进行调整，即越是后期learning rate 越小。
### warmup
而bert中的learning rate的调整是两段线性调整学习率：前 10% steps 将learning rate 从 0 增长到 init_learning_rate，然后，再一致递减 到0
而warmup为何有效？
1. 可以避免较早的对mini-batch过拟合，即较早的进入不好的局部最优而无法跳出；
2. 保持模型深层的稳定性
具体可以参考[warmup 为什么有效](https://www.zhihu.com/question/338066667/answer/771252708)

此外，由于warmup要求前期保持较小的更新，所以Adam中由于前期会导致更新变小而需要进行的bias correct也可以去掉了。这也就是最初留下到那个问题到答案

# 总结
bert在 pretraining 为了让模型收敛到一个较好的点，不但在优化器 Adam 上使用了与 L2 regularization等效的weight decay，为了避免模型前期过早拟合进入local minimal，使用了warmup 策略。
bert作者也建议在进行fine-tuning时，使用与bert源码中相同的优化器，我也做了一些实验，提升有大概不到0.5个点（没有细调），所以在下游任务上可以尝试使用。

# 关于头图
摄于圆明园荷花池