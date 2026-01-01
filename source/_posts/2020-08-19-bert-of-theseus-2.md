---
author:
  nick: 小蛋子
  link: 'https://github.com/xv44586'
title: Knowledge Distillation (1) &#58; 模块替换之bert-of-theseus-下篇
date: 2020-08-19 22:12:28
categories: NLP
tags:
  - Distillation
  - BERT
  - speed-up
cover: /2020/08/09/bert-of-theseus/theseus.png
---
<!-- toc -->

上一篇[模块替换之bert-of-theseus-上篇](https://xv44586.github.io/2020/08/09/bert-of-theseus/)中介绍了bert-of-theseus论文的主要思路，并贴了两组实验的结论，这篇是对上篇的后续一些思考与实验。

# 复现时的问题
在复现时，遇到最大的问题就是结果不稳定。首先每次训练predecessor时，其最优结果就会有上下1个点左右的波动，而因为theseus 中引入了随机数
来概率替换对应block，所以结果上一言难尽，有时能比12层bert低0.6个点, 有时只能达到直接3层fine tuning 的效果，于是我做了些观察与思考。

## 思考1：为什么失效
在训练theseus model时，其中抽出的successor在每个epoch结束后在验证集上的结果有时会很高，基本到达只比三层fine-tuning低6个点，有时又很
低，基本不到0.1%, 第一种明显是successor在theseus中训练太多，以至于接近直接fine tuning，而另一种情况下可能是successor训练不充足，
也可能是替换次数太少导致没有被训练，而且大多数情况下successor的验证集上都是不到0.1%。
为了验证第二种情况下是否是未替换导致successor在做fine tuning，我将successor进行单独fine tuning后,将得到的classifier 拼回predecessor，
发现此时在验证集上d结果只下降了2个点，所以此时大概率是替换次数过少，基本没有训练到successor，所以导致结果不好，而这里开始我以为是我
实现问题，后来来来回回检查了一周，也没发现问题，于是我就想换一种更稳定的方式。

## 思考二 ：bert-of--theseus有效的本质是什么
熟悉bert的同学肯定对warm up不陌生，而warm up之所以有效，我认为比较重要的一点是如果在最初的steps中，模型提前拟合了样本，进入了一个局部最优区域，后期无论你怎么迭代他都跳不出来，而由已经<code>fine tuned predecessor</code>带着一起再进行训练，也和warm up有些相似，即用小的
步子带着你朝着更优的方向走几步，跳出来，让你有进入更好的局部最优点的可能，此外，概率替换的思路也与<code>Dropout</code>有几分相似，让successor
有一定的几率参与训练，从而让successor在缺少predecessor的情况下也有一定的robust。
[苏剑林的博客](https://spaces.ac.cn/archives/7575)里也提到了替换的数学形式：
$$
\\begin{equation}\\begin{aligned} 
&\\varepsilon^{(l)}\\sim U(\\{0, 1\\})\\\\ 
&x^{(l)} = x_p^{(l)} \\times \\varepsilon^{(l)} + x_s^{(l)} \\times \\left(1 - \\varepsilon^{(l)}\\right)\\\\ 
&x_p^{(l+1)} = F_p^{(l+1)}\\left(x^{(l)}\\right)\\\\ 
&x_s^{(l+1)} = F_s^{(l+1)}\\left(x^{(l)}\\right) 
\\end{aligned}\\end{equation}
$$
同时，他也提到$\epsilon$能否不取非0即1，那既然我们是想让successor在task方向上warm up一下，那直接相加，即此时 $\epsilon = k$, 
k是常数也是可以的。此时只要调节k 就能避免successor训练不充分或太充分的情况了，模型也就稳定了，可以满足我们的要求了。

## 实验1
实验代码其实比较容易修改，只需将BinaryRandomChoice 层替换为相加即可。具体代码在[classification_ifytek_bert_of_theseus](https://github.com/xv44586/toolkit4nlp/blob/master/examples/classification_ifytek_bert_of_theseus.py)
中可以看到。

```python
class ProportionalAdd(Layer):
    """将两层的结果乘比例后相加，output = (input_1 * proportion + input_2 * (1 - proportion)) / 2
    """

    def __init__(self, proportion=0.5, **kwargs):
        super(ProportionalAdd, self).__init__(**kwargs)
        self.supports_masking = True
        self.proportion = proportion

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            return mask[1]

    def call(self, inputs):
        source, target = inputs
        source = source * self.proportion
        target = target * (1 - self.proportion)
        output = (source + target)/2
        return K.in_train_phase(output, target)

    def compute_output_shape(self, input_shape):
        return input_shape[1]
```
文本分类：CLUE的iflytek数据集

|            | 直接微调        | BERT-of-Theseus      |
|------------|-----------------|----------------------|
| **层数**   | 完整12层 / 前6层 / 前3层                    | 6层 / 3层            |
| **效果**   | 60.11% / 58.99% / 57.96%                    | 59.7% / 59.5%        |

结果上看确实更稳定了，也更好一点点了，基本比predecessor低<code>0.5%~1%</code> .


## 思考三：直接在predecessor 上抽successor行不行？
既然我们说bert-of-theseus有效的原因是在task 的方向进行了warm up，那predecessor已经在task上fine tuned了，能不能<code>直接抽取某几
层作为successor来直接fine tuning?</code>此外，之前我们也说了，predecessor与successor的classifer差距很小，那我们能不能改变successor
的classifer的学习率，让他进一步学习，来弥补一部分前三层无法拟合的分布呢？

## 实验2
具体实验代码[two-stage-fine-tuning](https://github.com/xv44586/toolkit4nlp/blob/master/examples/two_stage_fine_tuning.py)
实验时尝试了<code>随机初始化classifier/predecessor classifier初始化classifier/ 放大classifier lr</code>组合策略，最后的结果就不贴了，基本都没有
超过3层bert fine tuning的效果。

# 总结
尝试分析了bert-of-theseus复现中的问题，并尝试了一些修复方案，同时，实验测试了theseus model的必要性，最后结论是binary random choice
策略不如 proportion add 策略稳定，同时，theseus是必须的。

# 关于头图
[论文原作者配图](https://github.com/JetRunner/BERT-of-Theseus)