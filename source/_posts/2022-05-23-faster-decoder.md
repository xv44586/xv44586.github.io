---
author:
  nick: 小蛋子
  link: 'https://github.com/xv44586'
title: faster-decoder之 decoder解码加速
date: 2022-05-23 10:25:43
categories: NLP
tags:
 - faster decoder
 - T5
 - simbert
cover: /2022/05/23/faster-decoder/Psyduck.jpeg
---
<!-- toc -->

# 1 背景
Transformer 模型舍弃了 step by step 的时序循环，依靠 self-attention 的并行结构获得了远超同量级 LSTM 网络的训练速度。即使做auto-regresisve 任务时，通过attention-mask 机制依然可以像encoder 一样并行计算。然而在解码时，却任然需要step by step 的进行，即需要知道上一个time step 的结果后才能进行下一个time step 的解码。此外，通常我们的解码策略是在获得模型结果后在内存中计算的，需要不停的将结果从GPU load 进 CPU 然后计算，这就进一步的拖慢了解码速度。而通常我们在部署时，首选的tf-serving 需要将结果通过网络传输，这将进一步的拖慢解码速度。而针对解码慢的问题，主要的加速方案有：
1. 将解码策略放在GPU 上计算，这样将避免结果在GPU/CPU 之间转换与等待；
2. attention cache，根据attention 层的特点，对attention 中的 $K$ / $V$ 进行cache，避免对之前的time step 进行重复计算，将attention 层的计算由 $O(n^2)$  降低到 $O(n)$。
3. transformer 计算最耗时的是attention 层中的softmax，尝试使用一些线性函数进行近似替换

# 2 attention cache
三种方案中，GPU 上进行解码需要一些底层技术进行开发，暂时没能力，而替换softmax 方案则会或多或少的损失一些精度，本文都不做进一步讨论。本文聚焦在attention cache 方案上，加速的同时又“不会”损失精度。
## 2.1 原理
attention 的计算公式：

$$
A = Softmax(QK^{T})* V
$$

在解码时，我们是step by step 进行的，所以，我们将时刻 t 的attention 写出来：

$$
A = Softmax(Q_{t}K^{T})* V
$$

即：对于时刻t 来说，attention 只需要当前的 $Q_{t}$ 时刻信息，$K$ / $V$ 的所有时刻信息进行计算。而 $Q_{t}$ 的计算只需要 $Token_{t-t}$ 即可，如何加速计算的关键就剩下如何更加高效的计算 $K$ / $V$.

## encoder-decoder cross-attention
对于encoder-decoder cross-attention 来说，对应的 $K$ / $V$ 都来自encoder 的outputs，所以直接将其整个进行cache 即可，而无需每步都重新计算。

## self-attention
而当attention 是self-attention 时，对于时刻 $t$ 来说，此时的 $K$ / $V$ 为 $K$ / $V$ 的前 $t$ 时刻信息，即 $K_{\leq t}$ / $V_{\leq t}$ .此时的 attention 计算为：

$$
A = Softmax(Q_{t}K_{\leq t}^{T})* V_{\leq t}
$$

而 $K_{t}$/$V_{t}$ 的计算只与 $Token_t$ 有关，与其他时刻的 $Token$ 无关，且不论是时刻 $t$ 还是时刻 $t+1$,对应的 $K_{t-1}$ / $V_{t-1}$ 的计算结果都是一样的。因此，每个时刻都对 $K_{\leq t}$ / $V_{\leq t}$ 全部计算是低效且浪费的。

由于 $K_t$ / $V_t$ 有只需 $Token_t$ 计算且不同时刻结果“一致”的特点，我们将每个时刻的 $K_t$ / $V_t$ 进行cache，在进行attention 计算时使用cache 中的 $K_{\leq t}$ / $V_{\leq t}$
即可。
此外，由于使用了attention cache 后，每次解码输入只需要 $Token_t$ 而非 $Token_{\leq t}$ ，这样将其他层的计算量也会随之降低。
PS：由于decoder 中为了实现auto-regressive 而采用了下三角的attention mask，因此，不同时刻的attention mask 是不同的，这会导致不同时刻的 $K_t$ / $V_t$ 的结果略有不同（约e-10)，但是这并不影响最终端到端的结果。 

## 2.2 实现
attention 层在实现时，除了进行attention 计算的同时，还会包含attention mask 和 position bias 两种信息，其中，attention mask 来实现auto regressive，即当前位置的attention 只能包含当前位置及之前的信息；position bias 则包括各种position 信息的实现，所以在使用attention cache 后，还需要对这两种信息进行“纠正”。

### 1. attention 层修改

具体实现时，对于encoder-decoder cross-attention, 我们之间将encoder outputs 计算一次后进行cache，每次进行解码时作为inputs 送人decoder；

对于self-attention ，我们在得到 $Q_{t}$/$K_{t}$/$V_{t}$ 后，将 $K_t$/$V_t$ 与之前的 $K_{\leq t-1}$/ $V_{\leq t-1}$ cache 进行拼接，构造出完整的$K_{\leq t}$ / $V_{\leq t}$, 然后将$Q_t$ / $K_{\leq t}$ / $V_{\leq t}$ 进入self-attention 层进行计算。

### 2. attention mask 的“纠正”

由于attention mask 的作用是防止当前位置看到其后位置的信息，而在使用cache 后，当前位置即最后时刻的位置，所以此时的attention mask 已没有存在的必要，直接取消即可；PS: 由于这里直接取消了attention mask，而attention mask 的实现通常是通过加上一个 负无穷(-e12) 来实现的，所以加了cache 后的outputs 与没加之前会有一定的差异，大概在e-10 量级。

### 3. position bias 的“纠正”

由于position bias 通常是通过inputs 的长度进行计算的，而加了attention cache 后，每次的inputs 的长度变为1 了（当前时刻的$Token_t$），所以此时的position bias 恒等于长度为1 的序列。为了还原他原始的position bias，我们使用拼接了cache 后的$K_{\leq t}$ 进行计算完整序列的position bias， 然后取出当前query 在完整序列中位置对应的position bias 即可。

### 4. 解码实现
此外，在解码函数上，也需要进行相应的修改，以获得当前时刻的$K_t$/$V_t$ , 然后与之前时刻的所有 $K_{\leq t-1}$ / $V_{\leq t-1}$ cache 进行拼接，为下一个时刻计算做准备。

# 3 onnx
由于tensorflow 会对当前显卡的显存全部占用，所以一个显卡只能启动一个tensorflow 进程，这样就导致当一个模型的显存不需要占用所有显存即可解码时，使用tensorflow 会浪费一部分显存，这里我们将其转为onnx ，这样只需要占用模型需要的显存即可，避免显存浪费。即一个显卡可以起多个解码进程。


# 4 demo
在bert4keras 的基础上，对 T5/Roformer 进行了实现，具体代码参考：[faster-decoder](https://github.com/xv44586/faster-decoder)


# 关于头图
网上流传的某个可达鸭形象😄