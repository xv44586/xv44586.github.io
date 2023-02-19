---
author:
  nick: 小蛋子
  link: 'https://github.com/xv44586'
title: few-shot视角下的fine-tuning
date: 2023-02-01 12:46:43
categories: NLP
tags:
  - LLM
  - In-context learning
  - fine-tuning
cover: /2023/02/01/fine-tuning-at-few-shot/himalayas.JPG
---
<!-- toc -->
LLM 目前的使用方式主要是zero-shot/few-shot,其是从context中的examle 数量来区分的，如果按这个角度进一步概括目前的fine-tuning 方案，感觉是个有趣的视角。
![](/2023/02/01/fine-tuning-at-few-shot/few-shot.PNG)
# few-shot
目前LLM 的使用方式主要是zero-shot/few-shot，而通常few-shot 的性能也比zero-shot 要好，而且随着example 的数量的增加，few-shot 的性能也可能进一步提升；zero-shot 时只给出task input 效果可能不佳，通常需要给出对应的task description，而更”精准“ 的task description 通常也能得到更好的zero-shot 效果。
所以，提升LLM zero-shot /few-shot 的性能，主要的方式有两个：
1. 给出更”好“的task description
2. 给出更多的example

# fine-tuning
## supervised fine-tuning
对于小模型来说，其in-context learning 能力较弱，性能不理想。为了提升模型的性能，一个可以尝试的方向是增加context 中的example 数量，但是由于模型的context 窗口有限，不支持我们无限制的将更多的example 塞进context 中，所以我们变通一下，采用通过更新参数的方式将example 塞进“context”。此时可以看作是无限example（inf-shot)；
## instruction-tuning
通常在supervised fine-tuning 时，我们是将一个任务的example “塞”进context，而instruction-tuning 可以看作是同时将多个任务的example “塞”进context，为了在使用时区分应该当前context 里应该使用哪些example，我们在不同的task 前面增加对应的description，作为判断依据。即：同时将多个任务的多个example 塞进模型的context 中，使用时通过不同的task description 来区分当前context 内的example 应该是哪些。
## prompt tuning
由于pretrain model 的任务是预测下一个token，而非处理用户的指令（instruction），为了提升模型zero-shot 的性能，一个可以尝试的方法就是找到模型视角下更好的task description（pattern），prompt-tuning 的思路即通过大量的监督样本，尝试寻找到更适应模型的task description，然后期望这个task description 能提高模型的zero-shot 的性能。
## Reinforcement learning with human feedback
prompt-tuning 时，我们尝试找到模型视角下更好的task description，而这个方法显然是不利于交互的，更好的方式是让模型理解人类视角下的task description。rlhf 就是按照这个思路，让模型反过来更好的理解人类视角下的task description，使得交互更方便。

# 启发
从这个视角看，我们发现即使是小模型，也是有一定的in-context learning 的能力的，只是不够强，所以我们需要更多的example 他才能发挥出更好的效果；
pretrain+fine-tuning 的模式之所以能work，是因为pretrain 后的model 有in-context learning 的能力，in-context learning 并不要求task 的形式与pretrain 的一致，所以我们才能在pretrain 的基础上根据下游任务的不同，来构造不同的fine-tuning 过程，在few-shot 视角下，其对应的是在context 中”塞“进更多的example；
in-context learning 的增强，对应的是所需的样本逐渐减少，从supervised fine-tuning 的大量样本到few-shot 的少量样本最终到zero-shot 的不需要提供样本，只需提供任务描述。

# 总结
本文是笔者最近思考pretrain + fine-tuning 模式为什么能work 时，通过在few-shot 视角下的一个解释。通过该思路，笔者尝试将目前的fine-tuning 主流思路统一起来。

# 关于头图
太空视角下的喜马拉雅山脉
