---
author:
  nick: 小蛋子
  link: 'https://github.com/xv44586'
title: 模型增强（2）&#58; 从label下手
date: 2020-09-13 00:09:24
categories:  NLP
tags:
  - Classification
cover: /2020/09/13/classification-label-augment/lake.jpeg

---
<!-- toc -->

上篇[Knowledge Distillation (2): distilling knowledge of bert](https://xv44586.github.io/2020/08/31/bert-01/)中我们提到，模型压缩时一个很重要的知识是soft labels，
并且提出了一种简单的方式：自蒸馏（self-distillation），而从label 的角度来看，可以看作是一种label augmentation，即构造了一个新的label，为模型新增了一个任务，通过新任务的学习，来提高模型对原来任务的性能。本文就label augmentation 继续脑洞。

# 构造新label
构造新label，其实本质上是构造一个与当前任务相关的新的任务，而对应的label则是通过当前样本通过某种方式获得，获得的label至少要比随机好，否则只会帮倒忙。

## 自监督
构造新label，我们可以借鉴自监督的方式，如Mask Language Model，AutoRegressive，而BERT中已使用来MLM，UniLM中也告诉我们增加Seq2Seq的AR 任务对NLU任务提高不显著，不过今年的论文[Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](http://arxiv.org/abs/2004.10964) 实验证明了进一步预训练是能进一步提升下游任务的性能。而当前任务是文本分类，MLM也许不是很合适，所以Seq2Seq 的方式可以尝试。
具体的，我们让模型学习目标类别的同时，希望模型能同时生成样本的描述字段（或者人为给定的某种相关性短语），即利用类别对应描述字段构造一个seq2seq任务。

## 相似性
对于同一个类别的样本，他们必然有某种相似性，至少比与其他类别的样本更相似。而何如构造样本呢？
一种简单的方式是对每个样本都从类当中抽取一个样本与他组成一对，然后让每个<code>i</code>样本与<code>i+1</code>样本相似。这种方式由于每次样本都是shuffle 的，只要让batch size 小于label number，一个batch 内同时出现多个同一类别的样本概率就会很小。
既然在构造seq2seq任务时，我们使用来label对应的描述，此时我们也可以继续尝试使用：每个样本构造一个新的样本，新样本由label对应描述与label id组成。

# 实验结果
两组实验结果如下：

| seq2seq | similarity |
|---------|------------|
| 59.91%  | 56.9%      |

可以看到，对于构造seq2seq 任务，其结果与直接fine-tuning 结果基本一致，这也符合预期。而构造相似性任务，其结果直接fine-tuning 结果相比反而更差了。原因可能是样本不均衡，所以同一batch 内有较高概率出现同一类别的样本，同时通过让样本与同一样本相似来间接相似，这种方式可能有些曲折了，不过最根本的原因应该还是batch 内同一类别样本的出现干扰了学习。
具体实验代码可以查阅[classification_ifytek_auxiliary_seq2seq_task](https://github.com/xv44586/toolkit4nlp/blob/master/examples/classification_ifytek_auxiliary_seq2seq_task.py) 和[classification_ifytek_with_similarity](https://github.com/xv44586/toolkit4nlp/blob/master/examples/classification_ifytek_with_similarity.py)
# 总结
本文只是由于之前实验想到的尝试对label 做增强来实现模型增强的尝试，最后两组实验都没取得什么好的结果。

# 关于头图
摄于翠湖湿地