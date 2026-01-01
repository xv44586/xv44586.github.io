---
author:
  nick: 小蛋子
  link: 'https://github.com/xv44586'
title: 模型增强（1）&#58; 利用NLG 增强QA 任务性能
date: 2020-08-22 09:38:13
categories: NLP
tags:
  - QA
  - NLG
  - UniLM
cover: /2020/08/22/qa-augmentation/cat.JPG
---
<!-- toc -->

# 背景
上周打算把UniLM在[toolkit4nlp](https://github.com/xv44586/toolkit4nlp)的基础上实现一下，又刷了一遍[论文](https://arxiv.org/pdf/1905.03197.pdf),发现作者提到用UniLM做问题生成，来增强QA任务的性能，觉得很有意思，所以想尝试一下。

# UniLM
因为这篇 UniLM 是主角，所以简单介绍一下该模型。该模型是通过灵活使用 attention mask ，将 NLG 与 NLU 任务统一在来一起，所以叫 unified LM，
他的做法是将 left-to-right/right-to-left/masked lm/seq2seq lm/放在一个框架里训练，从而让模型兼具 NLU 与 NLG 的能力。
![](/2020/08/22/qa-augmentation/lm.png)
而为了达到这个训练，只需要在 bert 的基础上根据不同的 lm 调整 attention mask 即可。所以利用 bert 做 NLG 时，只需要调整 attention mask 
为 seq2seq lm 对应mask即可。

# 数据增强
通常增强都是同义词/近义词替换，subsequence的随机删除/重复/互换等，我之前在做百度比赛时尝试过随机删除和随机两个片段互换位置，提升不是
非常大而论文里大问题生成带来大提升还是相当大的：
![](/2020/08/22/qa-augmentation/table9.png)
仔细想一下，由于attention机制，互换只是改变了position embedding部分内容，而这部分的互换对模型的影响是很弱的；随机删除可能会破坏语义，
所以增加模型robust的同时可能会降低模型性能。而问题生成，则可以看作是同义词/近义词替换的句子级别替换，所以理论上能带来不错的提升。
从对抗的角度来看，生成的问题在语义上与原问题基本一致，这也正好符合<code>输入的微小改变</code>，从而让模型在这种带有微小扰动的前提下仍然能很好的预测。

# 实验
既然UniLM具有很强的NLG能力能力，那就有很多不同的玩法。首先，可以训练一个模型，来针对 context 和 answer 生成对应的问题，来对问题进行
<code>"换个问法"</code>，其次，既然可以对问题<cdoe>"换个问法"</code>,自然也可以<code>"换个问题"</code>,也就是根据 context 生成新的问题
和答案。另外，由于是扩增训练数据，所以有一个技巧是做生成是将 train data 与 dev data 互换，不过由于我用的是百度比赛数据，dev data 太少，
所以我是 train + dev。

## 问题生成
问题生成时，就是将 context 与 answer 拼接，然后生成对应的question。具体样本形如：<code> [CLS] answer + context [SEP] question [SEP]</code> .
模型直接用bert base权重按UniLM的seq2seq方式来构建，可以看到效果还是很不错的，比如：
<blockquote>
context：报雅思或者托付培训班,一般情况下要900元左右。 雅思和托福考试可以自学: 一、基础知识准备:单词、基本语法、长难句分析; 二、板块训练:听说读写,四个板块; 三、合理备考计划,可以参见中国上别人经验结合自己的自身条件; 四、效果强化跟踪,使用合理的备考软件或者是自测题目随时跟踪自己的学习状态
question：雅思班价格
answer: ['900元', '900元左右']
generate question: 雅思班报名多少钱
</blockquote>

<blockquote>context：USB电压有5伏 USB一般4根线, 中间两根是数据线, 左右各为 +- 线 只要不短路是不会烧主板该插口的 ,我想你应该这样做,手机的线的一端直接插入手提电脑,另一头剪掉头子,从线中分离出四根线, 用万用表测出(红色+和其它色如黑-)剩下两根用胶布包扎(不用)然后 在这两根线上(正电极中最好串一50到100欧电阻)后接入一支高亮度发光二极管就成功了.
question：usb线电压 
answer: ['5伏']
generate question: usb线电压 </blockquote>

解码时，有两种选择：随机抽样与 beam search 。随机抽样可以增加问题的多样性，并且可以生成多个问题；beam search近似最优，得到一个最优的
问题。由于我们是使用 train data 训练模型，在对 train data 生成新的问题时，beam search 将可能产生很多一摸一样的问题，这样将降低新增
数据的量；而随机抽样能产生很多新的问题，但可能新生成的问题与答案并不配套，还需要一些后处理之后才能真正拿来用。这里两种方式都拿来做实验，
并对生成的问题做一个简单的过滤：新生成的问题与原问题中有70%以上的字是重合的。
| base line | beam search | random sample |
|-----------|-------------|---------------|
| 80.39%    | 81.0%       | 79.8%         |

random sample的样本经过了很多次过滤之后才能基本达到baseline的效果，所以生成的问题如果"问非所答"，对最终的效果反而是不好的，这也符合预期。

## 问题答案对生成
问题答案对生成时，由于答案是在 context 内的，相对问题生成简单一些，所以我们先生成答案，再根据 context 和生成的 answer 来生成对应的
question。不过为了让问题答案对更丰富多样，解码答案时我们采用随机抽样，而生成问题时，为了让问题尽量准确，我们采用 beam search。
样本形如 <code>[CLS]context[SEP]answer[SEP][question][SEP]</code>，生成的效果如下：
<blockquote>
context：您好，孕妇整个孕期体重增加12.5公斤左右，在增加的这12.5公斤中，胎儿的体重占3公斤左右，胎盘和羊水约是2公斤左右。在孕早期（怀孕3个月以内）增加2公斤，中期（怀孕3－6个月）以及末期（怀孕7－9个月）各增加5公斤左右。所以怀孕6个月体重增加7公斤左右比较正常。希望我的回答对你有所帮助。
question：孕妇6个月体重增加多少 
answer: 7公斤左右
generate question: 孕妇6个月体重增加多少 
generate answer: 12.5公斤左右
</blockquote>
不过也由于train data 参与训练，所以很多生成的问题答案对与原始问题答案对一致，如果有更多的外部数据，可以利用外部数据来训练。
| base line | beam search | random sample | question answer generation |
|-----------|-------------|---------------|----------------------------|
| 80.39%    | 81.0%       | 79.8%         | 81.76%                     |

# 总结
通过生成新的问题与新的问题答案对能在一定程度上提高qa 任务的性能，在生成问题时，用beam search 得到的新问题虽然量少但由于更准确，所以
能带来一定的提升；用随机采样生成的问题会有部分与答案无关的或者语义有点不通顺的问题，所以可能反而会导致性能降低；问题答案对的生成时，
先生成相对简单的回答再生成对应问题，能对性能带来不错的提升，在做qa相关任务时，可以尝试使用一下。
实验代码：
[qa_baseline](https://github.com/xv44586/toolkit4nlp/blob/master/examples/qa_baseline.py)
[qa_question_generation_seq2seq](https://github.com/xv44586/toolkit4nlp/blob/master/examples/qa_question_generation_seq2seq.py)
[qa_question_answer_generation_seq2seq](https://github.com/xv44586/toolkit4nlp/blob/master/examples/qa_question_answer_generation_seq2seq.py)

# 关于头图
看瓜的怒气小猫