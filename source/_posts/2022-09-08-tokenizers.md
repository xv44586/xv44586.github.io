---
author:
  nick: 小蛋子
  link: 'https://github.com/xv44586'
title: tokenizers 总结
date: 2022-09-08 15:23:52
categories: NLP
tags: 
  - BPE
  - WordPiece
  - Unigram
cover: /2022/09/08/tokenizers/cannot_code.PNG
---
<!-- toc -->
# 前言
tokenizer 目前主流的方式是subword level，至于char level /word level 都由于粒度问题已被主流抛弃。目前subword level 的tokenizer BPE, Bytes BPE, WordPiece, Unigram, SentencePiece,下面简单总结一下各个方法。 

## BPE
bpe 的方案是通过统计词频来确定两个相邻的pair subwords 要不要合并，具体做法：
1.统计pre-tokenize 的word 的词频；
```bash
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```
2.使用词典对word 进行切分：
```bash
# base vocabulary: ["b", "g", "h", "n", "p", "s", "u"]
("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)
```
3.统计相邻两个subword pair 词频，将top-k 高的pair 合并生成新的subword，添加进vocabulary，同时，如果当前的subword  只会同pair 一起出现，则同时将vocabulary 中对应subword 删除。
```bash
# count pair
h + u = 10 + 5 = 15
u + g = 10 + 5 +  = 20
...
# merge top k
set k = 1
ug -> vocabulary
base vocabulary: ["b", "g", "h", "n", "p", "s", "u", "ug"]
​
loop until vocabulary match vocab_size
​
```

## Bytes BPE
Bytes BPE 与BPE基本相同，唯一不同的是：BPE 中会存在UNK 的情况，为了解决unk 的问题，一个非常天才的想法是将所有text 先转为<code>bytes</code> ，这样就不会存在unk 的问题，尤其是在多语言中，这种方式可以大大缩减词表大小;此外即使不是目标语言训练的模型也可以拿来使用。通常词表大小包括256 个基本bytes + <end|of|text> + vocab-size,如gpt2 的词表为50257: 256 base bytes tokens + <end|of|text> + 50,000 merges.
此外，训练bytes bpe 时，通常我们还会选择先将文本进行normalize，这部分后面会进一步说明。

## WordPiece
WordPiece 与BPE 也非常相似，区别主要在于merge 的策略：BPE 中选择频率最高的pair 进行合并，WordPiece 则选择使用语言模型来进行选择：
$$
L = logP(S) = \\sum^Nlog(P_i)
$$
对于两个subword： $t_x$, $t_y$ ，合并后为 $t_z$ ，则合并前后的增益：
$$
Loss = logP(t_z) - (logP(t_x) + logP(t_y))
$$
通过计算合并增益是否增大来决定是否合并subword pair.

## Unigram
Unigram 与 上述的方法都略有不同：Unigram 不再是通过合并base vocabulary 中的subword 来新增，他选择在初始化时初始化一个非常大的subword set，通过计算是否需要将一个subword 切分为多个base subword （remove 这个subword）来减小vocabulary size 直到达到vocab size。
这里有一个假设：句子之间是独立的，subword 与 subword 之间是独立的。对应的句子的语言模型似然值就是其subword 的概率的乘积。目标是保存vocab size 的同时语言模型似然值最大。
$$
x^* = argmax_{x \\in U}P(\\overrightarrow{x})
$$
整个求解过程是一个简单的EM 或者说一个迭代过程：
0.建立一个足够大的种子subword vocabulary，可以用字典树构建可以是所有字符的组合，也可以用bpe 构建；
1.（期望E）统计vocabulary 中每个subword 的频率，计算其对应概率值；
2.（最大化M）根据其概率，使用维特比算法返回其语言模型似然值最大化下的最佳分割方案；
3.计算最佳分割方案下每个新子词的loss，这里的loss 是指将当前subword 从vocabulary 中移除时，对应的语言模型似然值，即
$$
L = − \\sum^Nlog (\\sum_{x∈S(x_i)}p(x)) 
$$
4.丢弃掉loss 前x% 对应的subword；
5.重复2-4阶段，直到vocabulary 达到指定的vocab size。

## SentencePiece
SentencePiece 其实并不是一个新的tokenizer 方法，他其实是一个实现了BPE/Unigram tokenizer 的一个集合，不过他有一些创新的地方。
上述方法中有一些问题：
1.都有字，子词或词的概念，然而在很多语言中并没有这样的概念；
2.都默认需要自己进行pre-tokenize，如英语则利用“空格”作为词的分割符，中文则一般选择jieba 进行pre-tokenize，这个过程不同的语言有自己的一套做法，不统一；
3.token 格式不统一。以英文为例，表示token 时会有 ##xx, xx/s 这种，表示subword 是否出现在词的首尾，然而中文中是没有这种概念的；
4.解码困难，如BPE解码时需要进行一些标准化，最常见的是去除标点符号。而我们解码后是 [new] [york]两个token，我们并不知道原来的词是 newyork/new york/new-york 中的哪一个.
SentencePiece 的做法：
[
SentencePiece treats the input text just as a sequence of Unicode characters. Whitespace is also handled as a normal symbol. To handle the whitespace as a basic token explicitly, SentencePiece first escapes the whitespace with a meta symbol "▁" (U+2581) as follows.](https://github.com/google/sentencepiece)

即首先将空格转换为一个标准的字符"▁",然后将text 转换为unicode，其实这里的unicode 是NFKC-based normalization后的unicode，至于unicode 标准化，可以参考[unicode文本标准化](https://blog.csdn.net/weixin_43866211/article/details/98384017) ，虽然通常我们使用NFKC 标准化，但sentencepiece 内部四种方法都实现了。
通过上述的空格转换加normalize，所有的语言经过转换后就有统一的格式了，这样多语言的问题就彻底的与token 切分解偶了，tokenizer 就有了一个完全端到端的解决方案。

## train from scratch
训练一个tokenizer model 主要有两个仓库可以参考： [huggingface/tokenizers](https://github.com/huggingface/tokenizers)   和  [google/sentencepiece](https://github.com/google/sentencepiece).
其中tokenizers 支持bpe/bytes bpe/unigram/wordpiece, sentencepiece 支持bpe/unigram.两者都支持四种标注化方法。
此外，tokenizers 不支持自定义的pre_tokenizer的保存，如中文时我们常用的jieba.lcut ；bytes bpe不支持big dataset 的训练，1T  内存训练100G 文本也会因内存不足被killed，（一个办法是缩小语料训练，因为没有oov 的问题，基本上小语料下训练也能用）。
在实际使用时，通常会结合huggingface/transformers 一起使用，这里也看了一下transformers/tokenizers 的实现。原始的GPT2 中的tokenizer 是没有做normalize 的，所以 transformers中的GPT2Tokenizer 也是没有做normalize 的，而通常我们自己训练的bytes bpe 是会加一个normalize 的过程，所以如果是通过huggingface/tokenizers 训练的tokenier，迁移到transformers 时需要注意normalize 是否实现。

# 推荐
当目标语言为中文时，推荐使用WordPiece + jieba 的方案；而是多语言场景时，推荐使用SentencePieceBPE/SentencePieceUnigram.
无论哪种合并/切分 subword 的策略，我们的初衷是: 
<code>尽量不切分常用词，而是将不常用词切分为常用的子词.</code>
而中文中，有明确的字/词概念，却没有子词的概念（如英文中有"app", "##le", 中文却没有"苹" "##果"），而转bytes 后对子词更友好，此外，中文通常需要3个bytes（GBK）或者4个bytes（Chinese-Japanese character set），对于一个中文的字，很有可能需要大于1个token 来表示，反而会增加tokenize 后序列的长度，对模型的训练与使用不利；此外，中文中空格也没有切分词/句子 的语义，保留空格反而会由于各种空格的错误使用带来问题，最终的推荐方案就是jieba +  Word Piece/SentencePieceUnigram。
而多语言场景下，推荐使用SentencePieceBPE，他提供一个端到端的方案，而不需要再根据不同语言进行不同的pre-tokenize/subword 格式，此外，SentencePiece 都是bytes 粒度的，这样既能大大缩减词表又能避免unk 的情况。

# 补充阅读
[tokenizer summary-huggingface](https://huggingface.co/docs/transformers/tokenizer_summary)
[Difference Between NFD, NFC, NFKD, and NFKC Explained with Python Code-medium](https://towardsdatascience.com/difference-between-nfd-nfc-nfkd-and-nfkc-explained-with-python-code-e2631f96ae6c)
[深入理解NLP Subword算法：BPE、WordPiece、ULM-知乎](https://zhuanlan.zhihu.com/p/86965595)
[natural_language_understanding/subword_units/subword_units-github.io](https://everdark.github.io/k9/notebooks/ml/natural_language_understanding/subword_units/subword_units.nb.html#121_expectation-maximization)

# 关于头图
本人真实写照🐶