

[TOC]



## **1. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks(UKP-TUDA)**

![sentencebert_1](/Users/chenhao/Documents/NOTE/Transformer Note/images/sentencebert_1.png)

**文章主要贡献：**

1. 指出了直接用原始Bert做sentence embedding效果不好
2. 在paper的introduction部分有很多sentence embedding方法的介绍
3. 提出了利用类似<u>双塔Bert模型</u>做相似性的想法，不同的场景使用了不同的embedding策略和不同的损失函数

**直接使用原始Bert做sentence embedding效果不好**

引用paper中的一句话：

```
The most commonly used approach is to average the BERT output layer (known as BERT embeddings) or by using the out-put of the first token (the [CLS] token). As we will show, this common practice yields rather bad sentence embeddings, often worse than averaging GloVe embeddings .
```

另外一句话：

```
A large disadvantage of the BERT network structure is that no independent sentence embeddings are computed, which makes it difficult to derive sentence embeddings from BERT.
```

**SBert**

如图所示，文章并没有做pre-training，而是直接在别人预训练好的模型上针对自己的特定任务fine-tuning

paper提出了3种pooling strategies:

1. Using the output of CLS-token
2. computing the mean of all output vectors(MEAN-strategies) <font color=#FF0000 >默认使用策略</font>
3. computing a Max-over-time of the output vectors(MAX-strategies)

依据不同的training data使用了3种不同的structures and objective functions

Classification Objective Function: $o=softmax(W_t(u,v,|u-v|))$

Regression Objective Function: 如图2利用cosine相似性计算,然后求MSE

Triplet Objective Function: $max(||s_a-s_p||-||s_a-s_n||+\epsilon, 0)$ 其中，$s_p$正样例到$s_a$的距离比$s_n$负样例到$s_a$的距离远

**Setup**

1. er用的语料是pair sentence
2. batch_size=16, Adam optimizer learning rate 2e-5, linear learning rate warm-up over 10% training data, default pooling strategy is MEAN

------



## **2. Improving Multilingual Sentence Embedding using Bi-directional Dual Encoder with Additive Margin Softmax(google AI)**



![sentencebert_3](/Users/chenhao/Documents/NOTE/Transformer Note/images/sentencebert_3.png)

**有关上图的备注：**

> [Guo *et al.*, 2018] proposed a new approach using a <u>dual- encoder architecture instead of a encoder-decoder one</u>. The dual-encoder architecture optimizes the cosine similarity between the source and target sentences directly. Here, we extend this approach by using a bidirectional dual-encoder with additive margin softmax, which significantly improves the model performance.

**文章主要贡献：**



1. 指出了NMT任务的实质的将source and target sentence映射到相同的embedding空间，然后做rank问题，因此使用翻译的语料来学习sentence embedding.
2. 介绍了Dual encoder Model
3. 介绍了在做机器翻译的时候的Bidirectional Dual Encoder
4. 介绍了 Dual Encoder with Additive Margin Softmax
5. 介绍了Translation retrieval pipeline的大体流程



**Dual Encoder Model:**

如上图所示，介绍了一种instead 传统的encoder-decoder方式的 dual-encoder框架模型，虽然不是他提出来的，

将机器翻译问题当做为一个rank问题，对于一个source $x_i$ translation retrieval can be modeled as a ranking problem to place $y_i$ from $\mathcal{Y}$,  $P(y_i|x_i)$可以被表示为：
$$
P\left(y_{i} \mid x_{i}\right)=\frac{e^{\phi\left(x_{i}, y_{i}\right)}}{\sum_{\bar{y} \in \mathcal{Y}} e^{\phi\left(x_{i}, \bar{y}\right)}}
$$
在优化 $P(y_i|x_i)$的过程中，同一个batch的其他样本可以看做为负样本：

>  during training by sampling negatives, yn, among translation pairs in the same batch

$$
P_{\text {approx}}\left(y_{i} \mid x_{i}\right)=\frac{e^{\phi\left(x_{i}, y_{i}\right)}}{e^{\phi\left(x_{i}, y_{i}\right)}+\sum_{n=1, n \neq i}^{N} e^{\phi\left(x_{i}, y_{n}\right)}}
$$

相似性$\phi$使用dot-product，最终的损失函数可表示为：
$$
\mathcal{L}_{s}=-\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{\phi\left(x_{i}, y_{i}\right)}}{e^{\phi\left(x_{i}, y_{i}\right)}+\sum_{n=1, n \neq i}^{N} e^{\phi\left(x_{i}, y_{n}\right)}}
$$
**Bidirectional Dual Encoder**

针对机器翻译任务，好的embedding空间是双向的， form both $x_i$ to $y_i$ and $y_i$ to $x_i$
$$
\mathcal{L}_{s}^{\prime}=-\frac{1}{N} \sum_{i=1}^{N} \frac{e^{\phi\left(y_{i}, x_{i}\right)}}{e^{\phi\left(y_{i}, x_{i}\right)}+\sum_{n=1, n \neq i}^{N} e^{\phi\left(y_{i}, x_{n}\right)}}
$$
双向的损失可表示为：
$$
\overline{\mathcal{L}}_{s}=\mathcal{L}_{s}+\mathcal{L}_{s}^{\prime}
$$
**Dual Encoder with Additive Margin Softmax**

![sentencebert_4](/Users/chenhao/Documents/NOTE/Transformer Note/images/sentencebert_4.png)

使用的是别人的方法，paper：<u>Additive margin softmax for face veri- fication</u>

 <font color=#FF0000 >其实本质就是将同一类别间的embedding距离缩小，不同类别embedding空间的距离拉大了</font>

$softmax$公式没怎么变，后续再研究原理，这里先列出针对多语言翻译场景的公式，就是在相似性函数上加上了指数$-m$：
$$
\phi^{\prime}\left(x_{i}, y_{j}\right)=\left\{\begin{array}{ll}
\phi\left(x_{i}, y_{j}\right)-m & \text { if } i=j \\
\phi\left(x_{i}, y_{j}\right) & \text { if } i \neq j
\end{array}\right.
$$
损失函数随之改变：
$$
\mathcal{L}_{a m s}=-\frac{1}{N} \sum_{i=1}^{N} \frac{e^{\phi\left(x_{i}, y_{i}\right)-m}}{e^{\phi\left(x_{i}, y_{i}\right)-m}+\sum_{n=1, n \neq i}^{N} e^{\phi\left(x_{i}, y_{n}\right)}}
$$
翻译的双向损失：
$$
\overline{\mathcal{L}}_{a m s}=\mathcal{L}_{a m s}+\mathcal{L}_{a m s}^{\prime}
$$
**The Parallel Corpus Retrieval Pipeline**

1. 使用dual-encoder  model计算出source 和 target 的embedding
2. all embedding build 完成后，接下来就是通过最近邻算法求出最优的

**Setup**

1. 使用了character-level和word-level的vocbulary,所有语言共享200k的词表
2. encoder 使用的是3-layer transformer, 8heads, hidden_size 512, filter_size 512x4=2048
3. 输出层使用了4 pooling layers的结合，max pooling, mean pooling, first token pooling, attention pooling 拼接然后映射到500的空间
4. margin  softmax 的$m$值设置为 0.3

## **3. Language-agnostic BERT Sentence Embedding(google)**

基于Bert训练多语言的sentence embedding， <u>上篇是直接fine-tuning产生sentence embedding, 而这里是pre-training 了一个模型</u>

![sentencebert_2](/Users/chenhao/Documents/NOTE/Transformer Note/images/sentencebert_2.png)



**文章主要贡献：**

1. 从另一个角度阐述了原始Bert不能做sentence embedding的原因
2. 利用了多语言和机器翻译的语料类似sentenceBert的双塔模式来训练,这里它将这种双塔模型称作 <font color=#FF0000 >dual-encoder</font>
3. 使用了一种在多机多卡下构造训练负样本的方法，类似DCR，这里称作<font color=#FF0000 >Additive Margin Softmax</font>

**原始Bert不能做sentence embedding**

> However, pretrained MLMs do not intrinsically produce good sentence-level embeddings. Rather, the production of sentence embeddings from MLMs must be learned via fine-tuning, similar to other downstream task. 

其实说的就是MLM任务不能产生很好的sentence-level embedding，同时点到了如果要产生较好的sentence embedding就该像bert通常的那种fine-tunin下游任务一样，针对具体的任务通过fine-tuning产生sentence-level embedding

**Bidirectional Dual Encoder with Additive Margin Softmax**

DCR里面我记得是将一个batch中，一个正样本，其他$N-1$个样本作为负样本。如下面的公式：
$$
\mathcal{L}=-\frac{1}{N} \sum_{i=1}^{N} \frac{e^{\phi\left(x_{i}, y_{i}\right)-m}}{e^{\phi\left(x_{i}, y_{i}\right)-m}+\sum_{n=1, n \neq i}^{N} e^{\phi\left(x_{i}, y_{n}\right)}}
$$
$\phi\left(x_{i}, y_{i}\right)=cosine(x,y)$ ,如果将$x$视为source $y$视为target，那么在李航的那本书里面提过这是个L2R的问题，上述公式目标是将$y$排在最前面。

同理如果将$y$视为source $x$视为target，那么目标是将x排在最前面，将两个损失加起来才是一个样本的最终损失，即：
$$
\overline{\mathcal{L}}=\mathcal{L}+\mathcal{L}^{\prime}
$$
这里的$m$其实就是 Margin softmax，具体的一些细节后续再研究

**Cross-Accelerator Negative Sampling**

这里更近一步，因为在训练的时候用的是多机多卡，在多机多卡的框架下，模型同时训练的不仅是一个batch，而是多个batch同时train, 这里将多个batch里面的其他样本都视为负样本，paper中的表达：

> under this strategy the sentences from all cores are broadcast as negatives for the examples assigned to other cores. This allows us to fully realize the benefits of dis- tributed training.

**Pre-training and parameter sharing**

在Monolingual Data上面训练MLM任务，在Bilingual translation pairs上面训练翻译任务

使用了3个stage的stacking算法，首先学习$\frac{L}{4}$ layers， 再学习$\frac{L}{2}$ layers, 最后学习$L$ layers, 早期stage的参数被copy到后期stage的任务中

**Setup**

1. corpus: We have two types of data: monolingual data and bilingual translation pairs.

   Monolingual Data: 来自wiki和爬虫，we remove short lines < 10 characters and those > 5000 characters.利用一个句子分类器，过滤掉了那些质量不好的sentence

   Bilingual translation pairs: 来自 web pages，同样利用一些方法做了筛选和过滤，分为 GOOD 和 BAD 句子

2. 因为是多语言所以The final vocabulary size is 501,153

3. encoder使用Bert base model, 12 layers, 12 heads, 768 hidden_size

4. 使用[CLS]作为sentence embedding

5. source和target的最大句长max-seq_len 为64, AdamW optimize learning rate is 1e-5