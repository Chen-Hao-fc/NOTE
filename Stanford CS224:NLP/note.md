## n-gram

N-gram遇到的问题

Example：students open their __

1. 比如说 student open their $\underline{iphone}$   iphone这个词如果从来没有出现在train data中，那么n-gram probability 就是0？

   解决办法： smoothing

   给词表中的每个词增加很小的probability，代表其可能发生

2. 如果students open their 在train data中从来没有发生过，则这句话出现的概率就是0？

   解决办法：backoff

   4-grams 发生的概率为0，那么考虑trigrams，即考虑open their_发生的概率

## perplexity和cross-entropy的关系

### perplexity（越小越好）

标准的Language Models的metrix是perplexity
$$
perplexity = \prod_{t=1}^T (\frac{1}{P_{LM}(x^{(t+1)}|x^{t}, ...,x^1)})^\frac{1}{T}
$$
其中 $\frac{1}{T}$是用来Noramalized的，

<font color=red>对于训练好的模型，测试集中的句子都是正常句子，那么训练好的模型在才是的上的困惑度越小越好</font>

### cross-entropy

在generated LM中在t位置预测next word即 $x^{(t+1)}$时，利用Cross-entropy来计算

令:predicted probability distribution is $\hat{y}^{(t)}$ and the true next word $y^{(t)}$(one-hot for $x^{(t+1)}$) 
$$
J^{(t)}(\theta)=CE(y^{(t)}, \hat{y}^{(t)})=-\sum_{w\in V}y^{(t)}_{w}log\hat{y}_{w}^{(t)}=-log\hat{y}_{x^{(t+1)}}^{(t)}
$$
注意：这里有个向量对应元素相乘其中一个是one-hot的概念

接着平均每个位置上的损失overall loss:
$$
J(\theta)=\frac{1}{T}\sum_{t=1}^TJ^{(t)}(\theta)=\frac{1}{T}\sum_{t=1}^T-log\hat{y}_{x^{(t+1)}}^{(t)}
$$
在language model中困惑度这个公式是等价于 exp 交叉熵$J(\theta)$的：
$$
exp(J(\theta))=exp(\frac{1}{T}\sum_{t=1}^T-log\hat{y}_{x^{(t+1)}}^{(t)})=\prod_{t=1}^T(\frac{1}{\hat{y}_{x^{(t+1)}}})^{\frac{1}{T}}=perplexity
$$
<font color=red>因此在生成式模型中，优化交叉熵其实就是优化其在训练集的困惑度，所以也可以用困惑度作为损失函数</font>

## multi-layer RNN

在训练多层RNN时，可以加入resnet来避免梯度消失