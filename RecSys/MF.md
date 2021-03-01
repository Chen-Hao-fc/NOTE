## **NCF: Neural Collaborative Filtering**

文章主要是利用两个向量内积拟合 user-item矩阵,提出了现有矩阵分解的局限性，文章举例：

![cnf_1](/Users/chenhao/Documents/NOTE/RecSys/images/cnf_1.png)

如图(a)，$u_4$和$u_1$最相似，其次是$u_3$最后是$u_2$,然而在图(b)中，如果将$p_4$拉想$p_1$的话，则$p_4$ closer to $p_2$ than $p_3$

但是这显然和图(a)的数据不符，这说明MF使用简单的内积在低维向量空间中用来估计user和item的interaction存在局限性。

于是文章提出使用nerual network代替内积对user-item的interaction来建模，大体的framework如下图：

![ncf_2](/Users/chenhao/Documents/NOTE/RecSys/images/ncf_2.png)

在基础框架的基础上，文章提出了两种结构：

1. **GMF: Generalized Matrix Factorization**

   一般的dot product:  $\phi_{1}\left(\mathbf{p}_{u}, \mathbf{q}_{i}\right)=\mathbf{p}_{u} \odot \mathbf{q}_{i}$

​	   GMF的dot product: $\hat{y}_{u i}=a_{\text {out}}\left(\mathbf{h}^{T}\left(\mathbf{p}_{u} \odot \mathbf{q}_{i}\right)\right)$

​		其中：$h$表示给内积加权，$a_{out}$表示激活函数

2. **MLP: Multi-Layer Perceptron**
   $$
   \begin{aligned} \mathbf{z}_{1} &=\phi_{1}\left(\mathbf{p}_{u}, \mathbf{q}_{i}\right)=\left[\begin{array}{c}\mathbf{p}_{u} \\ \mathbf{q}_{i}\end{array}\right] \\ \phi_{2}\left(\mathbf{z}_{1}\right) &=a_{2}\left(\mathbf{W}_{2}^{T} \mathbf{z}_{1}+\mathbf{b}_{2}\right) \\ & \ldots \ldots \\ \phi_{L}\left(\mathbf{z}_{L-1}\right) &=a_{L}\left(\mathbf{W}_{L}^{T} \mathbf{z}_{L-1}+\mathbf{b}_{L}\right) \\ \hat{y}_{u i} &=\sigma\left(\mathbf{h}^{T} \phi_{L}\left(\mathbf{z}_{L-1}\right)\right) \end{aligned}
   $$
   其中：$\left[\begin{array}{l}\mathbf{p}_{u} \\ \mathbf{q}_{i}\end{array}\right]$表示向量拼接，经过多层感知机后得到 logits

3. **Fusion of GMF and MLP**

   ![ncf_3](/Users/chenhao/Documents/NOTE/RecSys/images/ncf_3.png)



公式如下：
$$
\begin{aligned} \phi^{G M F} &=\mathbf{p}_{u}^{G} \odot \mathbf{q}_{i}^{G} \\ \phi^{M L P} &=a_{L}\left(\mathbf{W}_{L}^{T}\left(a_{L-1}\left(\ldots a_{2}\left(\mathbf{W}_{2}^{T}\left[\begin{array}{l}\mathbf{p}_{u}^{M} \\ \mathbf{q}_{i}^{M}\end{array}\right]+\mathbf{b}_{2}\right) \ldots\right)+\mathbf{b}_{L}\right)\right.\\ \hat{y}_{u i} &=\sigma\left(\mathbf{h}^{T}\left[\begin{array}{l}\phi^{G M F} \\ \phi^{M L P}\end{array}\right]\right) \end{aligned}
$$
将两个方式结合拼接，加权，激活后得到结果，也就是ensemble了一下。

文章最后也点出，做了pre-training, 先将两个模型分别train,然后再结合train

除此之外，作者也采用了negative sampling的方法进行train

------



## NCF vs MF

**Neural Collaborative Filtering vs. Matrix Factorization Revisited** 作者是Google推荐大佬Steffen Rendle(FM的提出者)，来自RECSystem2020

文章主要向16年何向蓝提出的NCF提出质疑,全文就表名一个主题：<font color=red>在NCF 中使用MLP代替原始MF的dot product效果差</font>

```
we revisit the experiments of the NCF paper that popularized learned similarities using MLPs. First, we show that with a proper hyperparameter selection, a simple dot product substantially outperforms the proposed learned similarities. Second, while a MLP can in theory approximate any function, we show that it is non-trivial to learn a dot product with an MLP.
```

在introduction的末尾作者点出，MLP拟合dot product的前提是:<font color=red>large dataset 或者 embedding dimension is very small</font>

```
To summarize, this paper argues that MLP-based similarities for combining embeddings should be used with care. While MLPs can approximate any continuous function, their inductive bias might not be well suited for a similarity measure. Unless the dataset is large or the embedding dimension is very small, a dot product is
likely a better choice.
```

文章提出NCF效果好，可能是cherry-picked

```
the results for NeuMF and MLP in [17] were cherry-picked in the following sense: the metrics are reported for the best iteration selected on the test set.
```

**文章也说明了 卷积，RNN, attention structuers 这些网络不能被MLP所replace,所以因为dot product简单，所以它就能被MLP所拟合吗？**

在实际的实时推荐应用中，dot product的时间复杂度为$O(d)$,而MLP的复杂度至少是$O(d^2)$

除此之外，作者还说命了dot product在DNN中的广泛应用

1. Dot products at the Output Layer of DNNs

   在通常的DNN中，$x$代表input, $f(x)$表示将$x$ map to a representation or a embedding $f(x)\in R^d$,紧接着会乘一个class matrix 

   $Q \in R^{n \times d}$ to obtain a scalar score for each of the $n$ class, 然后获得的这个voctor会被送进loss function，作者想说的是，$Q$ 可以看做users matrix, $f(x)$看做 items matrix这个$Q^Tf(x)$的过程其实就是user-item dot product的过程。

2. CNN，transformer这些在DNN上有很好的效果，虽然原理上他们能被MLP拟合，但是MLP却不能达到这么好的结果。同时表名已经有文章指出MLP不能model multiplications，他们在近似dot product和MLP时候发现，在较好的embedding dimension时效果还好，但是维度到达100后，效果会差。