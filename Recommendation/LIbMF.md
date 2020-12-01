## 简述

这是台大[lib系列](%3Cahref="https://www.csie.ntu.edu.tw/~cjlin/libmf/"%3Ehttps://www.csie.ntu.edu.tw/~cjlin/libmf/%3C/a%3E)也是一篇Systems的最佳论文，虽然偏偏工程，在算法上没有太大的亮点，但是在时间和内存优化上面效果很好。

## background

矩阵分解的主要目标是

$min_{P,Q}=\sum_{((ur)\in{R})}((\gamma_{u,v} - p_u^Tq_v)^2 + \lambda_P||p_u||^2 + \lambda_Q||q_v||^2)$

如果取矩阵中一个数可表示为：

$(\gamma_{u,v} - p_u^Tq_v) + \lambda_Pp_u^Tp_u+\lambda_Qq_v^Tq_v$

而$p_u$和$q_v$的梯度下降公式为：

$p_u\leftarrow p_u + \eta(e_{u,v}q_v-\lambda_Pp_u)$

$q_v\leftarrow q_v + \eta(e_{u,v}p_u-\lambda_Qq_v)$

其中：$e_{u,v}=\gamma_{u,v}-p_u^Tq_v$

虽然SG已经成功用在了矩阵分解上面，但是还是不能用在大数据上面，可以看公式在优化$p_u$的时候用到$q_v$,而在优化$q_v$的时候用到刚才的$p_u$，论文描述这是一个sequential问题，因此它是一个不能并行的问题。而为了提高随机梯度的速度，必定要用到并行，因此后续有很多随机梯度并行算法。

\###### 现有的并行的随机梯度下降方法

\* **HogWild**

多个线程之间互相独立，各自更新自己的，然后定时的拉取每个线程的结果，然后更新所有的参数，可是很明显，这个过程中有明显的over-write问题。

\* **DSGD**

将矩阵分为多个相互独立的block,这些block中的值可以并行更新。（<u>we say two blocks are independent to each other if they share neither any common column nor any common row of the matrix</u>）如图：![4f1f99adaf5700b99df7ab5cf387a5a1.jpeg](evernotecid://153AB4E7-343B-4EA3-894B-7B6515E116C4/appyinxiangcom/25354291/ENResource/p68)

图中每个图都可以是矩阵被分解为多个block的结果，这些block之间不share相同的row和column

下图可以说明这些block之间可以相互并行的原因：![edf72596e2cdcdeacdded4b2bccc8d63.jpeg](evernotecid://153AB4E7-343B-4EA3-894B-7B6515E116C4/appyinxiangcom/25354291/ENResource/p69)

但是每次多个线程结束后，P和Q都必须相互交换他们的信息，比如说图3(b),是两次不同的block分割，左图中在优化$i_0$的时候，需要用到$q_0$,但是在右图不同的block分割中，在优化$i_3$的时候也要用到$q_0$，因此将矩阵分解成多个相互独立的block,在一次优化过程中每个block之间可以并行进行，但是进行到下一轮，重新分解block的时候，需要等到上一轮block的所有都更新完，才能进行。

\* CCD++(一种坐标下降法，这里就不多介绍了)

\###### 现有并行的随机梯度下降法（parallel SG）的问题所在

论文指出主要存在两个问题 locking problem 和 memory discontinuity

\* **locking problem**

对于一个并行的算法，如果想最大化它的效果，最重要的keepin all threads busy, 这里的locking problem指的是，一个线程数据少处理的快，而在等待其他线程的结束。而造成这样的原因主要是矩阵数据分布不均匀。如图：![24694b4846ad06973bff637f8a74e0db.jpeg](evernotecid://153AB4E7-343B-4EA3-894B-7B6515E116C4/appyinxiangcom/25354291/ENResource/p70)

目前主要的解决办法是random shuffling，但是打乱也有问题，即是数据每个block中数据已经分布均匀了，但是更新的时候也存在快慢问题。

\* **memory discontinuity**

论文里面点到：<u>When a program accesses data in memory discontinuously, it suffers from a high cache miss rate and performance degradation</u>意思是说如果存到内存的数据是不连续处理的，将会suffer from一个cache miss问题，而这会严重影响程序的 performance

论文中指出了HogWild和DSGD都采用了从R中随机选择(random method)去更新的策略，如下图：![20b7e42fe0185bb5c7fa6598cdacd2ba.jpeg](evernotecid://153AB4E7-343B-4EA3-894B-7B6515E116C4/appyinxiangcom/25354291/ENResource/p71)

因此是随机选择R中的数据更新，导致了P和Q中的数据也是随机update,杂乱无章，而这个过程中可能导致memory discontinuity

\###### 我们提出的方法 FPSG(fast parallel SG)

论文突出了两种方法去解决上述问题 lock-free scheduling 和 partial random method

\* **lock-free scheduling**

定义了一个概念：<u>free block : if it is independent from all blocks being processed Otherwise it is a non-free block </u>

当一个进程完成一个block后，则scheduler将会给他派遣一个新的block让它来处理，这个新的block必须满足一下准则：

\1. it is a free block

\2. Its number of past updates is the smallest among all free blocks(就是派遣一个最小的，也就是待更新数目较少的让它处理)

因为我们希望保留一些小的blocks供那些快速处理完的进程继续处理，因此如果我们有s个进程，我们一般讲矩阵分解为(s+1)x(s+1)个block。

上面我们说过造成locking problem的原因还有就是矩阵中数据分布不均匀所致，所以在shuffle之后，为了描述每个block数据的平衡程度，我们定义了一个degree of imbalance(DOI)的概念用来check数据中需要备update的个数。

\* **partial random method**

对应上述说的random method 论文采用order method 将要更新的R的数据按照顺序存储，如图：

![e80d543a601646a333d36884d2ea2b14.jpeg](evernotecid://153AB4E7-343B-4EA3-894B-7B6515E116C4/appyinxiangcom/25354291/ENResource/p72)

就是说如果我们将R中的数据按照如图的格式存储，逐行update的过程中，P中的向量更新是按照顺序的。

<u>在FPSG中，we propose a partial random method, which select ratings in a block orderly but randomizes th selection of blocks</u>(随机选择block但是在每个block内部的数据是按照顺序存储的)。之所以还有随机的成分，主要是因为在某个block处理完之后，该进程它需要选择一个smallest block去处理，但是如果此刻some blocks 有相同的待更新数目，这个时候就是随机选择一个，如图：![8a553e44467cabf1e64b6cbf4b66d189.jpeg](evernotecid://153AB4E7-343B-4EA3-894B-7B6515E116C4/appyinxiangcom/25354291/ENResource/p73)

\###### overview of FPSG

\1. shuffle R矩阵（随机打乱）

\2. 将R矩阵分割至少(s+1)x(s+1)个block， 其中s为线程数目

\3. 每个block内存对user或者item 排序（也就是说的order method）

\4. 构建scheduler去指导s个进程work