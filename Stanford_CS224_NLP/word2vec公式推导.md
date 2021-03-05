# 跟着stanford老爷子推导word2vec公式

## Likelihood 

$$
L(\theta)=\prod_{t=0}^T\prod_{-m\leq j\leq m \\ j\neq0}P(w_{t+j}|w_{t};\theta)
$$

t代表位置，m表示context大小

## objective function

$$
J(\theta)=-\frac{1}{T}logL(\theta)=-\frac{1}{T}\sum_{t=0}^T\sum_{-m\leq j\leq m \\ j\neq0}logP(w_{t+j}|w_{t};\theta)
$$

## 怎样计算 $P(w_{t+j}|w_{t};\theta)$ ?

Answer We will use two vectors per words w:

$v_{w}$ when $w$ is a center word

$u_{w}$ when $w$ is a context word

Then for a center word $c$ and a context word $o$：
$$
P(o|c)=\frac{exp(u_{o}^T v_{c})}{\sum_{w\in V}exp(u_{w}^T v_{c})}
$$

## 求导

$$
\begin{align}
&\frac{\partial }{\partial v_c}log\frac{exp(u_{o}^T v_{c})}{\sum_{w\in V}exp(u_{w}^T v_{c})} \\
&=\frac{\partial }{\partial v_c}log(exp(u_{o}^T v_{c})) -\frac{\partial }{\partial v_c}log(\sum_{w\in V}exp(u_{w}^T v_{c})) \\
&=\frac{\partial }{\partial v_c}u_{o}^T v_{c} - \frac{1}{\sum_{w\in V}exp(u_{w}^T v_{c})}\frac{\partial }{\partial v_c}\sum_{w\in V}exp(u_{w}^T v_{c}) \\
&=u_o - \frac{1}{\sum_{w\in V}exp(u_{w}^T v_{c})} \sum_{w\in V}\frac{\partial }{\partial v_c}exp(u_{w}^T v_{c}) \\
&=u_o - \frac{1}{\sum_{w\in V}exp(u_{w}^T v_{c})} \sum_{w\in V}exp(u_{w}^T v_{c})\frac{\partial }{\partial v_c} u_{w}^T v_{c} \\
&=u_o - \frac{1}{\sum_{w\in V}exp(u_{w}^T v_{c})} \sum_{w\in V}exp(u_{w}^T v_{c})\cdot u_{w} \\
&=u_o - \frac{\sum_{x\in V}exp(u_{x}^T v_{c})\cdot u_{x}}{\sum_{w\in V}exp(u_{w}^T v_{c})}  \\
&=u_o - \sum_{x\in V}\frac{exp(u_{x}^T v_{c})}{\sum_{w\in V}exp(u_{w}^T v_{c})}\cdot u_x \\
&=u_o - \sum_{x=1}^{V}P(x|c)\cdot u_x
\end{align}
$$

从上述公式看出，在原始的word2vec中，影响词向量$v_c$梯度的是其context向量$u_o$和vocabulary中<font color=#008000>所有词</font>的probability $P(x|c)$和其对应的向量$u_x$

<font color=red>TODO</font>: $u_o - \frac{1}{\sum_{w\in V}exp(u_{w}^T v_{c})} \sum_{w\in V}exp(u_{w}^T v_{c})\cdot u_{w}$在计算这里的时候，我差点把分子分母约掉了，后来发现这里分子分母不是一个东西，是不能约掉的，相对来说分子多了一个$u_w$使得分子和分母完全不是一个东西。

