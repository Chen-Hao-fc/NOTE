$r_{i j}=\frac{\sum_{k} \text {Sim}\left(u_{i}, u_{k}\right) r_{k j}}{\text { number of ratings }}$







$\operatorname{Sim}\left(u_{i}, u_{k}\right)=\frac{\sum_{j}\left(r_{i j}-r_{i}\right)\left(r_{k j}-r_{k}\right)}{\sqrt{\sum_{j}\left(r_{i j}-r_{i}\right)^{2} \sum_{j}\left(r_{k j}-r_{k}\right)^{2}}}$







$\min _{p, q, b_{u}, b_{i}} \sum\left(r_{u i}-\left(p_{u}^{T} q_{i}+\mu+b_{u}+b_{i}\right)\right)^{2}$



Linear: $\hat{y}(x)=w_{0}+\sum_{i=1}^{n} w_{i} x_{i}$
$L R: \hat{y}(x)=\frac{1}{1+w_{0} \exp \left(-w^{T} x\right)}$



$Linear:\hat{y}(x)=w_{0}+\sum_{i=1}^{n} w_{i} x_{i}$

$L R: w_{wide}=\frac{1}{1+w_{0} \exp \left(-w^{T} x\right)}$

$\mathrm{FM}: \hat{y}(x)=w_{0}+\sum_{i=1}^{n} w_{i} x_{i}+\sum_{i=1}^{n} \sum_{j=i+1}^{n}\left\langle v_{i}, v_{j}\right\rangle x_{i} x_{j}$
$$
x=[x_{user};x_{item}]=[x_1,x_2,...x_n]
$$
$\hat{y}(x)=\sigma\left(\mathrm{w}_{w i d e}^{T}[\mathrm{x}, \phi(\mathrm{x})]+\mathrm{w}_{\text {deep}}^{T} a^{\left(l_{f}\right)}+b\right)$



$\hat{y}(x)=\sigma(y_{FM}+y_{DNN})$

$y_{FM}=<w,x>+\sum_{j_1=1}^d\sum_{j_2=j_1+1}^d<V_i,V_j>x_{j_1}.x_{j_2}$



$\boldsymbol{v}_{U}(A)=f\left(\boldsymbol{v}_{A}, \boldsymbol{e}_{1}, \boldsymbol{e}_{2}, \ldots, \boldsymbol{e}_{H}\right)=\sum_{j=1}^{H} a\left(\boldsymbol{e}_{j}, \boldsymbol{v}_{A}\right) \boldsymbol{e}_{j}=\sum_{j=1}^{H} \boldsymbol{w}_{j} \boldsymbol{e}_{j}$







$\underbrace{p(y=1, z=1 \mid x)}_{p C T C V R}=\underbrace{p(y=1 \mid x)}_{p C T R} \times \underbrace{p(z=1 \mid y=1, x)}_{p C V R}$

$p(z=1 \mid y=1, x)=\frac{p(y=1, z=1 \mid x)}{p(y=1 \mid x)}$

$\begin{aligned} L\left(\theta_{c v r}, \theta_{c t r}\right) &=\sum_{i=1}^{N} l\left(y_{i}, f\left(x_{i} ; \theta_{c t r}\right)\right) \\ &+\sum_{i=1}^{N} l\left(y_{i} \& z_{i}, f\left(x_{i} ; \theta_{c t r}\right) \times f\left(x_{i} ; \theta_{c v r}\right)\right) \end{aligned}$

$S^{k}(x)=\left[E_{(k, 1)}^{T}, E_{(k, 2)}^{T}, \ldots, E_{\left(k, m_{k}\right)}^{T}, E_{(s, 1)}^{T}, E_{(s, 2)}^{T}, \ldots, E_{\left(s, m_{s}\right)}^{T}\right]^{T}$

$w^{k}(x)=\operatorname{Softmax}\left(W_{g}^{k} x\right)$

$g^{k}(x)=w^{k}(x) S^{k}(x)$

