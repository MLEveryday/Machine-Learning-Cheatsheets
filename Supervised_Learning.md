# 1. 有监督学习
## 1.1 介绍
给定数据点集合 $\{x^{(1)},...,x^{(m)}\}$ 和输出集合 $\{y^{(1)},...,y^{(m)}\}$，我们想建立一个分类器，让它学习如何从 $x$ 预测 $y$。

 - 预测结果的类型

不同预测结果的类型见下表：

|模型|输出|例子|
|:--:|:--:|:--:|
|回归|连续|线性回归|
|分类器|类别|逻辑回归、支持向量机、朴素贝叶斯|


 - 模型的类型
 
 |模型|目标|例子|
|:--:|:--:|:--:|
|判别式|估计 $P(y\vert{x})$|各种回归、支持向量机|
|生成式|预测 $P(x\vert{y})$ <br>用于推断 $P(y\vert{x})$|GDA，朴素贝叶斯|

## 1.2 备注和一般概念
- 假设

假设 $h_\theta$ 是我们选择的模型。对于给定的输入 $x^{(i)}$，模型预测结果是 $h_\theta(x^{(i)})$。

- 损失函数

损失函数 $L:(z,y)\in{\Bbb{R}}\times{Y}\mapsto{L(z,y)}\in{\Bbb{R}}$ 把预测值 $y$ 和真实值 $z$ 作为输入，输出他们的差异程度。常见的损失函数见下表：

|二乘法|Logistic|Hinge|交叉熵|
|:--:|:--:|:--:|:--:|
|$\frac{1}{2}(y-z)^2$|$\log(1+\exp(-yz))$|$\max(0,1-yz)$|$-[y\log(z)+(1-y)\log(1-z)]$|
|![][1]|![][2]|![][3]|![][4]|
|线性回归|逻辑回归|支持向量机|神经网络|


- 代价函数

代价函数 $J$ 通常用于表示模型的性能，和损失函数 $L$ 一起，定义如下：

$$
\boxed{J_\theta=\sum\limits_{i=1}^mL(h_\theta(x^{(i)},y^{(i)}))}
$$

- 梯度下降法

学习速度 $\alpha\in\Bbb{R}$，梯度下降法更新规则用学习速度和代价函数表示：

$$
\boxed{\theta\leftarrow\theta-\alpha\nabla{J}(\theta)}
$$

![此处输入图片的描述][5]
>备注：随机梯度下降算法(SGD)基于每个训练数据更新参数，而批量梯度下降是基于批量数据。

- 似然函数

模型的似然函数 $L(\theta)$ 通过可能性最大化找到最优的参数 $\theta$。 实践中，我们用对数似然函数 $\ell(\theta)=\log(L(\theta))$，更容易优化：

$$
\boxed{\theta^{opt}=\arg \max\limits_\theta L(\theta)}
$$

- 牛顿算法

牛顿算法是通过 $\ell'(\theta)=0$ 查找 $\theta$。它的更新规则如下：

$$
\boxed{\theta\leftarrow\theta-\dfrac{\ell'(\theta)}{\ell''(\theta)}}
$$

>备注：多维泛化，也被称作牛顿迭代法，更新规则如下：

$$
\theta\leftarrow\theta-(\nabla_\theta^2\ell(\theta)^{-1}\nabla_\theta{\ell(\theta)})
$$

## 1.3 线性模型
### 1.3.1 线性回归
我们假设 $y|x;\theta\sim{\mathcal{N}(\mu,\sigma^2)}$。

- 正则方程

矩阵 $X$,代价函数最小值 $\theta$ 是一个闭式方案：

$$
\boxed{\theta=(X^TX)^{-1}X^Ty}
$$

- 最小二乘法

学习速率 $\alpha$，对于 $m$ 各点的训练集，最小二乘法更新规则同Widrow-Hoﬀ学习法，如下：

$$
\boxed{\forall, \theta_j\leftarrow+\alpha\sum_{i=1}^m[y^(i)-h_\theta(x^{(i)})]x_j^{(i)}}
$$

- 局部加权回归

局部加权回归，即LWR，是线性回归的变形，在代价函数中给每个训练数据权重 $\omega^{(i)}(x)$，参数 $\tau\in\Bbb{R}$：

$$
\boxed{\omega^{(i)}(x)=\exp(-\dfrac{(x^{(i)}-x)^2}{2\tau^2})}
$$

### 1.3.2 逻辑回归
- 激活函数

激活函数 $g$，即逻辑函数，定义如下：

 $$
 \forall{z}\in\Bbb{R}, \boxed{g(z)=\dfrac{1}{1+e^{-z}}\in[0,1]}
 $$

- 逻辑回归

假设 $y|x;\theta\in{Bernoulli}(\phi)$，如下：

$$
\boxed{\phi=p(y=1|x;\theta)=\dfrac{1}{1+\exp(-\theta^Tx)}=g(\theta^Tx)}
$$

> 备注：逻辑回归中没有闭式方案。

- SOFTMax回归

SOFTMax回归，也被称为多元逻辑回归，用于当输出类别多于2个时。按照惯例，设 $\theta_K=0$，每个类 $i$ 的伯努力参数 $\phi_i$：

$$
\boxed{\phi_i=\dfrac{\exp(\theta_i^Tx)}{\sum\limits_{j=1}^K\exp(\theta_j^Tx)}}
$$

### 1.3.3 广义线性模型
- 指数族

如果一个类型人不可以用自然参数表示，也被称为正则参数或链接函数，$\eta$，素数统计量 $T(\eta)$ 和对数划分函数 $\alpha(\eta)$ 如下：

$$
\boxed{p(y;\eta)=b(y)\exp(\eta{T}(y)-\alpha(\eta))}
$$

> 备注：通常情况 $T(y)=y$。同样的，$\exp(-\alpha(\eta))$ 可以看作正则化参数，使得概率结果是1。

常用的指数分布见下表：

|分布|$\eta$|$T(y)$|$\alpha(\eta)$|$b(\eta)$|
|:--:|:--:|:--:|:--:|:--:|
|伯努力|$\log(\frac{\phi}{1-\phi})$|$y$|$\log(1+\exp(\eta))$|1|
|高斯|$\mu$|$y$|$\frac{\eta^2}{2}$|$\frac{1}{\sqrt{2\pi}}\exp(-\frac{y^2}{2})$|
|泊松|$\log(\lambda)$|$y$|$e^\eta$|$\frac{1}{y!}$|
|几何|$\log(1-\phi)$|$y$|$\log(\frac{e^\eta}{1-e^\eta})$|1|


- GLM假设

GLM的目标是预测一个随机变量 $y$，如函数 $x\in\Bbb{R}$，基于下面3个假设
$$
(1)\ \boxed{y|x;\theta\sim{ExpFamily(\eta)}}\ (2)\ \boxed{h_\theta(x)=E[y|x;\theta]}\ (3)\  \boxed{\eta=\theta^Tx}
$$
>备注：普通最小二乘法和逻辑回归是GLM的特例。
## 1.4 支持向量机
支持向量机是为了使最小距离最大化。

- 最优边界分类器

最优边界分类器 $h$：

$$
h(x)=sign(\omega^Tx-b)
$$
其中 $(\omega,b)\in\Bbb{R}^n\times\Bbb{R}$ 是下面两个优化问题的解：

$$
\boxed{\min\frac{1}{2}\|\omega\|^2 \ \ {y^{(i)}(\omega^Tx^{(i)}-b)\geqslant1}}
$$

![][6]
>备注：线定义 $\boxed{\omega^Tx-b=0}$。

- 铰链损耗

支持向量机的铰链损耗定义如下：

$$
\boxed{L(z,y)=[1-yz]_+=\max(0,1-yz)}
$$

- 核

特征图 $\phi$，核 $K$ 定义如下：

$$
\boxed{K(x,z)=\phi(x^T)\phi(z)}
$$

实践中，核 $K$ 定义 $K(x,z)=\exp(-\frac{\|x-z\|^2}{2\sigma^2})$,叫做高斯核，应用广泛：
![此处输入图片的描述][7]
>备注：我们使用“核技巧”去计算损失函数，因为我们不需要知道明确的图 $\phi$，通常非常复杂。相反的，只需要$K(x,z)$的值。

- 拉格朗日

我们定义拉格朗日 

$$
\boxed{\mathcal{L}(\omega,b)=f(\omega)+\sum\limits_{i=1}^l\beta_ih^i(\omega)}
$$

>备注：系数 $\beta_i$ 成为拉格朗日乘数。

## 1.5 生成学习
生成模型首先尝试去学数据通过估计 $P(x|y)$，可以通过贝叶斯规则估计 $P(y|x)$。
### 1.5.1 高斯判别分析
- 设置

高斯判别分析假设 $y$ 并且 $x|y=0$ 、$x|y=1$：

$$
\boxed{y\sim{Bernoulli(\phi)}}\\
\boxed{x|y=0\sim\mathcal{N}(\mu_0,\sum)}\\
\boxed{x|y=1\sim\mathcal(m_1,\sum)}
$$

- 估计

最大似然估计统计如下：

|$\hat{\phi}$|$\hat{\mu_j}(j=0,1)$|$\widehat{\sum}$|
|:--:|:--:|:--:|
|$\dfrac{1}{m}\sum\limits_{i=1}^m1_{\{y^{(i)}=1\}}$|$\dfrac{\sum_{i=1}^m1_{{\{y^{(i)}=j\}}}{x^{(i)}}}{\sum_{(i=1)}^m1_{\{y^{(i)}=j\}}}$|$\dfrac{1}{m}\sum\limits_{i=1}^m(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T$|

### 1.5.2 朴素贝叶斯
- 假设

朴素贝叶斯假设每个数据点的特征都相互独立：

$$
\boxed{P(x|y)=P(x_1,x_2,...|y)=P(x_1|y)P(x_2|y)...=\prod\limits_{i=1}^nP(x_i|y)}
$$

- 求解

最大似然估计给出了如下方案，其中 $k\in\{0,1\},l\in[1,L]$：

$$
\boxed{P(y=k)=\frac{1}{m}\times\#\{j|y^{(j)}=k\}}
$$
$$
\boxed{P(x_i=l|y=k)=\dfrac{\#\{j|y^{(j)=k}\ and\ x_i^{(j)}=l\}}{\#\{j|y^{(j)}=k\}}}
$$

>备注：朴素贝叶斯广泛用于文字分类。

## 1.6 其他非参数化方法

- k近邻法

数据点的特性由它周围 $k$ 个邻居决定。
>备注：参数 $k$ 越高，偏差越大；参数 $k$ 越低，变量越高。

## 1.7 学习理论

- 联合界

假设 $A_1,...,A_k$ 是 $k$ 个事件：

$$
\boxed{P(A_1\cup{...}\cup{A_k})\leqslant{P}(A_1)+...+P(A_k)}
$$

![][8]

- Hoe－Pid不等式



  [1]: https://s1.ax1x.com/2018/09/04/iSNF5F.png
  [2]: https://s1.ax1x.com/2018/08/24/P7qMkR.png
  [3]: https://s1.ax1x.com/2018/08/24/P7qlfx.png
  [4]: https://s1.ax1x.com/2018/08/24/P7q81K.png
  [5]: https://s1.ax1x.com/2018/08/24/P7qG6O.png
  [6]: https://s1.ax1x.com/2018/08/24/P7qOE9.png
  [7]: https://s1.ax1x.com/2018/08/25/PHt3wt.png
  [8]: https://s1.ax1x.com/2018/08/25/PHNGu9.png
