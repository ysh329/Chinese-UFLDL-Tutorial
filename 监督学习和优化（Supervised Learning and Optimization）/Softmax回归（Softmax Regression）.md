# Softmax回归（Softmax Regression）  
## 
## 介绍（Introduction）  

Softmax回归（或称为多元逻辑斯特回归）是通常的逻辑斯特回归用来处理多类分类问题的更一般化形式。在逻辑斯特回归中，我们假定标签都是二元的：即 $y^{(i)} \in \{0,1\}$ 。我们曾用这样的一个分类器来做两类的手写数字分类。然而，Softmax 回归可让我们处理K类（的分类问题），其中类标签 $y^{(i)} \in \{1,K\}$ 。  

不妨再回归一下逻辑斯特回归，我们有m个已标记样本的训练集 $\{ (x^{(1)}, y^{(1)}), \ldots, (x^{(m)}, y^{(m)}) \}$ ，其中（每个样本的）输入特征是 $x^{(i)} \in \Re^{n}$ 。在先前的逻辑斯特回归中，我们的分类设定的是两类，所以类标签是 $y^{(i)} \in \{0,1\}$ ，我们假设采取这样的形式：  

$$
\begin{align}
h_\theta(x) = \frac{1}{1+\exp(-\theta^\top x)},
\end{align}
$$  


其中，模型参数 $\theta$ 被用来训练以最小化代价函数  

$$
\begin{align}
J(\theta) = -\left[ \sum_{i=1}^m y^{(i)} \log h_\theta(x^{(i)}) + (1-y^{(i)}) \log (1-h_\theta(x^{(i)})) \right]
\end{align}
$$  

在Softmax回归的设定中，（与前文中两类分类相反）我们的兴趣在多类分类，正因如此标签 $y$ 可以取 $K$ 个不同的值，而不仅限于（两类分类中的）两个值。因此，训练集样本 $\{ (x^{(1)}, y^{(1)}), \ldots, (x^{(m)}, y^{(m)}) \}$ 的类别标签值有 $y^{(i)} \in \{1, 2, \ldots, K\}$ 。（注意：我们通常起始类从 $1$ 开始，而不是 $0$ ）。举个例子，在 $MNIST$ 数字识别任务（译者注： MNIST 是一个手写数字识别库，由NYU 的Yann LeCun 等人维护。http://yann.lecun.com/exdb/mnist/ ）中，我们有 $K = 10$ ，即不同的类别个数是10个。  


给出测试输入 $x$ ，我们希望我们的假设可以针对同一样本在不同的 $k$ （其中，$k = 1, ..., K$）值下估计概率 $P(y=k | x)$ 的值。也就是说，我们想要估计类标签取 $K$ 个不同的值时的概率。由此，我们的假设将会输出 $K$ 维向量（该向量元素值和为 $1$ ），它给出的是 $K$ 个估计的概率值。更具体地说，我们的假设 $h_{\theta}(x)$ 会采取这样的形式：  

$$
\begin{align}
h_\theta(x) =
\begin{bmatrix}
P(y = 1 | x; \theta) \\
P(y = 2 | x; \theta) \\
\vdots \\
P(y = K | x; \theta)
\end{bmatrix}
=
\frac{1}{ \sum_{j=1}^{K}{\exp(\theta^{(j)\top} x) }}
\begin{bmatrix}
\exp(\theta^{(1)\top} x ) \\
\exp(\theta^{(2)\top} x ) \\
\vdots \\
\exp(\theta^{(K)\top} x ) \\
\end{bmatrix}
\end{align}
$$  

这里，$\theta^{(1)}, \theta^{(2)}, \ldots, \theta^{(K)} \in \Re^{n}$ 是我们模型的参数。需要注意的是，$\frac{1}{ \sum_{j=1}^{K}{\exp(\theta^{(j)\top} x) } }$ 这一项对分布进行了标准化（$normalize$），所以其（最终）会加和为一项。  


为方便起见，我们也写 $\theta$ 来表示我们模型的所有参数。当你实现 Softmax 回归时，$n$ 行 $K$ 列的矩阵 $\theta$ 其实也是一列列 $\theta^{(k)}$ 所组成的，即  

$$
\theta = \left[\begin{array}{cccc}| & | & | & | \\
\theta^{(1)} & \theta^{(2)} & \cdots & \theta^{(K)} \\
| & | & | & |
\end{array}\right].
$$  


## 代价函数（Cost Function）  

我们现在来描述 Softmax 回归的代价函数。在下面的方程中， $1\{\cdot\}$ 被称为“指示器函数”（$indicator function$ ，译者注：老版教程中译为“示性函数”），即 $1\{值为真的表达式\} = 1$ ，$1\{值为假的表达式\} = 0$ 。例如， $1\{2+2=4\}$ 求出的数值为 $1$ ；而 $1\{1+1=5\}$ 求出的数值为 $0$ 。我们的代价函数将会是：  

$$
\begin{align}
J(\theta) = - \left[ \sum_{i=1}^{m} \sum_{k=1}^{K}  1\left\{y^{(i)} = k\right\} \log \frac{\exp(\theta^{(k)\top} x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)})}\right]
\end{align}
$$  

值得注意的是，我们也可以将逻辑斯特回归的代价函数等价地写成这样类似的形式：  

$$
\begin{align}
J(\theta) &= - \left[ \sum_{i=1}^m   (1-y^{(i)}) \log (1-h_\theta(x^{(i)})) + y^{(i)} \log h_\theta(x^{(i)}) \right] \\
&= - \left[ \sum_{i=1}^{m} \sum_{k=0}^{1} 1\left\{y^{(i)} = k\right\} \log P(y^{(i)} = k | x^{(i)} ; \theta) \right]
\end{align}
$$  

除了我们需要将 $K$ 个不同的类标签可能值相加外， Softmax 的代价函数（与逻辑斯特回归的代价函数）是相似的。需要注意的是，在 Softmax 回归中我们有：  

$$
P(y^{(i)} = k | x^{(i)} ; \theta) = \frac{\exp(\theta^{(k)\top} x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)}) }
$$  

We cannot solve for the minimum of J(θ) analytically, and thus as usual we’ll resort to an iterative optimization algorithm. Taking derivatives, one can show that the gradient is:  

我们不能分析式地得到 $J(\theta)$ 的最小值。因此，如往常一样，我们使用迭代优化算法来求解。对其求导，其梯度为：  

$$
\begin{align}
\nabla_{\theta^{(k)}} J(\theta) = - \sum_{i=1}^{m}{ \left[ x^{(i)} \left( 1\{ y^{(i)} = k\}  - P(y^{(i)} = k | x^{(i)}; \theta) \right) \right]  }
\end{align}
$$  

回想 $\nabla_{\theta^{(k)}}$ 符号的含义。尤其需要注意的是， $\nabla_{\theta^{(k)}} J(\theta)$ 其本身就是一个向量，所以，其第 $j$ 个元素即 $\frac{\partial J(\theta)}{\partial \theta_{lk}}$ ，它是关于 $\theta^{(k)}$ 的第 $j$ 个元素的偏导数。  

有了这个导数公式，然后可以将其插入到一个优化包中并最小化 $J(\theta)$ 。  


##Softmax回归的参数属性（Properties of Softmax regression parameterization）  

Softmax 回归有一个不同寻常的特性，那就是参数冗余（ $redundant$ ）。为了解释这个特性，我们假设有参数向量 $\theta^{(j)}$ ，我们对该向量减去某个固定的向量 $\psi$ ，此时（向量中地每个元素） $\theta^{(j)}$ 就被$\theta^{(j)} - \psi$ （其中 $j = 1, ..., k$ ）替代了。现在，我们的假设在计算类标签的概率（表示）为  

$$
\begin{align}
P(y^{(i)} = k | x^{(i)} ; \theta)
&= \frac{\exp((\theta^{(k)}-\psi)^\top x^{(i)})}{\sum_{j=1}^K \exp( (\theta^{(j)}-\psi)^\top x^{(i)})}  \\
&= \frac{\exp(\theta^{(k)\top} x^{(i)}) \exp(-\psi^\top x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)}) \exp(-\psi^\top x^{(i)})} \\
&= \frac{\exp(\theta^{(k)\top} x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)})}.
\end{align}
$$  

换句话说，从每个（元素） $\theta^{(j)}$ 中减去 $\psi$ 一点也不会影响到我们的假设预测！这表明了 Softmax 回归的参数中是有多余的。更正式地说，我们的 Softmax 模型是过参数化的（ $overparameterized$ ），意味着对任何一个拟合数据的假设而言，有多种参数取值很有可能得到同样的假设 $h_\theta$，即从输入 $x$ 得到同样的预测结果。  

进一步说，如果成本函数（ $cost function$ ）$J(\theta)$ 被某组参数 $(\theta^{(1)}, \theta^{(2)},\ldots, \theta^{(k)})$ 最小化，那么对任意的 $\psi$ ，成本函数也可以被 $(\theta^{(1)} - \psi, \theta^{(2)} - \psi,\ldots, \theta^{(k)} - \psi)$ 最小化。因此， $J(\theta)$ 的最小值时的参数并不唯一。（有趣的是， $J(\theta)$ 仍是凸的，并且在梯度下降中不会遇到局部最优问题，但是 $Hessian$ 矩阵是奇异或者不可逆的，这将会导致在牛顿法的直接实现上遇到数值问题。）  

Notice also that by setting ψ=θ(K), one can always replace θ(K) with θ(K)−ψ=0⃗  (the vector of all 0’s), without affecting the hypothesis. Thus, one could “eliminate” the vector of parameters θ(K) (or any other θ(k), for any single value of k), without harming the representational power of our hypothesis. Indeed, rather than optimizing over the K⋅n parameters (θ(1),θ(2),…,θ(K)) (where θ(k)∈Rn), one can instead set θ(K)=0⃗  and optimize only with respect to the K⋅n remaining parameters.  



# 与逻辑斯特回归的关系（Relationship to Logistic Regression）  

在 $K=2$ 特例中，一个可以证明的是 Softmax 回归简化为了逻辑斯特回归。这表明了 Softmax 回归是逻辑斯特回归的一般化形式。更具体地说，当 $K=2$，Softmax 回归的假设输出为  

$$
\begin{align}
h_\theta(x) &=

\frac{1}{ \exp(\theta^{(1)\top}x)  + \exp( \theta^{(2)\top} x^{(i)} ) }
\begin{bmatrix}
\exp( \theta^{(1)\top} x ) \\
\exp( \theta^{(2)\top} x )
\end{bmatrix}
\end{align}
$$  

Taking advantage of the fact that this hypothesis is overparameterized and setting ψ=θ(2), we can subtract θ(2) from each of the two parameters, giving us  

我们可以利用假设是过参数化的这一事实，设定 $\psi = \theta^{(2)}$ ，我们可以从这两个参数中的每个减去 $\theta^{(2)}$ 。

$$
\begin{align}
h(x) &=

\frac{1}{ \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) + \exp(\vec{0}^\top x) }
\begin{bmatrix}
\exp( (\theta^{(1)}-\theta^{(2)})^\top x )
\exp( \vec{0}^\top x ) \\
\end{bmatrix} \\

&=
\begin{bmatrix}
\frac{1}{ 1 + \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) } \\
\frac{\exp( (\theta^{(1)}-\theta^{(2)})^\top x )}{ 1 + \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) }
\end{bmatrix} \\

&=
\begin{bmatrix}
\frac{1}{ 1  + \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) } \\
1 - \frac{1}{ 1  + \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) } \\
\end{bmatrix}
\end{align}
$$  

Thus, replacing θ(2)−θ(1) with a single parameter vector θ′, we find that softmax regression predicts the probability of one of the classes as 11+exp(−(θ′)⊤x(i)), and that of the other class as 1−11+exp(−(θ′)⊤x(i)), same as logistic regression.  

## 练习 1C（Exercise 1C）  
Starter code for this exercise is included in the Starter code GitHub Repo in the ex1/ directory.  

In this exercise you will train a classifier to handle all 10 digits in the MNIST dataset. The code is very similar to that used for Exercise 1B except that it will load the entire MNIST train and test sets (instead of just the 0 and 1 digits), and the labels y(i) have 1 added to them so that y(i)∈{1,…,10}. (The change in the labels allows you to use y(i) as an index into a matrix.)  

The code performs the same operations as in Exercise 1B: it loads the train and test data, adding an intercept term, then calls minFunc with the softmax_regression_vec.m file as the objective function. When training is complete, it will print out training and testing accuracies for the 10-class digit recognition problem.  

Your task is to implement the softmax_regression_vec.m file to compute the softmax objective function J(θ;X,y) and store it in the variable f. You must also compute the gradient ∇θJ(θ;X,y) and store it in the variable g. Don’t forget that minFunc supplies the parameters θ as a vector. The starter code will reshape θ into a n-by-(K-1) matrix (for K=10 classes). You also need to remember to reshape the returned gradient g back into a vector using g=g(:);  

You can start out with a for-loop version of the code if necessary to get the gradient right. (Be sure to use the gradient check debugging strategy covered earlier!) However, you might find that this implementation is too slow to run the optimizer all the way through. After you get the gradient right with a slow version of the code, try to vectorize your code as well as possible before running the full experiment.  

Here are a few MATLAB tips that you might find useful for implementing or speeding up your code (though these may or may not be useful depending on your implementation strategy):  

1.Suppose we have a matrix A and we want to extract a single element from each row, where the column of the element to be extracted from row i is stored in y(i), where y is a row vector. We can use the sub2ind() function like this:  
I=sub2ind(size(A), 1:size(A,1), y);  
values = A(I);  

This code will take each pair of indices (i,j) where i comes from the second argument and j comes from the corresponding element of the third argument, and compute the corresponding 1D index into A for the (i,j)‘th element. So, I(1) will be the index for the element at location (1,y(1)), and I(2) will be the index for the element at (2,y(2)).  


2.When you compute the predicted label probabilities y^(i)k=exp(θ⊤:,kx(i))/(∑Kj=1exp(θ⊤:,jx(i))), try to use matrix multiplications and bsxfun to speed up the computation. For example, once θ is in matrix form, you can compute the products for every example and the first 9 classes using a=θ⊤X. (Recall that the 10th class is left out of θ, so that a(10,:) is just assumed to be 0.)  