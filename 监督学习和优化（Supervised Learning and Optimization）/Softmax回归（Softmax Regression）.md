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

在Softmax回归的设定中，（与前文中两类分类相反）我们的兴趣在多类分类，正因如此标签 $y$ 可以取 $K$ 个不同的值，而不仅限于（两类分类中的）两个值。因此，训练集样本 $\{ (x^{(1)}, y^{(1)}), \ldots, (x^{(m)}, y^{(m)}) \}$ 的类别标签值有 $y^{(i)} \in \{1, 2, \ldots, K\}$ 。（注意：我们通常起始类从 $1$ 开始，而不是 $0$ ）。举个例子，在 MNIST 数字识别任务中，我们有 $K = 10$ ，即不同的类别个数是10个。  


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

我们现在来描述 softmax 回归的代价函数。在下面的方程中， $1\{\cdot\}$ 被称为“指示器函数”（$indicator function$），即 $1\{真实的陈述\} = 1$ ，$1\{虚假的陈述\} = 0$ 。例如， $1\{2+2=4\}$ 求出的数值为 $1$ ；而 $1\{1+1=5\}$ 求出的数值为 $0$ 。我们的代价函数将会是：  

$$
\begin{align}
J(\theta) = - \left[ \sum_{i=1}^{m} \sum_{k=1}^{K}  1\left\{y^{(i)} = k\right\} \log \frac{\exp(\theta^{(k)\top} x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)})}\right]
\end{align}
$$  

值得注意的是，我们也可以将逻辑斯特回归的代价函数等价地写成这样的形式：  

$$
\begin{align}
J(\theta) &= - \left[ \sum_{i=1}^m   (1-y^{(i)}) \log (1-h_\theta(x^{(i)})) + y^{(i)} \log h_\theta(x^{(i)}) \right] \\
&= - \left[ \sum_{i=1}^{m} \sum_{k=0}^{1} 1\left\{y^{(i)} = k\right\} \log P(y^{(i)} = k | x^{(i)} ; \theta) \right]
\end{align}
$$  

除了我们需要将 $K$ 个不同的类标签可能值相加外， softmax 的代价函数（与逻辑斯特回归的代价函数）是相似的。需要注意的是，在 Softmax 回归中我们有：  

$$
P(y^{(i)} = k | x^{(i)} ; \theta) = \frac{\exp(\theta^{(k)\top} x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)}) }
$$  

We cannot solve for the minimum of J(θ) analytically, and thus as usual we’ll resort to an iterative optimization algorithm. Taking derivatives, one can show that the gradient is:  

我们不能解决 $J(\theta)$ 的最小化问题

$$
\begin{align}
\nabla_{\theta^{(k)}} J(\theta) = - \sum_{i=1}^{m}{ \left[ x^{(i)} \left( 1\{ y^{(i)} = k\}  - P(y^{(i)} = k | x^{(i)}; \theta) \right) \right]  }
\end{align}
$$  

Recall the meaning of the ”∇θ(k)” notation. In particular, ∇θ(k)J(θ) is itself a vector, so that its j-th element is ∂J(θ)∂θlk the partial derivative of J(θ) with respect to the j-th element of θ(k).  

Armed with this formula for the derivative, one can then plug it into a standard optimization package and have it minimize J(θ).  




##softmax回归的参数属性（Properties of softmax regression parameterization）  
Softmax regression has an unusual property that it has a “redundant” set of parameters. To explain what this means, suppose we take each of our parameter vectors θ(j), and subtract some fixed vector ψ from it, so that every θ(j) is now replaced with θ(j)−ψ (for every j=1,…,k). Our hypothesis now estimates the class label probabilities as  

$$
\begin{align}
P(y^{(i)} = k | x^{(i)} ; \theta)
&= \frac{\exp((\theta^{(k)}-\psi)^\top x^{(i)})}{\sum_{j=1}^K \exp( (\theta^{(j)}-\psi)^\top x^{(i)})}  \\
&= \frac{\exp(\theta^{(k)\top} x^{(i)}) \exp(-\psi^\top x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)}) \exp(-\psi^\top x^{(i)})} \\
&= \frac{\exp(\theta^{(k)\top} x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)})}.
\end{align}
$$  

In other words, subtracting ψ from every θ(j) does not affect our hypothesis’ predictions at all! This shows that softmax regression’s parameters are “redundant.” More formally, we say that our softmax model is ”‘overparameterized,”’ meaning that for any hypothesis we might fit to the data, there are multiple parameter settings that give rise to exactly the same hypothesis function hθ mapping from inputs x to the predictions.  

Further, if the cost function J(θ) is minimized by some setting of the parameters (θ(1),θ(2),…,θ(k)), then it is also minimized by (θ(1)−ψ,θ(2)−ψ,…,θ(k)−ψ) for any value of ψ. Thus, the minimizer of J(θ) is not unique. (Interestingly, J(θ) is still convex, and thus gradient descent will not run into local optima problems. But the Hessian is singular/non-invertible, which causes a straightforward implementation of Newton’s method to run into numerical problems.)  

Notice also that by setting ψ=θ(K), one can always replace θ(K) with θ(K)−ψ=0⃗  (the vector of all 0’s), without affecting the hypothesis. Thus, one could “eliminate” the vector of parameters θ(K) (or any other θ(k), for any single value of k), without harming the representational power of our hypothesis. Indeed, rather than optimizing over the K⋅n parameters (θ(1),θ(2),…,θ(K)) (where θ(k)∈Rn), one can instead set θ(K)=0⃗  and optimize only with respect to the K⋅n remaining parameters.  



# 与逻辑斯特回归的关系（Relationship to Logistic Regression）  
In the special case where K=2, one can show that softmax regression reduces to logistic regression. This shows that softmax regression is a generalization of logistic regression. Concretely, when K=2, the softmax regression hypothesis outputs  

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