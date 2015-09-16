# 逻辑斯特回归（Logistic Regression）  
在先前的学习中，我们学习了预测连续数值的方法（如预测房价），如把输入值（如房屋大小）传给线性的函数。有时候，我们反而希望预测离散变量（Discrete Variable），如预测网格中像素强度是代表一个“0”位还是一个“1”位。此时，这便是一个分类问题，逻辑斯特回归（Logistic Regression）对于学习做这样的（分类）决策来说是一种简单的方法。  

在线性回归中，我们尝试使用线性函数 $y = h_{\theta}(x) = \theta^{\top}x$ 来对第 $i$ 个样本 $x^{(i)}$ 预测其（可能的） $y^{(i)}$ 值。这显然不是一个解决二值类标签（ $y^{(i)}∈{\{0,1\}}$ )预测（问题）的好办法。在逻辑斯特回归（Logistic Regression）中，我们使用了一个（与先前学到的）不同的假设空间（Hypothesis Class）来尝试预测样本所属的类 $“1”$ 以及与其相对的类 $“0”$ 的概率。具体而言，我们将会尝试使用形式如下的函数进行学习：  

$$ P(y = 1 \mid x) = h_{\theta}(x) = \frac{1}{1+\exp{(-\theta^{\top}x)}} \equiv \sigma(\theta^{\top}x), $$
$$ P(y = 0 \mid x) = 1 - P(y = 1 \mid x) = 1 - h_{\theta}(x). $$

函数 $\sigma(z) = \frac{1}{1 + \exp{(-z)}}$ 通常被称为 “ $sigmoid$ ” 或 “ $logistic$ ” （音译：逻辑斯特）函数——它是 $S$ 形的函数，其函数值 $\theta^{\top}x$ 被映射到 $[0,1]$ 区间上，所以我们也可将其值看成是概率。我们的目标是找到一个 $\theta$ 值，使其能满足：当 $x$ 属于 “1” 类时，$P(y = 1 \mid x ) = h_{\theta}(x)$ 的值很大；当 $x$ 属于 “0” 类时，$P(y = 0 \mid x ) = h_{\theta}(x)$ 的值很大。对于一组两类标记 ${\{(x^{(i)}, y^{(i)}); i = 1, ..., m\}}$ 的训练样本，我们使用下面的成本函数（Cost Function）来评估这个假设 $h_{\theta}$ 的好坏：  
$$ J(\theta) = -{\sum_{i}}\left (y^{(i)}\log{(h_{\theta}(x^{(i)}))} + (1-y^{(i)}) \log{(1 - h_{\theta}(x^{(i)}))}\right ). $$

需要注意的是，在上式的加和形式中，对每个训练样本，两项中只有一项是非零的（这取决于标记 $y^{(i)}$ 是 0 还是 1 ）。当 $y^{(i)} = 1$ 时，最小化成本函数意味着我们需要使 $h_{\theta}(x)$ 变大，而当 $y^{(i)} = 0$ 时，（正如前文所讲）我们也想要 $1 - h_{\theta}$ 变大。对于一个逻辑斯特回归（Logistic Regression）的完整解释以及成本函数（Cost Function）的推导过程，可以参考 <a href="http://cs229.stanford.edu/notes/cs229-notes1.pdf">CS229课程之监督学习</a>部分。  

现在，我们有了可以测量可拟合训练数据的假说（或称为“假设函数”） $h_{\theta}$ 好坏的成本函数（Cost Function）。我们可以学习分类训练数据，通过最小化 $J(\theta)$ （的方法）来找参数 $\theta$ 的最优值。当我们完成了这一过程，我们便可对新的测试点通过计算所属“1”类和“0”类最可能的概率，进行分类。如果 $P(y=1 \mid x)>P(y=0 \mid x)$ ，那么该样本就将标记为“1”类，否则（$P(y=1 \mid x) < P(y=0 \mid x)$ ）标记为“0”类。其实，这好比检查 $h_{\theta} > 0.5 $ 是否成立。  

为了最小化 $J(\theta)$ ，我们可以使用类似线性回归（Linear Regression）的工具。我们需要提供一个可以在任意参数 $\theta$ 值时，可计算出 $J(\theta)$ 和（其微分结果的） $\triangledown_{\theta} J(\theta)$ 的函数。在给定参数 $\theta_{j}$ 时， $J(\theta)$ 的微分结果是：  

$$ \frac{\partial J(\theta)}{\partial \theta_{j}} = \sum_{i}x_{j}^{(i)} \left ( h_{\theta}(x^{(i)} - y^{(i)} )  \right ). $$

若写成向量形式，其整个梯度可表示为：

$$ \triangledown_{\theta} J(\theta) = \sum_{i}x^{(i)} \left ( h_{\theta}(x^{(i)} - y^{(i)} )  \right ). $$

除了当前的假设函数 $h_{\theta} = \sigma(\theta^{\top}x)$ ，这里的梯度计算与线性回归基本相同。  

## 练习 1B（Exercise 1B）

本次练习的初学者代码已经在<a href="https://github.com/amaas/stanford_dl_ex">初学者代码（Starter Code）的 GitHub Rep</a> 中的 ex1/ 目录中。  

In this exercise you will implement the objective function and gradient computations for logistic regression and use your code to learn to classify images of digits from the MNIST dataset as either “0” or “1”. Some examples of these digits are shown below:

在本次练习中，您将会实现逻辑斯特回归（Logistic Regression）的目标函数（Objective Function）以及梯度计算（Gradient Computation），并使用您的代码从 <a href="http://yann.lecun.com/exdb/mnist/">MNIST 数据集</a> 中，学习分类数字（“0”或“1”的）图片。如下是列举的一些数字图片：  

<img src="./images/Mnist_01.png">  

Each of the digits is is represented by a 28x28 grid of pixel intensities, which we will reformat as a vector x(i) with 28*28 = 784 elements. The label is binary, so y(i)∈{0,1}.



You will find starter code for this exercise in the ex1/ex1b_logreg.m file. The starter code file performs the following tasks for you: