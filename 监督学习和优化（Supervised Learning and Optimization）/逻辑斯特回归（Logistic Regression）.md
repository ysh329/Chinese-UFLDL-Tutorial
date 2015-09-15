# 逻辑斯特回归（Logistic Regression）  
在先前的学习中，我们学习了预测连续数值的方法（如预测房价），如把输入值（如房屋大小）传给线性的函数。有时候，我们反而希望预测离散变量（Discrete Variable），如预测网格中像素强度是代表一个“0”位还是一个“1”位。此时，这便是一个分类问题，逻辑斯特回归（Logistic Regression）对于学习做这样的（分类）决策来说是一种简单的方法。  

在线性回归中，我们尝试使用线性函数 $y = h_{\theta}(x) = \theta^{\top}x$ 来对第 $i$ 个样本 $x^{(i)}$ 预测其（可能的） $y^{(i)}$ 值。这显然不是一个解决二值类标签（ $y^{(i)}∈{\{0,1\}}$ )预测（问题）的好办法。在逻辑斯特回归（Logistic Regression）中，我们使用了一个（与先前学到的）不同的假设空间（Hypothesis Class）来尝试预测样本所属的类 $“1”$ 以及与其相对的类 $“0”$ 的概率。具体而言，我们将会尝试使用形式如下的函数进行学习：

$$ P(y = 1 \mid x) = h_{\theta}(x) = \frac{1}{1+\exp{(-\theta^{\top}x)}} \equiv \sigma(\theta^{\top}x) $$
$$ P(y = 0 \mid x) = 1 - P(y = 1 \mid x) = 1 - h_{\theta}(x) $$

 it is an S-shaped function that “squashes” the value of θ⊤x into the range [0,1] so that we may interpret hθ(x) as a probability. Our goal is to search for a value of θ so that the probability P(y=1|x)=hθ(x) is large when x belongs to the “1” class and small when x belongs to the “0” class (so that P(y=0|x) is large). For a set of training examples with binary labels {(x(i),y(i)):i=1,…,m} the following cost function measures how well a given hθ does this:

函数 $\sigma(z) = \frac{1}{1 + \exp{(-z)}}$ 通常被称为 “sigmoid” 或 “logistic” （音译：逻辑斯特）函数——它是形如S形的函数