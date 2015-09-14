<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
# 线性回归（Linear Regression）  
## 问题描述  
我们（不妨）回顾一下（这些知识点），我们将从如何实现线性回归（linear regression）开始。在这一节的主要思想是知道什么是目标函数（objective functions），计算其梯度（gradients）以及通过一组参数来优化目标（函数）。这些基本的工具将会构建（在稍后教程中我们要讲到的）复杂的算法。想要更多学习资料的读者可以在参考<a href="http://cs229.stanford.edu/notes/cs229-notes1.pdf">监督学习讲座笔记</a>。  

在线性回归中，我们的目标是从一个输入值向量x∈Rn，去预测目标值y。例如，我们要预测房价，其中y表示房子的（美元）价格，xj（j是上角标，表示该房屋样本的第j个特征）代表向量x的第j个元素，表示房子的第j个特征，我们用特征来描述一个房子（如房子的面积，卧室的数目等）。假设我们现有很多房屋的数据（特征），其中比方说要表示第i个房子的特征，表示为x(i)（i是下角标，表示该房屋样本是数据集里的第i个样本），其房价表示为y(i)。
$$x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$$
\\(x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}\\)
