# Softmax Regression
## 
## 介绍（Introduction）
Softmax regression (or multinomial logistic regression) is a generalization of logistic regression to the case where we want to handle multiple classes. In logistic regression we assumed that the labels were binary: y(i)∈{0,1}. We used such a classifier to distinguish between two kinds of hand-written digits. Softmax regression allows us to handle y(i)∈{1,…,K} where K is the number of classes.

Recall that in logistic regression, we had a training set {(x(1),y(1)),…,(x(m),y(m))} of m labeled examples, where the input features are x(i)∈Rn. With logistic regression, we were in the binary classification setting, so the labels were y(i)∈{0,1}. Our hypothesis took the form:






## 代价函数（Cost Function）
We now describe the cost function that we’ll use for softmax regression. In the equation below, 1{⋅} is the ”‘indicator function,”’ so that 1{a true statement}=1, and 1{a false statement}=0. For example, 1{2+2=4} evaluates to 1; whereas 1{1+1=5} evaluates to 0. Our cost function will be:






##softmax回归的参数属性（Properties of softmax regression parameterization）
Softmax regression has an unusual property that it has a “redundant” set of parameters. To explain what this means, suppose we take each of our parameter vectors θ(j), and subtract some fixed vector ψ from it, so that every θ(j) is now replaced with θ(j)−ψ (for every j=1,…,k). Our hypothesis now estimates the class label probabilities as







# 与逻辑斯特回归的关系（Relationship to Logistic Regression）
In the special case where K=2, one can show that softmax regression reduces to logistic regression. This shows that softmax regression is a generalization of logistic regression. Concretely, when K=2, the softmax regression hypothesis outputs



## 练习 1C（Exercise 1C）
Starter code for this exercise is included in the Starter code GitHub Repo in the ex1/ directory.