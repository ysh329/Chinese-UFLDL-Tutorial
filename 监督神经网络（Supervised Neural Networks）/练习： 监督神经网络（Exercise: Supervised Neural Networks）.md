# 练习： 监督神经网络（Exercise: Supervised Neural Networks）  

In this exercise, you will train a neural network classifier to classify the 10 digits in the MNIST dataset. The output unit of your neural network is identical to the softmax regression function you created in the Softmax Regression exercise. The softmax regression function alone did not fit the training set well, an example of underfitting. In comparison, a neural network has lower bias and should better fit the training set. In the section on Multi-Layer Neural Networks we covered the backpropagation algorithm to compute gradients for all parameters in the network using the squared error loss function. For this exercise, we need the same cost function as used for softmax regression (cross entropy), instead of the squared error function.  

本次练习中，您将训练一个神经网络分类器，并在 MNIST 数据集上对 10 种手写数字进行分类。神经网络的输出单元与您在 Softmax 回归练习中创建的是相同的。Softmax 回归的函数本身并不适合拟合训练集，其中一个原因是欠拟合。
相比之下，神经网络具有较低的偏差，应更好地拟合训练集。在多层神经网络的部分中，我们覆盖的反向传播算法计算梯度的所有参数在网络中使用的平方误差损失函数。对于这个练习，我们需要同样的成本函数用于Softmax回归（交叉熵），而不是平方误差函数。  


The cost function is nearly identical to the softmax regression cost function. Note that instead of making predictions from the input data x the softmax function takes as input the final hidden layer of the network hW,b(x). The loss function is thus,

代价函数与 Softmax 回归的代价函数基本一样。需要注意的是，