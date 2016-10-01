## MATLAB 文件指引（MATLAB Modules）

### MATLAB 文件指引（MATLAB Modules）

#### 稀疏自编码器 | [sparseae_exercise.zip](http://ufldl.stanford.edu/wiki/resources/sparseae_exercise.zip)

checkNumericalGradient.m - 检查 computeNumericalGradient 的计算结果是否正确

computeNumericalGradient.m - 计算函数的数值梯度(待实现)

display_network.m - 可视化自动编码器的图像或滤波器的结果

initializeParameters.m - 随机初始化稀疏自动编码器的权重值

sampleIMAGES.m - 从图像矩阵中采样大小为 $8
\times 8$ 的小图(待实现)

sparseAutoencoderCost.m - 计算稀疏自动编码器中代价函数的函数值（即代价）和梯度

train.m - 用来训练和测试稀疏自动编码器的框架

### MNIST 数据集使用向导 | [mnistHelper.zip](http://ufldl.stanford.edu/wiki/resources/mnistHelper.zip)

loadMNISTImages.m - 返回包含原始 MNIST 图像的矩阵

loadMNISTLabels.m - 返回包含原始 MNIST 图像标签的矩阵

### 主成分分析与白化 | [pca_exercise.zip](http://ufldl.stanford.edu/wiki/resources/pca_exercise.zip)

display_network.m - 可视化自动编码器的图像或滤波器的结果

pca_gen.m - 白化练习框架

sampleIMAGESRAW.m - 返回一个 $8 \times 8$ 的的原始（未经白化过的）小图像

### SoftMax 回归 | [softmax_exercise.zip](http://ufldl.stanford.edu/wiki/resources/softmax_exercise.zip)

checkNumericalGradient.m - 检查 computeNumericalGradient 的计算结果是否正确

display_network.m - 可视化自动编码器的图像或滤波器的结果

loadMNISTImages.m - 返回包含原始 MNIST 图像的矩阵

loadMNISTLabels.m - 返回包含原始 MNIST 图像标签的矩阵

softmaxCost.m - 计算 Softmax 目标函数的代价和梯度

softmaxTrain.m - 给定参数下训练一个 Softmax 模型

train.m - 本练习的训练框架
