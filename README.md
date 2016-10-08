为了极佳的阅读体验，您可点击 [这里](https://github.com/ysh329/Chinese-UFLDL-Tutorial/archive/master.zip) 将本文档下载到本地，并安装 [Haroopad](http://pad.haroopress.com/user.html#download) 进行阅读。

# 非监督特征学习与深度学习 中文教程

中文版的新版 UFLDL 教程（项目地址： www.github.com/ysh329/Chinese-UFLDL-Tutorial ），该版本翻译自 [UFLDL Tutorial](http://deeplearning.stanford.edu/tutorial/) ，是新版教程的翻译。也可参考 [旧版 UFLDL 中文教程](http://ufldl.stanford.edu/wiki/index.php/UFLDL教程) 。翻译过程中有一些数学公式，使用 [Haroopad](http://pad.haroopress.com/user.html#download) 编辑和排版， Haroopad 是一个优秀的离线 [MarkDown](https://en.wikipedia.org/wiki/Markdown) 编辑器，支持 [Tex](https://en.wikipedia.org/wiki/TeX) 公式编辑，支持多平台（Win/Mac/Linux），目前还在翻译中，翻译完成后会考虑使用 Tex 重新排版。  

自己对新版 UFLDL 教程翻译过程中，发现的英文错误，见 [新版教程英文原文勘误表](./新版教程英文原文勘误表.md) 。  

**注： UFLDL 是非监督特征学习及深度学习（Unsupervised Feature Learning and Deep Learning）的缩写，而不仅指深度学习（Deep Learning）。**  

-  翻译者：Shuai Yuan ，部分小节参考旧版翻译进行修正和补充。
-  若有翻译错误，请直接 [New issue](https://github.com/ysh329/Chinese-UFLDL-Tutorial/issues/new) 或 [发邮件](Mailto:ysh329@sina.com) ，感谢！  

>更多详细参考资料，见 [计算机科学](https://github.com/bayandin/awesome-awesomeness) ， [人工智能](https://github.com/owainlewis/awesome-artificial-intelligence) ， [机器学习](https://github.com/josephmisiti/awesome-machine-learning) ， [深度学习](https://github.com/ChristosChristofidis/awesome-deep-learning) ， [强化学习](https://github.com/aikorea/awesome-rl) ， [深度强化学习](https://github.com/junhyukoh/deep-reinforcement-learning-papers) ， [公开数据集](https://github.com/ChristosChristofidis/awesome-public-datasets) 。


# 欢迎来到新版 UFLDL 中文教程！

说明：本教程将会教给您非监督特征学习以及深度学习的主要思想。通过它，您将会实现几个特征学习或深度学习的算法，看到这些算法为您（的工作）带来作用，以及学习如何将这些思想应用到适用的新问题上。

本教程假定您已经有了基本的机器学习知识（具体而言，熟悉监督学习，逻辑斯特回归以及梯度下降法的思想）。如果您不熟悉这些，我们建议您先去 [机器学习课程](http://openclassroom.stanford.edu/MainFolder/CoursePage.php?course=MachineLearning) 中去学习，并完成其中的第II，III，IV章节（即到逻辑斯特回归）。

材料由以下人员提供：Andrew Ng, Jiquan Ngiam, Chuan Yu Foo, Yifan Mai, Caroline Suen, Adam Coates, Andrew Maas, Awni Hannun, Brody Huval, Tao Wang, Sameep Tandon

## 获取初学者代码（Starter Code）

### 初学者代码

您可以获得初学者所有练习的代码从 [该Github的代码仓库](https://github.com/amaas/stanford_dl_ex) 。  

有关的数据文件可以从 [这里](http://ai.stanford.edu/~amaas/data/data.zip) 下载。 下载到的数据需要解压到名为<font color=red>`“common”`</font>的文件夹中（以便初学者代码的使用）。


# 目录

**每个小节后面的<font color=red>\[old\]\[new]\[旧\]</font>分别代表：旧版英文、新版英文、旧版中文三个版本。若没有对应的版本则用<font color=red>\[无\]</font>代替。**

* **预备知识（Miscellaneous）**

  * [MATLAB 文件指引（MATLAB Modules）](./预备知识（Miscellaneous ）/MATLAB　文件指引（MATLAB Modules）.md)\[[old](http://ufldl.stanford.edu/wiki/index.php/MATLAB_Modules)\]\[无\]\[无\]

  * [代码风格（Style Guide）](./预备知识（Miscellaneous ）/代码风格（Style Guide）.md)\[[old](http://ufldl.stanford.edu/wiki/index.php/Style_Guide)\]\[无\]\[无\]

  * [预备知识推荐（Useful Links）](./预备知识（Miscellaneous ）/预备知识推荐（Useful Links）.md)\[[old](http://ufldl.stanford.edu/wiki/index.phssp/Useful_Links)\]\[无\]\[无\]

  * [推荐读物（UFLDL Recommended Readings）](./预备知识（Miscellaneous ）/推荐读物（UFLDL Recommended Readings）.md)\[[old](http://deeplearning.stanford.edu/wiki/index.php/UFLDL_Recommended_Readings)\]\[无\]\[无\]

* **监督学习与优化（Supervised Learning and Optimization）**

  *  [线性回归（Linear Regression）](./监督学习和优化（Supervised Learning and Optimization）/线性回归（Linear Regression）.md)\[无\]\[[new](http://ufldl.stanford.edu/tutorial/supervised/LinearRegression)\]\[无\]

  *  [逻辑斯特回归（Logistic Regression）](./监督学习和优化（Supervised Learning and Optimization）/逻辑斯特回归（Logistic Regression）.md)\[[old](http://deeplearning.stanford.edu/wiki/index.php/Logistic_Regression_Vectorization_Example)\]\[[new](http://ufldl.stanford.edu/tutorial/supervised/LogisticRegression)\]\[[旧](http://ufldl.stanford.edu/wiki/index.php/%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E7%9A%84%E5%90%91%E9%87%8F%E5%8C%96%E5%AE%9E%E7%8E%B0%E6%A0%B7%E4%BE%8B)\]

  *  [向量化（Vectorization）](./监督学习和优化（Supervised Learning and Optimization）/向量化（Vectorization）.md)\[[old](http://deeplearning.stanford.edu/wiki/index.php/Vectorization)\]\[[new](http://ufldl.stanford.edu/tutorial/supervised/Vectorization)\]\[[旧](http://ufldl.stanford.edu/wiki/index.php/%E7%9F%A2%E9%87%8F%E5%8C%96%E7%BC%96%E7%A8%8B)\]

  *  [调试：梯度检查（Debugging: Gradient Checking）](./监督学习和优化（Supervised Learning and Optimization）/调试：梯度检查（Debugging：Gradient Checking）.md)\[[old](http://deeplearning.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization)\]\[[new](http://ufldl.stanford.edu/tutorial/supervised/DebuggingGradientChecking)\]\[[旧](http://ufldl.stanford.edu/wiki/index.php/%E6%A2%AF%E5%BA%A6%E6%A3%80%E9%AA%8C%E4%B8%8E%E9%AB%98%E7%BA%A7%E4%BC%98%E5%8C%96)\]

  *  [Softmax 回归（Softmax Regression）](./监督学习和优化（Supervised Learning and Optimization）/Softmax回归（Softmax Regression）.md)\[[old](http://deeplearning.stanford.edu/wiki/index.php/Softmax_Regression)\]\[[new](http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression)\]\[[旧](http://ufldl.stanford.edu/wiki/index.php/Softmax%E5%9B%9E%E5%BD%92)\]

  *  [调试：偏差和方差（Debugging: Bias and Variance）](./监督学习和优化（Supervised Learning and Optimization）/检查：偏差和方差（Debugging：Bias and Variance）.md)\[无\]\[[new](http://ufldl.stanford.edu/tutorial/supervised/DebuggingBiasAndVariance)\]\[无\]

  *  [调试：优化器和目标（Debugging: Optimizers and Objectives）](./监督学习和优化（Supervised Learning and Optimization）/调试：优化器和目标（Debugging：Optimizers and Objectives）.md)\[无\]\[[new](http://ufldl.stanford.edu/tutorial/supervised/DebuggingOptimizersAndObjectives)\]\[无\]

* **监督神经网络（Supervised Neural Networks）**

  *  [多层神经网络（Multi-Layer Neural Networks）](./监督神经网络（Supervised Neural Networks）/多层神经网络（Multi-Layer Neural Networks）.md)\[[old](http://deeplearning.stanford.edu/wiki/index.php/Neural_Networks)\]\[[new](http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks)\]\[[旧](http://ufldl.stanford.edu/wiki/index.php/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)\]

   *  [神经网络向量化（Neural Network Vectorization）](./监督神经网络（Supervised Neural Networks）/神经网络向量化（Neural Network Vectorization）.md)\[[old](http://ufldl.stanford.edu/wiki/index.php/Neural_Network_Vectorization)\]\[无\]\[[旧](http://ufldl.stanford.edu/wiki/index.php/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%90%91%E9%87%8F%E5%8C%96#.E5.8F.8D.E5.90.91.E4.BC.A0.E6.92.AD)\]

   *  [练习：监督神经网络（Exercise: Supervised Neural Network）](./监督神经网络（Supervised%20Neural%20Networks）/练习：%20监督神经网络（Exercise:%20Supervised%20Neural%20Networks）.md)\[无\]\[[new](http://ufldl.stanford.edu/tutorial/supervised/ExerciseSupervisedNeuralNetwork)\]\[无\]

* **监督卷积网络（Supervised Convolutional Neural Network）**

  *  [使用卷积进行特征提取（Feature Extraction Using Convolution）](./监督卷积网络（Supervised Convolutional Neural Network）/使用卷积进行特征提取（Feature Extraction Using Convolution）.md)\[[old](http://deeplearning.stanford.edu/wiki/index.php/Feature_extraction_using_convolution)\]\[[new](http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution)\]\[[旧](http://ufldl.stanford.edu/wiki/index.php/%E5%8D%B7%E7%A7%AF%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96)\]

  *  [池化（Pooling）](./监督卷积网络（Supervised Convolutional Neural Network）/池化（Pooling）.md)\[[old](http://deeplearning.stanford.edu/wiki/index.php/Pooling)\]\[[new](http://ufldl.stanford.edu/tutorial/supervised/Pooling)\]\[[旧](http://ufldl.stanford.edu/wiki/index.php/%E6%B1%A0%E5%8C%96)\]

   * [练习：卷积和池化（Exercise: Convolution and Pooling）](./监督卷积网络（Supervised Convolutional Neural Network）/练习：卷积和池化（Exercise: Convolution and Pooling）.md)\[无\]\[[new](http://ufldl.stanford.edu/tutorial/supervised/ExerciseConvolutionAndPooling)\]\[无\]

  *  [优化方法：随机梯度下降（Optimization: Stochastic Gradient Descent）](./监督卷积网络（Supervised Convolutional Neural Network）/优化方法：随机梯度下降（Optimization: Stochastic Gradient Descent）.md)\[无\]\[[new](http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent)\]\[无\]

  *  [卷积神经网络（Convolutional Neural Network）](./监督卷积网络（Supervised Convolutional Neural Network）/卷积神经网络（Convolutional Neural Network）.md)\[无\]\[[new](http://ufldl.stanford.edu/tutorial/supervised/ConvolutionalNeuralNetwork)\]\[无\]

   * [练习：卷积神经网络（Excercise: Convolutional Neural Network）](./监督卷积网络（Supervised Convolutional Neural Network）/练习：卷积神经网络（Excercise: Convolutional Neural Network）.md)\[无\]\[[new](http://ufldl.stanford.edu/tutorial/supervised/ExerciseConvolutionalNeuralNetwork)\]\[无\]

* **无监督学习（Unsupervised Learning）**

  * [自动编码器（Autoencoders）](./无监督学习（Unsupervised Learning）/自动编码器（Autoencoders）.md)\[[old](http://deeplearning.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity)\]\[[new](http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders)\]\[[旧](http://ufldl.stanford.edu/wiki/index.php/%E8%87%AA%E7%BC%96%E7%A0%81%E7%AE%97%E6%B3%95%E4%B8%8E%E7%A8%80%E7%96%8F%E6%80%A7)\]

   * [线性解码器（Linear Decoders）](./无监督学习（Unsupervised Learning）/线性解码器（Linear Decoders）.md)[[old](http://ufldl.stanford.edu/wiki/index.php/Linear_Decoders)][无][[旧](http://ufldl.stanford.edu/wiki/index.php/%E7%BA%BF%E6%80%A7%E8%A7%A3%E7%A0%81%E5%99%A8)]

   * [练习：使用稀疏编码器学习颜色特征（Exercise:Learning color features with Sparse Autoencoders）](./无监督学习（Unsupervised Learning）/练习：使用稀疏编码器学习颜色特征（Exercise:Learning color features with Sparse Autoencoders）.md)[[old](http://ufldl.stanford.edu/wiki/index.php/Exercise:Learning_color_features_with_Sparse_Autoencoders)][无][无]

  * [主成分分析白化（PCA Whitening）](./无监督学习（Unsupervised Learning）/主成分分析白化（PCA Whitening）.md)\[[old](http://deeplearning.stanford.edu/wiki/index.php/Implementing_PCA/Whitening)\]\[[new](http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening)\]\[[旧](http://ufldl.stanford.edu/wiki/index.php/%E5%AE%9E%E7%8E%B0%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90%E5%92%8C%E7%99%BD%E5%8C%96)\]

   * [练习：实现 2D 数据的主成分分析（Exercise:PCA in 2D）](./无监督学习（Unsupervised Learning）/练习：实现 2D 数据的主成分分析（Exercise:PCA in 2D）.md)[[old](http://ufldl.stanford.edu/wiki/index.php/Exercise:PCA_in_2D)][无][无]

   * [练习：主成分分析白化（Exercise: PCA Whitening）](./无监督学习（Unsupervised Learning）/练习：主成分分析白化（Exercise: PCA Whitening）.md)\[[old](http://ufldl.stanford.edu/wiki/index.php/Exercise:PCA_and_Whitening)\]\[[new](http://ufldl.stanford.edu/tutorial/unsupervised/ExercisePCAWhitening)\]\[无\]

  * [稀疏编码（Sparse Coding）](./无监督学习（Unsupervised Learning）/稀疏编码（Sparse Coding）.md)\[[old](http://deeplearning.stanford.edu/wiki/index.php/Sparse_Coding)\]\[[new](http://ufldl.stanford.edu/tutorial/unsupervised/SparseCoding/)\]\[[旧](http://ufldl.stanford.edu/wiki/index.php/%E7%A8%80%E7%96%8F%E7%BC%96%E7%A0%81)\]

   * [稀疏自编码符号一览表（Sparse Autoencoder Notation Summary）](./无监督学习（Unsupervised Learning）/稀疏自编码符号一览表（Sparse Autoencoder Notation Summary）.md)\[[old](http://ufldl.stanford.edu/wiki/index.php/Sparse_Autoencoder_Notation_Summary)\]\[无\]\[[旧](http://ufldl.stanford.edu/wiki/index.php/%E7%A8%80%E7%96%8F%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%E7%AC%A6%E5%8F%B7%E4%B8%80%E8%A7%88%E8%A1%A8)\]

   * [稀疏编码自编码表达（Sparse Coding: Autoencoder Interpretation）](./无监督学习（Unsupervised Learning）/稀疏编码自编码表达（Sparse Coding: Autoencoder Interpretation）.md)[[old](http://ufldl.stanford.edu/wiki/index.php/Sparse_Coding:_Autoencoder_Interpretation)][无][[旧](http://ufldl.stanford.edu/wiki/index.php/%E7%A8%80%E7%96%8F%E7%BC%96%E7%A0%81%E8%87%AA%E7%BC%96%E7%A0%81%E8%A1%A8%E8%BE%BE)]

   * [练习：稀疏编码（Exercise:Sparse Coding）](./无监督学习（Unsupervised Learning）/练习：稀疏编码（Exercise:Sparse Coding）.md)[[old](http://ufldl.stanford.edu/wiki/index.php/Exercise:Sparse_Coding)][无][无]

  * [独立成分分析（ICA）](./无监督学习（Unsupervised Learning）/独立成分分析（ICA）.md)\[[old](http://ufldl.stanford.edu/wiki/index.php/Independent_Component_Analysis)\]\[[new](http://ufldl.stanford.edu/tutorial/unsupervised/ICA)\]\[[旧](http://ufldl.stanford.edu/wiki/index.php/%E7%8B%AC%E7%AB%8B%E6%88%90%E5%88%86%E5%88%86%E6%9E%90)\]

   * [练习：独立成分分析（Exercise:Independent Component Analysis）](./无监督学习（Unsupervised Learning）/练习：独立成分分析（Exercise:Independent Component Analysis）.md)\[[old](http://ufldl.stanford.edu/wiki/index.php/Exercise:Independent_Component_Analysis)\]\[无\]\[无\]

  * RICA（RICA）\[无\]\[[new](http://ufldl.stanford.edu/tutorial/unsupervised/RICA)\]\[无\]

   * [练习：RICA（Exercise: RICA）](./无监督学习（Unsupervised Learning）/练习：RICA（Exercise: RICA）.md)\[无\]\[[new](http://ufldl.stanford.edu/tutorial/unsupervised/ExerciseRICA)\]\[无\]

  * 附1：[数据预处理（Data Preprocessing）](./无监督学习（Unsupervised Learning）/数据预处理（Data Preprocessing）.md)\[[old](http://ufldl.stanford.edu/wiki/index.php/Data_Preprocessing)\]\[无\]\[[旧](http://ufldl.stanford.edu/wiki/index.php/%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86)\]

  * 附2：[用反向传导思想求导（Deriving gradients using the backpropagation idea）](./无监督学习（Unsupervised Learning）/用反向传导思想求导（Deriving gradients using the backpropagation idea）.md)\[[old](http://ufldl.stanford.edu/wiki/index.php/Deriving_gradients_using_the_backpropagation_idea)\]\[无\]\[[旧](http://ufldl.stanford.edu/wiki/index.php/%E7%94%A8%E5%8F%8D%E5%90%91%E4%BC%A0%E5%AF%BC%E6%80%9D%E6%83%B3%E6%B1%82%E5%AF%BC)\]

* **自我学习（Self-Taught Learning）**

  * [自我学习（Self-Taught Learning）](./自我学习（Self-Taught Learning）/自我学习（Self-Taught Learning）.md)\[[old](http://deeplearning.stanford.edu/wiki/index.php/Self-Taught_Learning)\]\[[new](http://ufldl.stanford.edu/tutorial/selftaughtlearning/SelfTaughtLearning)\]\[[旧](http://ufldl.stanford.edu/wiki/index.php/%E8%87%AA%E6%88%91%E5%AD%A6%E4%B9%A0)\]

   * [练习：自我学习（Exercise: Self-Taught Learning）](./自我学习（Self-Taught Learning）/练习：自我学习（Exercise: Self-Taught Learning）.md)[[old](http://ufldl.stanford.edu/wiki/index.php/Exercise:Self-Taught_Learning)][[new](http://ufldl.stanford.edu/tutorial/selftaughtlearning/ExerciseSelfTaughtLearning)][无]

  * [深度网络概览（Deep Networks: Overview）](./自我学习（Self-Taught Learning）/深度网络概览（Deep Networks: Overview）.md)[[old](http://ufldl.stanford.edu/wiki/index.php/Deep_Networks:_Overview)][无][[旧](http://ufldl.stanford.edu/wiki/index.php/%E6%B7%B1%E5%BA%A6%E7%BD%91%E7%BB%9C%E6%A6%82%E8%A7%88)]

  * [栈式自编码算法（Stacked Autoencoders）](./自我学习（Self-Taught Learning）/栈式自编码算法（Stacked Autoencoders）.md)[[old](http://ufldl.stanford.edu/wiki/index.php/Stacked_Autoencoders)][无][[旧](http://ufldl.stanford.edu/wiki/index.php/%E6%A0%88%E5%BC%8F%E8%87%AA%E7%BC%96%E7%A0%81%E7%AE%97%E6%B3%95)]

  * [微调多层自编码算法（Fine-tuning Stacked AEs）](./自我学习（Self-Taught Learning）/微调多层自编码算法（Fine-tuning Stacked AEs）.md)[[old](http://ufldl.stanford.edu/wiki/index.php/Fine-tuning_Stacked_AEs)][无][[旧](http://ufldl.stanford.edu/wiki/index.php/%E5%BE%AE%E8%B0%83%E5%A4%9A%E5%B1%82%E8%87%AA%E7%BC%96%E7%A0%81%E7%AE%97%E6%B3%95)]

   * 练习：用深度网络实现数字分类（Exercise: Implement deep networks for digit classification）[[old](http://ufldl.stanford.edu/wiki/index.php/Exercise:_Implement_deep_networks_for_digit_classification)][无][无]

* **其它官方暂未写完的小节（Others）**

  * 卷积训练（Convolutional training）

  * 受限玻尔兹曼机（Restricted Boltzmann Machines）

  * 深度置信网络（Deep Belief Networks）

  * 降噪自编码器（Denoising Autoencoders）

  * K 均值（K-means）

  * 空间金字塔/多尺度（Spatial pyramids / Multiscale）

  * 慢特征分析（Slow Feature Analysis）

  * 平铺卷积网络（Tiled Convolution Networks）