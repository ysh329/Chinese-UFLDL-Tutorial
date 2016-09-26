# Chinese-UFLDL-Tutorial

中文版的新版UFLDL教程（项目地址： www.github.com/ysh329/Chinese-UFLDL-Tutorial ），该版本翻译自 <a href="http://deeplearning.stanford.edu/tutorial/" target="_blank">UFLDL Tutorial</a> ，是新版教程的翻译。也可参考 <a href="http://ufldl.stanford.edu/wiki/index.php/UFLDL教程" target="_blank">老版本的UFLDL中文教程</a> 。翻译过程中有一些数学公式，使用 <a href="http://pad.haroopress.com/user.html#download" target="_blank">Haroopad</a> 编辑和排版， Haroopad 是一个很不错的离线 MarkDown 编辑器，支持 <a href="https://en.wikipedia.org/wiki/TeX" target="_blank">Tex</a> 公式编辑，支持多平台（Win/Mac/Linux），目前还在翻译中，翻译完成后会考虑使用 Tex 重新排版。  



自己对新版UFLDL教程翻译过程中，发现的英文错误，见 <a href="./新版教程英文原文勘误表.md" target="_blank">新版教程英文原文勘误表</a> 。  

**注：UFLDL是非监督特征学习及深度学习（Unsupervised Feature Learning and Deep Learning）的缩写，而不仅指深度学习（Deep Learning）。**  

-  翻译者：Shuai Yuan  
-  若有翻译错误，请发邮件至 <a href="Mailto:ysh329@sina.com" target="_blank">ysh329@sina.com</a> ，感谢！  

>更多详细参考资料，见 <a href="https://github.com/bayandin/awesome-awesomeness" target="_blank">计算机科学</a> ， <a href="https://github.com/owainlewis/awesome-artificial-intelligence" target="_blank">人工智能</a> ， <a href="https://github.com/josephmisiti/awesome-machine-learning" target="_blank">机器学习</a> ， <a href="https://github.com/ChristosChristofidis/awesome-deep-learning" target="_blank">深度学习</a> ， <a href="https://github.com/aikorea/awesome-rl" target="_blank">强化学习</a> ， <a href="https://github.com/junhyukoh/deep-reinforcement-learning-papers" target="_blank">深度强化学习</a> ， <a href="https://github.com/ChristosChristofidis/awesome-public-datasets" target="_blank">公开数据集</a> 。



# 欢迎来到UFLDL教程！

说明：本教程将会教给您非监督特征学习以及深度学习的主要思想。通过它，您将会实现几个特征学习或深度学习的算法，看到这些算法为您（的工作）带来作用，以及学习如何将这些思想应用到适用的新问题上。



本教程假定您已经有了基本的机器学习知识（具体而言，熟悉监督学习，逻辑斯特回归以及梯度下降法的思想）。如果您不熟悉这些，我们建议您先去 <a href="http://openclassroom.stanford.edu/MainFolder/CoursePage.php?course=MachineLearning" target="_blank">机器学习课程</a> 中去学习，并完成其中的第II，III，IV章节（即到逻辑斯特回归）。



材料由以下人员提供：Andrew Ng, Jiquan Ngiam, Chuan Yu Foo, Yifan Mai, Caroline Suen, Adam Coates, Andrew Maas, Awni Hannun, Brody Huval, Tao Wang, Sameep Tandon

## 获取初学者代码（Starter Code）

### 初学者代码

您可以获得初学者所有练习的代码从 <a href="https://github.com/amaas/stanford_dl_ex" target="_blank">该Github的代码仓库</a> 。  

有关的数据文件可以从 <a href="http://ai.stanford.edu/~amaas/data/data.zip" target="_blank">这里</a> 下载。 下载到的数据需要解压到名为<font color=red>`“common”`</font>的文件夹中（以便初学者代码的使用）。



# 目录

* 监督学习与优化（Supervised Learning and Optimization）

  *  \[无\]\[new]\[无\]<a href="./监督学习和优化（Supervised Learning and Optimization）/线性回归（Linear Regression）.md" target="_blank">线性回归（Linear Regression）</a>

  *  \[old\]\[new]\[<a href="http://ufldl.stanford.edu/wiki/index.php/%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E7%9A%84%E5%90%91%E9%87%8F%E5%8C%96%E5%AE%9E%E7%8E%B0%E6%A0%B7%E4%BE%8B" target="_blank">旧</a>\]<a href="./监督学习和优化（Supervised Learning and Optimization）/逻辑斯特回归（Logistic Regression）.md" target="_blank">逻辑斯特回归（Logistic Regression）</a>

  *  \[old\]\[new]\[<a href="http://ufldl.stanford.edu/wiki/index.php/%E7%9F%A2%E9%87%8F%E5%8C%96%E7%BC%96%E7%A8%8B" target="_blank">旧</a>\]<a href="./监督学习和优化（Supervised Learning and Optimization）/向量化（Vectorization）.md" target="_blank">向量化（Vectorization）</a>

  *  \[old\]\[new]\[<a href="http://ufldl.stanford.edu/wiki/index.php/%E6%A2%AF%E5%BA%A6%E6%A3%80%E9%AA%8C%E4%B8%8E%E9%AB%98%E7%BA%A7%E4%BC%98%E5%8C%96" target="_blank">旧</a>\]<a href="./监督学习和优化（Supervised Learning and Optimization）/调试：梯度检查（Debugging：Gradient Checking）.md" target="_blank">调试：梯度检查（Debugging: Gradient Checking）</a>

  *  \[old\]\[new]\[<a href="http://ufldl.stanford.edu/wiki/index.php/Softmax%E5%9B%9E%E5%BD%92" target="_blank">旧</a>\]<a href="./监督学习和优化（Supervised Learning and Optimization）/Softmax回归（Softmax Regression）.md" target="_blank">Softmax回归（Softmax Regression）</a>

  *  \[old\]\[new]\[无\]<a href="./监督学习和优化（Supervised Learning and Optimization）/检查：偏差和方差（Debugging：Bias and Variance）.md" target="_blank">调试：偏差和方差（Debugging: Bias and Variance）</a>

  *  \[old\]\[new]\[无\]<a href="./监督学习和优化（Supervised Learning and Optimization）/调试：优化器和目标（Debugging：Optimizers and Objectives）.md" target="_blank">调试：优化器和目标（Debugging: Optimizers and Objectives）</a>

* 监督神经网络（Supervised Neural Networks）

  * \[old\]\[new]\[<a href="http://ufldl.stanford.edu/wiki/index.php/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C" target="_blank">旧</a>\]<a href="./监督神经网络（Supervised Neural Networks）/多层神经网络（Multi-Layer Neural Networks）.md" target="_blank">多层神经网络（Multi-Layer Neural Networks）</a>

  * \[old\]\[new]\[旧\]<a href="./监督神经网络（Supervised%20Neural%20Networks）/练习：%20监督神经网络（Exercise:%20Supervised%20Neural%20Networks）.md" target="_blank">练习：监督神经网络（Exercise: Supervised Neural Network）</a>

* 监督卷积网络（Supervised Convolutional Neural Network）

  * \[old\]\[new]\[<a href="http://ufldl.stanford.edu/wiki/index.php/%E5%8D%B7%E7%A7%AF%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96" target="_blank">旧</a>\]<a href="./监督卷积网络（Supervised Convolutional Neural Network）/使用卷积进行特征提取（Feature Extraction Using Convolution）.md" target="_blank">使用卷积进行特征提取（Feature Extraction Using Convolution）</a>

  * \[old\]\[new]\[<a href="http://ufldl.stanford.edu/wiki/index.php/%E6%B1%A0%E5%8C%96" target="_blank">旧</a>\]<a href="./监督卷积网络（Supervised Convolutional Neural Network）/池化（Pooling）.md" target="_blank">池化（Pooling）</a>

  * \[old\]\[new]\[旧\]练习：卷积和池化（Exercise: Convolution and Pooling）

  * \[old\]\[new]\[旧\]<a href="./监督卷积网络（Supervised Convolutional Neural Network）/优化方法：随机梯度下降（Optimization: Stochastic Gradient Descent）.md" target="_blank">优化方法：随机梯度下降（Optimization: Stochastic Gradient Descent）</a>

  * \[old\]\[new]\[旧\]<a href="./监督卷积网络（Supervised Convolutional Neural Network）/卷积神经网络（Convolutional Neural Network）.md" target="_blank">卷积神经网络（Convolutional Neural Network）</a>

  * \[old\]\[new]\[旧\]练习：卷积神经网络（Excercise: Convolutional Neural Network）

* 无监督学习（Unsupervised Learning）

  * \[old\]\[new]\[<a href="http://ufldl.stanford.edu/wiki/index.php/%E8%87%AA%E7%BC%96%E7%A0%81%E7%AE%97%E6%B3%95%E4%B8%8E%E7%A8%80%E7%96%8F%E6%80%A7" target="_blank">旧</a>\]自动编码器（Autoencoders）

  * \[old\]\[new]\[<a href="http://ufldl.stanford.edu/wiki/index.php/%E5%AE%9E%E7%8E%B0%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90%E5%92%8C%E7%99%BD%E5%8C%96" target="_blank">旧</a>\]PCA Whitening（PCA Whitening）

  * \[old\]\[new]\[旧\]练习：PCA Whitening（Exercise: PCA Whitening）

  * \[old\]\[new]\[<a href="http://ufldl.stanford.edu/wiki/index.php/%E7%A8%80%E7%96%8F%E7%BC%96%E7%A0%81" target="_blank">旧</a>\]稀疏编码（Sparse Coding）

  * \[old\]\[new]\[无\]ICA（ICA）

  * \[old\]\[new]\[无\]RICA（RICA）

  * \[old\]\[new]\[无\]练习：RICA（Exercise: RICA）

* 自学学习（Self-Taught Learning）

  * \[old\]\[new]\[<a href="http://ufldl.stanford.edu/wiki/index.php/%E8%87%AA%E6%88%91%E5%AD%A6%E4%B9%A0" target="_blank">旧</a>\]自我学习（Self-Taught Learning）

  * \[old\]\[new]\[旧\]练习：自我学习（Exercise: Self-Taught Learning）
