# 使用卷积进行特征提取（Feature Extraction Using Convolution）  
##  
## 概览（Overview）  

在之前的练习中，练习问题涉及到的图片其分辨率都偏低，例如小图像修补程序和小图像的手写数字识别。而在本节中，我们将会开发一种方法，它能够扩展先前学到的方法在更实际的大图像数据集上。  

## 全连接网络（Fully Connected Networks）  
In the sparse autoencoder, one design choice that we had made was to “fully connect” all the hidden units to all the input units. On the relatively small images that we were working with (e.g., 8x8 patches for the sparse autoencoder assignment, 28x28 images for the MNIST dataset), it was computationally feasible to learn features on the entire image. However, with larger images (e.g., 96x96 images) learning features that span the entire image (fully connected networks) is very computationally expensive–you would have about 104 input units, and assuming you want to learn 100 features, you would have on the order of 106 parameters to learn. The feedforward and backpropagation computations would also be about 102 times slower, compared to 28x28 images.  


在稀疏编码，一个设计选择，我们已经是“完全连接”所有的隐藏单元的所有输入单元。在我们的工作与相对较小的图像（例如，8x8的补丁的稀疏编码的任务，28x28图像MNIST数据集），它是学习的特点对整个图像计算是可行的。然而，更大的图像（例如，96x96图像）学习跨越整个图像特征（完全连接网络）的计算是非常昂贵的–你要有104个输入单元，假设你想了解100的功能，你就会对106参数的顺序学习。前向和反向传播计算也会慢了大约102倍，比28x28图像。

## 局部连接网络（Locally Connected Networks）  
One simple solution to this problem is to restrict the connections between the hidden units and the input units, allowing each hidden unit to connect to only a small subset of the input units. Specifically, each hidden unit will connect to only a small contiguous region of pixels in the input. (For input modalities different than images, there is often also a natural way to select “contiguous groups” of input units to connect to a single hidden unit as well; for example, for audio, a hidden unit might be connected to only the input units corresponding to a certain time span of the input audio clip.)  

This idea of having locally connected networks also draws inspiration from how the early visual system is wired up in biology. Specifically, neurons in the visual cortex have localized receptive fields (i.e., they respond only to stimuli in a certain location).  

## 卷积（Convolutions）  
Natural images have the property of being ”‘stationary”’, meaning that the statistics of one part of the image are the same as any other part. This suggests that the features that we learn at one part of the image can also be applied to other parts of the image, and we can use the same features at all locations.  

More precisely, having learned features over small (say 8x8) patches sampled randomly from the larger image, we can then apply this learned 8x8 feature detector anywhere in the image. Specifically, we can take the learned 8x8 features and ”‘convolve”’ them with the larger image, thus obtaining a different feature activation value at each location in the image.  

To give a concrete example, suppose you have learned features on 8x8 patches sampled from a 96x96 image. Suppose further this was done with an autoencoder that has 100 hidden units. To get the convolved features, for every 8x8 region of the 96x96 image, that is, the 8x8 regions starting at (1,1),(1,2),…(89,89), you would extract the 8x8 patch, and run it through your trained sparse autoencoder to get the feature activations. This would result in 100 sets 89x89 convolved features.  