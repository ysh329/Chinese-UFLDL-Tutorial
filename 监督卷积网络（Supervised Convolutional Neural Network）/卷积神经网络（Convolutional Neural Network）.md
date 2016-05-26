# 卷积神经网络（Convolutional Nerual Network）  
##  
## 概述（Overview）  

卷积神经网络（ $CNN$ ）是有一个或多个卷积层（常伴有下采样步骤）并后面跟一个或多个全连接层的标准<a href="../监督神经网络（Supervised Neural Networks）/多层神经网络（Multi-Layer Neural Networks）.md" target="_blank">多层神经网络</a>。卷积神经网络在体系结构的设计利用了输入图像的二维结构（其它的二维输入还有语音信号等）。  

卷积神经网络的实现是借助局部连接和在其之后的绑定权重，其中池化操作有平移不变特征。卷积神经网络的另一个优点在于更容易训练，并且卷积神经网络的参数虽多但却比相同隐含单元的全连接网络要少。  

在本节内容中，我们将会讨论卷积神经网络的架构和用来计算模型中不同参数梯度的反向传播算法。<a href="../监督卷积网络（Supervised Convolutional Neural Network）/使用卷积进行特征提取（Feature Extraction Using Convolution）.md" target="_blank">卷积</a>和<a href="../监督卷积网络（Supervised Convolutional Neural Network）/池化（Pooling）.md" target="_blank">池化</a>更详细的具体操作见本教程的各自章节。  

## 网络架构（Architecture）

一个卷积神经网络是由很多个卷积层、后接可选的（一层或多层）下采样层以及后接的全连接层组成的。卷积层的输入是一个规模为 $m \times m \times r$ 的图像，其中（第一个） $m$ 是图像的高度，第二个 $m$ 是宽度， $ r $ 是图像的通道个数。，例如一个 $RGB$ 图像的通道数 $ r=3 $ 。卷积层有 $k$ 个规模为 $n \times n \times q$ 的滤波器（或称为核），其中 $n$ 小于图片的维度（译者注：图片的高度或宽度）， $q$ 既可以是通道个数 $r$ ，也可能对于不同的滤波器（或称为核）而不同。  

过滤器的规模增加了局部连接的结构（译者注：不是很理解这句话的深层含义：The size of the filters gives rise to the locally connected structure.），原图被（这种结构）卷积成为规模为 $m-n+1$ 的 $k$ 个特征图（译者注：这里的“特征图”，即 $feature\ map$，是二维的。其中 $m-n+1$ 是一个特征图的宽度或者高度，原图是正方形，所以这里特征图的边长为 $m-n+1$ ）。  

之后，每个（特征）图通常在 $p \times p$ 的相邻区域进行平均值或最大值下采样（译者注： $CNN$ 中的下采样即池化），对于相邻区域 $p$ 的值范围从 $2$ （对于小图片值为 $2$ ，例如 $MNIST$ 手写图片数据集）并通常不超过 $5$ （对于大图片）。  

在下采样层的之前或之后，（译者注：会对结果）应用一个附加的偏置和 $S$ 型的非线性（译者注：的映射）。  

下图描述了卷积神经网络中一个完整的卷积和下采样层。一样颜色的（译者注：神经）单元有着连接权重。  

<center><img src="./images/Cnn_layer.png"></center>
<center>图1：一个卷积神经网络的第一层的池化过程。
相同颜色的神经单元有连接权重，不同颜色的神经元表示不同的滤波器（图）  
（译者注：这里不同颜色的神经元代表着不同的滤波器或核，  
<font color=red>注意：滤波器和核是一个意思，滤波器和通道不是一个意思</font>）。
</center>  

在卷积层后可能有任意个全连接层。在一个标准的<a href="../监督神经网络（Supervised Neural Networks）/多层神经网络（Multi-Layer Neural Networks）.md" target="_blank">多层神经网络</a>中，被密集连接（译者注：即全连接）的这些层是一样的。  

## 反向传播（Back Propagation）  
Let δ(l+1)
be the error term for the (l+1)-st layer in the network with a cost function J(W,b;x,y) where (W,b) are the parameters and (x,y) are the training data and label pairs. If the l-th layer is densely connected to the (l+1)-st layer, then the error for the l-th layer is computed as



