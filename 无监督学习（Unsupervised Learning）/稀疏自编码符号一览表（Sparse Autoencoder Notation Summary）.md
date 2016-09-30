## 稀疏自编码符号一览表（Sparse Autoencoder Notation Summary）  
注：本章节翻译完全参考旧版 UFLDL 中文教程。  

下面是我们在推导 稀疏自编码（ sparse autoencoder ）时使用的符号一览表：  

|符号            | 含义   |
| :-------------: | -------------------------|
|$\textstyle x$ |训练样本的输入特征， $\textstyle x \in \Re^{n}.$|
|$\textstyle y$	| 输出值/目标值. 这里 $\textstyle y$ 可以是向量. 在 autoencoder 中， $\textstyle y=x.$ |
|$\textstyle (x^{(i)}, y^{(i)})$ | 第 $\textstyle i$ 个训练样本 |
|$\textstyle h_{W,b}(x)$ | 输入为 $\textstyle x$ 时的假设输出，其中包含参数 $\textstyle W,b.$ 该输出应当与目标值 $\textstyle y$ 具有相同的维数. |
|$\textstyle W^{(l)}_{ij}$ | 连接第 $\textstyle l$ 层 $\textstyle j$ 单元和第 $\textstyle l+1$ 层 $\textstyle i$ 单元的参数. |
|$\textstyle b^{(l)}_{i}$ |	第 $\textstyle l+1$ 层 $\textstyle i$ 单元的偏置项. 也可以看作是连接第 $\textstyle l$ 层偏置单元和第 $\textstyle l+1$ 层 $\textstyle i$ 单元的参数. |
|$\textstyle \theta $ | 参数向量. 可以认为该向量是通过将参数 $\textstyle W,b$ 组合展开为一个长的列向量而得到. |
|$\textstyle a^{(l)}_i$ | 网络中第 $\textstyle l$ 层 $\textstyle i$ 单元的激活（输出）值. <br/> 另外，由于 $\textstyle L_1$ 层是输入层，所以 $\textstyle a^{(1)}_i = x_i.$ |
|$\textstyle f(\cdot)$ | 激活函数. 本文中我们使用 $\textstyle f(z) = \tanh(z).$ |
|$\textstyle z^{(l)}_i$ | 第 $\textstyle l$ 层 $\textstyle i$ 单元所有输入的加权和. 因此有 $\textstyle a^{(l)}_i = f(z^{(l)}_i).$ |
|$\textstyle \alpha$ | 学习率 |
|$\textstyle s_l$ | 第 $\textstyle l$ 层的单元数目（不包含偏置单元）. |
|$\textstyle n_l$ |	网络中的层数. 通常 $\textstyle L_1$ 层是输入层，$\textstyle L_{n_l}$ 层是输出层. |
|$\textstyle \lambda$ | 权重衰减系数. |
|$\textstyle \hat{x}$ | 对于一个 autoencoder ，该符号表示其输出值；亦即输入值 $\textstyle x$ 的重构值. 与 $\textstyle h_{W,b}(x)$ 含义相同. |
|$\textstyle \rho$ | 稀疏值，可以用它指定我们所需的稀疏程度. |
|$\textstyle \hat\rho_i$ | （ sparse autoencoder 中）隐藏单元 $\textstyle i$ 的平均激活值. |
|$\textstyle \beta$ |（ sparse autoencoder 目标函数中）稀疏值惩罚项的权重. |