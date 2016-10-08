## 微调多层自编码算法（Fine-tuning Stacked AEs）

注：本章节翻译参考旧版 UFLDL 中文教程。

### 1. 介绍（Introduction）

微调（ Fine-tunning ）是深度学习中的常用策略，可以大幅提升一个栈式自编码神经网络的性能表现。从更高的视角来讲，微调将栈式自编码神经网络的所有层视为一个模型，这样在每次迭代中，网络中所有的权重值都可以被优化。

### 2. 一般策略（General Strategy）

幸运的是，实施微调栈式自编码神经网络所需的工具都已齐备！为了在每次迭代中计算所有层的梯度，我们需要使用稀疏自动编码一节中讨论的反向传播算法。因为反向传播算法可以延伸应用到任意多层，所以事实上，该算法对任意多层的栈式自编码神经网络都适用。

### 3. 使用反向传播法进行微调（Finetuning with Backpropagation）

为方便读者，以下我们简要描述如何实施反向传播算法：


1. 进行一次前馈传递，对 $\textstyle L_2$ 层、 $\textstyle L_3$ 层直到输出层 $\textstyle L_{n_l}$ ，使用前向传播步骤中定义的公式计算各层上的激活值（激励响应）。

2. 对输出层（第 $\textstyle n_l$ 层），令
$$
\begin{align}
\delta^{(n_l)} = - (\nabla_{a^{n_l}}J) \bullet f'(z^{(n_l)})
\end{align}
$$
>**符号说明**
>
>$\delta^{(n_l)}$ ：输出层（第 $n_l$ 层）误差
>$f'(z^{(n_l)})$ ：对输出层函数 $f$ 的导数，传入上一层输出的结果 $z^{(n_l)}$
>$\nabla_{a^{n_l}}J$ ：输出层目标函数 $J$ 关于所求参数 $W$ 和 $b$ 的偏导数
>
当使用 SoftMax 分类器时， SoftMax 层满足： $\nabla J = \theta^T(I-P)$ ，其中 $\textstyle I$ 为输入数据对应的类别标签， $\textstyle P$ 为条件概率向量。
3. 对 $\textstyle l = n_l-1, n_l-2, n_l-3, \ldots, 2$ 令
$$
\begin{align}
\delta^{(l)} = \left((W^{(l)})^T \delta^{(l+1)}\right) \bullet f'(z^{(l)})
\end{align}
$$
>**符号说明**
>
>$W^{(l)}$：第 $l$ 层与第 $l+1$ 层间的网络权重参数
>$\delta^{(l)}$：误差从输出层经反向传播到第 $l$ 层的误差
4. 计算所需的偏导数：
$$
\begin{align}
\nabla_{W^{(l)}} J(W,b;x,y) &= \delta^{(l+1)} (a^{(l)})^T, \\
\nabla_{b^{(l)}} J(W,b;x,y) &= \delta^{(l+1)}.
\end{align}
$$

$$
\begin{align}
J(W,b) &= \left[ \frac{1}{m} \sum_{i=1}^m J(W,b;x^{(i)},y^{(i)}) \right]
\end{align}
$$



>注：我们可以认为输出层 SoftMax 分类器是附加上的一层，但是其求导过程需要单独处理。具体地说，网络“最后一层”的特征会进入 SoftMax 分类器。所以，第二步中的导数由 $\delta^{(n_l)} = - (\nabla_{a^{n_l}}J) \bullet f'(z^{(n_l)})$ 计算，其中 $\nabla J = \theta^T(I-P)$ （其中， $\textstyle I$ 为输入数据对应的类别标签， $\textstyle P$ 为条件概率向量）。
