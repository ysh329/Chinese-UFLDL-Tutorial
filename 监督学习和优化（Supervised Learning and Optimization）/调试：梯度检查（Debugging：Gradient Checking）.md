# 调试：梯度检查（Debugging: Gradient Checking）
## 

迄今为止，我们已经实现了较为简单的算法，这些算法都是直截了当地计算目标函数和梯度的，在 MATLAB 里实现必要的计算。在后面的章节中，我们将看到更复杂的模型（例如神经网络的反向传播算法）。对于这些模型来说，其梯度计算会变得难以调试，及难以得到正确结果。有时，一个微小的错误可以（使模型成功地）学习（到某些东西），（尽管表现能力不如正确的情形）但这看起来又是难以置信地合理。因此，即使微小的错误，也难说对所有的（结果或最终结果）有不好（的影响）。在本节中，我们将描述一种在数值层面上检查你的代码在导数计算部分的正确性。通过进行导数检查这一过程，将使你显著增加你在代码正确性上的信心。  

假设我们想要最小化带有参数 $\theta$ 的函数 $J(\theta)$。在这个例子中，假设有 $\textstyle J : \Re \mapsto \Re$，以便 $\textstyle \theta \in \Re$。如果我们使用 <font color=red>`minFunc`</font> 或一些其它的优化算法，我们在通常在此之前已经实现了某个 $g(\theta)$ 函数，该函数是根据（计算 $J(\theta)$ 的导数） $\textstyle \frac{d}{d\theta}J(\theta)$ 得到的。  

我们怎样检查我们的 $g$ 实现的是正确的呢？  
让我们再来回顾一下导数的数学定义：  

$$
\begin{align}
\frac{d}{d\theta}J(\theta) = \lim_{\epsilon \rightarrow 0}
\frac{J(\theta+ \epsilon) - J(\theta-\epsilon)}{2 \epsilon}.
\end{align}
$$
因此，对任何特定的 \theta 参数值，我们可以用下面这个方法在数值上近似这个导数值:  
$$
\begin{align}
\frac{J(\theta+{\rm EPSILON}) - J(\theta-{\rm EPSILON})}{2 \times {\rm EPSILON}}
\end{align}
$$

在实践中，我们设置 ${\rm EPSILON}$ 为一小常量，通常设置为 $10^{-4}$ 。 （ ${\rm EPSILON}$ 的值域范围尽管很大，但我们不设置 ${\rm EPSILON}$ “非常”小，比如 $10^{-20}$ ，因为这会导致数值舍入误差。）  

因此，对给定函数 $g(\theta)$ ，它应计算 $\textstyle \frac{d}{d\theta}J(\theta)$ ，我们现在就通过下面这个式子从数值角度上来验证其正确性  
$$
\begin{align}
g(\theta) \approx
\frac{J(\theta+{\rm EPSILON}) - J(\theta-{\rm EPSILON})}{2 \times {\rm EPSILON}}.
\end{align}
$$

以这两个值彼此的接近程度将取决于 $J$。假设 $\textstyle {\rm EPSILON} = 10^{-4}$。通常，你会发现的上式中左手边和右手边（算下来的结果中）一致的位数至少4位（但也经常更多）。  

现在，考虑一下参数 $\theta$ 是一个向量，而非单个实数的情况（为了我们想要学到的 $n$ 个参数），并且有 $\textstyle J: \Re^n \mapsto \Re$ 。现在，我们概括了导数检查过程，其中参数 $\theta$ 可能是一个向量（如在线性回归和逻辑回归的例子中的）。如果我们正在通过几个向量或者矩阵来做优化，我们总能将这些参数“打包”进一个“长”的向量中去。在这里，我们可以用同样的方法来检查我们的导数。（这也可以使用现成的优化包来完成）。  

假设我们有了根据计算导数 $\textstyle \frac{\partial}{\partial \theta_i} J(\theta)$ 结果的函数 $\textstyle g_i(\theta)$；我们想要检查 $g_{i}$ 是否输出了正确的导数值。我们有 $\textstyle \theta^{(i+)} = \theta + {\rm EPSILON} \times \vec{e}_i$，其中  

$$
\begin{align}
\vec{e}_i = \begin{bmatrix}0 \\ 0 \\ \vdots \\ 1 \\ \vdots \\ 0\end{bmatrix}
\end{align}
$$

是第 $i$ 个基向量（是与 $\theta$ 参数同维度的向量，在该向量中第 $i$ 个位置元素值为 $“1”$ ，其余全部为 $“0”$）。所以，除了其第 $i$ 个元素被 ${\rm EPSILON}$ 增加外，参数 $\textstyle \theta^{(i+)}$ 与 $\theta$ 是相同的。同理， $\textstyle \theta^{(i-)} = \theta - {\rm EPSILON} \times \vec{e}_i$ 是参数 $\theta$ 向量在第 $i$ 个元素的位置被 ${\rm EPSILON}$ 相减的向量。  


现在，我们可以从数值上，对每个 $i$ 检查以验证 $\textstyle g_i(\theta)$ 的正确性：  

$$
\begin{align}
g_i(\theta) \approx
\frac{J(\theta^{(i+)}) - J(\theta^{(i-)})}{2 \times {\rm EPSILON}}.
\end{align}
$$

## 梯度检查代码（Gradient checker code）
本次练习，将尝试实现上述方法来检查您的线性回归（Linear Regression）和逻辑斯特回归（Logistic Regression）函数的梯度。另外，您也可以使用提供的 <font color=red>`ex1/ grad_check.m`</font> 文件（其中带有的参数与 <font color=red>`minFunc`</font> 类似），对众多随机选择的 $i$ 做 $\frac{\partial J(\theta)}{\partial \theta_i}$ 导数值的检查。

