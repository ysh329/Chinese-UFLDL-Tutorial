# 调试：梯度检查（Debugging: Gradient Checking）  
## 

迄今为止，在 MATLAB 中已经实现了通过计算目标函数的导数来计算梯度的算法（这种求梯度的方法叫做解析解）。在后续章节中，将看到更复杂的模型（例如神经网络的反向传播算法）。对于这些模型，梯度的计算会变得难以调试，并难以得到正确结果。有时，代码中的微小错误也可以使模型学习到东西，尽管表现稍稍不如完全正确的代码。因此，即使代码中微小的错误，也难说对最终结果有不好的影响。在本节中，将描述一种在数值层面（这种求梯度的方法叫做数值解）上检查你的代码在导数计算部分的正确性。通过用数值解来验证导数求得的梯度结果，可以增加您在代码正确性上的信心。  

>译者注：`解析解`指能够根据题意，得出在一定条件下的能够以数学表达式直接表达出来的的解。而`数值解`指在题中所给出的条件下难以用数学表达式表达出来，或者能够表达出来但需要每个给定自变量值下的数字结果，而通过计算（手算或计算机计算）的出来的以表格或图形表示的结果。`数值解`一般是近似结果，它与微分方程的真实结果有偏差（参考： <a href="http://www.zybang.com/question/c84bf6ec2427e4dbe662cec27d31c8b3.html" target="_blank">百度知道</a> ）。  

假设想要最小化带有参数 $\theta$ 的函数 $J(\theta)$ 。在这个例子中，假设有 $\textstyle J : \Re \mapsto \Re$ ， $\textstyle \theta \in \Re$ 。如果使用 <font color=red>`minFunc`</font> 或其它优化算法，在此之前已实现了某个 $g(\theta)$ 函数的代码，函数 $g(\theta)$ 是计算 $J(\theta)$ 的导数，即 $\textstyle \frac{d}{d\theta}J(\theta)$ （解析解）。  

怎样检查 $g$ 的代码实现是正确的呢？  
再来回顾一下导数的数学定义：  

$$
\begin{align}
\frac{d}{d\theta}J(\theta) = \lim_{\epsilon \rightarrow 0}
\frac{J(\theta+ \epsilon) - J(\theta-\epsilon)}{2 \epsilon}.
\end{align}
$$  

因此，对任何特定的 $\theta$ 参数值，可以用下面这个方法（数值解）检查与导数值（解析解）是否接近:  

$$
\begin{align}
\frac{J(\theta+{\rm EPSILON}) - J(\theta-{\rm EPSILON})}{2 \times {\rm EPSILON}}
\end{align}
$$  

在实践中，设置 ${\rm EPSILON}$ 为一小常量，通常设置为 $10^{-4}$ 。 （ ${\rm EPSILON}$ 的值域范围尽管很大，但不设置 ${\rm EPSILON}$ “非常”小，比如 $10^{-20}$ ，因为这会产生计算机的舍入误差。）  

>译者注：`舍入误差`，由于计算机的字长有限，进行数值计算的过程中，对计算得到的中间结果数据要使用“四舍五入”或其他规则取近似值，因而使计算过程有误差。这种误差称为舍入误差（参考： <a href="http://baike.baidu.com/link?url=2vnU5YEAPZ5Te7VlIaRabbpMeLbYyQMTAPKhfocm_lBYA_9VF8FE1P2ZiRjXHk2Ze4Dloe6JCZH5f4KCPyQU5_" target="_blank">百度百科</a> ）。  

因此，对给定目标函数的导数 $g(\theta)$ ，它计算的是 $\textstyle \frac{d}{d\theta}J(\theta)$ （即解析解），可以通过下面这个式子从数值角度（即数值解）来验证导数求得的解（即解析解）的正确性  

$$
\begin{align}
g(\theta) \approx
\frac{J(\theta+{\rm EPSILON}) - J(\theta-{\rm EPSILON})}{2 \times {\rm EPSILON}}.
\end{align}
$$  

以这两个值彼此的接近程度将取决于 $J$ 。假设 $\textstyle {\rm EPSILON} = 10^{-4}$。通常，你会发现的上面这个约等式中的左手边和右手边两个计算出的结果，一致的位数至少4位（但也经常更多）。  

现在，考虑一下参数 $\theta$ 是一个向量，而非单个实数的情况（为了想要学到的 $n$ 个参数），并且有 $\textstyle J: \Re^n \mapsto \Re$ 。现在，概括了导数检查过程，其中参数 $\theta$ 可能是一个向量（如在线性回归和逻辑回归的例子中的）。如果正在通过几个向量或者矩阵来做优化，可以将这些参数“打包”进一个“长”的向量中去。在这里，可以用同样的方法来检查导数。（这也可以使用现成的优化包来完成）。  

假设有目标函数 $J(\theta)$ 的导数 $\textstyle \frac{\partial}{\partial \theta_i} J(\theta)$ 的计算并化简出的结果： $\textstyle g_i(\theta)$ ；想要检查通过导数算出的梯度 $g_{i}$ 是否输出了正确的导数值（即梯度值）。有 $\textstyle \theta^{(i+)} = \theta + {\rm EPSILON} \times \vec{e}_i$，其中  

$$
\begin{align}
\vec{e}_i = \begin{bmatrix}0 \\ 0 \\ \vdots \\ 1 \\ \vdots \\ 0\end{bmatrix}
\end{align}
$$  

$\vec{e}_i$ 是第 $i$ 个基向量（ $\vec{e}_i$ 是与 $\theta$ 参数同维度的向量，在 $\vec{e}_i$ 向量中第 $i$ 个位置的元素值为 $“1”$ ，其余全部为 $“0”$）。所以，除了其第 $i$ 个元素被 ${\rm EPSILON}$ 增加外，参数 $\textstyle \theta^{(i+)}$ 与 $\theta$ 是相同的。同理， $\textstyle \theta^{(i-)} = \theta - {\rm EPSILON} \times \vec{e}_i$ 是参数 $\theta$ 向量在第 $i$ 个位置的元素被 ${\rm EPSILON}$ 相减得到的向量。  


现在，可以从数值上（数值解的角度），对第 $i$ 个参数的梯度 $\textstyle g_i(\theta)$ 进行检查（译者注：检查的是模型参数向量中每一个参数的梯度，从数值解的角度来验证解析解），以验证解析解的正确性：  

$$
\begin{align}
g_i(\theta) \approx
\frac{J(\theta^{(i+)}) - J(\theta^{(i-)})}{2 \times {\rm EPSILON}}.
\end{align}
$$  

## 梯度检查代码（Gradient checker code）  
本次练习，将尝试实现上述方法来检查您的线性回归（Linear Regression）和逻辑斯特回归（Logistic Regression）函数的梯度。另外，您也可以使用提供的 <font color=red>`ex1/ grad_check.m`</font> 文件（其中带有的参数与 <font color=red>`minFunc`</font> 类似），对众多随机选择的 $i$ 做 $\frac{\partial J(\theta)}{\partial \theta_i}$ 导数值的检查。
