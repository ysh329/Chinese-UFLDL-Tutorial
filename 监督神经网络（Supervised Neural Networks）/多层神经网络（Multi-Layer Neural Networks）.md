# 多层神经网络（Multi-Layer Neural Network）  
##  

考虑一个监督学习问题，即我们有机会得到带标签的训练样本 $(x^{(i)}, y^{(i)})$ 。神经网络给出了一种方式用来定义复杂以及非线性的假设形式 $h_{W,b}(x)$ ，该形式带有参数 $W, b$，这些参数可被用来拟合我们的数据。  

为描述神经网络，我们将开始描述一个最为简单的神经网络 —— 只有一个神经元。我们将用下面这幅图来表示这样的单个神经元：  
<center><img src="./images/SingleNeuron.png" width=300/></center>  

这个神经元是一个可以输入 $x1, x2, x3$ 的计算单元（其中， $+1$ 是截距项），并且它将会输出 $\textstyle h_{W,b}(x) = f(W^Tx) = f(\sum_{i=1}^3 W_{i}x_i +b)$ 。其中， $f : \Re \mapsto \Re$ 被称为“激活函数”。在这些笔记中，我们会选择 $f(\cdot)$ 作为我们的 S 型函数：  

$$
f(z) = \frac{1}{1+\exp(-z)}.
$$  

因此，单个神经元对应的是被定义为逻辑斯特回归的输入-输出映射。  

尽管这些笔记将会用在 $S$ 型函数，但很有必要说明的是，其它常见的 $f$ 的选择可以是双曲正切或正切函数：  

$$
f(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}.
$$  

最近的研究发现了一种与众不同的激活函数，整流线性激活函数（the rectified linear function），在实际中对于深层神经网络的效果更好。这种激活函数与 $S$ 型函数和双曲正切函数 $tanh$ 不同，因为其上限值不确定的，而且不是连续可微的。下面给出整流线性激活函数：  

$$
f(z) = \max(0,x).
$$  

下面是 $S$ 型函数，双曲正切函数 $tanh$ 以及整流线性函数：  

<center><img src="./images/Activation_functions.png"></center>  

双曲正切函数 $tanh(z)$ 是 $S$ 型函数的一个缩放版本，其输出范围是 $[-1,1]$ ，而不是 $[０,1]$ 。整流线性函数是一个分段的线性函数，当输入 $z$ 值小于 $0$ 时，其函数值为 $
0$ 。  

值得注意的是，不像其它地方（包括开放式的课堂教学视频，以及部分 CS229 的课程），我们不按照惯例 $x_0=1$ ，相反，截距项是由参数 $b$ 单独处理。  

最后我们要说的是在后面将会很有用的一个等式： $f(z) = 1/(1+\exp(-z))$ 是一个 $S$ 型函数，其导数为 $f'(z) = f(z) (1-f(z))$ （如果 $f$ 是双曲函数 $tanh$ ，则其导数为 $f'(z) = 1- (f(z))^2$ ）。您也可以自己通过 $S$ 型函数（或双曲函数 $tanh$）的定义尝试求导。整流线性函数在输入 $z \leq 0$ 时梯度为 $0$ ，其它取值时为 $1$ 。当输入值 $z=0$ 时，梯度是不确定的，但这并不会在实际中引起什么问题，因为在优化过程中，我们是建立在大量的训练样本之上来计算梯度的平均值的。  


## 神经网络模型（Neural Network model）　　

神经网络是通过将众多简单的神经元连接在一起得到的，一个神经元的输出可以是另一个的输入。例如，这里是一个小神经网络：  

<center><img src="./images/Network331.png" width=300 /></center>  

在这幅图中，我们用圆圈来表示网络的输入。在圈里被表为 “+1” 的圆圈称为偏置单元，对应于截距项。网络最左边的那一层称为输入层，而输出层即最右层（在这个例子中，输出层只有一个节点）。介于最左和最右的中间层称为隐藏层，因为它的值是无法在训练集中观察到的。由此，我们可以说，该例子中的神经网络有 3 个输入单元（不把计偏置单元计算在内）， 3 个隐藏单元，和 1 个输出单元。  

We will let nl denote the number of layers in our network; thus nl=3 in our example. We label layer l as Ll, so layer L1 is the input layer, and layer Lnl the output layer. Our neural network has parameters (W,b)=(W(1),b(1),W(2),b(2)), where we write W(l)ij to denote the parameter (or weight) associated with the connection between unit j in layer l, and unit i in layer l+1. (Note the order of the indices.) Also, b(l)i is the bias associated with unit i in layer l+1. Thus, in our example, we have W(1)∈R3×3, and W(2)∈R1×3. Note that bias units don’t have inputs or connections going into them, since they always output the value +1. We also let sl denote the number of nodes in layer l (not counting the bias unit).

$n_l$ 表示网络的层数；因此，在我们的例子中 $n_l = 3$ 。我们把层 $l$ （第一层）表示为 $L_1$，所以层 $L_1$ 为输入层，输出层用 $L_{n_l}$ 来表示。我们的神经网络参数 $(W,b) = (W^{(1)}, b^{(1)}, W^{(2)}, b^{(2)})$ ，参数 $W^{(l)}_{ij}$ 表示第 $l$ 层的第 $j$ 个单元与第 $$
单位J L层之间的连接相关联，并在单位我层L + 1。（请注意指数的顺序），我是一个与单位有关的偏见，我在1层。因此，在我们的例子中，我们有W（1）∈R3×3，和W（2）∈R1×3。请注意，偏置单元没有输入或连接进入它们，因为它们总是输出的值+ 1。我们也让SL表示节点层L的数量（不包括偏置单元）。  

We will write a(l)i to denote the activation (meaning output value) of unit i in layer l. For l=1, we also use a(1)i=xi to denote the i-th input. Given a fixed setting of the parameters W,b, our neural network defines a hypothesis hW,b(x) that outputs a real number. Specifically, the computation that this neural network represents is given by:







