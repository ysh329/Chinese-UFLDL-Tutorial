## 推荐读物（UFLDL Recommended Readings）

如果您正在学习 UFLDL (非监督特征学习与深度学习)，那么您可以考虑下面这份阅读清单。给出这份推荐阅读清单的前提是，我们假设您已经对 [CS229](http://cs229.stanford.edu/) 这门课上的机器学习基础知识（也包括讲座笔记）有所掌握。

基础知识:
* [CS294A](http://cs294a.stanford.edu) 神经网络/稀疏自编码 教程（其中大部分现已在本架教程中，但练习作业仍旧在 CS294A 的课程网站上。）
* [[1]](http://www.naturalimagestatistics.net/) Natural Image Statistics book, Hyvarinen et al.  
** 这本书很长，您可跳过您熟悉的章节。** 重要章节: 5 (主成分分析与白化; 您可能已有所了解), 6 (稀疏编码), 7 (独立成分分析), 10 (ISA), 11 (TICA), 16 (temporal models).  
* [[2]](http://redwood.psych.cornell.edu/papers/olshausen_field_nature_1996.pdf) Olshausen and Field. Emergence of simple-cell receptive field properties by learning a sparse code for natural images Nature 1996. (稀疏编码)
* [[3]](http://www.cs.stanford.edu/~ang/papers/icml07-selftaughtlearning.pdf)  Rajat Raina, Alexis Battle, Honglak Lee, Benjamin Packer and Andrew Y. Ng. 自我学习：从未标记数据中迁移学习. ICML 2007


自动编码器: 
* [[4]](http://www.cs.toronto.edu/~hinton/science.pdf)  Hinton, G. E. and Salakhutdinov, R. R. 用神经网络对数据降维. Science 2006. 代码在 [这里]( http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html).
* [[5]](http://books.nips.cc/papers/files/nips19/NIPS2006_0739.pdf) Bengio, Y., Lamblin, P., Popovici, P., Larochelle, H. 神经网络的贪婪逐层训练. NIPS 2006 
* [[6]](http://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf) Pascal Vincent, Hugo Larochelle, Yoshua Bengio and Pierre-Antoine Manzagol. 用降噪自编码器提取合成出健壮特征 ICML 2008.  
**(他们有一个好模型，但然后合理化为一个概率模型。忽略向后合理化的概率模型[Section 4].)** 


深度学习有效性分析: 
* [[7]](http://www.cs.toronto.edu/~larocheh/publications/deep-nets-icml-07.pdf) H. Larochelle, D. Erhan, A. Courville, J. Bergstra, and Y. Bengio. 多因素变化下的深层结构问题的实证分析. ICML 2007.
**(Someone read this and let us know if this is worth keeping,. [Most model related material already covered by other papers, it seems not many impactful conclusions can be made from results, but can serve as reading for reinforcement for deep models])** 
* [[8]](http://www.jmlr.org/papers/volume11/erhan10a/erhan10a.pdf) Dumitru Erhan, Yoshua Bengio, Aaron Courville, Pierre-Antoine Manzagol, Pascal Vincent, and Samy Bengio. 为何非监督的预训练可帮助深度学习？ JMLR 2010  
* [[9]](http://cs.stanford.edu/~ang/papers/nips09-MeasuringInvariancesDeepNetworks.pdf) Ian J. Goodfellow, Quoc V. Le, Andrew M. Saxe, Honglak Lee and Andrew Y. Ng. 测量深度网络的不变性. NIPS 2009. 


径向基网络:
* [[10]](http://deeplearning.net/tutorial/rbm.html) 径向基网络教程. 
** 但请忽略 Theano 代码示例** 
(有人问我这条是否应该之后移除，虽对了解后来网络来说可能用处不大，但对于了解深度学习还是有用的。[看来还是留下比较好，对于不知道径向基网络的读者来说是一个很好的介绍，可以复现 Hinton 在 06 年时 Science 上的三通路径向基网络])


卷积网络:
* [[11]](http://deeplearning.net/tutorial/lenet.html) 卷积网络教程.
**但请忽略 Theano 代码.**


应用:
* 计算机视觉
** [[12]](http://www.ifp.illinois.edu/~jyang29/ScSPM.htm) Jianchao Yang, Kai Yu, Yihong Gong, Thomas Huang. 基于稀疏编码线性空间金字塔匹配的图像分类, CVPR 2009 ** 
[[13]](http://people.csail.mit.edu/torralba/publications/cvpr2008.pdf) A. Torralba, R. Fergus and Y. Weiss. Small codes and large image databases for recognition.  CVPR 2008.
* 语音识别
**[[14]](http://www.cs.stanford.edu/people/ang/papers/nips09-AudioConvolutionalDBN.pdf) 基于卷积深度置信网络无监督特征学习的语音识别, Honglak Lee, Yan Largman, Peter Pham and Andrew Y. Ng. In NIPS 2009.**


自然语言处理:
* [[15]](http://www.iro.umontreal.ca/~lisa/publications2/index.php/attachments/single/57) Yoshua Bengio, Réjean Ducharme, Pascal Vincent and Christian Jauvin, 一个神经概率语言模型. JMLR 2003.
* [[16]](http://ronan.collobert.com/pub/matos/2008_nlp_icml.pdf) R. Collobert and J. Weston. 自然语言处理的统一架构：多任务学习的深度神经网络. ICML 2008.
* [[17]](http://www.socher.org/uploads/Main/SocherPenningtonHuangNgManning_EMNLP2011.pdf) Richard Socher, Jeffrey Pennington, Eric Huang, Andrew Y. Ng, and Christopher D. Manning. 半监督递归编码器预测情绪的分布. EMNLP 2011
* [[18]](http://www.socher.org/uploads/Main/SocherHuangPenningtonNgManning_NIPS2011.pdf) Richard Socher, Eric Huang, Jeffrey Pennington, Andrew Y. Ng, and Christopher D. Manning. 基于动态池化和递归展开自动编码器的释义检测. NIPS 2011
* [[19]](http://www.cs.toronto.edu/~hinton/absps/threenew.pdf) Mnih, A. and Hinton, G. E. 统计语言建模新的三种图模型. ICML 2007


高阶内容:
* 慢特征分析:
* [[20]](http://itb.biologie.hu-berlin.de/~wiskott/Publications/BerkWisk2005c-SFAComplexCells-JoV.pdf) 基于慢特征分析生成一个复杂细胞的完整特性. Journal of Vision, 2005.
* 预测稀疏分解
* [[21]](http://cs.nyu.edu/~koray/publis/koray-psd-08.pdf) Koray Kavukcuoglu, Marc'Aurelio Ranzato, and Yann LeCun, "稀疏编码算法中的快速推理及其在目标识别中的应用", Computational and Biological Learning Lab, Courant Institute, NYU, 2008. 
* [[22]](http://cs.nyu.edu/~koray/publis/jarrett-iccv-09.pdf) Kevin Jarrett, Koray Kavukcuoglu, Marc'Aurelio Ranzato, and Yann LeCun, "最佳的多阶段目标识别体系结构是什么？", In ICCV 2009

均值与协方差联合模型
* [[23]](http://www.cs.toronto.edu/~ranzato/publications/ranzato_aistats2010.pdf) M. Ranzato, A. Krizhevsky, G. Hinton. 考虑三路受限玻尔兹曼机模型的自然图像建模. In AISTATS 2010.
* [[24]](http://www.cs.toronto.edu/~ranzato/publications/ranzato_cvpr2010.pdf) M. Ranzato, G. Hinton, 基于第三阶受限玻尔兹曼机分解的像素均值和协方差建模. CVPR 2010 
**(someone and tell us if you need to read the 3-way RBM paper before the mcRBM one [我认为没必要, 实际上 CVPR 论文更易理解.])**
* [[25]](http://www.cs.toronto.edu/~hinton/absps/mcphone.pdf) Dahl, G., Ranzato, M., Mohamed, A. and Hinton, G. E. 基于均值协方差的受限玻尔兹曼机的电话语音识别. NIPS 2010.
* [[26]](http://www.nature.com/nature/journal/v457/n7225/pdf/nature07481.pdf) Y. Karklin and M. S. Lewicki, 复杂细胞属性在自然场景中的学习到泛化, Nature, 2008.
**(someone tell us if this should be here.  Interesting algorithm + nice visualizations, though maybe slightly hard to understand. [seems a good reminder there are other existing models])**


概述
* [[27]](http://www.iro.umontreal.ca/~bengioy/papers/ftml_book.pdf) Yoshua Bengio. 为人工智能学习深度架构. FTML 2009. 
**(该领域更宽视角的描述, 但技术细节无需深究. 当您已经读完该领域的一些文章后就会发现很好理解.)**


实战指导:
* [[28]](http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf) Geoff Hinton. 训练受限玻尔兹曼机的指导. UTML TR 2010–003. 
**这是一篇实战指导 (如果您尝试实现受限玻尔兹曼机不妨一读，但若不是请跳过因为这不是一篇教程).**
* [[29]](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) Y. LeCun, L. Bottou, G. Orr and K. Muller. 高效反向传播. 神经网络: 诀窍, Springer, 1998
**如果您尝试实现反向传播; 否这不建议阅读**


其它:
* [[30]](http://www.eecs.umich.edu/~honglak/teaching/eecs598/schedule.html) Honglak Lee 的课程
* [[31]](http://www.cs.toronto.edu/~hinton/deeprefs.html) 来自 Geoff 的教程