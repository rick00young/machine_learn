#对于LSA、PLSA与LDA的关系的理解

3个模型就我的理解都是计算那2个矩阵，一个为word对应topic的矩阵，一个为topic对应doc的矩阵，只是计算方式不一样，2个矩阵的呈现方式也不一样。

* LSA采用暴力SVD矩阵分解，如果维数大了，矩阵大了，没法计算而已。
* PLSA把LSA变成从概率的角度理解，把LSA的2个矩阵做归一化处理后，就可以看成PLSA的word对于topic的概率分布和文档对于topic的概率分布，采用的是EM的方式学习，先随机初始化这2个分布，计算其后验概率，然后反过来利用较大似然又来计算这2个分布，不断迭代直到收敛。可计算性方面大大提高，但是也很繁琐。
* LDA在PLSA的基础上，利用贝叶斯估计，引入先验分布，相当于多了一个条件，而后采用Gibbs sampling的方式来学习参数，收敛后计算得到word对于topic的概率分布和文档对于topic的概率分布。