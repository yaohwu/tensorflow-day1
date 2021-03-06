# 机器学习

## 理论
### 背景：

从AI（Artificial Intelligence）研究中兴起，对于一些直接编码做不到的事情，通过学习型的算法来间接实现。

### 关系：

- 人工智能：为机器赋予人的智能；
- 机器学习：一种实现人工智能的方式；
- 深度学习：一种实现机器学习的技术（神经网络）。

![ai_ml_dl](https://github.com/yaohwu/tensorflow-day1/blob/master/ai_ml_dl.png)

### 分类：

##### 监督学习：
是用正确答案已知的数据来进行模型训练，也就是用标记过的数据，有标签的数据。
例如：给出一堆照片，明确知道照片上是什么，训练之后可以照片上的物体（分类问题）；
给出房价和时间的数据，训练之后用于预测房价（回归问题）。

分类（有监督）问题：类别化的目标参数（标签），离散的输出，寻找的是决策边界；
回归问题：连续的目标参数（标签），连续的输出，寻找的是最优拟合。


1. 数据生成和分类；
生成训练集和验证集，训练集用于模型训练，验证集用于模型验证和评价；
2. 训练；
3. 验证和优化模型；
验证很好理解，就是输入验证集看输出结果是否准确；模型优化就是调整一些参数，例如 模型训练时训练层级、训练快慢等。
4. 应用。

对数据集进行标记的成本是非常高的。因此，必须确保使用网络得到的收益比标记数据和训练模型的消耗要更高。

##### 无监督学习：
使用未被标记的数据进行学习，不知道输入数据对应的输出结果是什么。无监督学习只能默默的读取数据，自己寻找数据的模型和规律。
例如：聚类（未知照片分类）和异常检测（一个新开的银行账户转了一大笔钱？你的账户经常是在上海登录的，却突然在西安上线了？）。


##### 半监督学习：

半监督学习训练中使用的数据，只有一小部分是标记过的，而大部分是没有标记的。因此和监督学习相比，半监督学习的成本较低，但是又能达到较高的准确度。



##### 增强学习：

强化学习也是使用未标记的数据，但是可以通过某种方法知道你是离正确答案越来越近还是越来越远（即奖惩函数）；
增强学习算法的目标就是使奖惩函数回报最大化，这个回报也可以说是一种延迟的标签。

例如：Flappy bird，Mario，AlphaZero。



#### TF DEMO

#### tensor flow 综述

是一个使用数据流图进行数值计算的开源软件库。图中的节点代表数学运算， 而图中的边则代表在这些节点之间传递的多维数组（tensor张量）。


#### code
