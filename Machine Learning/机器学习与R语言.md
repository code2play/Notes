# R语言知识点

## 科学计算

### 特殊运算符

* 整除：%/%
* 取模：%%
* 幂：^
* 矩阵乘法：%*%

### 特殊数字

* 无穷：Inf
* 缺失值：NA
* 非数字：NaN

## 常用函数

> https://www.douban.com/note/244258549/

## 一些感觉比较重要的课后问题

* 如何检查变量x是否等于pi？
  * `all.equal(x, pi)`
* `all.equal()`和`identical()`的区别
  * `all.equal`是近似相等，`identical`是完全相等
* 给变量赋值的方法
  * `=, <-, assign()`
* 数字的三个内置类的名称是什么？
  * `integer, double, numeric`
* 指出至少三个用于检视变量内容的函数
  * `summary(), str(), View()`
* 如何删除用户工作区中的所有变量？
  * `rm(list=ls())`
* 描述两种命名向量元素的方式
  * 给向量的names赋值或在创建向量时给元素命名
* 向量索引中的四种类型是什么？
  * 正整数、负整数、逻辑向量、元素名称向量
* 尽可能多地说出的几种创建数据框子集的方法
  * 下标`[]`或`[[]]`、美元符号`$`、`subset()`
* 如何创建一个数据框，使得它的列名既非唯一又非有效？
  * 设置`check.names=FALSE`
* 你会使用哪个函数将一个数据框追加到另一个之后？
  * `rbind()`
* 全局环境的另一个名字是什么？
  * 用户工作区
* 列举三个能够输出函数形参名称的函数
  * `formals(), args(), formalArgs()`
* 尽可能多地指出用于格式化数字的函数
  * `formatC(), sprintf(), format(), prettyNum()`
* 如何把数字变量转换成类别变量？
  * 利用`cut()`
* 如果为if传入条件NA会发生什么？
  * 报错
* 如果给ifelse传入条件NA会发生什么？
  * 返回NA
* 什么类型的变量可以作为switch函数的第一个参数传入？
  * 整数、字符串
* 列出尽可能多的apply函数家族中的成员函数
  * `apply, lapply, vapply, sapply, tapply, mapply, rapply`
* lapply、vapply和sapply之间的区别是什么？
  * lapply返回列表，vapply可以控制输出格式，sapply为vapply的简化版，能自动调整输出格式
* 在plyr包中，**ply中的星号意味着什么？
  * 输入格式和输出格式

# 机器学习过程
1. 数据输入
2. 抽象化(训练模型？)
3. 一般化(应用模型？)

# 将机器学习应用于数据中的步骤
1. 收集数据
2. 探索和准备数据
3. 基于数据训练模型
4. 评价模型
5. 改进模型

# 选择机器学习算法
1. 考虑输入的数据
2. 考虑机器学习算法的类型
3. 为数据匹配合适的算法

# 预处理
## Shuffle
```R
data_rand <- data[order(runif(size)), ]
```

## 归一化 (Min-Max)
```R
normalize <- function(x) {
    return((x - min(x)) / (max(x) - min(x)))
}
lapply(x, normalize)
```

## 标准化 (Z-score)
```R
scale(x, center = TRUE, scale = TRUE)
# center=TRUE 将一组数中每个数减去平均值
# scale=TRUE 将一组数中每个数除以标准差
```

## 语料库
```R
library(tm)
corpus <- Corpus(VectorSource(text))
# Corpus()创建了一个R对象来存储文本文档
# VectorSource()来指示函数Corpus()使用向量text的信息
```

# Algorithm
## KNN
### Pros
+ 简单有效
+ 对数据分布没有要求
+ 训练快
### Cons
- 不产生模型，在发现特征之间关系上的能力有限
- 分类阶段很慢、需要大量的内存
- 名义变量（特征）和缺失数据需要额外处理
### Code
```R
library(class)
pred <- knn(train, test, cl, k = 1, l = 0)
# cl 训练集label
# l 决定种类的最小票数
```
### Note
* 基于近邻方法的分类算法被认为是懒惰学习算法，因为从技术上来说，没有抽象化的步骤。抽象过程和一般化过程都被跳过了，这就破坏了前面给出的学习的定义。

## Naive Bayes
### Pros
+ 简单、快速、有效
+ 能处理好噪声数据和缺失数据
+ 需要用来训练的例子相对较少，但同样能处理好大量的例子
+ 很容易获得一个预测的概率估计值
### Cons
- 依赖于一个常用的错误假设，即一样的重要性和独立特征
- 应用在含有大量数值特征的数据集时并不理想
- 概率的估计值相对于预测的类而言更加不可靠
### Code
```R
library(e1071)
model <- naiveBayes(formula, data, laplace = 0, ...)
# or
model <- naiveBayes(x, y, laplace = 0, ...)
pred <- predict(model, test)
```
### Note
* 拉普拉斯估计：假设已经有两次试验，一次成功和一次失败。即分子加1分母加2。

## Decision Tree
### Pros
+ 适用于大多数问题，比其他复杂模型更有效
+ 高度自动化的学习过程，可以处理数值型数据、名义特征以及缺失数据
+ 只能使用最重要的特征
+ 可以用于只有相对较少的训练案例的数据或者有相当多的训练案例的数据
+ 树较小的情况下，模型容易解释
### Cons
- 在根据具有大量水平的特征进行分类时往往是有偏的
- 很容易过拟合或欠拟合
- 依赖于轴平行分割，所以在对一些关系建立模型时会有困难
- 训练数据中的小变化可能会导致决策树逻辑的较大变化
- 大的决策树可能很难理解，给出的决策可能违反直觉
### Code
```R
library(C50)
model <- C5.0(x, y, trail = 1, cost = NULL)
# trail boosting迭代次数
summary(model)  # 可以看到树的结构
pred <- predict(model, test, type)
# type为‘class’或‘prob’返回类别或概率
```
### Note
* 预剪枝：一旦决策树达到一定数量的决策，或者决策节点仅含有少量的案例，我们就停止树的增长
    * 优点：避免了做不必要的工作
    * 缺点：没有办法知道决策树是否会错过细微但很重要的模式，这种细微的模式只有决策树生长到足够大时才能学习到
* 后剪枝：如果一棵决策树生长得过大，就根据节点处的错误率使用修剪准则将决策树减小到更合适的大小。
    * 优点：比预剪枝法更有效

## Rule Learning
### *OneR*
#### Pros
+ 可以生成一个单一的、易于理解的、人类可读的经验法则(大拇指规则？)
+ 往往表现得出奇的好
+ 可以作为更复杂算法的一个基准
#### Cons
- 只使用了单一特征
- 可能会过于简单
### *RIPPER (Repeated Incremental Pruning to Produce Error Reduction)*
#### Pros
+ 生成易于理解的、人类可读的规则
+ 对大数据集和噪声数据集有效
+ 通常比决策树产生的模型更简单
#### Cons
- 可能会导致违反常理或者专家知识的规则
- 处理数值型数据不太理想
- 性能有可能不如更复杂的模型
### Code
```R
library(RWeka)
# 1R分类器，能够识别对于目标类最具有预测性的单一特征
oneR <- OneR(formula, data)
summary(oneR)
# JRip()是基于Java实现的RIPPER规则学习算法
jrip <- JRip(formula, data)
summary(jrip)
pred <- predict(jrip, test)
```

## Linear Regression
### Pros
+ 是数值型数据建模最常用的方法
+ 适用于几乎所有数值型数据
+ 提供了特征(变量)与结果之间关系的强度与大小的估计
### Cons
- 对数据做出了很强的假设
- 该模型的形式必须由使用者事先指定
- 不能很好地处理缺失数据
- 只能处理数值特征，所以分类数据需要额外的处理
- 需要一些统计知识来理解模型
### Code
```R
model <- lm(formula, data)
pred <- predict(model, test)
```

## 回归树 (CART: Classification and Regression Trees)
### Pros
+ 将决策树的优点与对数值型数据建立模型的能力相结合
+ 能自动选择特征，允许该方法与大量特征一起使用
+ 不需要使用者事先指定模型
+ 拟合某些类型的数据可能会比线性回归好得多
+ 不要求用统计的知识来解释模型
### Cons
- 不像线性回归那样常用
- 需要大量的训练数据
- 难以确定单个特征对于结果的总体净影响
- 可能比回归模型更难解释
### Code
```R
library(rpart)
m.rpart <- rpart(formula, data)
p.rpart <- predict(model, test)
# 可视化决策树
library(rpart.plot)
rpart.plot(m.rpart, digits = 4, fallen.leaves = TRUE, type = 3, extra = 101)
```

## 模型树
### Code
```R
library(RWeka)
m.m5p <- M5P(quality ~ ., data = wine_train)
summary(m.m5p)
p.m5p <- predict(m.m5p, wine_test)
```

## Neural Network
### Pros
+ 适用于分类和数值预测问题
+ 属于最精确的建模方法
+ 对数据的基本关系几乎不需要做出假设
### Cons
- 计算量大，训练缓慢，特别是在网络拓扑结构复杂的情况下
- 很容易过拟合或欠拟合
- 结果可能不好解释
### Code
```R
NN <- neuralnet(formula, data, hidden)
plot(NN)
pred <- compute(NN, test)$net.result
```

## SVM (Support Vector Machine)
### Pros
+ 可用于分类或者预测
+ 不会过多地受到噪声数据的影响，且不容易出现过拟合
+ 准确度高
+ 比神经网络容易使用
### Cons
- 寻找最好的模型需要测试不同的和函数和参数组合
- 训练缓慢，尤其是数据集具有大量特征或案例时
- 结果不好解释
### Code
```R
library(kernlab)
svm <- ksvm(formula, data, kernel)
# rbfdot Radial Basis kernel "Gaussian"
# polydot Polynomial kernel
# vanilladot Linear kernel
pred <- predict(svm, test)
```

## Apriori
### Pros
+ 非常适合处理极其大量的事务性数据
+ 规则中的结果很容易理解
+ 对于“数据挖掘”和发现数据库中的意外的知识很有用
### Cons
- 对于小的数据集不是很有帮助
- 需要努力地将对数据的洞察和常识区分开
- 很容易从随机模式中得出虚假的结论
### Code
```R
library(arules)
rules <- apriori(groceries, parameter = list(support, confidence, minlen))
```
### Note
* 支持度：$support(X) = count(X)/N$
* 置信度：$confidence(X->Y) = support(X,Y)/support(X)$

## K-Means
### Pros
+ 使用简单的原则来确定可以用非统计术语解释的类
+ 有较高的灵活性，并且通过简单的整理就可以修正以克服其所有缺点
+ 有效且运行良好
### Cons
- 有些过于简单
- 随机初始化，因此不确保能找到最佳的类
- 需要合理猜测数据有多少个自然类
### Code
```R
clusters <- kmeans(data, k) 
clusters$size
clusters$centers
clusters$cluster
```

# 模型评估
## CorssTable (F1 score)
```R
library(gmodels)
CrossTable(x=test_labels, y=test_pred, prop.chisq=FALSE, prop.c = FALSE, prop.r = FALSE)
# prop.chisq=FALSE 不显示卡方值
# prop.c和prop.r为FALSE删除表中列与行的百分比
```

## 相关系数
```R
cor(pred, label)
```

## MAE (Mean Absolute Error)
```R
MAE <- function(actual, predicted) {
    mean(abs(actual - predicted))  
}
MAE(label, pred)
```

# 数据可视化
## Word Cloud
```R
library(wordcloud)
wordcloud(words, freq, scale=c(4,.5), min.freq=3, max.words=Inf,
    random.order=TRUE, random.color=FALSE)
# scale 调整词云中单词的最大字体和最小字体
```

## 相关系数矩阵
```R
cor(data[c(...)])
corrgram(x,order=, lower.panel= , upper.panel=,text.panel=,diag.panel=,…)
```

## 散点图矩阵
```R
pairs(data[c(...)])
library(psych)
pairs.panels(data[c(...)])
```

## 散点图
```R
plot(x, y, color)
points(x, y, color)
```

## 向日葵散点图
```R
sunflowerplot(x,y=,number, rotate=,size=,seg.col=,seg.lwd=,...)
```

## 条形图
```R
barplot(height, beside =, horiz =, , ...)
```

## 直方图
```R
hist(x, breaks, freq=TRUE, probability=FALSE)
# breaks 分段区间
# freq 是否以频数作图
# probability 是否以概率密度作图
```

## 箱型图
```R
boxplot(x, ..., range=, width=, varwidth=FALSE, notch=FALSE, names=, horizontal=FALSE, add=FALSE, ...)
boxplot(formula, data, ..., subset, na.action = NULL) 
```

## 小提琴图
```R
vioplot(x,...,range=,horizontal=,border="black",rectCol="black",colMed="white", pchMed=19,add=FALSE,wex=1,drawRect=TRUE) 
```

## Cleveland点图
```R
dotchart(x, labels = NULL,...)
```

## 饼图
```R
dotchart(x, labels = NULL,...)
```

## 马赛克图
```R
mosaicplot(x, dir = NULL, type = c("pearson", "deviance", "FT"), ...) 或   mosaicplot(formula, data =, ...)
```

## 星状图
```R
stars(x,full=TRUE,scale=,radius=,labels=,locations=,len=,key.loc=,key.labels=,key.xpd=,flip.labels=,draw.segments=,col.segments=,col.stars=, frame.plot=,...)
```

## 等高图
```R
contour(x=,y=,z,nlevels=,levels=,labels= ,method=,...)
```

## 热图
```R
heatmap(x,Rowv=,Colv=,distfun=,hclustfun=,scale=c() ,...)
```