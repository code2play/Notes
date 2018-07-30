# 协同过滤

## 相似度计算

相似度计算是协同过滤中核心的步骤，对于向量$\bold{a}$和$\bold{b}$可以用以下方法计算其相似度：

### 欧式距离

$$
sim = ||\bold{a}-\bold{b}||
$$

### 余弦相似度

$$
sim = \frac{\bold{a}\cdot \bold{b}}{||\bold{a}||\cdot||\bold{b}||}
$$

### 皮尔逊相关系数

$$
sim = \frac{cov(\bold{a}, \bold{b})}{\sigma_{\bold{a}}\sigma_{\bold{b}}} \\
cov(\bold{a}, \bold{b}) = E[(\bold{a}-\mu_{\bold{a}})(\bold{b}-\mu_{\bold{b}})]
$$

### Jaccard相似度

如果向量$\bold{a}, \bold{b}$的值只有0和1，可将其看作两个集和$A, B$
$$
sim = \frac{|A \cap B|}{|A \cup B|}
$$

## User-Based CF

对于用户$a$没有评价过而用户$b$评价过的物品$i$，可通过如下方法预测$a$对$i$的评分：
$$
\bold{a}_i = \frac{\sum_{b=1}^n sim(\bold{a}, \bold{b}) \cdot \bold{b}_i}{\sum_{b=1}^n sim(\bold{a}, \bold{b})}
$$
然后将预测评分较高的推荐给该用户。

## Item-Based CF



## Content-Based CF



## 评价指标







