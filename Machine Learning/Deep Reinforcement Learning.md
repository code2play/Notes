# Deep Reinforcement Learning

## Notations

- $s$: state
- $o$: observation
- $a$: action
-  $\tau$: trajectory $s_1, a_1, \ldots, s_T, a_T$
-  $\theta$: parameter
- $\pi_\theta(a_t|s_t)$: policy respect to $\theta$

## Imitation Learning

模仿学习是指从示教者提供的范例中学习，一般提供人类专家的决策数据 $\{\tau_1, \tau_2, \ldots, \tau_m\}$，每个决策包含状态和动作序列$\tau_i = <s_1^i,a_1^i,s_2^i,a_2^i,\ldots,s_{n_ni+1}^i>$，将所有「状态-动作对」抽取出来构造新的集合 $\mathcal{D}=\left\{ (s_1,a_1),(s_2,a_2),(s_3,a_3),\ldots \right\}$。

之后就可以把状态作为特征（feature），动作作为标记（label）进行分类（对于离散动作）或回归（对于连续动作）的学习从而得到最优策略模型。模型的训练目标是使模型生成的状态-动作轨迹分布和输入的轨迹分布相匹配。

但是由于复合误差（compounding errors），训练结果往往有较大偏差。

<img src="assets/compounding errors.jpg" style="width:600px">

有两个原因使得我们经常不能很好地学习专家行为：

1. 非 Markov 行为，意思是专家所做的行为可能不完全依赖于当前的观测。一个解决方法是根据之前所有的观测决定动作，将 $\pi_\theta(a_t|o_t)$ 变成 $\pi_\theta(\mathbf{a}_t|\mathbf{o}_1,\ldots,\mathbf{o}_t)$ ，可以使用 RNN (LSTM) 实现。
2. 多峰 (multimodal) 行为。解决方法：
   1. 高斯分布的混合
   2. Implicit Density Model
   3. Autoregressive Discretization

### Data Augmentation

生成更多数据以减少误差。

### DAgger (Dataset Aggregation)

让 $p_{\pi_\theta}(\mathbf{o}_t)$ 去接近 $p_{\mathrm{data}}(\mathbf{o}_t)$ 比较困难，倒不如转头去对 $p_{\mathrm{data}}(\mathbf{o}_t)$ 做点手脚让它能贴近 $p_{\pi_\theta}(\mathbf{o}_t)$

一个简化版本的DAgger算法是这样的：

1. 从人工提供的数据集 $\mathcal{D}=\{\mathbf{o}_1,\mathbf{a}_1,\mathbf{o}_2,\mathbf{a}_2,\ldots,\mathbf{o}_N,\mathbf{a}_N\}$ 中训练出策略 $\pi_\theta(\mathbf{a}_t|\mathbf{o}_t)$
2. 运行策略 $\pi_\theta(\mathbf{a}_t|\mathbf{o}_t)$ 来获得一个新的数据集 $\mathcal{D}_\pi=\{\mathbf{o}_1,\ldots,\mathbf{o}_M\}$
3. 人工来对数据集 $\mathcal{D}_\pi$ 进行标注，得到一系列 $\mathbf{a}_t$
4. 合并数据集，$\mathcal{D}\leftarrow\mathcal{D}\cup\mathcal{D}_{\pi}$，返回第一步

该算法的难点在第三步需要人为打标记。可以使用更复杂的 LQR 算法自动打标记。

## Markov Chain

$$
\mathcal{M} = \{\mathcal{S}, \mathcal{T}\} \\
\mathcal{S} - \text{state space} \\
\mathcal{T} - \text{transition operator} 
$$

state $s \in \mathcal{S}$ (discrete or continuous)

let 
$$
\mu_{t,i} = p(s_t=i) \\
\mathcal{T}_{i,j} = p(s_{t+1}=i|s_t=j)
$$
then
$$
\overrightarrow{\mu}_{t+1} = p(s_{t+1}=i) = \sum_{j=1}^{\mathcal{S}} p(s_{t+1}=i|s_t=j)p(s_t=j) =\mathcal{T}\overrightarrow{\mu}_t
$$

## Markov Decision Process (MDP)

$$
\mathcal{M} = \{\mathcal{S}, \mathcal{A}, \mathcal{T},r\} \\
\mathcal{S} - \text{state space} \\
\mathcal{A} - \text{action space} \\
\mathcal{T} - \text{transition operator} \\
r - \text{reward function } r:\mathcal{S}\times\mathcal{A}\to \mathbb{R}
$$

action $a \in \mathcal{A}$ (discrete or continuous),  $r(s_t, a_t) $ is the reward

let
$$
\mu_{t,j} = p(s_t=j) \\
\xi_{t,k} = p(a_t=k) \\
\mathcal{T}_{i,j,k} = p(s_{t+1}=i|s_t=j,a_t=k)
$$
then
$$
\mu_{t+1,i} = \sum_{j,k}\mathcal{T}_{i,j,k}\mu_{t,j}\xi_{t,k}
$$


## Partially Observable Markov Decision Process (POMDP)

$$
\mathcal{M} = \{\mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{T}, \mathcal{E}, r\} \\
\mathcal{S} - \text{state space} \\
\mathcal{A} - \text{action space} \\
\mathcal{O} - \text{observation space} \\
\mathcal{T} - \text{transition operator} \\
\mathcal{E} - \text{emission probability } p(o_t|s_t) \\
r - \text{reward function } r:\mathcal{S}\times\mathcal{A}\to \mathbb{R}
$$

部分可观察的马尔可夫决策过程（POMDP）是MDP的泛化。在POMDP模型中，系统（这里的系统可以用具体的机器人系统来代替）的动态响应和MDP类似（如状态转移矩阵），但是系统并不能直接观测到当前的状态，就是说系统不确定自己现在处于哪个状态。所以，系统需要对环境做一个感知，来确定自己处于哪个状态。 

## Reinforcement Learning

### Goal of RL

#### Finite Horizon

$$
\underbrace{p_{\theta}(s_1, a_1, \ldots, s_T, a_T)}_{\pi_\theta(\tau)} = p(s_1) \prod_{t=1}^T \pi_\theta(a_t|s_t)p(s_{t+1}|s_t,a_t) \\
\theta^* = \arg \max_{\theta} E_{\tau \sim p_\theta(\tau)} \biggl[\sum r(s_t,a_t) \biggr] = \arg \max_\theta \sum_{t=1}^T E_{(s_t,a_t)\sim p_\theta(s_t,a_t)}[r(s_t,a_t)]
$$

其中 $p_\theta(s_t, a_t)$ state-action marginal

由于 $p((s_{t+1},a_{t+1})|(s_t,a_t)) = p(s_{t+1}|s_t,a_t)\pi_\theta(a_{t+1}|s_{t+1})$ ，$\pi_\theta(\tau)$ 可看作 (s,a) 的马尔可夫链

#### Infinite Horizon

$\mathcal{T}\to \infin​$ 时，若 $t+1​$ 和 $t​$ 时刻状态相同则
$$
\mu = \mathcal{T}\mu \\
(\mathcal{T}-\bold{I})\mu = 0
$$
即 $\mu$ 是 $\mathcal{T}$ 特征值 1 对应的特征向量。$\mu$ 最终变为平稳分布（stationary distribution）$ \mu = p_\theta(s,a)$。 此时
$$
\theta^* = \arg\max_\theta \frac{1}{T} E_{\tau \sim p_\theta(\tau)} \biggl[\sum r(s_t,a_t) \biggr] = \arg\max_\theta E_{(s,a)\sim p_\theta(s,a)}[r(s,a)]
$$

### Structure of RL Algorithms

<img src="assets/RL.jpg" style="width:500px">

### Q Function & Value Function

- Q Function

$$
Q^\pi(s_t, a_t) = \sum_{t'=t}^TE_{\pi_\theta}[r(s_{t'}, a_{t'})|s_t, a_t]
$$

​	表示在状态 $s_t$ 执行行动 $a_t$ 后根据策略 $\pi$ 未来总收益的条件期望

- Value Function

$$
V^\pi(s_t) = \sum_{t'=t}^TE_{\pi_\theta}[r(s_{t'}, a_{t'})|s_t] = E_{a_t \sim \pi(a_t|s_t)}[Q^\pi(s_t,a_t)]
$$

​	表示状态 $s_t$ 后根据策略 $\pi$ 未来总收益的条件期望

- Usage
  - 目标函数可重写为 $E_{s_1\sim p(s_1)}[V^\pi (s_1)]$
  - 如果我们现在有一个策略 $\pi$，且我们知道 $Q^\pi(s,a)$，那么我们可以构造一个新的策略 $\pi'(a|s) = 1 \text{ if } a=\arg\max_a Q^\pi(s,a)$，这个策略至少和 $\pi$ 一样好（且可能更好），是因为这一个策略最大化未来的收益。这一点与当前的 $\pi$ 是什么没有关系
  - 我们可以增加“好的行动”发生的概率。注意到，$V^\pi(s) = E[Q^\pi(s,a)]$ 代表了在策略 $\pi(a|s)$ 下的行动平均水平，所以如果 $Q^\pi(s,a)>V^\pi(s)$，就能说明 $a$ 是高于平均水平的行动。那么我们便可以改动策略，使得这样的行动发生的概率上升


### Types of RL Algorithms

- **策略梯度法 (policy gradient) **：这类算法直接对目标函数关于参数求梯度。本质是一阶最优化算法，求解无约束优化问题的通用方法
- **值函数方法**：这类方法尝试去近似估计**最优策略下的**值函数或Q函数，而并不揣测策略函数是什么。注意此时策略并需要不显式表达出来，只需要选择使得Q函数最大的行动即可（或者值函数类似于动态规划中的手段）
- **演员-评论家 (actor-critic) 方法**：这类方法尝试去近似估计**当前策略下的**值函数或Q函数，并用这个信息求一个策略的梯度，改进当前的策略。所以也可以看作是策略梯度法和值函数方法的一个混合体
- **基于模型 (model-based) 的增强学习方法**与上面的几类都不同。它需要去**估计转移概率**来作为模型，描述物理现象或者其他的系统动态。有了模型以后，可以做很多事情。譬如可以做行动的安排（不需要显式的策略），可以去计算梯度改进策略，也可以结合一些模拟或使用动态规划来进行无模型训练


### Tradeoff

- off-policy: 可以在不用现在的策略去生成新样本的情况下，就能改进我们的策略。其实就是能够使用其他策略生成过的历史遗留数据来使得现在的策略更优
- on-policy: 算法指的是每次策略被更改，即便只改动了一点点，也需要去生成新的样本。在线算法用于梯度法通常会有一些问题，因为梯度算法经常会走梯度步，些许调整策略就得生成大量样本

<img src="assets/RL tradeoff.jpg" style="width:800px">

**Note**: efficient $\ne$ running time

## Policy gradients

