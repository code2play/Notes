# Spark

## Introduction

- 最大的误区：Spark替代Hadoop
- Spark借鉴MR发展而来，继承分布式并行 计算优点并改进了MR缺陷
- MR计算结果需落地磁盘，Spark中间数据放在内存中
- MR只提供Map和Reduce两种操作， Spark提供了Transformations和Actions两大类操作

## 体系结构

- Client 可以在集群内也可以在集群外
- 1个 application 包含1个 driver 和多个 executor
- 1个 executor 里面运行的 tasks 都属于同1个application

## 执行计划

与MR的不同之处：

- MR中，用户直接面对 task，mapper 和 reducer 的职责分明：一个进行分块处理， 一个进行 aggregate
- MR中数据依赖和task执行流程是一致且固定 的，只需实现 map 和 reduce函数即可
- Spark中数据依赖可以非常灵活
- Spark将数据依赖和具体 task的执行流程分 开，通过将逻辑执行计划转换成物理执行计划

执行计划：

- 首先建立逻辑执行计划（RDD依赖关系）
  - 逻辑执行计划描述了数据流：经过哪些 transformation，中间生成哪些 RDD 及 RDD 之间的依赖关系
  - 逻辑执行计划表示数据上的依赖关系，不是task的执行流程
  - 从数据源读取数据创建最初的 RDD，如本地 file、 内存数据结构（parallelize）、 HDFS、HBase等
  - 对 RDD 进行一系列的 transformation操作
  - 对final RDD 进行 action 操作，每个 partition 计算 后产生结果 result
  - 将 result 回送到 driver 端，进行最后的 f(list[result]) 计算
    - 如count() 实际包含了action() 和 sum() 两步
- 把逻辑执行计划转换为物理执行计划（划分 Job，形成Stage和Task）
- 执行Task