# Hadoop

## Introduction

大数据技术特点：**健壮、可扩展、简单方便**

大数据技术主要应用场景：

- 搜索引擎，为大规模的网页快速建立索引。
- 大数据存储，利用分布式存储能力，建立数据备份、数据仓库等。
- 大数据处理，利用分布式处理能力，例如数据挖掘、数据分析等。

**Hadoop** 框架的核心是 ***HDFS, MapReduce, Yarn***
- **HDFS** 是分布式文件系统，提供海量数据的存储
- **MapReduce** 是分布式数据处理模型，提供数据计算
- **Yarn **是资源管理和调度工具

Hadoop安装模式：

- 单机模式：
  - 占用资源最少的模式
  - 完全运行在本地
  - 不使用Hadoop文件系统
  - 不加载任何守护进程
- 伪分布模式：
  - “单节点集群”模式
  - 所有的守护进程都运行在同一台机子上
  - 代码调试
  - 可以查看HDFS的输入/输出，以及各守护进程
- 全分布模式：真正的分布式集群配置，用于生产环境

### 其他大数据工具

- [100个工具就问你怕不怕](http://www.bigdatas.cn/portal.php?mod=view&aid=1936)

- **Flume**

  Cloudera提供的一个高可用的、高可靠的、分布式的海量日志采集、聚合和传输的系统。Flume支持在日志系统中定制各类数据发送方，用于收集数据。同时，Flume支持对数据进行简单处理，并写入各种数据接受方（可定制）。

- **Spark**

  一个高速、通用大数据计算处理引擎。拥有Hadoop MapReduce所具有的优点，但不同的是Job的中间输出结果可以保存在内存中，从而不再需要读写HDFS，因此Spark能更好地适用于数据挖掘与机器学习等需要迭代的MapReduce的算法。它可以与Hadoop和Apache Mesos一起使用，也可以独立使用。

- **Hive**

  是基于Hadoop的一个数据仓库工具，可以将结构化的数据文件映射为一张数据库表，并提供简单的sql查询功能，可以将sql语句转换为MapReduce任务进行运行。 其优点是学习成本低，可以通过类SQL语句快速实现简单的MapReduce统计，不必开发专门的MapReduce应用，十分适合数据仓库的统计分析。

- **Shark**

  即Hive on Spark，本质上是通过Hive的HQL解析，把HQL翻译成Spark上的RDD操作，然后通过Hive的metadata获取数据库里的表信息，实际HDFS上的数据和文件，会由Shark获取并放到Spark上运算。Shark的特点就是快，完全兼容Hive，且可以在shell模式下使用rdd2sql()这样的API，把HQL得到的结果集，继续在scala环境下运算，支持自己编写简单的机器学习或简单分析处理函数，对HQL结果进一步分析计算。

- **SparkSQL**

  前身是Shark，SparkSQL抛弃原有Shark的代码并汲取了一些优点，如内存列存储（In-Memory Columnar Storage）、Hive兼容性等。由于摆脱了对Hive的依赖性，SparkSQL无论在数据兼容、性能优化、组件扩展方面都得到了极大的方便。

- **HBase**

  是Hadoop的数据库，一个分布式、可扩展、大数据的存储。是为有数十亿行和数百万列的超大表设计的，是一种分布式数据库，可以对大数据进行随机性的实时读取/写入访问。提供类似谷歌Bigtable的存储能力，基于Hadoop和Hadoop分布式文件系统（HDFS）而建。

- **Azkaban**

  一款基于Java编写的任务调度系统任务调度，来自LinkedIn公司，用于管理他们的Hadoop批处理工作流。Azkaban根据工作的依赖性进行排序，提供友好的Web用户界面来维护和跟踪用户的工作流程。

- **Kafka**

  一种高吞吐量的分布式发布订阅消息系统，它可以处理消费者规模网站中的所有动作流数据，目前已成为大数据系统在异步和分布式消息之间的最佳选择。

- **ZooKeeper**

  一个分布式的应用程序协调服务，是Hadoop和Hbase的重要组件。它是一个为分布式应用提供一致性服务的工具，让Hadoop集群里面的节点可以彼此协调。ZooKeeper现在已经成为了 Apache的顶级项目，为分布式系统提供了高效可靠且易于使用的协同服务。

### 集群管理工具

- **YARN**

  一种新的Hadoop资源管理器，它是一个通用资源管理系统，可为上层应用提供统一的资源管理和调度，解决了旧MapReduce框架的性能瓶颈。它的基本思想是把资源管理和作业调度/监控的功能分割到单独的守护进程。

- **Mesos**

  由加州大学伯克利分校的AMPLab首先开发的一款开源群集管理软件，支持Hadoop、ElasticSearch、Spark、Storm 和Kafka等架构。对数据中心而言它就像一个单一的资源池，从物理或虚拟机器中抽离了CPU，内存，存储以及其它计算资源， 很容易建立和有效运行具备容错性和弹性的分布式系统。

- **Ambari**

  作为Hadoop生态系统的一部分，提供了基于Web的直观界面，可用于配置、管理和监控Hadoop集群。目前已支持大多数Hadoop组件，包括HDFS、MapReduce、Hive、Pig、 Hbase、Zookeper、Sqoop和Hcatalog等。

- **ZooKeeper ???** 

## MapReduce

### 优缺点

- 优点
  - 易于编程：简单的实现一些接口，类似于写一个简单的串 行程序，就可以完成一个分布式程序，这个分布式程序可以 分布到大量廉价的 PC 机器运行
  - 良好的扩展性：计算资源不够的时候，可以通过简单的增 加机器扩展计算能力
  - 高容错性：例如其中一台机器挂了，会自动把上面的计算 任务转移到另外一个节点上运行，不至于任务运行失败，整 个过程不需要人工参与
  - 离线处理PB 级以上海量数据：适合离线处理
- 缺点
  - 实时计算： MapReduce 无法像 Mysql 一样，在毫秒或者 秒级内返回结果
  - 流式计算： MapReduce 自身的设计特点决定了数据源必 须是静态的，不能动态变化，而流式计算的输入数据是动态 的
  - DAG计算：涉及多次迭代计算，MapReduce 可以做，但 是每个MapReduce 作业的输出结果都会写入到磁盘，造成
    大量的磁盘IO，导致性能低下

### MR流程(以Word Count为例)

#### Input

- HDFS 中的数据以 **Split** 方式作为 MapReduce 的 输入
- 通常1个 Split 对应1个 **Block** (HDFS 术语)，也可能对应多个Block，具 体是由 InputFormat 和压缩格式决定的

#### Map Task

##### InputFormat

- 对输入数据格式进行解析，默认为 TextInputFormat，key代表每行偏移量，value代表每行数据内容

  ```Java
  public int run(String[] args) throws Exception {
      Job job = Job.getInstance(conf, "WordCount");// 新建一个任务
      job.setJarByClass(WordCount.class);
  
      Path in = new Path(args[0]);	// 输入路径
      FileInputFormat.addInputPath(job, in);
  
      // ...
  }
  ```

##### Mapper

- 对输入的 key, value 处理，转换成新的 key, value 输出

  ```Java
  public static class TokenizerMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
      private final static IntWritable one = new IntWritable(1);
      private Text word = new Text();
  
      public void map(LongWritable key, Text value, Context context)
          	throws IOException,InterruptedException {
          StringTokenizer itr = new StringTokenizer(value.toString());
          while (itr.hasMoreTokens()) {
              word.set(itr.nextToken());
              context.write(word, one);
          }
      }
  }
  
  public int run(String[] args) throws Exception {
      // ...
  	
      job.setMapperClass(TokenizerMapper.class);
      
      // ...
  }
  ```

##### Combiner (Optional)

- Combiner 可以看做局部的 Reducer (local reducer) 

- Combiner 在 Job 中的设置：

  ```java
  public int run(String[] args) throws Exception {
      // ...
      
      job.setCombinerClass(IntSumReducer.class);
      
      // ...
  }
  ```

##### Partitioner (Optional)

- 数据分组， Mapper 的输出 key 会经过 Partitioner 分组选择不同的 Reducer 。默认Partitioner 会对 map 输出的 key 进行 hash 取模，比如有6个Reduce Task， 它就是模（mod）6，如果key的hash值为0，就选择第0个 Reduce Task。这样不同的map 对相同key，它的 hash 值取模是一样的

#### Reduce Task

##### Shuffle

- Reduce Task 远程拷贝每个 map 处理的结果， 从每个 map 中读取一部分结果，每个 Reduce Task 拷贝哪些数据，是由 Partitioner 决定的

##### Sort

- 读取完数据后，会按照key排序，相同的key被分到一组

##### Reducer

- 对输入的key、value处理，转换成新 的key、value输出

  ```Java
  public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
      private IntWritable result = new IntWritable();
  
      public void reduce(Text key, Iterable<IntWritable> values, Context context)
              throws IOException, InterruptedException {
          int sum = 0;
          for (IntWritable value : values) {
              sum += value.get();
          }
          result.set(sum);
          context.write(key, result);
      }
  }
  
  public int run(String[] args) throws Exception {
      // ...
  	
  	job.setReducerClass(IntSumReducer.class);
      
      // ...
  }
  ```

##### OutputFormat

- 数据输出格式， Reducer 的结果将按照 OutputFormat 格式输出，默认为 TextOutputFormat ，以WordCount为例，这里的key为单词，value为词频数

  ```java
  public int run(String[] args) throws Exception {
      // ...
      
      Path out = new Path(args[1]);	// 输出路径
      FileOutputFormat.setOutputPath(job, out);
      
      job.setOutputKeyClass(Text.class);
      job.setOutputValueClass(IntWritable.class);
  
      return job.waitForCompletion(true) ? 0 : 1;	// 等待作业完成退出
  }
  ```

### 常用算法

#### 平均值

#### TopN

#### 最大值

#### Map Join

#### Reduce Join



