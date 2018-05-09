# Hadoop

## Introduction

大数据技术特点：**健壮、可扩展、简单方便**

大数据技术主要应用场景：

- 搜索引擎，为大规模的网页快速建立索引
- 大数据存储，利用分布式存储能力，建立数据备份、数据仓库等
- 大数据处理，利用分布式处理能力，例如数据挖掘、数据分析等
- 批量处理 (Batch Processing)
  - 侧重于处理海量数据，处理速度可忍受，时间可能在数十分钟到数小时
  - MR
- 历史数据交互式查询 (Interactive Query)
  - 时间在数秒到数十分钟之间
  - Presto
- 实时流数据处理 (Streaming Processing)
  - 通常在数十毫秒到数秒之间
  - Storm

**Hadoop** 框架的核心是 ***HDFS, MapReduce, Yarn***
- **HDFS** 是分布式文件系统，提供海量数据的存储
- **MapReduce** 是分布式数据处理模型，提供数据计算
- **Yarn** 是资源管理和调度工具

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

- **Storm** 流式计算

- **Spark** 内存计算、流式计算、图计算

- **Sqoop** 从关系数据库导入数据到Hadoop，并可直接导入到HDFS或Hive

- **Flume** 将流数据或日志数据导入HDFS

- **Hive** 在Hadoop中扮演==数据仓库==的角色，Hive使用类SQL语法进行数据操作，适合分析一段时间内的数据

- **HBase** 是面向列的==数据库==，运行在HDFS之上， HBase以BigTable为蓝本，可以快速在数十亿行数据中随机存取数据，适合实时查询

- **ZooKeeper** 协调集群成员

- **Oozie** 提供管理工作流程和依赖的功能，将多个 MapReduce作业连接到一起，定制彼此间依赖，有定时功能

- **Ambari** 可对Hadoop集群提供监控、部署、配置、升级和管理等核心功能

- **Azkaban**

  一款基于Java编写的任务调度系统任务调度，来自LinkedIn公司，用于管理他们的Hadoop批处理工作流。Azkaban根据工作的依赖性进行排序，提供友好的Web用户界面来维护和跟踪用户的工作流程

- **Kafka**

  一种高吞吐量的分布式发布订阅消息系统，它可以处理消费者规模网站中的所有动作流数据，目前已成为大数据系统在异步和分布式消息之间的最佳选择

- **ZooKeeper**

  一个分布式的应用程序协调服务，是Hadoop和Hbase的重要组件。它是一个为分布式应用提供一致性服务的工具，让Hadoop集群里面的节点可以彼此协调。ZooKeeper现在已经成为了 Apache的顶级项目，为分布式系统提供了高效可靠且易于使用的协同服务

### 集群管理工具

- **YARN**

  一种新的Hadoop资源管理器，它是一个通用资源管理系统，可为上层应用提供统一的资源管理和调度，解决了旧MapReduce框架的性能瓶颈。它的基本思想是把资源管理和作业调度/监控的功能分割到单独的守护进程。

- **Mesos**

  由加州大学伯克利分校的AMPLab首先开发的一款开源群集管理软件，支持Hadoop、ElasticSearch、Spark、Storm 和Kafka等架构。对数据中心而言它就像一个单一的资源池，从物理或虚拟机器中抽离了CPU，内存，存储以及其它计算资源， 很容易建立和有效运行具备容错性和弹性的分布式系统。

- **Ambari**

- **ZooKeeper**

## MapReduce

### 优缺点

- 优点
  - 易于编程：简单的实现一些接口，类似于写一个简单的串 行程序，就可以完成一个分布式程序，这个分布式程序可以 分布到大量廉价的 PC 机器运行
  - 良好的扩展性：计算资源不够的时候，可以通过简单的增 加机器扩展计算能力
  - 高容错性：例如其中一台机器挂了，会自动把上面的计算 任务转移到另外一个节点上运行，不至于任务运行失败，整 个过程不需要人工参与
  - 离线处理PB 级以上海量数据：适合离线处理
- 缺点
  - 实时计算： MapReduce 无法像 Mysql 一样，在毫秒或者秒级内返回结果
  - 流式计算： MapReduce 自身的设计特点决定了数据源必须是静态的，不能动态变化，而流式计算的输入数据是动态 的
  - DAG计算：涉及多次迭代计算，MapReduce 可以做，但是每个MapReduce 作业的输出结果都会写入到磁盘，造成
    大量的磁盘IO，导致性能低下

### 流程(以Word Count为例)

![MR](images/MR.png)

#### Input

- HDFS 中的数据以 **Split** 方式作为 MapReduce 的 输入（Mapper 的个数由 Split 数目确定）
- 通常1个 Split 对应1个 **Block** (HDFS 术语)，也可能对应多个Block，具体是由 InputFormat 和压缩格式决定的

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

- 多输入：

  - MultipleInputs 允许为每条输入路径指定 InputFormat 和 Mapper

  ```java
  MultipleInputs.addInputPath(job, InputPath, TextInputFormat.class, Mapper.class);
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
- Reducer 的个数由 Partitioner 决定

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

- 多输出：

  ```java
  public static class Reducer extends Reducer<Text, IntWritable, Text, IntWritable> {
      private IntWritable result = new IntWritable();
      private MultipleOutputs<Text, IntWritable> multipleOutputs;
  
      @Override
      protected void setup(Context context) throws IOException, InterruptedException {
          // setup方法中构造一个 MultipleOutputs的实例
          multipleOutputs = new MultipleOutputs<Text, IntWritable>(context);
      }
  
      protected void reduce(Text key, Iterable<IntWritable> values, Context context)
              throws IOException, InterruptedException {
          // ...
          
          // 使用 MultipleOutputs实例输出， 而不是 context.write()方法，分别输出键、值、和文件名
          multipleOutputs.write(key, result, name);
      }
  
      @Override
      protected void cleanup(Context context) throws IOException, InterruptedException {
          multipleOutputs.close();
      }
  }
  
  ```

#### 数据压缩

- 目的：

  - 降低磁盘IO
  - 降低网络IO
  - 降低存储成本

- 常用压缩格式：

  |  格式  | 压缩率 |  速度  | 是否hadoop自带 | linux命令 | 换成压缩格式后原应用程序是否要修改 |
  | :----: | :----: | :----: | :------------: | :-------: | :--------------------------------: |
  |  gzip  |  很高  | 比较快 |  是，直接使用  |    有     |     和文本处理一样，不需要修改     |
  | snappy | 比较高 |  很快  |  否，需要安装  |   没有    |     和文本处理一样，不需要修改     |

  （ gzip常用于不经常分析的历史数据 ）

### 单元测试 

MRUnit 框架是Cloudera公司专为Hadoop MapReduce写的单元测试框架。MRUnit针对不同测试对象使用不同的Driver：

- MapDriver：针对单独的Map测试
- ReduceDriver：针对单独的Reduce测试
- MapReduceDriver：将map和reduce串起来测试

### 常用算法

#### 平均值

计算每个用户的平均访问时间：

- Sample Input:

  > user1	2
  >
  > user2	2
  >
  > user3	3
  >
  > user1	4
  >
  > user2	5
  >
  > user3	6

- Sample Output:

  > user1	3
  >
  > user2	3.5
  >
  > user3	4.5

```java
public static class AverageCountMapper extends Mapper<LongWritable, Text, Text, Text> {
    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String line = value.toString();
        String[] parameters = line.split("\\s+");
        context.write(new Text(parameters[0]), new Text(parameters[1]));
    }
}

public static class AverageCountCombiner extends Reducer<Text, Text, Text, Text> {
    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        Double sum = 0.00;
        int count = 0;
        for (Text item : values) {
            sum = sum + Double.parseDouble(item.toString());
            count++;
        }
        context.write(new Text(key), new Text(sum + "-" + count));
    }
}

public static class AverageCountReducer extends Reducer<Text, Text, Text, Text> {
    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        Double sum = 0.00;
        int count = 0;
        for (Text t : values) {
            String[] str = t.toString().split("-");
            sum += Double.parseDouble(str[0]);
            count += Integer.parseInt(str[1]);
        }
        double average = sum / count;
        context.write(new Text(key), new Text(String.valueOf(average)));
    }
}
```

#### TopN

计算前3个访问最多的页面：

- Sample Input:

  > url1	123456
  >
  > url2	42
  >
  > url3	3457
  >
  > url4	3490
  >
  > url5	983457

- Sample Output:

  > url5	983457
  >
  > url1	123456
  >
  > url4	3490

- 注意：

  - 使用map的cleanup输出
  - map输出key为空确保只有一个reducer，不然结果是局部最大值 

```java
public static final int k = 3;

public static class TopNMapper extends Mapper<LongWritable, Text, NullWritable, Text> {
    private TreeMap<Integer, String> map = new TreeMap<Integer, String>();
    
    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String line = value.toString();
        String[] parameters = line.split("\\s+");
        Integer clicks = Integer.parseInt(parameters[1]);
        map.put(clicks, value.toString());
        if (map.size() > k) {
            map.remove(map.firstKey());
        }
    }

    protected void cleanup(Context context) throws IOException, InterruptedException {
        for (String text : map.values()) {
            if (text.toString() != null && !text.toString().equals("")) {
                context.write(NullWritable.get(), new Text(text));
            }
        }
    }
}

public static class TopNReducer extends Reducer<NullWritable, Text, NullWritable, Text> {
    private TreeMap<Integer, String> map = new TreeMap<Integer, String>();

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        for (Text item : values) {
            String value[] = item.toString().split("\t");
            Integer clicks = Integer.parseInt(value[1]);
            map.put(clicks, item.toString());
            if (map.size() > k) {
                map.remove(map.firstKey());
            }
        }
        for (String text : map.values()) {
            context.write(NullWritable.get(), new Text(text));
        }
    }
}

```

#### 最大值

计算每个页面最长访问时间：

- Sample Input:

  > url1 	15
  >
  > url2	18
  >
  > url3	12
  >
  > url2	9
  >
  > url3	6
  >
  > url1	18

- Sample Output:

  > url1	18
  >
  > url2	18
  >
  > url3	12

```java
public static class MaxNumMapper extends Mapper<LongWritable, Text, Text, Text> {

    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String line = value.toString();
        String[] parameters = line.split("\\s+");
        String urlId = parameters[0];
        context.write(new Text(urlId), new Text(String.valueOf(parameters[1])));
    }
}

public static class MaxNumReducer extends Reducer<Text, Text, Text, Text> {
    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        long max = 0L;
        for (Text text : values) {
            if (Long.parseLong(text.toString()) > max) {
                max = Long.parseLong(text.toString());
            }
        }
        context.write(key, new Text(String.valueOf(max)));
    }
}

```

#### Map Join

Map Join 适用于较小文件可以直接读入内存的情况。 Map Join 速度最快。

- Sample Input:

  > Station.txt:
  >
  > 011990-99999	SIHCCAJAVRI        CA
  > 012650-99999	TRNSET-HANSMOEN    AR  

  > Temperature.txt
  >
  > 012650-99999	194903241200	111
  > 012650-99999	194903241800	78
  > 011990-99999	195005150700	0
  > 011990-99999	195005151200	22
  > 011990-99999	195005151800	-11

```java
public static class MapJoinMapper extends Mapper<Object, Text, Text, Text> {
    private HashMap<String, String> stationMap = new HashMap<String, String>();
    private String TemperatureStr;
    private String[] TemperatureItems;
    private String station; // 不包括stationId
    private Text outPutKey = new Text();
    private Text outPutValue = new Text();

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        BufferedReader br;
        String station;
        // 把缓存的小文件读取出来并缓存到hashmap中
        Path[] paths = context.getLocalCacheFiles();
        for (Path path : paths) {
            String pathStr = path.toString();
            if (pathStr.endsWith("Station.txt")) {
                br = new BufferedReader(new FileReader(pathStr));
                while (null != (station = br.readLine())) {
                    String[] stationItems = station.split("\\s+");
                    if (stationItems.length == 3) {// 去掉脏数据
                        // 缓存到一个map
                        stationMap.put(stationItems[0], stationItems[1] + "\t" + stationItems[2]);
                    }
                }
            }
        }
    }

    @Override
    protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        TemperatureStr = value.toString();
        // 过滤脏数据
        if (TemperatureStr.equals("")) {
            return;
        }
        TemperatureItems = TemperatureStr.split("\\s+");
        if (TemperatureItems.length != 3) {
            return;
        }

        // 判断当前记录的joinkey字段是否在stationMap中
        station = stationMap.get(TemperatureItems[0]);
        if (null != station) {// 如果缓存中存在对应的joinkey，那就做连接操作并输出
            outPutKey.set(TemperatureItems[0]);
            outPutValue.set(station + "\t" + TemperatureItems[1] + "\t" + TemperatureItems[2]);
            context.write(outPutKey, outPutValue);
        }
    }
}

public int run(String[] args) throws Exception {
    Configuration conf = new Configuration();
    Job job = Job.getInstance(conf, "mapjoin");

    // 把小表的数据缓存起来(Station.txt)
    job.addCacheFile(new Path(args[0]).toUri());

    job.setJarByClass(MapJoin.class);
    job.setMapperClass(MapJoinMapper.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Text.class);

    FileInputFormat.addInputPath(job, new Path(args[1]));
    FileOutputFormat.setOutputPath(job, new Path(args[2]));
    return job.waitForCompletion(true) ? 0 : 1;
}

```

#### Reduce Join

适用于需要 Join 的文件都很大的情况。

```Java
private final static String STATION_FILE = "Station.txt";
private final static String TEMPERATURE_FILE = "Temperature.txt";

public static class ReduceJoinMapper extends Mapper<Object, Text, Text, Text> {
    private Text joinKey = new Text();
    private Text combineValue = new Text();

    /**
     * 为来自不同文件的key/value加标记。
     */
    @Override
    protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        String pathName = ((FileSplit) context.getInputSplit()).getPath().toString();
        // 如果数据来自于STATION_FILE，加一个STATION_FILE的标记
        if (pathName.endsWith(STATION_FILE)) {
            String[] valueItems = value.toString().split("\\s+");
            // 过滤掉脏数据
            if (valueItems.length != 3) {
                return;
            }
            joinKey.set(valueItems[0]);
            combineValue.set(STATION_FILE + valueItems[1] + "\t" + valueItems[2]);
        } else if (pathName.endsWith(TEMPERATURE_FILE)) {
            // 如果数据来自于TEMPERATURE_FILE，加一个TEMPERATURE_FILE的标记
            String[] valueItems = value.toString().split("\\s+");
            // 过滤掉脏数据
            if (valueItems.length != 3) {
                return;
            }
            joinKey.set(valueItems[0]);
            combineValue.set(TEMPERATURE_FILE + valueItems[1] + "\t" + valueItems[2]);
        }
        context.write(joinKey, combineValue);
    }
}

public static class ReduceJoinReducer extends Reducer<Text, Text, Text, Text> {
    private List<String> stations = new ArrayList<String>();
    private List<String> temperatures = new ArrayList<String>();
    private Text result = new Text();

    @Override
    protected void reduce(Text key, Iterable<Text> values, Context context)
            throws IOException, InterruptedException {
        // 一定要清空数据
        stations.clear();
        temperatures.clear();
        // 相同key的记录会分组到一起，我们需要把相同key下来自于不同文件的数据分开
        for (Text value : values) {
            String val = value.toString();
            if (val.startsWith(STATION_FILE)) {
                stations.add(val.replaceFirst(STATION_FILE, ""));
            } else if (val.startsWith(TEMPERATURE_FILE)) {
                temperatures.add(val.replaceFirst(TEMPERATURE_FILE, ""));
            }
        }

        for (String station : stations) {
            for (String temperature : temperatures) {
                result.set(station + "\t" + temperature);
                context.write(key, result);
            }
        }
    }
}

```

#### Semi Join

Reduce Join 的优化，利用缓存过滤掉一部分无法 Join 的数据。

```java
private final static String STATION_FILE = "Station.txt";
private final static String TEMPERATURE_FILE = "Temperature.txt";

public static class SemiJoinMapper extends Mapper<Object, Text, Text, Text> {
    private Set<String> joinKeys = new HashSet<>();
    private Text joinKey = new Text();
    private Text combineValue = new Text();

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        BufferedReader br;
        String station;
        Path[] paths = context.getLocalCacheFiles();
        for (Path path : paths) {
            String pathStr = path.toString();
            if (pathStr.endsWith("Station.txt")) {
                br = new BufferedReader(new FileReader(pathStr));
                while (null != (station = br.readLine())) {
                    String[] stationItems = station.split("\\s+");
                    if (stationItems.length == 3) {
                        joinKeys.add(stationItems[0]);
                    }
                }
            }
        }
    }

    /**
     * map直接判断当前记录的stationId在缓存中是否存在
     */
    @Override
    protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        String pathName = ((FileSplit) context.getInputSplit()).getPath().toString();
        // 如果数据来自于STATION_FILE，加一个STATION_FILE的标记
        if (pathName.endsWith(STATION_FILE)) {
            String[] valueItems = value.toString().split("\\s+");
            // 过滤掉脏数据
            if (valueItems.length != 3) {
                return;
            }
            // 过滤掉无法join的记录
            if (joinKeys.contains(valueItems[0])) {
                joinKey.set(valueItems[0]);
                combineValue.set(STATION_FILE + valueItems[1] + "\t" + valueItems[2]);
            }
        } else if (pathName.endsWith(TEMPERATURE_FILE)) {
            // 如果数据来自于TEMPERATURE_FILE，加一个TEMPERATURE_FILE的标记
            String[] valueItems = value.toString().split("\\s+");
            // 过滤掉脏数据
            if (valueItems.length != 3) {
                return;
            }
            // 过滤掉无法join的记录
            if (joinKeys.contains(valueItems[0])) {
                joinKey.set(valueItems[0]);
                combineValue.set(TEMPERATURE_FILE + valueItems[1] + "\t" + valueItems[2]);
            }
        }
        context.write(joinKey, combineValue);
    }
}

public static class SemiJoinReducer extends Reducer<Text, Text, Text, Text> {
    private List<String> stations = new ArrayList<String>();
    private List<String> temperatures = new ArrayList<String>();
    private Text result = new Text();

    @Override
    protected void reduce(Text key, Iterable<Text> values, Context context)
            throws IOException, InterruptedException {
        // 一定要清空数据
        stations.clear();
        temperatures.clear();
        // 相同key的记录会分组到一起，我们需要把相同key下来自于不同文件的数据分开
        for (Text value : values) {
            String val = value.toString();
            if (val.startsWith(STATION_FILE)) {
                stations.add(val.replaceFirst(STATION_FILE, ""));
            } else if (val.startsWith(TEMPERATURE_FILE)) {
                temperatures.add(val.replaceFirst(TEMPERATURE_FILE, ""));
            }
        }

        for (String station : stations) {
            for (String temperature : temperatures) {
                result.set(station + "\t" + temperature);
                context.write(key, result);
            }
        }
    }
}

```

### 运行机制

- 和HDFS一样，MapReduce也是采用Master/Slave的架构
- MapReduce包含4个部分：Client, JobTracker, TaskTracker和Task
- Client:
  - 将JAR文件、配置参数Configuration、计算分片、 Distributed Cache 文件存储在HDFS
  - 向 JobTracker 申请JobId
- JobTracker: 负责资源监控和作业调度
  - 监控所有TaskTracker 与job的健康状况，一旦发现失败，就将相应的任务转移到其他节点
  - 跟踪任务的执行进度、资源使用量等信息，并将这些信息告诉作业调度器
  - 调度器会在资源出现空闲时，选择合适的任务使用这些资源
- TaskTracker:
  - 周期性地通过Heartbeat 将本节点上资源的使用情况 和任务的运行进度汇报给JobTracker
  - 接收JobTracker 发送过来的命令并执行相应的操作（ 如启动新任务、杀死任务等）
  - 使用“slot”等量划分本节点上的资源量，“slot”代表计算 资源（CPU、内存等）分配给Task 使用
- Task:
  - 分为Map Task 和Reduce Task 两种，均由 TaskTracker 启动
  - Map Task 和Reduce Task 分别使用Map slot 和 Reduce slot
- 容错：
  - JobTracker：存在单点故障，一旦出现故障，整个集群 就不可用，出现故障之后重启一下，再把作业重新提交就可 以了，它不会像 HDFS 那样出现数据的丢失
  - TaskTracker：周期性向 JobTracker 汇报心跳，如果一 定时间内没有汇报，JobTracker 就认为该TaskTracker 挂掉 了，或者TaskTracker上运行的Task失败次数太多，就会把 上面所有任务调度到其它TaskTracker上
  - Task：MapTask和ReduceTask 也可能运行挂掉，比如内 存超出了或者磁盘挂掉了，TaskTracker会汇报JobTracker ，JobTracker会把该Task调度到其它节点上，但受到重试次数的限制

## YARN

### 优点

- YARN把JobTracker分为ResouceManager和 ApplicationMaster，ResouceManager专管整个集 群的资源管理和调度，而ApplicationMaster则负责 应用程序的任务调度和容错等
- YARN不再是一个单纯的计算框架，而是一个框 架管理器，用户可以将各种各样的计算框架移植到 YARN之上，由YARN进行统一管理和资源分配
- 对于资源的表示以内存和CPU为单位，比之前slot 更合理

### 运行机制

- YARN主要由RM、NM、AM和Container等4个组件构成
- ResourceManager
  - 处理客户端请求
  - 启动和监控ApplicationMaster
  - 监控NodeManager
  - 资源的分配与调度
- NodeManager
  - 管理单个节点上的资源
  - 处理来自ResourceManager的命令
  - 处理来自ApplicationMaster的命令
- ApplicationMaster
  - 为应用程序申请资源并分配给内部的任务
  - 任务的监控与容错
- Container
  - 对任务运行环境进行抽象，封装CPU、内存等资源

### 容错

- RM：HA方案避免单点故障
- AM：AM向RM周期性发送心跳，出故障后RM会启动新的 AM，受最大失败次数限制
- NM：周期性RM发送心跳，如果一定时间内没有发送， RM 就认为该NM 挂掉了，或者NM上运行的Task失败次数 太多，就会把上面所有任务调度到其它NM上
- Task：Task也可能运行挂掉，比如内存超出了或者磁盘 挂掉了，NM会汇报AM，AM会把该Task调度到其它节点上，但受到重试次数的限制

## HDFS

### 优缺点

- 优点：
  - 高容错，可构建在廉价机器上
    - 数据自动保存多个副本，通过增加副本，提高容错性
    - 某1个副本丢失以后，可以自动恢复、
  - 适合批处理
    - 移动计算而不是移动数据
    - 把数据位置暴露给计算框架
  - 适合大数据处理
    - 处理数据达到 GB、TB、甚至PB级别的数据
    - 能够处理百万规模以上的文件数量，数量相当之大
    - 能够处理10K节点的规模
  - 流式文件访问
    - 一次写入，多次读取，文件一旦写入不能修改
    - 保证数据的一致性
    - 高吞吐率
- 缺点：
  - 低延时数据访问
    - 毫秒级内数据存储
    - 毫秒级内读取数据
  - 小文件存储
    - 大量小文件会占用 NameNode 大量的内存来存储文件 、目录和块信息，而NameNode 的内存总是有限的
    - 小文件存储的寻道时间会超过读取时间，违反了HDFS 的设计目标
  - 并发写入、文件随机修改
    - 一个文件只能有一个写，不允许多个线程同时写。
    - 仅支持数据 append（追加），不支持文件的随机修改

### 体系结构

HDFS 采用Master/Slave的架构来存储数据，该架构主要由四个部分组成

![HDFS](images/HDFS.gif)

#### HDFS Client

- 文件切分，文件上传 HDFS 的时候，Client 将文件切分成一个一个的Block，然后进行存储
- 与 NameNode 交互，获取文件的位置信息
- 与 DataNode 交互，读取或者写入数据
- Client 提供一些命令来管理 HDFS，比如启动或者关闭HDFS
- Client 可以通过一些命令来访问 HDFS

#### NameNode

- Master，一个管理者，不实际存储数据
- 管理 HDFS 的名称空间
- 管理数据块（Block）映射信息
- 配置副本策略
- 处理客户端读写请求

#### DataNode

- Slave，NameNode 下达命令，DataNode 执行实际的操作
- 存储实际的数据块
- 执行数据块的读/写操作

#### SecondaryNameNode

- 辅助NameNode，分担NameNode工作量
- 定期合并 fsimage和edits，并推送给NameNode
- 执行合并时机
  - 根据配置文件设置的时间间隔 fs.checkpoint.period 默认3600秒
  - 根据配置文件设置edits log大小 fs.checkpoint.size 规定edits文件的最大值默 认是64MB
- 在紧急情况下，可辅助恢复 NameNode
- 并非 NameNode 的热备，当NameNode 挂掉的时候，它并不能马上替换 NameNode 并提供服务

### Robustness

- NameNode HA (High Availability) 解决 NameNode 出错

  - 集群启动时，可以同时启动2个NameNode。这些NameNode只有一个是active的，另一个属于standby状态。active状态意味着提供服务，standby状态意味着处于休眠状态，只进行数据同步，时刻准备着提供服务 
- NameNode Federation 
  - 有多个namenode，分别负责几个datanode。类似于美国的联邦机制
- Replication 解决 DataNode 出错

  - 副本因子一般为3
  - 副本存放策略需要对可靠性、写入带宽和读取带 宽进行权衡
  - 2个副本在1个机架的不同节点上, 另1个副本在 另1个机架节点上，其他副本随机存储
  - 针对读请求，HDFS提供离客户端最近的副本
  - Hadoop启动时，在安全模式下会检查副本数
- Cluster Rebalancing
- Checksum 保证数据完整性
- 快照

### 读写原理

- 读取：
  - 获取DistributedFileSystem实例
  - DistributedFileSystem通过RPC获得第一批block的locations，这些 locations按照hadoop拓扑结构排序，距离客户端近的排在前面
  - 客户端调用read方法，连接离客户端最近的datanode
  - 数据从datanode流向客户端
  - 第一个block块的数据读完，就会关闭指向第一个block块的datanode连接 ，接着读取下一个block块
  - 第一批block都读完，就会去namenode拿下一批block的locations，然后继续读，所有的block块都读完，就会关闭流
- 写入：
  - 客户端调用 DistributedFileSystem 的create方法，创建一个新文件
  - DistributedFileSystem 通过 RPC调用 NameNode，去创建一个没有 blocks关联的新文件
  - 创建前，NameNode 会做各种校验，比如文件是否存在，客户端有无权 限创建等
  - DFSOutputStream会把数据切成一个个小packet，然后排成队列 data queue，并问询 NameNode 这个新的 block 最适合存储的3个DataNode
  - DFSOutputStream 还有一个队列叫 ack queue，也是由 packet 组成， 等待DataNode的收到响应，当所有DataNode都表示已经收到，这时ack queue才会把对应的packet包移除掉
  - 客户端完成写数据后，调用close方法关闭写入流
  - 最后通知元数据节点写入完毕

### HDFS Shell

基本命令：

- 创建文件夹：hdfs dfs -mkdir /log/20161001 

- 上传文件或目录：hdfs dfs -put log.txt /log/20161001/ 或 hdfs dfs -copyFromLocal log.txt /log/20161001/

- 显示目录下的文件： hdfs dfs -ls /log/20161001/ 

  - 如果是文件，返回文件信息如下： 

    权限	 <副本数>	用户ID	组ID	文件大小	修改日期	修改时间	权限文件名

  - 如果是目录，返回目录信息如下：

    权限用户ID	组ID	修改日期	修改时间	目录名

- 递归显示目录下的文件：hdfs dfs -ls -R /log/20161001/

- 查看内容：hdfs dfs -text /log/20161001/log.txt 或-cat、-tail命令，但对于压缩文件只能用 -text 参数来查 看，否则是乱码 

- 复制到本地(默认当前目录)：hdfs dfs -get /log/20161001/log.txt /home/hadoop 或hdfs dfs -copyToLocal /log/20161001/log.txt
  /home/Hadoop 

- 删除文件： hdfs dfs -rm /log/20161001/log.txt 

- 删除文件夹： hdfs dfs –rm -r /log/20161001/log.txt

- 连接文件到本地： hdfs dfs -getmerge /user/hadoop/output/ /home/hadoop/merge --将output目录下的所有文件合并到本地merge文件中

- 复制文件： 

  - 将file1 文件复制到file2：hdfs dfs -cp /user/hadoop/file1 /user/hadoop/file2
  - 将文件file1,file2复制到dir目录下：hdfs dfs -cp /user/hadoop/file1 /user/hadoop/file2 /user/hadoop/dir

- 移动文件：

  - 将文件 file1移动到file2：hdfs dfs -mv /user/hadoop/file1 /user/hadoop/file2
  - 将file1 file2 file3 移动到dir目录下：hdfs dfs -mv /user/hadoop/file1 /user/hadoop/file2 /user/hadoop/file3 /user/hadoop/dir

管理命令：

- 查看正在运行的 Job：mapred job -list，-list all显示所有 job 
- 关闭正在运行的 Job： mapred job -kill job_2732108212572_0001 
- 检查 HDFS 块状态，查看是否损坏： hdfs fsck / 
- 检查 HDFS 块状态，并删除损坏的块： hdfs fsck / -delete
- 检查 HDFS 状态，包括 DataNode 信息： hdfs dfsadmin – report
- Hadoop 进入安全模式：hdfs dfsadmin -safemode enter
- Hadoop 离开安全模式： hdfs dfsadmin -safemode leave
- 平衡集群中的文件： sbin/start-balancer.sh

### 序列化与反序列化

- 序列化在分布式数据处理的两大领域经常出现：永久存储 和进程间通信
- 序列化就是将结构化对象（实例）转化为字节流（字符 数组）
- 反序列化就是将字节流转向结构化对象的逆过程
- Hadoop实现的序列化特点
  - 紧凑：紧凑的格式能充分利用网络带宽，而带宽 是数据中心最稀缺的资源之一
  - 快速：进程通信形成了分布式系统的骨架，所以 需要尽量减少序列化和反序列化的性能开销
  - 互操作：能支持不同语言写的客户端和服务端进行交互
- Java 基本类型和Hadoop 自带 Writable 的对应关系

| Java基本类型 |  Writable实现   | 字节 |
| :----------: | :-------------: | :--: |
|   boolean    | BooleanWritable |  1   |
|     byte     |  ByteWritable   |  1   |
|     int      |   IntWritable   |  4   |
|    float     |  FloatWritable  |  4   |
|     long     |  LongWritable   |  8   |
|    double    | DoubleWritable  |  8   |

- 除了上表Hadoop中的Writable类型，Text类型也 很常用

  - Text 类是一种 UTF-8 格式的Writable 类型
  - 可以将它理解为一种与 java.lang.String 类似的 Writable 类型
  - 可以通过 set(byte[] utf8) 方法设置 Text 实例

- Hadoop中定义了两个序列化接口：Writable 接口 和 Comparable 接口（可以合并成一个接口 WritableComparable）

  ```java
  public interface Writable{
      /**
  	* 将对象状态写入二进制格式的DataOutput流
  	*/
      public void write(DataOutput out) throws IOException;
      /**
      * 从二进制格式的DataInput流中读取字节流反序列化为对象
      */
      public void readFields(DataInput in) throws IOException;
  }
  
  public interface Comparable{
      /**
      * 将this对象和对象o进行比较
      */
      public int compareTo(T o);
  }
  ```



