# dl4j-tutorials

deeplearning4j 教程

视频教程列表：[Deeplearning4j - 入门视频](https://www.jianshu.com/p/566fc3db676b)

哔哩哔哩直达地址：https://space.bilibili.com/327018681/#/

 - **交流群： 289058486**
 - **入群问题： Deeplearning4j 源码在 github的地址（mac 系统QQ看不到群问题，入群记得添加答案）**

DeepLearning4J（DL4J）是一套基于Java语言的神经网络工具包，可以构建、定型和部署神经网络。DL4J与Hadoop和Spark集成，支持分布式CPU和GPU，为商业环境（而非研究工具目的）所设计。Skymind是DL4J的商业支持机构。

Deeplearning4j拥有先进的技术，以即插即用为目标，通过更多预设的使用，避免多余的配置，让非企业也能够进行快速的原型制作。DL4J同时可以规模化定制。DL4J遵循Apache 2.0许可协议，一切以其为基础的衍生作品均属于衍生作品的作者。

# Give Me a Favor

<center>
<img src="http://static.zybuluo.com/ZzzJoe/yflamvkjh2i7zn5qcp9wpj61/%E5%AF%92%E6%B2%A7.jpg" />
</center>

---

## 注意
因为使用的maven管理项目，所以第一次使用的时候更改maven配置。更改仓库地址为国内的阿里云

- [Deeplearning4j入门（零）- maven环境配置 - 寒沧](https://www.bilibili.com/video/av25768162)
- [settings.xml 文件下载](https://github.com/sjsdfg/dl4j-tutorials/blob/master/src/main/resources/setting/settings.xml)

```xml
<mirror>
	<id>nexus-aliyun</id>
	<mirrorOf>central</mirrorOf>
	<name>Nexus aliyun</name>
	<url>http://maven.aliyun.com/nexus/content/groups/public</url>
</mirror> 
```

### 使用maven把jar包导出为外部
```bash
mvn dependency:copy-dependencies -DoutputDirectory=target/lib
```

- [deeplearning4j-1.0.0beta离线jar包---百度云](https://pan.baidu.com/s/1pxuEmzypSvlguCftsMaZ3g)


## dl4j概览

1. [dl4j快速索引：网络层，功能和类](https://github.com/deeplearning4j/deeplearning4j-docs/blob/gh-pages/quickref.md)
2. [dl4j-example 概览](https://github.com/deeplearning4j/deeplearning4j-docs/blob/gh-pages/examples-tour.md)
3. [dl4j 神经网络评估](https://deeplearning4j.org/docs/latest/deeplearning4j-nn-evaluation)
4. [dl4j 版本发布日志](https://github.com/deeplearning4j/deeplearning4j-docs/blob/releasenotes_100a/releasenotes.md)
5. [Java api文档](https://deeplearning4j.org/api/v1.0.0-beta2/)
6. [skymind 官方博客](https://blog.skymind.ai/)
7. [Quickstart with Deeplearning4J](http://www.dubs.tech/guides/quickstart-with-dl4j/)
8. [旧版本官网github](https://github.com/deeplearning4j/deeplearning4j-docs/tree/gh-pages)
9. [skymind ai wiki](https://skymind.ai/wiki/)
10. [skymind开源数据集集合](https://skymind.ai/wiki/open-datasets)
11. [Java Deep Learning Projects: Implement 10 real-world deep learning using Deeplearning4j and opensource APIs](https://pan.baidu.com/s/1Y2VoO6kLd6RIHCVqpVKzlQ)

## 调参

1. [我搭的神经网络不work该怎么办！看看这11条新手最容易犯的错误](https://mp.weixin.qq.com/s?__biz=MzIxMjAzNDY5Mg==&mid=2650791830&idx=1&sn=da81a253d4753e78d0ad5040ecf3ca29&chksm=8f474a7db830c36bf083c8e22414b2b2ee0fd38dd60175921952ae01c669ce4a3d105f972b09&mpshare=1&scene=1&srcid=0913NEV8u5Rz7fdaIDjAPnvs#rd)
2. [nd4j 和 DeepLearning4j 性能调优 debug](https://deeplearning4j.org/docs/latest/deeplearning4j-config-performance-debugging)
3. [神经网络训练问题排查](https://deeplearning4j.org/cn/troubleshootingneuralnets)

## lesson1 nd4j基础操作

参考资料：

 1. [一天搞懂深度学习](https://pan.baidu.com/s/1FW8zqzE4rK7pCOsC46dhIQ) 
 1. [Deep Learning A Practitioner’s Approach](https://pan.baidu.com/s/1C1s2xMuDYJBd3kCB8bxlxA)
 2. https://nd4j.org/userguide
 3. [nd4j方法快速索引](https://www.jianshu.com/p/c4f6284946bf)

## lesson2 简易线性回归

参考资料：

 1. [深度神经网络简介][2]
 2. [译-第四章 可视化证明神经网络可以计算任何函数](http://www.jianshu.com/p/1d80023119cc)
 3. [A visual proof that neural nets can compute any function](http://neuralnetworksanddeeplearning.com/chap4.html)

## lesson3 简易数据分类
参考资料：

 1. [ETL用户指南][3]
 2. [MNIST基础教程，包含一些分类知识][4]
 3. [Deeplearning4j Smote 样本均衡实现](https://zhuanlan.zhihu.com/p/52258279)

## lesson4 Minst手写数字分类

参考资料：

 1. [MINST数据集](http://yann.lecun.com/exdb/mnist/)
 2. [神经网络学习的可视化、监测及调试方法](https://deeplearning4j.org/cn/visualization)


## lesson5 模型保存与读取

参考资料：

 1. [HDFS模型保存][5]
 2. [SparkDl4jMultiLayer模型存储](https://github.com/sjsdfg/deeplearning4j-issues/blob/master/markdown/deeplearning4j%E7%9B%B8%E5%85%B3/SparkNetwork%E6%A8%A1%E5%9E%8B%E5%AD%98%E5%82%A8.md)

## lesson6 Minst手写数字模型改进-CNN

参考资料：
 1. [关于深度学习之CNN经典论文原文(1950~2018)简介][9]
 2. [Visualizing and Understanding CNNs.pdf](https://github.com/sjsdfg/deeplearning4j-issues/blob/master/Visualizing%20and%20Understanding%20CNNs.pdf)
 3. [deep learning for computer vision with python(3 本)](https://pan.baidu.com/s/17UMo76p75piTcArqu0wXJQ) 密码：vr0r
 4. [对ResNet本质的一些思考](https://zhuanlan.zhihu.com/p/60668529)

在使用 GPU 加速之前请务必确认一下几点：
 1. 电脑是否为 **英伟达** GPU，即 GTX 系列，使用 AMD 显卡无法使用 GPU 加速
 2. 电脑是否安装了 cuda ，如果安装了 cuda 请确认安装的 cuda 版本和你 pom 中引入的 `nd4j.backend` 版本是否对应
 3. 电脑安装 cuda 之后请确保你的 IDE 已经感知到环境变量的变化，在 IDE 中的 `terminal` 使用 `nvcc -V` 命令查看。如不确定直接重启电脑即可
 
以下为 GPU 安装和使用教程：
 1. [Deeplearning4j-使用Cuda 9.1和 Cudnn7.1 加速模型训练](https://www.jianshu.com/p/8a7533c2c79a)
 2. [在Deeplearning4j中使用cuDNN](https://blog.csdn.net/u011669700/article/details/79028821)
 3. [【视频】Deeplearning4j入门 - （十）GPU加速训练 - 寒沧](https://www.bilibili.com/video/av24603590)
 
如想确定 DeepLearning4j 已经支持的 cuda 和 cudnn 的配套版本，请打开如下链接：
 1. [Using Deeplearning4j with cuDNN](https://deeplearning4j.org/cudnn) ：搜索 `CUDA Version` 字眼

## lesson7 RNN循环神经网络

参考资料
 1. 理解LSTM网络：https://www.jianshu.com/p/9dc9f41f0b29
 2. 循环网络和LSTM教程：https://deeplearning4j.org/cn/recurrentnetwork
 3. DL4J中的循环网络：https://deeplearning4j.org/cn/usingrnns
 4. [DeepLearning4j: LSTM Network Example](https://deeplearning4j.org/programmingguide/05_lstm)

## ObjectDetection 目标检测

参考资料：
 1. [DeepLearning4j-使用Java训练YOLO模型](https://blog.csdn.net/u011669700/article/details/79886619)
 2. [Java构建汽车无人驾驶：汽车目标检测](https://blog.csdn.net/u011669700/article/details/79432195)
 3. [基于深度学习的目标检测技术演进：R-CNN、Fast R-CNN、Faster R-CNN、YOLO、SSD](https://www.julyedu.com/question/big/kp_id/26/ques_id/2103)
 4. [【中文】Yolo v1全面深度解读 目标检测论文](https://www.bilibili.com/video/av23354360)
 5. [【中文】Mask R-CNN 深度解读与源码解析 目标检测 物体检测 RCNN object detection 语义分割](https://www.bilibili.com/video/av24795835)
 6. 目标检测自定义数据集：https://pan.baidu.com/s/1u5yYv5SmK_vgd1zq1PsteQ
 <div align="center"> <img src="https://upload-images.jianshu.io/upload_images/2137832-f04063fbdfdaab6e.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" /> </div>
 


## tensorflow 导入tf模型

参考资料：
 1. https://blog.csdn.net/u011669700/article/details/80025161

 ## baidudianshi 百度点石比赛 baseline demo

 参考资料：
  1. 比赛地址：http://dianshi.baidu.com/dianshi/pc/competition/22/rule
  2. 防止比赛结束，数据寻回链接：https://pan.baidu.com/s/1_M0yPejFTvxDFOn4780OPA
  3. Baseline 0.83 得分模型：https://pan.baidu.com/s/1i-v02HnMPQwjtm32fPp67A （已经保存 Updater 信息，可用于增量训练）
  4. 内存管理官方文档：https://deeplearning4j.org/docs/latest/deeplearning4j-config-memory
  5. 迁移学习官方文档：https://deeplearning4j.org/docs/latest/deeplearning4j-nn-transfer-learning
  6. 迁移学习推荐阅读博客：https://blog.csdn.net/wangongxi/article/details/75127131
  7. 早停法训练模型官方文档：https://deeplearning4j.org/docs/latest/deeplearning4j-nn-early-stopping
  8. [百度点石-“探寻地球密码”天宫数据利用大赛.md](https://github.com/sjsdfg/deeplearning4j-issues/blob/master/markdown/%E7%99%BE%E5%BA%A6%E7%82%B9%E7%9F%B3-%E2%80%9C%E6%8E%A2%E5%AF%BB%E5%9C%B0%E7%90%83%E5%AF%86%E7%A0%81%E2%80%9D%E5%A4%A9%E5%AE%AB%E6%95%B0%E6%8D%AE%E5%88%A9%E7%94%A8%E5%A4%A7%E8%B5%9B.md)
  9. [百度点石-“探寻地球密码”天宫数据利用大赛.pdf](https://github.com/sjsdfg/deeplearning4j-issues/blob/master/%E7%99%BE%E5%BA%A6%E7%82%B9%E7%9F%B3-%E2%80%9C%E6%8E%A2%E5%AF%BB%E5%9C%B0%E7%90%83%E5%AF%86%E7%A0%81%E2%80%9D%E5%A4%A9%E5%AE%AB%E6%95%B0%E6%8D%AE%E5%88%A9%E7%94%A8%E5%A4%A7%E8%B5%9B.pdf)

 ## 模型训练早停法

 ### 1. 创建 ModelSaver

 用于在模型训练过程中，指定最好模型保存的位置：

 1. InMemoryModelSaver：用于保存到内存中
 2. LocalFileModelSaver：用于保存到本地目录中，只能保存 `MultiLayerNetwork` 类型的网络结果
 3. LocalFileGraphSaver：用于保存到本地目录中，只能保存 `ComputationGraph` 类型的网络结果

 ### 2. 配置早停法训练配置项

 1. epochTerminationConditions：训练结束条件
 2. evaluateEveryNEpochs：训练多少个epoch 来进行一次模型评估
 3. scoreCalculator：模型评估分数的计算者
     - org.deeplearning4j.earlystopping.scorecalc.RegressionScoreCalculator 用于回归的分数计算
     - ClassificationScoreCalculator 用于分类任务的分数计算
 4. modelSaver：模型的存储位置
 5. iterationTerminationConditions：在每一次迭代的时候用于控制

 ### 3. 获取早停法信息
 ```Java
 //Conduct early stopping training:
 EarlyStoppingResult result = trainer.fit();
 System.out.println("Termination reason: " + result.getTerminationReason());
 System.out.println("Termination details: " + result.getTerminationDetails());
 System.out.println("Total epochs: " + result.getTotalEpochs());
 System.out.println("Best epoch number: " + result.getBestModelEpoch());
 System.out.println("Score at best epoch: " + result.getBestModelScore());
 
 //Print score vs. epoch
 Map<Integer,Double> scoreVsEpoch = result.getScoreVsEpoch();
 List<Integer> list = new ArrayList<>(scoreVsEpoch.keySet());
 Collections.sort(list);
 System.out.println("Score vs. Epoch:");
 for( Integer i : list){
     System.out.println(i + "\t" + scoreVsEpoch.get(i));
 }
 ```

 ## 迁移学习

 ### 1. 获取原有的网络结构

 ```Java
  // 构造数据模型
 ZooModel zooModel = VGG16.builder().build();
 ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();
 ```


 ### 2. 修改模型的训练部分超参数

  1. updater
  2. 学习率
  3. 随机数种子：用于模型的复现

 ```java
  FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                 .updater(new Nesterovs(0.1, 0.9))
                 .seed(123)
                 .build();
 ```

 ### 3. 修改网络架构

 #### 3.1 setFeatureExtractor

 用于指定那个层以下为非 frozen 层，非冻结层。


 #### 3.2 结构更改

 1. 一般只有不同网络层之间才会出现 shape 异常：需要根据异常信息调整我们的网络层结构和参数
 2. `removeVertexKeepConnections` 和 `addLayer` 或者是 `addVertex` 进行网络结构的更改

## 自定义网络层实现GRU

参考资料：
 1. https://github.com/Gerry-Pan/pan-dl4j

根据GRU前向公式推导反向公式，并在dl4j中实现。

## 整合DL4J训练模型与Web工程
参考资料：
 1. 博文地址：https://my.oschina.net/u/1778239/blog/1648854
 2. 源码地址：https://gitee.com/lxkm/dl4j-demo/tree/master/digitalrecognition

## 【深度学习】图像矫正、dl4j yolo和tesseract ocr

参考资料：
 1. 视频地址：https://tianchi.aliyun.com/forum/videoStream.html#postsId=5312
 2. 视频代码所在github：https://github.com/awaymeet/tesseract

## 人脸识别 - FaceRecognition

参考资料：
  1. https://github.com/fradino/FaceRecognition
  2. https://gitee.com/xshuai/FaceRecognition

## Deeplearning4j 实现 Attention

参考资料：
1. [直播实现视频 youtube(自备梯子)](https://www.youtube.com/watch?v=XrZ_Y4koV5A)
2. [Implementing NLP Attention Mechanisms with DeepLearning4j(搬运到国内bilibili)](https://www.bilibili.com/video/av37100054/)
3. [attention 实现源码](https://github.com/treo/dl4j_attention)
4. [Attention Mechanisms (Enterprise AI Virtual Meetup).pdf](https://pan.baidu.com/s/1BzrteMiqlvm_l7Cv54Yc4g)

## GAN

- [GAN 使用 MNIST 实例](https://github.com/sjsdfg/dl4j-tutorials/tree/master/src/main/java/gan)。群友 @城枫林 和 [@liweigu](https://github.com/liweigu) 提供
- [gan_deeplearning4j](https://github.com/hamaadshah/gan_deeplearning4j)

## 自制AI图像搜索引擎

  群友 @射水鱼 攥写了一本使用 DeepLearning4j 实现的《自制AI图像搜索引擎》

按章节详细讲述了图像搜索引擎各主要组成部分的原理和实现，并在最后一章带领大家使用DL4J从零开始逐步构建了一个基于深度学习的Web图像搜索引擎，使读者能够更透彻地理解图像检索的理论并具有独立地实现一个在线图像搜索引擎的实际能力。每章都在对相关理论和方法进行阐述的同时，使用基于Java语言的实现代码和详实的代码注释来对相关理论和方法进行复述。

- 书籍地址：[https://www.epubit.com/book/detail/30316](qq://txfile/#)
- 源码地址：[https://box.lenovo.com/l/LHh2vR](qq://txfile/#) 密码: 1aaa  

```xml
<dependency>
    <groupId>be.tarsos</groupId>
    <artifactId>TarsosLSH</artifactId>
    <version>${tarsosLSH.version}</version>
</dependency>
```

如果导入项目中有依赖缺失，下载以下 jar 包：

- [TarsosLSH-0.9 下载地址](https://pan.baidu.com/s/1sbmvbkab6K5tRF92U-ItHw) 提取码：88qv
- [TarsosLSH github地址，也可以自行编译](https://github.com/JorenSix/TarsosLSH)

使用 `<scope> system </scope>`进行本地的 jar 包导入，或者使用以下命令安装在本地的 maven 仓库中：

```bash
mvn install:install-file -Dfile=/path/to/jar -DgroupId=be.tarsos -DartifactId=TarsosLSH -Dversion=0.9 -Dpackaging=jar
```

## 强化学习 RL4j

参考资料：
 1. 简书文章：https://www.jianshu.com/p/4d7f23395e92
 2. gitee代码：https://gitee.com/re6g3y/DL4J-with-LIBGDX

 <div align="center"> <img src="https://upload-images.jianshu.io/upload_images/2137832-9a808a77f1cab0b9.gif?imageMogr2/auto-orient/strip"/> </div>

## Deeplearning4j 经典开源项目

 1. [ScalphaGoZero](https://github.com/maxpumperla/ScalphaGoZero):An independent implementation of DeepMind's AlphaGoZero in Scala, using Deeplearning4J (DL4J 实现阿尔法狗)
 2. https://github.com/tahaemara/yolo-custom-object-detector : 使用 YOLO 检测实时检测自定义数据集 - 魔方
 3. https://github.com/mccorby/PhotoLabeller : 安卓客户端实现分布式训练。 使用 Kotlin 实现
 4. https://github.com/tahaemara/real-time-sudoku-solver : 使用 dl4j 解决数独
 5. https://github.com/kaiwaehner/kafka-streams-machine-learning-examples : kafka 流训练
 6. https://github.com/fra82/textdigester : dl4j 实现文档总结

## 获取最新的Deeplearning4j(Snapshots And Daily Builds)

参考资料：
  1. https://deeplearning4j.org/docs/latest/deeplearning4j-config-snapshots

配置 `pom.xml` 文件
```XML
<repositories>
    <repository>
        <id>snapshots-repo</id>
        <url>https://oss.sonatype.org/content/repositories/snapshots</url>
        <releases>
            <enabled>false</enabled>
        </releases>
        <snapshots>
            <enabled>true</enabled>
            <updatePolicy>daily</updatePolicy>  <!-- Optional, update daily -->
        </snapshots>
    </repository>
</repositories>
```
自动获取 skymind 所提供的 jar 包编译更新

# Spark 读取数据

1. https://github.com/deeplearning4j/dl4j-examples/issues/689
2. [Spark Data Pipelines Guide](https://deeplearning4j.org/docs/latest/deeplearning4j-scaleout-data-howto)
3. [Spark 训练指南](https://deeplearning4j.org/docs/latest/deeplearning4j-scaleout-howto)

```java
ok, so there's 2 ways
(a) use SparkContext.parallelize (that's a standard spark op) - easy but bad performance (all preprocessing happens on master)
(b) write a better data pipeline that does the proper reading + conversion in parallel
```

# 额外资源

 1. [机器学习高质量数据集大合辑](https://mp.weixin.qq.com/s?__biz=MjM5MTQzNzU2NA==&mid=2651663921&idx=2&sn=300429e518d159bb7654e1771672429e&chksm=bd4c09a28a3b80b4aa961577a7f59229d23bbd5f88b50bec6de21b0f94bd2fd2b348d1d4eb04&mpshare=1&scene=23&srcid=1023m8ifSIuylq6VcBQKRkt7#rd)
 2. [中文开放聊天语料整理](https://github.com/codemayq/chaotbot_corpus_Chinese)
 3. [gitxiv:只提供有复现开源代码的论文](http://www.gitxiv.com/)
 4. [hadoop-winutils](https://github.com/steveloughran/winutils)：提供 hadoop 工具在 windows 平台下的 hadoop.dll和winutils.exe。便于 windows 下运行 spark-local 模式



 

[2]: https://deeplearning4j.org/cn/neuralnet-overview
[3]: https://deeplearning4j.org/cn/etl-userguide
[4]: https://deeplearning4j.org/cn/mnist-for-beginners
[5]: http://blog.csdn.net/u011669700/article/details/79113789
[6]: https://deeplearning4j.org/quickref
[7]: https://deeplearning4j.org/examples-tour
[8]: https://blog.csdn.net/u011669700/article/details/80139619
[9]: https://blog.csdn.net/qq_41185868/article/details/79995732
