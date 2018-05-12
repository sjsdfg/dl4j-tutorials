# dl4j-tutorials

deeplearning4j 教程

交流群： 289058486

---

## 注意
因为使用的maven管理项目，所以第一次使用的时候更改maven配置。更改仓库地址为国内的阿里云
```
<mirror>
	<id>nexus-aliyun</id>
	<mirrorOf>central</mirrorOf>
	<name>Nexus aliyun</name>
	<url>http://maven.aliyun.com/nexus/content/groups/public</url>
</mirror> 
```

## dl4j概览

1. [dl4j快速索引：网络层，功能和类][6]
2. [例子概览][7]


## lesson1 nd4j基础操作

参考资料：

 1. [Deep Learning A Practitioner’s Approach][1]
 2. https://nd4j.org/userguide
 3. [nd4j方法快速索引][8]

## lesson2 简易线性回归

参考资料：

 1. [深度神经网络简介][2]
 1. http://www.jianshu.com/p/1d80023119cc
 2. http://neuralnetworksanddeeplearning.com/chap4.html

## lesson3 简易数据分类
参考资料：

 1. [ETL用户指南][3]
 2. [MNIST基础教程，包含一些分类知识][4]

## lesson4 Minst手写数字分类


## lesson5 模型保存与读取

参考资料：

 1. [HDFS模型保存][5]

## lesson6 Minst手写数字模型改进-CNN

参考资料：
 1. [关于深度学习之CNN经典论文原文(1950~2018)简介][9]

## lesson7 RNN循环神经网络

参考资料
 1. 理解LSTM网络：https://www.jianshu.com/p/9dc9f41f0b29
 2. 循环网络和LSTM教程：https://deeplearning4j.org/cn/recurrentnetwork
 3. DL4J中的循环网络：https://deeplearning4j.org/cn/usingrnns

## ObjectDetection 目标检测

参考资料：
 1. https://blog.csdn.net/u011669700/article/details/79886619
 
## tensorflow 导入tf模型

参考资料：
 1. https://blog.csdn.net/u011669700/article/details/80025161
 
## 自定义网络层实现GRU

参考资料：
 1. https://github.com/Gerry-Pan/pan-dl4j

根据GRU前向公式推导反向公式，并在dl4j中实现。

## 整合DL4J训练模型与Web工程
参考资料：
 1. 博文地址：https://my.oschina.net/u/1778239/blog/1648854
 2. 源码地址：https://gitee.com/lxkm/dl4j-demo/tree/master/digitalrecognition


 
 


  [1]: http://download.csdn.net/download/u011669700/10026286
  [2]: https://deeplearning4j.org/cn/neuralnet-overview
  [3]: https://deeplearning4j.org/cn/etl-userguide
  [4]: https://deeplearning4j.org/cn/mnist-for-beginners
  [5]: http://blog.csdn.net/u011669700/article/details/79113789
  [6]: https://deeplearning4j.org/quickref
  [7]: https://deeplearning4j.org/examples-tour
  [8]: https://blog.csdn.net/u011669700/article/details/80139619
  [9]: https://blog.csdn.net/qq_41185868/article/details/79995732
