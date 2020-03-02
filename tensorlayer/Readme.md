## 底层框架

- tensorflow
- theano
- PyTorch

以上都是最基础的框架，其完备程度接近一套独立的技术，除了DL还可以写别的东西。
keras、tflearn和tensorlayer都是在此基础上搭建的二次封装库，其中keras可以使用theano和tf做后端，剩余两个只针对tf。
甚至Caffe也打算在PyTorch上完成2次开发，成为一个基于PyTorch的二次封装库


### keras
keras是最简单的，封装非常完善，底层不透明，随意切换后端、任意切换CPU\GPU而基本不用改动代码，可以快速上手。

缺点是不好设计复杂和特定结构的网络，计算效率略低。

### tensorlayer

tensorlayer(tl)是，他的好处是对tf做了一个恰到好处的补充。
tf有一个很大的问题就是太过底层，如果你要架构一个复杂网络，你和希望一些常规的部件可以被直接调用，这时可以用tl。
也清楚数据在这些封装好的函数背后是怎么流动的。
