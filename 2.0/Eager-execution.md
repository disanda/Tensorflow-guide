# 急速模式(Eager execution)

该模式是tf2.0后的新模式，是一种ML开发的发展趋势，类似PyTorch,参见：https://tensorflow.google.cn/guide/eager

## 梯度计算

主要用 tf.GradientTape(),一般在该方法返回对象打开的情况下输入方程表达式y=f(x)，一般用with tf.GradientTape() as xxxTape：语句,
之后进行梯度求导。如:
```py
x = tf.constant(3.0)
with tf.GradientTape() as g:
  g.watch(x)
  y = x * x
dy_dx = g.gradient(y, x) # Will compute to 6.0
```

