# RNN基础(tf-2.0)

由初始状态 h0(batch_size,state_size) 和 初始输入 input (batch_size,input_size)，得到下一个隐藏状态h1和当前的输出output。
以此类推可以得到后续的时间状态和输出

## tf 版

```py

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128) # state_size = 128
print(cell.state_size) # 128

inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size
h0 = cell.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态，形状为(batch_size, state_size)
output, h1 = cell(inputs, h0) #调用call函数

print(h1.shape) # (32, 128)

```

## keras

```py

import tensorflow as tf
from tensorflow.keras import layers

cell = tf.keras.layers.SimpleRNN(units = 128)


```
