## 简介

## 1.pad_sequences

用于序列的维度统一，对维度不足的予以填充，对维度多的予以截断

```
from tensorflow.keras.preprocessing import sequence
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
```

## 2.Embedding

类似一种降维，如word2vec，将一些相同的词性的词压缩要同一个向量内


## 参考
1.https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/6.2-understanding-recurrent-neural-networks.ipynb
2.https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526
