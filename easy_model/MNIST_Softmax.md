#最简单的TF运用，利用y=softmax(w*x+b)求分类
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data/",one_hot=True)

x = tf.placeholder('float',[None,784]) #该占位符第一维可以是任意长度，表示图像数据可以是28*28=784的n张图

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#w和b是需要学习的值，初始化为0

y = tf.nn.softmax(tf.matmul(x,W)+b)
#模型即计算公式,y是预测值 [*,784]*[784,10]+[10]
y_ = tf.placeholder('float',[None,10])
#y_是真实值

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_predict = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_predict,"float"))

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(10000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
    if i%1000 == 0:
    	acc=sess.run(accuracy,feed_dict={x:mnist.validation.images,y_:mnist.validation.labels})
    	#训练1000次后(更新的参数W,b)，对整体validation的测试
    	print('step %d is %g'%(i,acc))
    	#上两行代码等价于:
    	#print(accuracy.eval(feed_dict={x:mnist.validation.images,y_:mnist.validation.labels}))
    

accFinal=sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}) 
print(accFinal)
sess.close()
#准确率为0.9129
