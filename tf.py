import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np
import time

#input data
(x_train,t_train),(x_test,t_test) = cifar10.load_data()

#define
iterative = 20000 #總共訓練幾次
train_size = 50000 #測試資料大小
batch_size = 200 #抽出樣本的大小

#normalize
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

#one_hot
t_train = to_categorical(t_train)
t_test = to_categorical(t_test)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 32,32,3]) # 32x32*3
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs,[-1,32,32,3])

## conv1 layer ##
W_conv1 = weight_variable([5,5,3,32]) #patch 5x5,insize 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1) # output size 32*32*32
h_pool1 = max_pool_2x2(h_conv1) # output size = 16*16*32

## conv2 layer ##
W_conv2 = weight_variable([5,5,32,64]) # patch 5x5, insize 32, outsize 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #output size 16*16*64
h_pool2 = max_pool_2x2(h_conv2) #output size = 8*8*64

## func1 layer ##
W_fc1 = weight_variable([8*8*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,8*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,rate = 1 - keep_prob)

## func2 layer ##
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)
now = time.time()
for i in range(iterative):
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    sess.run(train_step, feed_dict={xs: x_batch, ys: t_batch, keep_prob: 0.5})
    loss = sess.run(cross_entropy, feed_dict={xs: x_batch, ys: t_batch, keep_prob: 0.5})
    if i % 50 == 0:
        tmp = int(time.time() - now)
        min_ = tmp/60
        sec_ = tmp%60
        print('次數　%000d, 測試資料準確率 %2.2f, 時間:%d:%02d' % (i, compute_accuracy(x_test[:5000], t_test[:5000])*100, min_, sec_))
        # print('次數:' , i , ',測試:' , compute_accuracy(x_test, t_test),',已使用時間:' , minute)