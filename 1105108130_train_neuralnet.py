# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
from two_layer_net import TwoLayerNet
from keras.datasets import cifar10
from keras.utils import to_categorical

np.random.seed(10)

(x_train,t_train),(x_test, t_test)=cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
t_train = to_categorical(t_train)
t_test = to_categorical(t_test)
x_test = x_test.reshape(-1,32*32*3)
x_train = x_train.reshape(-1,32*32*3)

network = TwoLayerNet(input_size=x_train.shape[1], hidden_size=200, output_size=t_train.shape[1])

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.2

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
