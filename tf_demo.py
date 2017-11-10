""" demo"""
import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 综述
# 是一个使用数据流图进行数值计算的开源软件库。图中的节点代表数学运算， 而图中的边则代表在这些节点之间传递的多维数组（tensor张量）。

# 1. 使用图（graph）来表示计算任务;
# 2. 使用类型化的多维数组（tensor）表示数据; 例如
a = 3  # a rank 0 tensor; this is a scalar with shape []
b = [1., 2., 3.]  # a rank 1 tensor; this is a vector with shape [3]
c = [[1., 2., 3.], [4., 5., 6.]]  # a rank 2 tensor; a matrix with shape [2, 3]
d = [[[1., 2., 3.]], [[7., 8., 9.]]]  # a rank 3 tensor with shape [2, 1, 3]
# 3. 在会话（Session）的上下文中执行图
# 4. 使用feed和fetch可以为任意操作赋值或者从中获取数据

# hello world
# 可计算图指一系列被排列成图节点的TensorFlow操作。每一个节点(操作)输入0或多个tensor产生0或多个tensor。
# 1. 构建
# 2. 运行
# 常量是一种没有输入，能输出它内部存储的值的节点
hello = tf.constant('Hello, TensorFlow!')
with tf.Session() as sess:
    print(sess.run(hello).decode())
# Hello, TensorFlow!

# 常量运算
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0, name="node2")  # 默认类型是tf.float32
print(node1.name)
print(node2.name)
print(node1, node2)
# Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
with tf.Session() as sess:
    print(sess.run([node1, node2]))
# [3.0, 4.0]
# node3 = tf.add(node1,node2)
node3 = node1 + node2
print(node3)
node4 = tf.constant([1.0, 2.0])
print(node4)
# Tensor("Add:0", shape=(), dtype=float32)
with tf.Session() as sess:
    print(sess.run(node3))
# 7.0

# 变量和常量
state = tf.Variable(0.0, name='counter')  # 初始值0.0 ,名字counter
print(state.name)
one = tf.constant(1.0)  # 常量
print(one)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)  # new_value参数替换state
init = tf.global_variables_initializer()  # 初始化变量,变量才会起作用
with tf.Session() as sess:
    sess.run(init)
    for i in range(3):  # 循环执行sess.run()三次,累加三次
        sess.run(update)
        print(sess.run(state))  # 不能直接输出state,必须使用sess.run(state)

# 传入值 placeholder

input1 = tf.placeholder(tf.float32)  # 需要保证placeholer的类型
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)  # 乘法运算
with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [[7.0]], input2: [[2.0]]}))  # 以feed_dict形式传入数值

a_1 = tf.placeholder(tf.float32)
b_1 = tf.placeholder(tf.float32)
times = tf.placeholder(tf.float32)
adder_node = a_1 + b_1
add_and_triple = adder_node * times

with tf.Session() as sess:
    print(sess.run(adder_node, {a_1: 3, b_1: 4.5}))
    # print(sess.run(adder_node, {a_1: [1, 3], b_1: [2, 4]}))
    print(sess.run(add_and_triple, {a_1: 3, b_1: 4.5, times: 3}))

# 构建模型

W_g = tf.Variable([.3], dtype=tf.float32)
b_g = tf.Variable([-.3], dtype=tf.float32)
x_g = tf.placeholder(tf.float32)
# y = 0.3 * x - 0.3
linear_model = W_g * x_g + b_g

# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(linear_model, {x_g: [1, 2, 3, 4, 5, 6]}))

# 已经有一个线性模型了，但是我们不知道这个模型的好坏，需要一个y提供一个准确的值来确认模型的好坏，通过loss值来衡量
y_g = tf.placeholder(tf.float32)

# 方差分析法
# 实际值与模型计算值 差的平方 [(y1-(Wx1+b))^2,(y1-(Wx1+b))^2,(y1-(Wx1+b))^2,(y1-(Wx1+b))^2]
squared_deltas = tf.square(linear_model - y_g)
# 求和
loss = tf.reduce_sum(squared_deltas)
# loss越接近于0 模型越符合

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(loss, {x_g: [1, 2, 3, 4], y_g: [0, -1, -2, -3]}))
    # 23.66 这个误差太大了

# 观察x_g和y_g的数据，我们"真智能"可以猜一下

fixW = tf.assign(W_g, [-1.])
fixb = tf.assign(b_g, [1.])

with tf.Session() as sess:
    sess.run(init)
    sess.run([fixW, fixb])  # 替换模型中的变量值
    print(sess.run(loss, {x_g: [1, 2, 3, 4], y_g: [0, -1, -2, -3]}))
    # loss 为0 可以说是一个完美的模型了,但是这不是机器学习

# 训练API train.API
# optimizer 类 能够根据传进去的loss的变化，将模型中的变量值逐渐调整
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# data
x_train = [1, 2, 3, 4]
y_train = [0, 1, 2, 3]

with tf.Session() as sess:
    sess.run(init)
    for i_g in range(1000):
        sess.run(train, {x_g: x_train, y_g: y_train})
    # evaluate training accuracy
    curr_W, curr_b, curr_loss = sess.run([W_g, b_g, loss], {x_g: x_train, y_g: y_train})
    print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))
    # loss 已经非常接近0了

# 整理一下代码
