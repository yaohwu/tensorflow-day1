"""linear_model"""
import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 构建模型
# data
x_train = [1, 2, 3, 4, 5]
y_train = [0, 1, 2, 3, 4]

with tf.name_scope("param"):
    W_g = tf.Variable([.3], dtype=tf.float32, name="W_g")
    b_g = tf.Variable([-.3], dtype=tf.float32, name="b_g")

with tf.name_scope("input"):
    x_g = tf.placeholder(tf.float32, name="x_g")
    y_g = tf.placeholder(tf.float32, name="y_g")

# y = 0.3 * x - 0.3

with tf.name_scope("linear"):
    linear_model = W_g * x_g + b_g
    tf.summary.histogram("c_W", W_g)
    tf.summary.histogram("c_b", b_g)

# 方差分析法 写loss函数
# 数学知识
with tf.name_scope("loss"):
    # 实际值与模型计算值 差的平方 [(y1-(Wx1+b))^2,(y1-(Wx1+b))^2,(y1-(Wx1+b))^2,(y1-(Wx1+b))^2]
    squared_deltas = tf.square(linear_model - y_g)
    # 求和
    loss = tf.reduce_sum(squared_deltas)
    tf.summary.scalar("loss", loss)
# loss越接近于0 模型越符合

# 训练API train.API
# optimizer 类 能够根据传进去的loss的变化，将模型中的变量值逐渐调整
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

# 选定可视化存储目录
if tf.gfile.Exists("./tmp/linear-logs"):
    tf.gfile.DeleteRecursively("./tmp/linear-logs")

# 初始化变量
init = tf.global_variables_initializer()

merged = tf.summary.merge_all()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./tmp/linear-logs", sess.graph)
    sess.run(init)
    for i_g in range(300):
        sess.run(train, {x_g: x_train, y_g: y_train})
        if i_g % 10 == 0:
            result = sess.run(merged, {x_g: x_train, y_g: y_train})
            writer.add_summary(result, i_g)
    # evaluate training accuracy
    curr_W, curr_b, curr_loss = sess.run([W_g, b_g, loss], {x_g: x_train, y_g: y_train})
    print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))

    # loss 已经非常接近0了
