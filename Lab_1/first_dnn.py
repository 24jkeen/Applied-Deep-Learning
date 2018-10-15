############################################################
#                                                          #
#  Code for Lab 1: Your First Fully Connected Layer  #
#                                                          #
############################################################


import tensorflow as tf
import os
import os.path
import numpy as np
import pandas as pd

sess = tf.Session()

data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", sep=",",
                   names=["sepal_length", "sepal_width", "petal_length", "petal_width", "iris_class"])
#

np.random.seed(0)
data = data.sample(frac=1).reset_index(drop=True)
#
# all_x = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
#
# all_y = pd.get_dummies(data.iris_class)
#


def sample(data, start, end):
    data = data.sample(frac=1).reset_index(drop=True)
    all_x = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    all_y = pd.get_dummies(data.iris_class)

    train_x = all_x[start:end]
    test_x = all_x[end:]
    train_y = all_y[start:end]
    test_y = all_y[end:]
    
    n_x = len(all_x.columns)
    n_y = len(all_y.columns)
    
    return train_x, test_x, train_y, test_y, n_x, n_y, all_y

train_x, test_x, train_y, test_y, n_x, n_y, all_y = sample(data, 0, 100)
#
#x = tf.placeholder(tf.float32, shape=[None, n_x])
#y = tf.placeholder(tf.float32, shape=[None, n_y])
#w = tf.Variable(tf.zeros(( n_x, n_y)))
#b = tf.Variable(tf.zeros((n_y)))
#
#
#y_hat = tf.nn.softmax(tf.subtract(tf.matmul( x, w  ), b))
#cost = tf.reduce_mean( -tf.reduce_sum(y * tf.log( y_hat), axis=1 )  )
#optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
#
#
#with tf.Session() as sess:
#	sess.run(tf.global_variables_initializer())
#	for epoch in range(10000):
#	    sess.run([optimizer], feed_dict={x: train_x, y: train_y})
#            if epoch % 100 == 0:
#		predictions = sess.run(y_hat, feed_dict={x: test_x, y: test_y})
#		predictions_array = np.zeros_like(predictions)
#		predictions_array[np.arange(len(predictions)), predictions.argmax(1)] = 1
#		predictions_accuracy = sum(list(map(lambda x: sum(x[0] * x[1]), zip(predictions_array, np.array(test_y)))))
#		print("The accuract for epoch {} is {}".format(epoch, predictions_accuracy/50))
#
#
g = tf.get_default_graph()
with g.as_default():
	logs_path = "./logs/"
	h1 = 10
	h2 = 20
	h3 = 10

	x = tf.placeholder(tf.float32, shape=[None, n_x])
	y = tf.placeholder(tf.float32, shape=[None, n_y])

	w_fc1 = tf.Variable(tf.truncated_normal([n_x, h1], stddev=0.1))
	b_fc1 = tf.Variable( tf.constant(0.1, shape=[h1]  ))
	h_fc1 = tf.nn.relu(tf.matmul(x, w_fc1) + b_fc1)

	w_fc2 = tf.Variable(tf.truncated_normal([h1, h2], stddev=0.1))
	b_fc2 = tf.Variable(tf.constant(0.1, shape=[h2]  ))
	h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)

	w_fc3 = tf.Variable(tf.truncated_normal([h2, h3], stddev=0.1))
	b_fc3 = tf.Variable( tf.constant(0.1, shape=[h3]  ))
	h_fc3 = tf.nn.relu(tf.matmul(h_fc2, w_fc3) + b_fc3)

	w = tf.Variable(tf.zeros(( h3, n_y)))
	b = tf.Variable(tf.zeros((n_y)))
	y_hat = tf.nn.softmax(tf.subtract(tf.matmul( h_fc3, w  ), b))

	cost = tf.reduce_mean( -tf.reduce_sum(y * tf.log( y_hat), axis=1 )  )

	with tf.name_scope('loss'):
	    cost_fcn = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_hat, scope="Cost_Function")
	    tf.summary.scalar('loss', cost_fcn)

	with tf.name_scope("accuracy"):
	    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1)),tf.float32))
	    tf.summary.scalar('accuracy', accuracy)

	merged = tf.summary.merge_all()
	optimizer = tf.train.AdagradOptimizer(0.1).minimize(cost_fcn)

	train_writer = tf.summary.FileWriter(logs_path + '/train')
	test_writer = tf.summary.FileWriter(logs_path + '/test')

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(3000):
		    _, summary_train = sess.run([optimizer, merged], feed_dict={x: train_x, y: train_y})
		    train_writer.add_summary(summary_train, epoch)
		    #test_writer.add_summary(summary_test, epoch)
		    if (epoch % 100) == 0:
			pred = sess.run(y_hat, feed_dict={x: test_x, y: test_y})
			acc = sess.run([accuracy],feed_dict={x: test_x, y: test_y})
			print("Accuracy at epoch {} is {} %".format(epoch,  str(100*acc[0])))






















