# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 21:59:33 2017

@author: ljm
"""

import tensorflow as tf



def imageprepare():
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    file_name='model/3.png'#导入自己的图片地址
    #in terminal 'mogrify -format png *.jpg' convert jpg to png
    im = Image.open(file_name).convert('L')


    im.save("model/_3.png")
    #im.show(im)
    tv = list(im.getdata()) #get pixel values

    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [ (255-x)*1.0/255.0 for x in tv] 
    #print(tva)
    return tva



x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, w) + b)
y_ = tf.placeholder('float', [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000) :
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})
    
#print(sess.run(b))
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))

saver = tf.train.Saver();
saver.save(sess, 'model/model.ckpt')

img = imageprepare()
with tf.Session() as sess :
    sess.run(tf.initialize_all_variables())
    saver.restore(sess, 'model/model.ckpt')
    #prediction = tf.argmax(y_, 1)
    #result = prediction.eval(feed_dict={x:[img]}, session = sess)
    print(sess.run(accuracy, feed_dict={x:img, y_:mnist.test.labels}))