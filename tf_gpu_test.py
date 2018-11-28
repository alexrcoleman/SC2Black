import tensorflow as tf
with tf.device('/gpu:0'):
    v1 = []
    v2 = []
    for x in range(1,20000):
        v1.append(1.0)
        v2.append(1.0)
    print(v1)
    print(v2)
    a = tf.constant(v1, shape=[10000,2], name='a')
    b = tf.constant(v2, shape=[2,10000], name='b')
    print("MULT:")
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print (sess.run(c))
