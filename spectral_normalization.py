import tensorflow as tf
from keras import backend as K

sess = tf.Session()

matrix = K.variable(value = [[-4.0, -33.0, -2.0],
       [-1.0,  0.0,  1.0],
       [ 2.0,  3.0,  14.0]], name='matrix')

matrix_shape = matrix.shape.as_list()

u = K.variable(value=K.random_normal([1, matrix_shape[1]]), name='u')
v = K.variable(value=K.random_normal([1, matrix_shape[0]]), name='v')

sess.run(tf.global_variables_initializer())

print("Original matrix is:\n %s\n" % (sess.run(matrix)))
iterations = 10
for i in range(iterations):
    v = K.update(v, K.l2_normalize(K.dot(u, K.transpose(matrix))))
    u = K.update(u, K.l2_normalize(K.dot(v, matrix)))

sigma = K.dot(K.dot(v, matrix), K.transpose(u))
matrix_norm = matrix / sigma

print("Sigma is: %s\n" %(sess.run(sigma)[0][0]))
print("The matrix after %s iterations is:\n %s\n" % (iterations, sess.run(matrix_norm)))
print("2 norm of the normalized matrix after %s iterations (tensorflow): %f\n" % (iterations, sess.run(tf.norm((matrix_norm), ord=2))))

m_n = sess.run(matrix_norm)
import numpy as np
print("2 norm (numpy):", np.sqrt(np.linalg.eigvalsh(m_n.dot(m_n.T))[-1]))