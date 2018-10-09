import tensorflow as tf

matrix = tf.Variable([[-4.0, -33.0, -2.0],
       [-1.0,  0.0,  1.0],
       [ 2.0,  3.0,  14.0]], tf.float32)

class Spectral_Normalization:
    def __init__(self, matrix, iterations=1):
        self.iterations = iterations
        self.matrix = matrix
        self.u = None
        self.v = None

    def initialize_vector(self):
        matrix_shape = matrix.shape.as_list()
        self.matrix = tf.reshape(matrix, [-1, matrix_shape[-1]])
        self.u = tf.get_variable("u", [1, matrix_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    def _update_u_v(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        print("Original matrix is:\n %s\n" % (sess.run(self.matrix)))

        self.initialize_vector()
        for i in range(self.iterations):
            self.v = tf.nn.l2_normalize(tf.matmul(self.u, tf.transpose(self.matrix)))
            self.u = tf.nn.l2_normalize(tf.matmul(self.v, self.matrix))

        sigma = tf.matmul(tf.matmul(self.v, self.matrix), tf.transpose(self.u))
        matrix_norm = tf.reshape(self.matrix / sigma, matrix.shape.as_list())

        sess.run(tf.global_variables_initializer())
        print("Sigma is: %s\n" %(sess.run(sigma)[0][0]))
        print("The matrix after the iterations is:\n %s\n" % (sess.run(matrix_norm)))
        print("L2 norm of the matrix after the iterations is: %f\n" %(sess.run(tf.norm((matrix_norm), ord=2))))

s = Spectral_Normalization(matrix, 11)
s._update_u_v()
