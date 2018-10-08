from __future__ import print_function

import numpy as np

matrix = np.array([[4, 33, 2],
       [1,  0,  1],
       [ 2,  3,  14]])




class Spectral_Normalization:
    def __init__(self, matrix, iterations):
        self.iterations = iterations
        self.matrix = matrix
        self.height = matrix.shape[0]
        self.width = matrix.shape[1]
        self._u = np.ndarray(shape = self.height)

    def l2_normalize(self, matrix):
        return matrix / np.linalg.norm(matrix)

    def initialize_vector(self):
        self._u = self.l2_normalize(np.random.normal(0, 1,(self.height)))

    def _update_u_v(self):
        self.initialize_vector()
        w = matrix
        u = self._u
        print("In round 0. u is: %s" % (u))
        height = w.shape[0]
        for i in range(self.iterations):
            v = self.l2_normalize(w.dot(u))
            u = self.l2_normalize(w.dot(v))
            print("In round %s. v is: %s u is: %s" %(i+1,v,u))

        self._u = u
        w = w / u.dot(w.dot(v))
        print("\nOriginal matrix is: \n %s" % (self.matrix))
        print("\nUpdated matrix is: \n %s" %(w))

s = Spectral_Normalization(matrix, 110)
s._update_u_v()