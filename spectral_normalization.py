from __future__ import print_function

import numpy as np

matrix = np.array([[-4, -33, -2],
       [-1,  0,  1],
       [ 2,  3,  14]])
def l2_normalize(matrix):
	return matrix / np.linalg.norm(matrix)

	   
class Spectral_Normalization:
	def __init__(self):
		self._u = np.ndarray(shape=3)
		
	def initialize_vector(self):
		height = matrix.shape[0]
		width = matrix.shape[1]

		self._u = l2_normalize(np.random.normal(0, 1,(height)))

	def _update_u_v(self):
		self.initialize_vector()
		w = matrix
		u = self._u
		
		height = w.shape[0]
		for _ in range(10):
			v = l2_normalize(w.dot(u))
			u = l2_normalize(w.dot(v))
			print("v: ")
			print(v)
			print("u: ")
			print(u)

		self._u = u
		w = w / u.dot(w.dot(v))
		print(w)
		
s = Spectral_Normalization()
s._update_u_v()