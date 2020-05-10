import numpy as np
class Arrow:
	def __init__(self, width):
		self.x = []
		self.x.append(0)
		self.w = width

	def degree_turn(self):
		return np.mean(self.x)-self.w/2

	def add_point(self,x):
		if len(self.x) > 100:
			self.x.pop(0)
		self.x.append(x)
