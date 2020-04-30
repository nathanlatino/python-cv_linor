class Trapeze:
	def __init__(self, width, height):
		self.hl1 = [0, height]
		self.hl2 = [0, height / 7 * 6]
		self.hl3 = [width / 3, height / 4 * 3]
		self.hr1 = [width / 2,height / 4 * 3]
		self.hr2 = [width, height / 7 * 6]
		self.hr3 = [width, height]
