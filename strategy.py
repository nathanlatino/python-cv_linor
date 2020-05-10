from abc import ABC, abstractmethod

import numpy as np

from point import Point


class Strategy(ABC):
	@abstractmethod
	def do_algorithme(self, data: list):
		pass

class means_lines(Strategy):
	def do_algorithme(self, lane_data: list):
		pass


class means_points(Strategy):
	def do_algorithme(self, lane_data: list):
		pass
