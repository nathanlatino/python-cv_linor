import cv2
import numpy as np
from numpy import ones, vstack
from numpy.linalg import lstsq

def draw_lines(img, lines, color=[255, 255, 255], thickness=3):
	for line in lines:
		coords = line[0]
		cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), color, thickness)

def draw_lanes(img, lines, color=[0, 255, 255], thickness=3):
	try:
		ys = []
		for i in lines:
			for ii in i:
				ys += [ii[1], ii[3]]
		min_y = min(ys)
		max_y = 600
		new_lines = []
		line_dict = {}

		for idx, i in enumerate(lines):
			for xyxy in i:
				# These four lines:
				# modified from http://stackoverflow.com/questions/21565994/method-to-return-the-equation-of-a-straight-line-given-two-points
				# Used to calculate the definition of a line, given two sets of coords.
				x_coords = (xyxy[0], xyxy[2])
				y_coords = (xyxy[1], xyxy[3])
				A = vstack([x_coords, ones(len(x_coords))]).T
				m, b = lstsq(A, y_coords)[0]

				# Calculating our new, and improved, xs
				x1 = (min_y - b) / m
				x2 = (max_y - b) / m

				line_dict[idx] = [m, b, [int(x1), min_y, int(x2), max_y]]
				new_lines.append([int(x1), min_y, int(x2), max_y])

		final_lanes = {}

		for idx in line_dict:
			final_lanes_copy = final_lanes.copy()
			m = line_dict[idx][0]
			b = line_dict[idx][1]
			line = line_dict[idx][2]

			if len(final_lanes) == 0:
				final_lanes[m] = [[m, b, line]]

			else:
				found_copy = False

				for other_ms in final_lanes_copy:

					if not found_copy:
						if abs(other_ms * 1.2) > abs(m) > abs(other_ms * 0.8):
							if abs(final_lanes_copy[other_ms][0][1] * 1.2) > abs(b) > abs(
									final_lanes_copy[other_ms][0][1] * 0.8):
								final_lanes[other_ms].append([m, b, line])
								found_copy = True
								break
						else:
							final_lanes[m] = [[m, b, line]]

		line_counter = {}

		for lanes in final_lanes:
			line_counter[lanes] = len(final_lanes[lanes])

		top_lanes = sorted(line_counter.items(), key=lambda item: item[1])[::-1][:2]

		lane1_id = top_lanes[0][0]
		lane2_id = top_lanes[1][0]

		def average_lane(lane_data):
			x1s = []
			y1s = []
			x2s = []
			y2s = []
			for data in lane_data:
				h = data[2][1] - data[2][3]
				if abs(h) > 50:
					x1s.append(data[2][0])
					y1s.append(data[2][1])
					x2s.append(data[2][2])
					y2s.append(data[2][3])

			return [int(np.mean(x1s)), int(np.mean(y1s)), int(np.mean(x2s)), int(np.mean(y2s))]

		return average_lane(final_lanes[lane1_id]), average_lane(final_lanes[lane2_id])

	except Exception as e:
		print("test ", str(e))
