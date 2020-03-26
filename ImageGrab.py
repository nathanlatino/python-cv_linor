import numpy as np
import pyscreenshot as ImageGrab
import cv2
from numpy import ones,vstack
from numpy.linalg import lstsq
from statistics import mean
import time

from metadata import MetaData


def roi(img, vertices):
    #blank mask:
    mask = np.zeros_like(img)
    # fill the mask
    cv2.fillPoly(mask, vertices, 255)
    # now only show the area that is the mask
    masked = cv2.bitwise_and(img, mask)
    return masked

def white_mask(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	low_white = np.array([100,0,210])
	upper_white = np.array([250, 255,255])

	mask = cv2.inRange(hsv, low_white, upper_white)
	return mask

def draw_lines(img, lines):
	for line in lines:
		coords = line[0]
		cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255,255,255], 3)

def morpho_process(img):
	# delete lines
	img2 = morph(img, (4,4))
	# strengthen intersections
	img2 = morph(img2, (9, 9), 3, 'd')
	img2 = morph(img2, (4, 4), 2)
	# close remaining blobs
	img2 = morph(img2, (12, 12))
	img2 = morph(img2, (12, 12), mode='d')
	return img2

def morph(img, size=(3,3), iteration=1, mode='e'):
	kernel = np.ones(size, np.uint8)
	if mode == 'e' :
		return cv2.erode(img, kernel, iterations=iteration)
	else:
		return cv2.dilate(img, kernel, iterations=iteration)


def draw_lanes(img, lines, color=[0, 255, 255], thickness = 3):
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
				x1s.append(data[2][0])
				y1s.append(data[2][1])
				x2s.append(data[2][2])
				y2s.append(data[2][3])
			return int(mean(x1s)), int(mean(y1s)), int(mean(x2s)), int(mean(y2s))

		l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
		l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])

		return [l1_x1, l1_y1, l1_x2, l1_y2], [l2_x1, l2_y1, l2_x2, l2_y2]

	except Exception as e:
		print(str(e))


def process_img(image, meta):
	original_image = image
	processed_img = white_mask(original_image)
	processed_img = morpho_process(processed_img)
	cv2.imshow("morph", processed_img)
	# processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# processed_img = cv2.Canny(processed_img, 200, 300)
	vertices = np.array([[0, meta.height],
						 [0, meta.height/7*6],
						 [meta.width/3, meta.height/4*3],
						 [meta.width/2, meta.height/4*3],
						 [meta.width, meta.height/7*6],
						 [meta.width, meta.height],
						 ],
						np.int32)
	processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)
	processed_img = roi(processed_img, [vertices])
	cv2.imshow("Processed image", processed_img)

	lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 150, np.array([]), 20, 15)
	try:
		# l1, l2 = draw_lanes(original_image, lines)
		for idx, i in enumerate(lines):
			for xyxy in i:
				cv2.line(original_image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), [255, 0, 0], 3)




		# cv2.line(original_image, (l1[0], l1[1]), (l1[2], l1[3]), [0, 255, 0], 10)
		# cv2.line(original_image, (l2[0], l2[1]), (l2[2], l2[3]), [0, 255, 0], 10)
	except Exception as e:
		print(str(e))
		pass
	# try:
	# 	for coords in lines:
	# 		coords = coords[0]
	# 		try:
	# 			cv2.line(processed_img, (coords[0], coords[1]), (coords[2], coords[3]), [255, 0, 0], 3)
	#
	#
	# 		except Exception as e:
	# 			print(str(e))
	# except Exception as e:
	# 	pass
	# draw_lines(processed_img, lines)

	return processed_img,original_image


def screen_record():
	# last_time = time.time()
	while True:
		screen = np.array(ImageGrab.grab((0,40,800,640)))
		# frame = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
		# cv2.putText(frame, f"time: {int((time.time() - last_time)*1000)} ms", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
		# last_time = time.time()
		frame = process_img(screen)
		cv2.imshow('window', frame)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break

def video_road(video):
	meta = MetaData(video)
	cap = cv2.VideoCapture(video)
	cpt_frame = 0
	while cap.isOpened():
		ret, frame = cap.read()
		cpt_frame += 1
		if ret:
			# screen = np.array(ImageGrab.grab((0,40,800,640)))
			# frame = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
			# cv2.putText(frame, f"time: {int((time.time() - last_time)*1000)} ms", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
			# last_time = time.time()
			processed_frame, original_image = process_img(frame, meta)
			# cv2.imshow('window', processed_frame)

			cv2.imshow('window', original_image)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	# screen_record()
	video_road("./resources/road.mp4")
	exit(0)
