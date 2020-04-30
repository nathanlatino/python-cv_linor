import numpy as np
import pyscreenshot as ImageGrab
import cv2

from arrow import Arrow
from pretreatment import roi, white_mask, morpho_process
from metadata import MetaData
from draw import draw_lanes, draw_lines
from trapeze import Trapeze


def process_img(image, meta, arrow):
	original_image = image

	# mask line
	processed_img = white_mask(original_image)

	# Erode, dilate
	processed_img = morpho_process(processed_img)

	# Canny
	threshold1 = 200
	threshold2 = 300
	processed_img = cv2.Canny(processed_img, threshold1, threshold2)

	# filtre gaussian
	kernel = (5,5)
	sigmax = 0
	processed_img = cv2.GaussianBlur(processed_img, kernel, sigmax)

	# mask zone trapeze
	trapeze = Trapeze(meta.width, meta.height)
	vertices = [np.array([trapeze.hl1, trapeze.hl2, trapeze.hl3,
						 trapeze.hr1, trapeze.hr2, trapeze.hr3], np.int32)]
	processed_img = roi(processed_img, vertices)


	# get lines with Hough algo
	rho = 1
	theta = np.pi / 180
	threshold = 150
	min_line_length = 20
	max_line_gap = 15
	lines = cv2.HoughLinesP(processed_img, rho, theta , threshold, np.array([]), min_line_length, max_line_gap)


	l1, l2 = draw_lanes(original_image, lines)
	# count = 0
	# for idx, i in enumerate(lines):
	# 	for xyxy in i:
	# 		count += 1
	# 		cv2.line(original_image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), [255, 0, 0], 3)
	# 		if count == 2:
	# 			break
	print("test l1:", l1)
	print("test l2:", l2)
	if l1 is None and l2 is None:
		l1 = [0, 0, 0, 0]
		l2 = [0, 0, 0, 0]
	m1 = (l1[3] - l1[1]) / (l1[2] - l1[0])
	b1 = l1[3] - m1 * l1[2]

	m2 = (l2[3] - l2[1]) / (l2[2] - l2[0])
	b2 = l2[3] - m2 * l2[2]

	x = (b1 - b2) / (m2 - m1)
	y = m1 * x + b1

	print(x)
	arrow.add_point(int(x))

	cv2.circle(original_image, (int(x), int(y)), 2, [0, 0, 255], 10)
	cv2.line(original_image, (l1[0], l1[1]), (l1[2], l1[3]), [0, 255, 0], 2)
	cv2.line(original_image, (l2[0], l2[1]), (l2[2], l2[3]), [0, 255, 0], 2)
	for coords in lines:
		coords = coords[0]
		cv2.line(processed_img, (coords[0], coords[1]), (coords[2], coords[3]), [255, 0, 0], 3)

	draw_lines(processed_img, lines)
	degree = int(arrow.degree_turn())
	d = 10
	middle = int(meta.width / 2)
	# print(middle)
	# print(degree)
	# print(middle + degree)
	cv2.arrowedLine(original_image, (middle, d), (middle + degree, d), [0, 0, 255], 2)
	return processed_img, original_image


def screen_record():
	# last_time = time.time()
	while True:
		screen = np.array(ImageGrab.grab((0, 40, 1920, 1024)))
		# frame = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
		# cv2.putText(frame, f"time: {int((time.time() - last_time)*1000)} ms", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
		# last_time = time.time()
		frame = process_img(screen)
		cv2.imshow('window', frame)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break


def video_record(video):
	meta = MetaData(video)
	cap = cv2.VideoCapture(video)
	arrow = Arrow(meta.width)
	cpt_frame = 0
	while cap.isOpened():
		ret, frame = cap.read()
		cpt_frame += 1
		if ret:
			processed_frame, original_image = process_img(frame, meta, arrow)
			cv2.imshow('window', original_image)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	# screen_record()
	video_record("./resources/road.mp4")
	exit(0)
