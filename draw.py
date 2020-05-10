import cv2

def draw_lines(img, lines, color=[255, 255, 255], thickness=3):
	for line in lines:
		coords = line[0]
		cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), color, thickness)

def draw_all_lines(img, lines, color=[255, 0, 0], thickness=3):
	for idx, i in enumerate(lines):
		for xyxy in i:
			cv2.line(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, thickness)


def draw_arrow(img, arrow, width, decalage=10, color=[0, 0, 255], thickness=2):
	degree = int(arrow.degree_turn())
	middle = int(width / 2)
	try:
		cv2.arrowedLine(img, (middle, decalage), (middle + degree, decalage), color, thickness)
	except:
		pass

def draw_infos(img, p, l1, l2) :
	cv2.circle(img, (int(p.x), int(p.y)), 2, [0, 0, 255], 10)
	cv2.line(img, (l1[0], l1[1]), (l1[2], l1[3]), [0, 255, 0], 2)
	cv2.line(img, (l2[0], l2[1]), (l2[2], l2[3]), [0, 255, 0], 2)
