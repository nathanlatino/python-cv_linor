import cv2

from metadata import MetaData

def process_img(frame, meta):
	img_ = cv2.resize(frame, (0,0), fx=1, fy=1)
	img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

	img2 = img1

	sift = cv2.xfeatures2d.SIFT_create()

	kp1, des1 = sift.detectAndCompute(img1, None)
	kp2, des2 = sift.detectAndCompute(img2, None)

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
			cv2.imshow('window', original_image)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	# screen_record()
	video_road("./resources/road.mp4")
	exit(0)
