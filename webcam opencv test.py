import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	base = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# Canny Edge Detector
	# img = cv2.Canny(base,100,200)
	
	# SIFT
	# sift = cv2.SIFT()
	# keypoints = sift.detect(base, None)
	# img = cv2.drawKeypoints(base, keypoints)

	# Harris Corner Detector
	base = np.float32(base) # Convert to float32 for cornerHarris
	corner = cv2.cornerHarris(base, 2, 3, 0.04)
	corner = cv2.dilate(corner, None) # Dilute to easily identify corners
	frame[corner > 0.05*corner.max()] = [0, 0, 255]
	img = frame

	# TODO Draw Hough Lines with through corners

	# Display the resulting frame
	cv2.imshow('frame',img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()