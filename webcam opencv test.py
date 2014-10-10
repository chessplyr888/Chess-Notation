import numpy as np
import cv2

class Square(object):
	"""docstring for Square"""
	def __init__(self, corners):
		super(Square, self).__init__()
		self.boxCorners[] = corners 
		

# TODO Change extends to insert
# This must happen because if the board is oriented diagonally, the sort method would break

def getEdgeCorners(corners):
	changeInX = corners[6[0]] - corners[0[0]]
	changeInY = corners[6[1]] - corners[0[1]]

	avgX = changeInX/7
	avgY = changeInY/7

	# Return a list of edges on opposite sides
	return [[corners[0[0]] - avgX, corners[0[1]] - avgY], [corners[6[0]] + avgX, corners[6[1]] + avgY]]

def edgeCorners(corners):
	# Extrapolate rows row0 - row6
	rows = []
	for i in range(0,6):
		rows.append((corners[0 + 7*i:6 + 7*i]))

	for i in enumerate(rows):
		temp = getEdgeCorners(rows[i])
		rows[i].insert(0, temp[0])
		rows[i].extend(temp[1])

	# Extrapolate cols col0 - col8
	for i in range(0,8): # Number of columns
		col = []
		for j in enumerate(corners):
			if j % 10 is i:
				col.append(corners[j])
		corners.extend(getEdgeCorners(col))

	# Final resort
	corners.sort()

	return corners


cap = cv2.VideoCapture(0)

pattern_size = (7,7) # Inner corners of a chessboard

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	base = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# findChessboardCorners + drawChessboardCorners
	# When finding corners, performance takes a major hit -> fps drops
	found, corners = cv2.findChessboardCorners(frame, pattern_size)
	print found, corners

	if found is True:
		cv2.drawChessboardCorners(frame, pattern_size, corners, found)

	# Display the resulting frame
	cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()