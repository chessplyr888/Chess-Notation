import numpy as np
import cv2



# Return the left and right edges of a given row/col
def getEdgeCorners(corners):
	length = len(corners)
	last = length - 1

	changeInX = corners[last[0]] - corners[0[0]]
	changeInY = corners[last[1]] - corners[0[1]]

	avgX = changeInX/len(corners)
	avgY = changeInY/len(corners)

	# Return a list of edges on opposite sides
	return [[corners[0[0]] - avgX, corners[0[1]] - avgY], [corners[last[0]] + avgX, corners[last[1]] + avgY]]

# Return a list of all the corers, including the edges of the board
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
	cols = []
	for i in range(0,8): # Number of columns
		cols[i] = []
		for j in enumerate(rows)
			cols[i].append(rows[j][i])

	for i in enumerate(cols):
		temp = getEdgeCorners(cols[i])
		cols[i].insert(0, temp[0])
		cols[i].extend(temp[1])

	# Consolidate cols into newCorners
	newCorners = []
	for i in range(0,8)
		for j in enumerate(cols)
			newCorners.append(cols[j][i])

	return newCorners


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

	fullCorners = edgeCorners(corners)

	if found is True:
		cv2.drawChessboardCorners(frame, pattern_size, fullCorners, found)

	# Display the resulting frame
	cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()