import numpy as np
import cv2

def checkBounds(axis, val):
	if val < 0:
			return 0
	if axis is "x":
		if val > cap.get(3):
			return cap.get(3)
	if axis is "y":
		if val > cap.get(4):
			return cap.get(4)
	return val


# Return the left/right or top/bottom edges of a given row/col
def getEdgeCornersRows(corners):
	length = len(corners)
	last = length - 1

	# print corners

	changeInX = corners[last][0][0] - corners[0][0][0]
	changeInY = corners[last][0][1] - corners[0][0][1]

	# print changeInX, changeInY, (changeInY/changeInX)

	avgX = changeInX/last
	avgY = changeInY/last

	lowX = checkBounds("x", corners[0][0][0] - avgX)
	lowY = checkBounds("y", corners[0][0][1] - avgY)
	highX = checkBounds("x", corners[last][0][0] + avgX)
	highY = checkBounds("y", corners[last][0][1] + avgY)

	# Return a list of edges on opposite sides
	return [[lowX, lowY], [highX, highY]]

def getEdgeCornersCols(corners):
	length = len(corners)
	last = length - 1

	changeInX = corners[last][0] - corners[0][0]
	changeInY = corners[last][1] - corners[0][1]

	avgX = changeInX/last
	avgY = changeInY/last

	lowX = checkBounds("x", corners[0][0] - avgX)
	lowY = checkBounds("y", corners[0][1] - avgY)
	highX = checkBounds("x", corners[last][0] + avgX)
	highY = checkBounds("y", corners[last][1] + avgY)

	# Return a list of edges on opposite sides
	return [lowX, lowY], [highX, highY]

# Return a list of all the corers, including the edges of the board
def edgeCorners(corners):
	# Corners should be an nparray of shape (49, 1, 2)
	newCorners = []

	print "ROWS"

	# print corners.shape, corners.ndim

	# Extrapolate rows row0 - row6
	rows = []
	for i in range(0,7):
		rows.append((corners[0 + 7*i:7 + 7*i]))
		# print corners[0 + 7*i:7 + 7*i]

	# print rows[0].shape, rows[0].ndim

	# print rows

	newRows = []

	for i in range(0, len(rows)):
		temp = getEdgeCornersRows(rows[i])
		# print temp[0]
		newRows.append(temp[0])
		for j in range(0, len(rows[i])):
			# print rows[i][j][0]
			newRows.append([rows[i][j][0][0], rows[i][j][0][1]])
		newRows.append(temp[1])

	# print newRows, len(newRows)

	# Extrapolate cols col0 - col8

	print "COLS"

	cols = []
	for i in range(0, 9):
		temp = []
		for j in range(0, len(newRows)):
			if j % 9 is i:
				temp.append((newRows[j]))
		cols.append(temp)

	for i in range(0, len(cols)):
		# print "before", cols[i], len(cols[i])
		temp = getEdgeCornersCols(cols[i])
		# print temp
		cols[i].insert(0, temp[0])
		cols[i].append(temp[1])
		# print "after", cols[i], len(cols[i])

	# Convert newCorners to numpy array
	
	# print len(cols), len(cols[0])

	for i in range(0, len(cols)):
		for j in range(0, len(cols[i])):
			newCorners.append(cols[i][j])

	# print newCorners, len(newCorners)

	fullCorners = np.array([newCorners])

	# fullCorners should be an nparray of shape (64, 1, 2)
	return fullCorners



cap = cv2.VideoCapture(0)
print cap.get(3), cap.get(4)

pattern_size = (7,7) # Inner corners of a chessboard
full_pattern_size = (9,9) # All corners of a chessboard

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
		fullCorners = edgeCorners(corners)
		# print corners.shape, corners.ndim, fullCorners, fullCorners.shape, fullCorners.ndim
		# Change shape from (1, 81, 2) to (81, 1, 2)
		# fullCorners.shape = (81, 1, 2)
		# print fullCorners, fullCorners.shape, fullCorners.ndim
		# cv2.drawChessboardCorners(frame, full_pattern_size, fullCorners, found)

		# Attempt without drawChessboardCorners, manually drawing the corners
		# for i in fullCorners:
		# 	x, y = i.ravel()
		# 	cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

		for i in range(0, len(fullCorners[0])):
			print fullCorners[0][i]
			x = int(fullCorners[0][i][0])
			y = int(fullCorners[0][i][1])
			cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)


	# Display the resulting frame
	cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
	cv2.imshow('frame', frame)
	# q to quit
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
