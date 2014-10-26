# TEST.PY IS A FILE FOR TESTING ALGORITHMS ON A SINGLE STATIC IMAGE INSTEAD OF A WEBCAM

# NEEDS TO BASED OFF TUPLES LOOK AT THE MATRIX FORMAT


import numpy as np
import cv2



# Return the left and right edges of a given row/col
def getEdgeCornersRows(corners):
	length = len(corners)
	last = length - 1

	# print corners

	changeInX = corners[last][0][0] - corners[0][0][0]
	changeInY = corners[last][0][1] - corners[0][0][1]

	# print changeInX, changeInY, (changeInY/changeInX)

	avgX = changeInX/length
	avgY = changeInY/length

	# Return a list of edges on opposite sides
	return [[corners[0][0][0] - avgX, corners[0][0][1] - avgY], [corners[last][0][0] + avgX, corners[last][0][1] + avgY]]

def getEdgeCornersCols(corners):
	length = len(corners)
	last = length - 1

	changeInX = corners[last][0] - corners[0][0]
	changeInY = corners[last][1] - corners[0][1]

	avgX = changeInX/length
	avgY = changeInY/length

	return [[corners[0][0] - avgX, corners[0][1] - avgY], [corners[last][0] + avgX, corners[last][1] + avgY]]

# Return a list of all the corers, including the edges of the board
def edgeCorners(corners):
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
	
	print len(cols), len(cols[0])

	for i in range(0, len(cols)):
		for j in range(0, len(cols[i])):
			newCorners.append(cols[i][j])

	# print newCorners, len(newCorners)

	fullCorners = np.array(newCorners)

	return fullCorners



pattern_size = (7,7) # Inner corners of a chessboard

# Load image
frame = cv2.imread("chessboard.jpg")

# Our operations on the frame come here
base = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# findChessboardCorners + drawChessboardCorners
# When finding corners, performance takes a major hit -> fps drops
found, corners = cv2.findChessboardCorners(frame, pattern_size)

fullCorners = edgeCorners(corners)
print fullCorners

if found is True:
	cv2.drawChessboardCorners(frame, pattern_size, fullCorners, found)

# Display the resulting frame
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.imshow('frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()