
import numpy as np
import cv2
import time


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

	rows = []
	for i in range(0,7):
		rows.append((corners[0 + 7*i:7 + 7*i]))
		# print corners[0 + 7*i:7 + 7*i]

	newRows = []

	for i in range(0, len(rows)):
		temp = getEdgeCornersRows(rows[i])
		# print temp[0]
		newRows.append(temp[0])
		for j in range(0, len(rows[i])):
			# print rows[i][j][0]
			newRows.append([rows[i][j][0][0], rows[i][j][0][1]])
		newRows.append(temp[1])

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
	for i in range(0, len(cols)):
		for j in range(0, len(cols[i])):
			newCorners.append(cols[i][j])

	fullCorners = np.array([newCorners])

	# fullCorners should be an nparray of shape (64, 1, 2)
	return fullCorners


def getContourList(corners):
	contours = []
	for i in range(0, 8):
		for j in range(0, 8):
			topLeft = corners[0][(i)*9 + (j)]
			topRight = corners[0][(i)*9 + (j + 1)]
			botLeft = corners[0][(i + 1)*9 + (j)]
			botRight = corners[0][(i + 1)*9 + (j + 1)]
			contours.append(np.array((topLeft, topRight, botRight, botLeft), dtype = np.int32))

	return contours

def boundingBox(points):
	x = []
	for i in range(0,4):
		x.append(points[i][0])

	y = [] 
	for i in range(0,4):
		y.append(points[i][1])

	xmin = min(x)
	xmax = max(x)
	ymin = min(y)
	ymax = max(y)

	return (xmin, ymin, xmax - xmin, ymax - ymin)

# KALYAN/SATHVIK FIND COLOR OF THE SQUARE
# Change this back to finding black/white from binary image
def getSquareColor(binaryROI):
	black = 0
	white = 0
	for i in range(0, len(binaryROI)):
		if binaryROI[i] == 0:
			black += 1
		else:
			white += 1
	return white > black


# KALYAN/SATHVIK START RESEARCHING KMEANS
# This method finds the important colors of the square (aka what colors are present - useful in determing what color peice is there)
def getPrimaryColors(frame, points):
        return frame

# INSTANTIATE VARIABLES HERE (TODO LATER) 
# 
# 
# 
# 
# 
# 
# cap = cv2.VideoCapture(0) # Primary (Built in) Camera
cap = cv2.VideoCapture(1) # Secondary (Attached) Camera
# MAXIMUM RESOLUTION FOR LOGITECH C920 FOR STRESS TESTING
# 921600 Pixels -> 3 X as many calculations as 640*480
cap.set(3, 1280)
cap.set(4, 720)
print cap.get(3), cap.get(4)

pattern_size = (7,7) # Inner corners of a chessboard
full_pattern_size = (9,9) # All corners of a chessboard
# 
# 
# 
# 
# 
# 
# 

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	base = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# findChessboardCorners + drawChessboardCorners
	# When finding corners, performance takes a major hit -> fps drops
	found, corners = cv2.findChessboardCorners(frame, pattern_size)
	print found

	if found is True:
		fullCorners = edgeCorners(corners)

		# Draws the corners
		for i in range(0, len(fullCorners[0])):
			# print fullCorners
			x = int(fullCorners[0][i][0])
			y = int(fullCorners[0][i][1])
			cv2.circle(frame, (x, y), 5, (0, 255, 0), -1) # Green Points

		# Get the contours
		contours = getContourList(fullCorners)

		# Draw the contours
		cv2.drawContours(frame, contours, -1, (0, 0, 255), 2) # Red Contours

		# Get the average color of the square
		colorOfSquare = []

		# Test Case For A Single Countour (Square)
		pointsInContour = []

		getImageTimeStart = time.time()
		x, y, w, h = boundingBox(contours[1])
		squareImg = cv2.rectangle(frame, (x,y), (x + w, y + h), (255, 0, 0), 2)		
		getImageTimeEnd = time.time()

		print "Load Image Time: %f" %(getImageTimeEnd - getImageTimeStart)

		getPointsTimeStart = time.time()
		for i in range(x, x + w):
			for j in range(y, y + h):
				if(cv2.pointPolygonTest(contours[1], (i, j), False) == 1):
					pointsInContour.append((i, j))
		getPointsTimeEnd = time.time()

		print "Calculate Points Time: %f" %(getPointsTimeEnd - getPointsTimeStart)

		# Get binary color mask of bounding box
		mask = frame[x: x + w, y: y + h]
		grayMask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
		# binaryMask = cv2.adaptiveThreshold(grayMask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
		thresh, binaryMask = cv2.threshold(grayMask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

		# Get binary mask of square region
		ROI = [] # ROI is Region of Interest
		for i in range(0, w):
			for j in range(0, h):
				ROI.append(binaryMask[i, j])

		# Get color of ROI
		squareColor = getSquareColor(ROI)
		if squareColor == True:
			print "white"
		else:
			print "black"

	# Display the resulting frame
	cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)
	cv2.imshow('frame', frame)

	# q to quit
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
