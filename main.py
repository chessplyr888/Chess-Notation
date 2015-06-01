
import numpy as np
import cv2
import time
from random import randint,choice
from copy import deepcopy
import math


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

	# fullCorners should be an nparray of shape (1, 81, 2)
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

def grayScale(ROI, base):
	red = 0;
	green = 0; 
	blue = 0;
	for i in ROI:
		red = base[ROI[i][0]][ROI[i][1]][0]
		green = base[ROI[i][0]][ROI[i][1]][1]
		blue = base[ROI[i][0]][ROI[i][1]][2]

		gray = (int)(0.2126*red + 0.7152*green + 0.0722*blue)

		base[ROI[i][0]][ROI[i][1]][0] = gray
		base[ROI[i][0]][ROI[i][1]][1] = gray
		base[ROI[i][0]][ROI[i][1]][2] = gray

	return base

# KALYAN/SATHVIK START RESEARCHING KMEANS
# This method finds the important colors of the square (aka what colors are present - useful in determing what color peice is there)
def getPrimaryColors(frame, ROI, k):
	colors=[]# will have length k
	
	# list of centers with all points in a cluster,
	# index 0 is the centroid of cluster (can be intialized as random triple or chosen from ROI
	# index 1+ are points in the cluster
	#centers=[ [( randint(0,255), randint(0,255), randint(0,255) )] for i in range(k)]
	centers=[ [( 255*(i+1)/(k+1), 255*(i+1)/(k+1), 255*(i+1)/(k+1) )] for i in range(k)]
	#centers=[ [ choice(ROI) ] for i in range(k)]
	#
	# print len(frame), frame.shape, ROI[0], ROI[1]
	tempROI = []
	for i in ROI:
		# print i
		x = frame[i[1], i[0]][0]
		y = frame[i[1], i[0]][1]
		z = frame[i[1], i[0]][2]
		tempROI.append([x, y, z])

	hasChanged=True
	while(hasChanged):
		#reset centers list
		for i in range(k):
			centers[i]=[centers[i][0]]
		#
		for p in tempROI: #assign points to cluster
			mindist=1000000000
			minindex=-1

			# print len(p), len(centers)
			for c in range(len(centers)):#c - which cluster
				dist=sum([ (p[i]-centers[c][0][i])**2 for i in range(3)])
				if(dist<mindist):
					mindist=dist
					minindex=c
			#
			centers[minindex].append(p)
		#
		hasChanged=False
		for c in centers:#modify center of clusters
			oldCenter=c[0]
			allPoints=c[1:]
			L=len(allPoints)
			if(L<=0): continue
			#
			colorTotals=[0,0,0]
			for p in allPoints:
				colorTotals[0]+=p[0]
				colorTotals[1]+=p[1]
				colorTotals[2]+=p[2]
			colorTotals[0]/=L
			colorTotals[1]/=L
			colorTotals[2]/=L

			# print oldCenter, colorTotals
			
			if((colorTotals[0] != oldCenter[0]) or (colorTotals[1] != oldCenter[1]) or (colorTotals[2] != oldCenter[2])): #if no change in centers, end
				hasChanged=True
				
			c[0]=tuple(colorTotals)
		# print hasChanged
		
	
	##
	colors=[ centers[c][0] for c in range(k) ]

	# print colors
	
	newRoi=deepcopy(tempROI)
	for p in range(len(tempROI)):
		mindist=1000000000
		for c in colors:
			dist=sum([ (tempROI[p][i]-c[i])**2 for i in range(3) ])
			if(dist<mindist):
				mindist=dist
				newRoi[p]=c
		
		
	return colors, newRoi


# Recursive hillclimb
def expandBoard(corners,s,allcorners):
	foundArray=[0]*8#number corners detected 0-top 1-bottom 2-left 3-right 4-7 corners
	excorners=corners[:]
	for i in range(s):
		corners[i]=0+corners[i]+0
	corners=[0]*(s+2)+corners+[0]*(s+2)
	
	dx=corners[0][1][0][0]-corners[0][0][0][0]
	dy=corners[1][0][0][1]-corners[0][0][0][1]
	
	#add corners in top row
	top=[]
	#
	bot=[]
	#
	left=[]
	#
	right=[]
	
	#add corners
	
	
	
	return excorners,foundArray#<-foundArray is 8 integer array
	#found array indicies:
	#0-top, 1-bottom, 2-left, 3-right, 4-topleft, 5-topright, 6-bottomleft, 7-bottomright
	#indicies 0-3 range from 0 to s
	#indicies 4-7 are either 0 or 1

def recursiveHillClimb(corners,s):
	allcorners=cv2.goodFeaturesToTrack(frame, 81, 0.01, 2)
	return recursiveHillClimbPhase2(corners, s, s*s, 1, allcorners)
	
# Recursive hillclimb
def recursiveHillClimbPhase2(corners, s, numfound, lastone, allcorners):#initial call -> numfound=s^2, lastone =1
	if(s==7):
		return corners, numfound
	excorners,foundArray=expandBoard(corners,s,allcorners)
	#
	#using the lastone arguement we make sure each board is processed exactly once
	#top left
	if(1>=lastone):
		newCorners1=[ [excorners[i][j][0] for i in range(0,s+1)] for j in range(0,s+1)]
		bestCorners1,bestValue1=recursiveHillClimbPhase2(newCorners1,s+1,numfound+foundArray[0]+foundArray[2]+foundArray[4],1,allcorners)
	#top right
	if(2>=lastone):
		newCorners2=[ [excorners[i][j][0] for i in range(0,s+1)] for j in range(1,s+2)]
		bestCorners2,bestValue2=recursiveHillClimbPhase2(newCorners2,s+1,numfound+foundArray[0]+foundArray[3]+foundArray[5],2,allcorners)
		if(bestValue2>bestValue1):
			bestCorners1=bestCorners2
			bestValue1=bestValue2
	#bottom left
	if(3>=lastone):
		newCorners3=[ [excorners[i][j][0] for i in range(1,s)] for j in range(0,s+1)]
		bestCorners3,bestValue3=recursiveHillClimbPhase2(newCorners3,s+1,numfound+foundArray[1]+foundArray[2]+foundArray[6],3,allcorners)
		if(bestValue3>bestValue1):
			bestCorners1=bestCorners3
			bestValue1=bestValue3
	#bottom right
	if(4>=lastone):
		newCorners4=[ [excorners[i][j][0] for i in range(1,s)] for j in range(1,s+2)]
		bestCorners4,bestValue4=recursiveHillClimbPhase2(newCorners4,s+1,numfound+foundArray[1]+foundArray[3]+foundArray[7],4,allcorners)
		if(bestValue4>bestValue1):
			bestCorners1=bestCorners4
			bestValue1=bestValue4
	
	return bestCorners1, bestValue1

def extrapolatePotentialCorners(pattern_size, corners):
	# Pattern size ranges from (1,1) to (7,7)
	diff = pattern_size[0] - full_pattern_size[0]
	
	length = len(corners)
	last = length - 1

	# print corners

	dx = (corners[last][0][0] - corners[0][0][0]) / pattern_size[0]
	dy = (corners[last][0][1] - corners[0][0][1]) / pattern_size[0]

	# potCoreners is the list of potential corners
	potCorners = []

	rows = []
	for i in range(0,pattern_size):
		rows.append((corners[0 + pattern_size*i:pattern_size + pattern_size*i]))
		# print corners[0 + pattern_size*i:pattern_size + pattern_size*i]

	# 7x7
	if diff == 0:
		potCorners = edgeCorners(corners)
		return potCorners
	# 6x6 - 4
	elif diff == 1:
		upperLeftCorner = [rows[0][0][0] + dx, rows[0][0][1] + dy]
		upperRow = [upperLeftCorner]
		potCorners.append(upperLeftCorner)
		for i in range(0, 8):
			potCorners.append([upperRow[i][0][0] + dx, upperRow[i][0][1]])

		# Parse through each row
		for i in range(0, 8):
			nextLeft = [upperLeftCorner[0][0][0], upperLeftCorner[0][0][1] - dy]
			potCorners.append(nextLeft)
			for j in range(0, pattern_size):
				potCorners.append([])

		return potCorners
	# 5x5 - 9
	elif diff == 2:
		return potCorners
	# 4x4 - 16
	elif diff == 4:
		return potCorners
	# 3x3 - 25
	elif diff == 5:
		return potCorners
	# 2z2 - 36
	elif diff == 6:
		return potCorners

# gives where the subsquare is in the board, needs to be rewritten
def findPositionofSquareRelativeToTheBoard(square, squareDimensions): # ([(x1,y1), (x2,y2), (x3,y3), (x4,y4)], sidelength = 2) 
	verticalShift = 0
	horizontalShift = 0
	# board's previous max x and max y
	# xMax, yMax
	#corner 1 is upper left, corner 2 is upper right, corner 3 is lower left, corner 4 is lower right 
	lengthX = x2 - x1
	lengthX2 = x4 - x3
	lengthX = (lengthX + lengthX2) / 2  # average gets a better value
	lengthY = y3 - y1
	lengthY2 = y4 - y2
	lengthY = (lengthY + lengthY2) / 2
	# assuming origin is at the lower left corner
	while x1 < xMax or x2 < xMax :
		x1 = x1 + lengthX
		x2 = x2 + lengthX
		verticalShift = verticalShift + 1
	verticalShift = verticalShift - 1
	while y2 < yMax or y4 < yMax :
		y2 = y2 + lengthY
		y4 = y4 + lengthY
		verticalShift = verticalShift + 1
	horizontalShift = horizontalShift - 1
	
	

	return (horizontalShift, verticalShift)


def getAllPossibleBoardCorners(square, squareDimensions): 
	# assume it's a 2x2
	length = (9 - squareDimensions)
	#corner 1 is upper left, corner 2 is upper right, corner 3 is lower left, corner 4 is lower right 
	x1,y1 = square[0]
	x2,y2 = square[1]
	x3,y3 = square[2]
	x4,y4 = square[3]

	lengthX = x2 - x1
	lengthX2 = x4 - x3
	lengthX = (lengthX + lengthX2) / (2 * (squareDimensions - 1))
	lengthY = y3 - y1
	lengthY2 = y4 - y2
	lengthY = (lengthY + lengthY2) / (2 * (squareDimensions - 1))

	#sidelength = (lengthX + lengthY) / 2  # average of both sides
	ratio = 8 / squareDimensions
	# assuming each square is at the upper left corner

	allPossibleCorners = []
	# for verticalShift in range (0,length):
	# 	for horizontalShift in range (0,length):
	# 		newX1 = x1 + horizontalShift
	# 		newY1 = y1 + verticalShift
	# 		newX2 = newX1 + ratio * lengthX
	# 		newY2 = newY1
	# 		newY3 = newY1 + ratio * lengthY
	# 		newX3 = newX1
	# 		newX4 = newX2
	# 		newY4 = newY3
	# 		allPossibleCorners.append([(newX1, newY1),(newX2, newY2),(newX3, newY3),(newX4, newY4)])


	for i in range (1, length):
		for j in range(1, length):
			tempFullCorners = []
			newTopLeft = [square[0][0] - (i * lengthX), square[0][1] - (j * lengthY)]
			for dx in range(0, 9):
				for dy in range(0, 9):
					newX = newTopLeft[0] + (dx * lengthX)
					if newX < 0:
						newX = 0
					if newX > cap.get(3):
						newX = cap.get(3)
					newY = newTopLeft[1] + (dy * lengthY)
					if newY < 0:
						newY = 0
					if newY > cap.get(4):
						newY = cap.get(4)
					tempFullCorners.append((newX, newY))

			allPossibleCorners.append(tempFullCorners)


	# problems might occur if there is a big difference in x1 and x3, y2 and y4
	return allPossibleCorners


def dist(x1, y1, x2, y2):
	distance = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
	return distance


def compareCorners(frame, potCorners):
	ShiTomasi = cv2.goodFeaturesToTrack(frame, 100, 0.01, 2)
	# print ShiTomasi

	finalCorners = []

	counter = 0
	maxCounter = 0


	# 100 px threshold
	threshold = 100

	for i in potCorners:
		for j in i:
			for k in ShiTomasi:
				x1 = j[0]
				y1 = j[1]
				x2 = k[0][0]
				y2 = k[0][1]
				distance = dist(x1, y1, x2, y2)
				# print distance
				if dist < threshold:
					counter = counter + 1
			if counter > maxCounter:
				maxCounter = counter
				finalCorners = i
		counter = 0

	return finalCorners

def getSubSquare(frame, ROI):
	pass





# INSTANTIATE VARIABLES HERE (TODO LATER) 
# 
# 
# 
# 
# 
# 
cap = cv2.VideoCapture(0) # Primary (Built in) Camera
# cap = cv2.VideoCapture(1) # Secondary (Attached) Camera
# MAXIMUM RESOLUTION FOR LOGITECH C920 FOR STRESS TESTING
# 921600 Pixels -> 3 X as many calculations as 640*480
cap.set(3, 640)
cap.set(4, 480)
print cap.get(3), cap.get(4)


temp_inner_corners = 6
pattern_size = (6,6) # Inner corners of a chessboard
full_pattern_size = (9,9) # All corners of a chessboard
# 
# 
# 
# 
# 
# 
# 

fullCorners = []

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# print frame.shape

	# Our operations on the frame come here
	base = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# findChessboardCorners + drawChessboardCorners
	# When finding corners, performance takes a major hit -> fps drops
	found, corners = cv2.findChessboardCorners(frame, pattern_size)
	
	print found

	if found == False:
		temp_inner_corners = temp_inner_corners - 1
		if temp_inner_corners < 4:
			pattern_size = (7, 7)
		else:
			pattern_size = (temp_inner_corners, temp_inner_corners)
	
	if found:
		tempSquare = [corners[0][0], corners[pattern_size[0] - 1][0], corners[pattern_size[0] * pattern_size[0] - pattern_size[0]][0], corners[pattern_size[0] * pattern_size[0] - 1][0]]
		potential_points = getAllPossibleBoardCorners(tempSquare, pattern_size[0]);

		if(len(potential_points) > 1):
			fullCorners = compareCorners(base, potential_points)
		else:
			fullCorners = potential_points[0]
		# print fullCorners


	while(found):
		# Capture frame-by-frame
		ret, frame = cap.read()

		# print frame.shape

		# Our operations on the frame come here
		base = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# findChessboardCorners + drawChessboardCorners
		# When finding corners, performance takes a major hit -> fps drops
		found, corners = cv2.findChessboardCorners(frame, pattern_size)
		
		print found

		if found == False:
			temp_inner_corners = temp_inner_corners - 1
			if temp_inner_corners < 4:
				pattern_size = (7, 7)
			else:
				pattern_size = (temp_inner_corners, temp_inner_corners)
		
		if found:
			tempSquare = [corners[0][0], corners[pattern_size[0] - 1][0], corners[pattern_size[0] * pattern_size[0] - pattern_size[0] + 1][0], corners[pattern_size[0] * pattern_size[0] - 1][0]]
			potential_points = getAllPossibleBoardCorners(tempSquare, pattern_size[0]);

			# print potential_points

			if(len(potential_points) > 1):
				fullCorners = compareCorners(base, potential_points)
			else:
				fullCorners = potential_points[0]
			print fullCorners

		# Draws the corners
		# print len(fullCorners)
		for i in range(0, len(fullCorners)):
			x = int(fullCorners[i][0])
			y = int(fullCorners[i][1])
			cv2.circle(frame, (x, y), 5, (0, 255, 0), -1) # Green Points

		cv2.imshow('newROI', frame)


		# # Get the contours
		# contours = getContourList(fullCorners)

		# # Draw the contours
		# cv2.drawContours(frame, contours, -1, (0, 0, 255), 2) # Red Contours

		# # Get the average color of the square
		# colorOfSquare = []

		# # Pixels of the entire board
		# fullBoardROI = []

		# # Test Case For A Single Countour (Square)
		# pointsInContour = []

		# # List of all bounding boxes
		# boxes = []

		# getBoxTimeStart = time.time()
		# for i in range(0, 64):
		# 	x, y, w, h = boundingBox(contours[i])
		# 	boxes.append([x, y, w, h])
		# 	squareImg = cv2.rectangle(frame, (x,y), (x + w, y + h), (255, 0, 0), 2)		
		# getBoxTimeEnd = time.time()

		# print "Load Image Time: %f" %(getBoxTimeEnd - getBoxTimeStart)

		# # Gets all the points in every contour and adds them to the custom full board ROI
		# getPointsTimeStart = time.time()
		# for k in range(0, 64):
		# 	for i in range(x, x + w):
		# 		for j in range(y, y + h):
		# 			if(cv2.pointPolygonTest(contours[k], (i, j), False) == 1):
		# 				fullBoardROI.append((i, j))
		# getPointsTimeEnd = time.time()

		# print "Calculate Points Time: %f" %(getPointsTimeEnd - getPointsTimeStart)



	# # K-Means N Color Image
	# primeColors, newROI = getPrimaryColors(frame, fullBoardROI, 4)

	# print primeColors
	
	# # print fullBoardROI
	# getPrintTimeStart = time.time()
	# p=0
	# for k in range(0, 64):
	# 	for i in range(x, x + w):
	# 		for j in range(y, y + h):
	# 			frame[i,j]=newROI[p]
	# 			p+=1
	# getPrintTimeEnd = time.time()

	# print "Calculate Print Full Board ROI Time: %f" %(getPrintTimeEnd - getPrintTimeStart)

	# Get binary color mask of fullBoardROI -> TODO
	# mask = frame[x: x + w, y: y + h]
	# grayMask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
	# binaryMask = cv2.adaptiveThreshold(grayMask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	# ret, binaryMask = cv2.threshold(grayMask, 127, 255, cv2.THRESH_BINARY)

	# Get binary mask of square region
	# ROI = [] # ROI is Region of Interest
	# for i in range(0, w):
	# 	for j in range(0, h):
	# 		ROI.append(binaryMask[i, j])

	# print ROI

	# WILL BECOME OBSOLETE
	# Get color of ROI
	# squareColor = getSquareColor(ROI)
	# if squareColor == True:
	# 	print "white"
	# else:
	# 	print "black"

	# Display the resulting frame
	cv2.namedWindow("newROI", cv2.WINDOW_AUTOSIZE)
	cv2.imshow('newROI', frame)

	# q to quit
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
