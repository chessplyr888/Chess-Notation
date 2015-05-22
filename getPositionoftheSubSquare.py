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
	lengthX = (lengthX + lengthX2) / 2
	lengthY = y3 - y1
	lengthY2 = y4 - y2
	lengthY = (lengthY + lengthY2) / 2
	#sidelength = (lengthX + lengthY) / 2  # average of both sides
	ratio = 8 / squareDimensions
	# assuming each square is at the upper left corner
	allPossibleCorners = []
	for verticalShift in range (0,length):
		for horizontalShift in range (0,length):
			newX1 = x1 + horizontalShift
			newY1 = y1 + verticalShift
			newX2 = newX1 + ratio * lengthX
			newY2 = newY1
			newY3 = newY1 + ratio * lengthY
			newX3 = newX1
			newX4 = newX2
			newY4 = newY3
			allPossibleCorners.append([(newX1, newY1),(newX2, newY2),(newX3, newY3),(newX4, newY4)])
	# problems might occur if there is a big difference in x1 and x3, y2 and y4
	return allPossibleCorners





