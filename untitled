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
	verticalShift = verticalShift - 1
	return (horizontalShift, verticalShift)

	
