from random import *
from copy import deepcopy



def getPrimaryColors(ROI, k):
	colors=[]# will have length k
	
	# list of centers with all points in a cluster,
	# index 0 is the centroid of cluster (can be intialized as random triple or chosen from ROI
	# index 1+ are points in the cluster
	#centers=[ [( randint(0,255), randint(0,255), randint(0,255) )] for i in range(k)]
	centers=[ [( 255*(i+1)//(k+1), 255*(i+1)//(k+1), 255*(i+1)//(k+1) )] for i in range(k)]
	#centers=[ [ choice(ROI) ] for i in range(k)]
	#
	hasChanged=True
	while(hasChanged):
		#reset centers list
		for i in range(k):
			centers[i]=[centers[i][0]]
		#
		for p in ROI: #assign points to cluster
			mindist=1000000000
			minindex=-1
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
			colorTotals[0]//=L
			colorTotals[1]//=L
			colorTotals[2]//=L
			
			if(colorTotals[0] != oldCenter[0] or colorTotals[1] != oldCenter[1] or colorTotals[2] != oldCenter[2]): #if no change in centers, end
				hasChanged=True
				
			c[0]=tuple(colorTotals)
		
		
	
	##
	colors=[ centers[c][0] for c in range(k) ]
	
	newRoi=deepcopy(ROI)
	for p in range(len(ROI)):
		mindist=1000000000
		for l in range(len(colors)):
			c=colors[l]
			dist=sum([ (ROI[p][i]-c[i])**2 for i in range(3) ])
			if(dist<mindist):
				mindist=dist
				if(l==0): newRoi[p]=(255,0,0)
				elif(l==1): newRoi[p]=(0,255,0)
				elif(l==2): newRoi[p]=(0,0,255)
				elif(l==3): newRoi[p]=(0,255,255)
				elif(l==4): newRoi[p]=(255,0,255)
				elif(l==5): newRoi[p]=(255,255,0)
				elif(l==6): newRoi[p]=(255,255,255)
				else: newRoi[p]=c
		
	return colors, newRoi


def main():
	file1=open('chessboard3.ppm','r')
	stng=file1.readline().strip()
	nums=file1.read().split()
	file1.close()
	image=[]
	for i in range(0,len(nums),3):
		red=int(nums[i])
		green=int(nums[i+1])
		blue=int(nums[i+2])
		
		image.append((red,green,blue))
		

	#roi=[ (randint(0,255),randint(0,255),randint(0,255)) for i in range(10)]
	colors,newRoi=getPrimaryColors(image, 4)
	#print(newRoi, colors, sep="\n")
	
	outfile=open('kmeanschess.ppm','w')
	outfile.write('P3 300 206 255\n')# change this line for every image
	for i in newRoi:
		for j in i:
			outfile.write(''+str(j)+' ')
		outfile.write('\n')
	outfile.close()




if __name__=='__main__': main()



