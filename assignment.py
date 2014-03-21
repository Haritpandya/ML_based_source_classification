from astropy.io import fits
from matplotlib import colors, cm, pyplot as plt
import numpy 
import pdb
dulist = fits.open('M86.fits')
imgdata = dulist[0].data
imgsize =  imgdata.shape
imgmean = numpy.mean(imgdata)
imgstd  = numpy.std(imgdata)
thresh  = imgmean + imgstd
M = numpy.zeros(shape=(3,3))
mup = numpy.zeros(shape=(3,3))

for i in range(1, imgsize[0]):
	for j in range(1, imgsize[1]):
		if imgdata[i][j] > thresh:		
			for i0 in range(0, 3):
				for j0 in range(0, 3):
					M[i0][j0] = M[i0][j0] + (imgdata[i][j]*(i**i0)*(j**j0))
centroid = numpy.array([M[1][0]/M[0][0],M[0][1]/M[0][0]])
mup[2][0] = M[2][0]/M[0][0] - centroid[0]**2
mup[0][2] = M[0][2]/M[0][0] - centroid[1]**2
mup[1][1] = M[1][1]/M[0][0] - centroid[0]*centroid[1]
th = 0.5 * numpy.arctan2((2*mup[1][1]),(mup[2][0]-mup[0][2]))
covMat = numpy.array([[mup[2][0],mup[1][1]],
[mup[1][1],mup[0][2]]])
norm = colors.LogNorm(imgdata.mean() + 0.5 * imgdata.std(), imgdata.max(), clip='True')
imgplot = plt.matshow(imgdata, cmap=cm.gray, norm=norm, origin="lower")
eigvals = numpy.linalg.eigvals(covMat)
axlen = eigvals[1]/5
axlen2 = eigvals[0]/5
plt.plot([centroid[1]-axlen*numpy.cos(th), centroid[1]+axlen*numpy.cos(th)], [centroid[0]-axlen*numpy.sin(th), centroid[0]+axlen*numpy.sin(th)], 'g-')
plt.plot([centroid[1]-axlen2*numpy.cos(th+numpy.pi/2), centroid[1]+axlen2*numpy.cos(th+numpy.pi/2)], [centroid[0]-axlen2*numpy.sin(th+numpy.pi/2), centroid[0]+axlen2*numpy.sin(th+numpy.pi/2)], 'r-')
circle=plt.Circle(centroid[::-1],5,color='g',fill=False);
fig = plt.gcf()
fig.gca().add_artist(circle)
plt.show()
fig.savefig('output.png')
