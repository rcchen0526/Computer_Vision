import numpy as np
import numpy.fft as fft
import cv2
import sys

class hybrid:
	def __init__(self, image1, image2):
		self.image1 = image1
		self.image2 = image2

	def run(self):
		row1, col1, rgb = self.image1.shape
		row2, col2, rgb = self.image2.shape
		b1,g1,r1 = cv2.split(image1)
		b2,g2,r2 = cv2.split(image2)
		low_filter = GaussianFilter(row1, col1, 10, False)
		image1_b = fft.ifft2(fft.ifftshift(fft.fftshift(fft.fft2(b1)) * low_filter))
		image1_g = fft.ifft2(fft.ifftshift(fft.fftshift(fft.fft2(g1)) * low_filter))
		image1_r = fft.ifft2(fft.ifftshift(fft.fftshift(fft.fft2(r1)) * low_filter))
		high_filter = GaussianFilter(row2, col2, 25, True)
		image2_b = fft.ifft2(fft.ifftshift(fft.fftshift(fft.fft2(b2)) * high_filter))
		image2_g = fft.ifft2(fft.ifftshift(fft.fftshift(fft.fft2(g2)) * high_filter))
		image2_r = fft.ifft2(fft.ifftshift(fft.fftshift(fft.fft2(r2)) * high_filter))
		b = image1_b + image2_b
		g = image1_g + image2_g
		r = image1_r + image2_r
		new_img = cv2.merge((np.real(b),np.real(g),np.real(r)))
		
		cv2.imwrite('output.jpg', new_img)



def GaussianFilter(row, col, cutoff, highPass):
	if row % 2:
		centerX = int(row/2) + 1
	else:
		centerX = int(row/2) 
	if col % 2:
		centerY = int(col/2) + 1
	else:
		centerY = int(col/2) 

	def gaussian(i,j):
		H = np.exp(-1.0 * ((i - centerX)**2 + (j - centerY)**2) / (2 * cutoff**2))
		return 1 - H if highPass else H

	return np.array([[gaussian(i,j) for j in range(col)] for i in range(row)])

if __name__ == "__main__":
	image1 = cv2.imread(sys.argv[1])
	image2 = cv2.imread(sys.argv[2])
	HB = hybrid(image1, image2)
	HB.run()
