
# coding: utf-8

# In[9]:

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import filters
import scipy.misc


# In[10]:

class ProcessImage:
    
    k = np.sqrt(2)  # Gaussian blur factor
    constFactor = np.sqrt(k**2 - 1)
    sigma = 1.6
    s = 5           # number of images per octave
    numOctave = 4   # number of octaves
    
    def __init__(self, arr):
        """Constructor: pass an array"""
        self.imgArr = np.array(arr)
        self.imgOrigin = np.array(arr)   # store a copy of original image
    
    def plot(self):
        plt.imshow(self.imgArr, cmap='Greys_r')
        plt.show()
    
    def saveIm(self, saveName):
        self.imgOrigin.save(saveName)
        
    def getSize(self):
        return self.imgArr.shape
        
    def gaussianBlur(self, sig):
        # sig is the standard deviation of the gaussian kernel
        return filters.gaussian_filter(self.imgArr, sig)

    def halfSize(self):
        """Take every second pixels in both rows and columns."""
        return self.imgArr[::2, ::2]


# In[11]:

def main():
    # 1. load original image
    origin = np.array(Image.open('Coco.jpg').convert('L'))
    im = ProcessImage(origin)
    im.getSize()

    # 2. creating a list with all gaussian blur images with original resolution
    scaleList = []
    # octave 1: \sigma --> k^4\sigma
    for i in range(ProcessImage.s):
        newArr = im.gaussianBlur(ProcessImage.sigma * ProcessImage.k**i)
        scaleList.append(ProcessImage(newArr))

    # octave 2:  k^2\sigma --> k^6*\sigma, 1/2 size
    # octave 3:  k^4\sigma --> k^8*\sigma, 1/4 size
    # octave 4:  k^6\sigma --> k^10*\sigma, 1/8 size
    factor = 4
    for j in range(3):
        for i in range(3):
            scaleList.append(ProcessImage(scaleList[-3].halfSize()))       # the image to halfSize is always at index -3
        for i in range(factor, factor+2):
            newSig = ProcessImage.k**i * ProcessImage.constFactor * ProcessImage.sigma
            newArr = scaleList[-1].gaussianBlur(newSig)
            scaleList.append(ProcessImage(newArr))
        factor += 2


    # test for sizes
    # [scaleList[j].getSize() for j in range(len(scaleList))]

    # 3. plot all octaves
    axarr = np.zeros((ProcessImage.numOctave, ProcessImage.s))
    for row in range(ProcessImage.numOctave):
        fig, axarr = plt.subplots(1, ProcessImage.s, sharey=True)
        plt.suptitle('Octave %d' % (row+1))
        for col in range(len(scaleList[:ProcessImage.s])):
            axarr[col].imshow(scaleList[row*ProcessImage.s+col].imgArr, cmap='Greys_r')
            currSigma = ProcessImage.k ** (2*row+col) * ProcessImage.sigma
            axarr[col].set_title('$\sigma = %0.4f$' % currSigma)
    plt.show()


# In[12]:

if __name__ == "__main__":
    main()

