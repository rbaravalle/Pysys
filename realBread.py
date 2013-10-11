# generates an Image from real samples, for Volume Rendering

import numpy as np
import binarization as bin
import Image
import os
import matplotlib
from matplotlib import pyplot as plt

def main():

    # dimentions of the image
    dim = 200

    I = Image.new('L',(dim,dim*dim),0.0)
    print I

    # get images from folder
    path = 'bread/'
    dirList=os.listdir(path)
        
    # apply the binarization for each image
    for i in range(len(dirList)):
        print dirList[i]
        ima = Image.open(path+dirList[i])
        gray = ima.convert('L').crop((0,0,dim,dim))
        im = bin.white(gray,dim,dim,40,1.15)
        plt.imshow(im, cmap=matplotlib.cm.gray)
        plt.show()
        I2 = Image.frombuffer('L',(dim,dim), np.uint8(255)*np.array(im).astype(np.uint8),'raw','L',0,1)
        I.paste(I2,(0,dim*i))


    # save final image
    I.save('realBread.png')

main()
