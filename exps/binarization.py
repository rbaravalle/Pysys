# sandbox multifractal implementation
# Rodrigo Baravalle - October 2012

#import random
from random import randrange,randint
from math import log
from scipy import ndimage
#from pylab import plot, title, show , legend
import matplotlib
from matplotlib import pyplot as plt
import time
import Image
import numpy as np
import sys
import os

total = 10*10      # number of pixels for averaging
P = 40             # window
cant = 10+1           # number of fractal dimensions (-1 x2)

filename = '../images/scanner/baguette/baguette17.tif'

# returns the sum of (summed area) image pixels in the box between
# (x1,y1) and (x2,y2)
def mww(x1,y1,x2,y2,intImg):
    sum = intImg[x2][y2]
    if (x1>= 1 and y1 >= 1):
        sum = sum + intImg[x1-1][y1-1]
    if (x1 >= 1):
        sum = sum - intImg[x1-1][y2];
    if (y1 >= 1):
        sum = sum - intImg[x2][y1-1]
    return sum/((x2-x1+1)*(y2-y1+1));

def sat(img,Nx,Ny,which):
    # summed area table, useful for speed up the computation by adding image pixels 
    intImg = [ [ 0 for i in range(Nx) ] for j in range(Ny) ]
    if(which == 'img'):
        intImg[0][0] = img.getpixel((0,0))
        
        arrNx = range(1,Nx)
        arrNy = range(1,Ny)
        for h in arrNx:
            intImg[h][0] = intImg[h-1][0] + img.getpixel((h,0))
        
        for w in arrNy:
            intImg[0][w] = intImg[0][w-1] + img.getpixel((0,w))
        
        for f in arrNx:
            for g in arrNy:
               intImg[f][g] = img.getpixel((f,g))+intImg[f-1][g]+intImg[f][g-1]-intImg[f-1][g-1]
    else:
        intImg[0][0] = img[0][0]
        
        arrNx = range(1,Nx)
        arrNy = range(1,Ny)
        for h in arrNx:
            intImg[h][0] = intImg[h-1][0] + img[h][0]
        
        for w in arrNy:
            intImg[0][w] = intImg[0][w-1] + img[0][w]
        
        for f in arrNx:
            for g in arrNy:
               intImg[f][g] = img[f][g]+intImg[f-1][g]+intImg[f][g-1]-intImg[f-1][g-1]

    return intImg

def white(img,Nx,Ny,vent,bias):
           
    im = np.zeros((Nx,Ny))
    
    intImg = sat(img,Nx,Ny,'img')
        
    arrNx = range(Nx)
    arrNy = range(Ny)

    for i in arrNx:
        for j in arrNy:
            if(mww(max(0,i-vent),max(0,j-vent),min(Nx-1,i+vent),min(Ny-1,j+vent),intImg) >= img.getpixel((i,j))*bias ): 
                im[j,i] = img.getpixel((i,j))

    #print "IM: ", im

    # do an opening operation to remove small elements
    return ndimage.binary_opening(im, structure=np.ones((2,2))).astype(np.int)

