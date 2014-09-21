import numpy as np
import Image
import random
import time
import sys 
from random import randint

I = Image.open(sys.argv[1])
dim = I.size[0]
Nz = I.size[1]/I.size[0]
arr = np.array(I).reshape((Nz,dim,dim))
N = dim
p = 0

def printImg(arr2):
    I3 = Image.new('L',(N,(N)*(Nz)),0.0)
    for w in range(Nz):
        II = Image.frombuffer('L',(N,N), np.array(arr2[0:N,0:N,w]).astype(np.uint8),'raw','L',0,1)
        I3.paste(II,(0,(N)*w))
    I3.save('cutted'+sys.argv[1])

def eatimage(arr):
    arr2 = arr
    for z in range(Nz):
        for y in range(N):
            for x in range(N):

               x1 = x-64
               y1 = N-y+94
               z1 = z-94
               if(x1*x1+y1*y1+z1*z1 < 120*160+randint(0,300)):
                   arr2[z,N-1-y,x] =  0
                   continue

               x1 = x-220
               y1 = N-y-65
               z1 = z-50
               if(x1*x1+y1*y1+z1*z1 < 100*100+randint(0,300)):
                   arr2[z,N-1-y,x] =  0
                   continue

               limit = np.sqrt((z/0.7-Nz/2)*(z/0.7-Nz/2)+(x/2.0-N/2)*(x/2.0-N/2)+(y/1-N/2)*(y/1-N/2))

               if(limit > 60+random.randint(0,2)):
                   arr2[z,N-1-y,x] =  0
                   continue

               arr2[z,N-1-y,x] = arr[z,N-1-y,x]

    printImg(arr2)


arr2 = eatimage(arr)
