import numpy as np
import Image
import random
import cprint, time
import sys 

I = Image.open(sys.argv[1])
dim = I.size[0]
Nz = I.size[1]/I.size[0]
arr = np.array(I).reshape((Nz,dim,dim))
N = dim

p = 0

maxNeigh = 15.0
c = 15#int(N*(0.9/100.0))
#print c

arr=cprint.cprint2(p,Nz,N,arr,c)

cprint.cprint(p,Nz,N,arr)

