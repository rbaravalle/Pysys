import numpy as np
import Image
import random

I = Image.open('imagenSystemPaper.png')
dim = I.size[0]
arr = np.array(I).reshape((dim,dim,dim))
N = dim
arr2 = np.zeros((N,N))

print N,N,N
for z in range(N):
    for y in range(N):
        for x in range(N):
           pepe = arr[z,x,y]

           #if(x*x+y*y+z*z > 90000+np.random.randint(0,2000)):
           #    print 0
           #    continue

           #x1 = x-64
           #y1 = y-104
           #z1 = z-124
           #if(x1*x1+y1*y1+z1*z1 < 20*20):
               #print 0
               #continue

           #x1 = x-44
           #y1 = y-124
           #z1 = z-84
           #if(x1*x1+y1*y1+z1*z1 < 20*20):
           #    print 0
           #    continue

           #x1 = x-94
           #y1 = y-144
           #z1 = z-144
           #if(x1*x1+y1*y1+z1*z1 < 60*60):
               #print 0
               #continue

           x1 = x-44
           y1 = y-54
           z1 = z+14
           if(x1*x1+y1*y1+z1*z1 < 110*110+np.random.randint(-400,400)):
               print 0
               continue

           #x1 = x-74
           #y1 = y-94
           #z1 = z-94
           #if(x1*x1+y1*y1+z1*z1 < 20*20):
           #    print 0
               #continue

           #if((z > 65 and z < 95) or (y*y > 250 and y < 55) or (x > 55 and x < 95)): print 0
           #else:
               #if(pepe == 255): pepe -= 255*random.randint(0,2)
           print pepe- 255*(np.random.randint(0,10)>6) #min(255,pepe)
           #arr2[x,y] = min(255,pepe)

#I = Image.frombuffer('L',(128,128), np.array(255-arr2).astype(np.uint8),'raw','L',0,1)

#I.save('pepe.png')

