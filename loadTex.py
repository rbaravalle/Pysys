import numpy as np
import Image
import random

I = Image.open('textures/imagenSystem.png')
dim = I.size[0]
arr = np.array(I).reshape((dim,dim,dim))
arr2 = np.zeros((128,128))

print 128,128,128
for z in range(128):
    for y in range(128):
        for x in range(128):
           pepe = arr[x,y,z]
           if((z > 65 and z < 95) or (y*y > 250 and y < 55) or (x > 55 and x < 95)): print 0
           else:
               if(pepe == 255): pepe -= 255*random.randint(0,1)
               print 255-min(255,pepe)
           arr2[x,y] = min(255,pepe)

#I = Image.frombuffer('L',(128,128), np.array(255-arr2).astype(np.uint8),'raw','L',0,1)

#I.save('pepe.png')

