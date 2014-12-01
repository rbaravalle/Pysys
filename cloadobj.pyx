import numpy as np
cimport numpy as np

ctypedef np.uint8_t DTYPE_t
ctypedef np.int32_t DTYPE_ti
ctypedef np.int8_t DTYPE_tii
ctypedef np.float32_t DTYPE_tf

import binvox
import time
import scipy.ndimage as ndimage

cdef extern from "math.h":
    int floor(float x)

def resize( np.ndarray[DTYPE_tf, ndim=3] model,int N, int Nz):
    cdef np.ndarray[DTYPE_tf, ndim=3] model2 = np.zeros((N,N,Nz)).astype(np.float32)

    cdef float x,y,z
    cdef int x2,y2,z2
    for x  from 0<=x<N:
        for y  from 0<=y<N:
            for z  from 0<=z<Nz:
                #model2[Nz-1-z,N-1-y,x] = model[x,z,y] # bunny
                x2 = floor(x*(255.0/(N-1)))
                y2 = floor(y*(255.0/(N-1)))
                z2 = floor(z*(255.0/(Nz-1)))
                model2[Nz-1-z,y,x] = model[x2,y2,z2]  # for bread2.vinbox
                #model2[Nz-1-z,N-1-y,x] = model[x2,z2,y2] # bunny
    return model2

# intersection between a cube field and a geometry
def intersect(field, geom,int N, int Nz):

    cdef float thresh = 4.4

    # threshold value
    # bubble intersection
    # based on the distance to the surface

    # distance field
    t = time.clock()
    print "Distance Field computing.."
    dfield = np.array(ndimage.distance_transform_edt(geom)).reshape(N,N,Nz)
    print "Distance Field computation time: ", t-time.clock()
    #mask = dfield > thresh

    #crumb = 255*(dfield > thresh) #255*np.array(geom-(255*mask*(255-field))).astype(np.uint8)
    #saveField(255*geom-crust,'crust.png')
    #crust = np.array(255*geom-crumb).astype(np.uint8)
    #printFile(crust ,"Ogre/output/media/fields/warpedC.field")

    #print field[210:220,240:250,128]
    #print mask[210:220,240:250,128]
    #print geom[210:220,240:250,128]

    # the bubbles in white (255-field) are taken into account
    # when they are away from the surface, so this is the 'crumb region'
    # crumb = mask * (255-field)
    # geom - crumb = crust
    t = time.clock()
    print "Crumb computing.."
    crumb = 255*(dfield > thresh)*(255-field)
    print "Crumb computation time: ", time.clock()-t
    

    print "Result computing.."
    result = 255*np.array(geom-(crumb)).astype(np.uint8)
    print "Result time: ", time.clock()-t
    return result, dfield
    #return 255*np.array(geom-(255*mask*(255-field))).astype(np.uint8)

