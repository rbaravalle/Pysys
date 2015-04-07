import numpy as np
cimport numpy as np
import scipy.ndimage as ndimage
import scipy
import time
import pylab
from matplotlib import pyplot as plt
import Image
import cloadobj
from cloadobj import orientate, resize,resizef,invresize,orientatef
import matplotlib

import warp

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t
ctypedef np.float32_t DTYPE_tf

cdef extern from "math.h":
    int round(float x)

def saveField(field,folder,filename):
    pp = 0
    N = field.shape[0]
    Nz = N
    I3 = Image.new('L',(N-2*pp,(N-2*pp)*(Nz)),0.0)

    if(True):
        base = folder+"/slice"#"accumulated/slice"
    else:
        base = 'warp2/warped/warpedslice'

    for w in range(Nz):
        II = Image.frombuffer('L',(N-2*pp,N-2*pp), np.array(field[pp:N-pp,pp:N-pp,w]).astype(np.uint8),'raw','L',0,1)
        II.save(base+str(w)+'.png')
        I3.paste(II,(0,(N-2*pp)*w))

    I3.save(folder+"/"+filename)
    print "Image "+folder+"/"+filename+" saved"

# field #(Nx,Ny,Nz)
# geom #(Nx,Ny,Nz)
# dfield #(256,256,256)
def bake(np.ndarray[DTYPE_t, ndim=3] field, np.ndarray[DTYPE_tf, ndim=3]  dfield, np.ndarray[DTYPE_t, ndim=3] geom, np.ndarray[DTYPE_tf, ndim=3] density, np.ndarray[DTYPE_tf, ndim=1]  temperatures,int N,int Nz, int k2):

    cdef float dist
    cdef int i,j,k,cant
    cdef float dmax = np.max(density)
    cdef float dmin = np.min(density)

    cdef np.ndarray[DTYPE_tf, ndim=3] gx, gy, gz, dfieldDeformed
    cdef np.ndarray[DTYPE_t, ndim=3] geomD
    cdef np.ndarray[DTYPE_tf, ndim=3] result = np.zeros((256,256,256)).astype(np.float32)

    # rising during baking is modulated by gravity, distance to the centre, and the density of particles at the point

    print "rise field..."
    #saveField(orientate(field,N,Nz),"accumulated","fieldPrev.png")
    
    # how many different temperatures
    cant = len(temperatures)
    cdef float f = ((cant-1)/np.max(dfield))

    print "Computing temperatures"
    for i from 0<=i<256:
        for j from 0<=j<256:
            for k from 0<=k<256:
                result[i,j,k] = temperatures[round(dfield[i,j,k]*f)]

    # For the paper
    if(False):
        I2 = Image.frombuffer('L',(256,256), (result[:,:,100]).astype(np.uint8),'raw','L',0,1)
        imgplot = plt.imshow(I2)
        plt.colorbar()
        matplotlib.rcParams.update({'font.size': 22})
        gx2, gy2 = np.gradient(result[:,:,100])
        gx2 = ndimage.filters.gaussian_filter(gx2,3)
        gy2 = ndimage.filters.gaussian_filter(gy2,3)
        #pylab.quiver(gx2,gy2)
        pylab.show()
        plt.show()

        if(True):
            I2 = Image.frombuffer('L',(256,256), (result[:,100,:]).astype(np.uint8),'raw','L',0,1)
            imgplot = plt.imshow(I2)
            plt.colorbar()
            gx2, gy2 = np.gradient(result[:,100,:])
            gx2 = ndimage.filters.gaussian_filter(gx2,3)
            gy2 = ndimage.filters.gaussian_filter(gy2,3)
            #pylab.quiver(gx2,gy2)
            pylab.show()
            plt.show()

            I2 = Image.frombuffer('L',(256,256), (result[100,:,:]).astype(np.uint8),'raw','L',0,1)
            imgplot = plt.imshow(I2)
            plt.colorbar()
            gx2, gy2 = np.gradient(result[100,:,:])
            gx2 = ndimage.filters.gaussian_filter(gx2,3)
            gy2 = ndimage.filters.gaussian_filter(gy2,3)
            #pylab.quiver(gx2,gy2)
            pylab.show()
            plt.show()

    print "Gradient..."
    gx, gy, gz = np.gradient(result)

    print "Gaussian's...x..."
    gx = ndimage.filters.gaussian_filter(gx.astype(np.float32),3)
    print "Gaussian's...y..."
    gy = ndimage.filters.gaussian_filter(gx.astype(np.float32),3)
    print "Gaussian's...z..."
    gz = ndimage.filters.gaussian_filter(gz.astype(np.float32),3)

    # gx,gy,gz #(Nx,Ny,Nz)
    print "Resize gx..."
    gx = cloadobj.resizef(gx,N,Nz)
    print "Resize gy..."
    gy = cloadobj.resizef(gy,N,Nz)
    print "Resize gz..."
    gz = cloadobj.resizef(gz,N,Nz)
    
    # dfield = resizef(dfield,N,Nz)
    field = warp.warp(field, gx,gy,gz, density,N, Nz,k2,dmax,dmin)
    saveField(orientate(field,N,Nz),"fieldrise1","fieldRise1.png")
    #density = warp.warp(density, gx,gy,gz, N, Nz,k2)
    print "rise geom..."
    geomD = warp.warpExpandGeom(geom,density,N,Nz,dmax,dmin)
    saveField(orientate(geomD,N,Nz),"accumulated","geomRise.png")
    # :s
    geomD = invresize(geomD,N,Nz)

    print "distance field..."
    dfieldDeformed = np.array(ndimage.distance_transform_edt(geomD/255)).reshape(256,256,256).astype(np.float32)
    # saveField(orientatef(2*dfieldDeformed,256,256),"accumulated","dfieldDeformed.png")

    
    field = warp.warpExpandGeom(field,density,N,Nz,dmax,dmin)

    saveField(orientate(field,N,Nz),"fieldrise2","fieldRise2.png")

    print "warp and return..."
    return field,geomD, dfieldDeformed

