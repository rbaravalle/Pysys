import numpy as np
import random
import Image
import ImageDraw
import os
import time
import scipy.ndimage as ndimage
import scipy
from matplotlib import pyplot as plt
import pylab


#from baking1D import calc
#from mvc import mvc # mean value coordinates
#import pyopencl as cl
import csv
import baking1D as bk

# Cython
#import pipelineCython
import proving
#import cbaking
import warp

# voxelizer
import binvox

N = 256
Nz = 256

def readCSV(where):
    arr = np.zeros((N+1,N+1)).astype(np.float32)
    with open(where, 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        i = 0
        for row in spamreader:
            arr[i] = row
            i = i+1
    return arr

# intersection between a cube field and a geometry
def intersect(field, geom):

    # threshold value
    # bubble intersection
    # based on the distance to the surface
    thresh = 4.4

    # distance field
    dfield = np.array(ndimage.distance_transform_edt(geom)).reshape(N,N,Nz)
    mask = dfield > thresh
    #print field[210:220,240:250,128]
    #print mask[210:220,240:250,128]
    #print geom[210:220,240:250,128]

    # the bubbles in white (255-field) are taken into account
    # when they are away from the surface, so this is the 'crumb region'
    # crumb = mask * (255-field)
    # geom - crumb = crust
    return 255*np.array(geom-(255*mask*(255-field))).astype(np.uint8)


# deform the 3D field
def deform(field, mask):

    # ...with the gradient of the distance field
    
    # distance field
    dfield = np.array(ndimage.distance_transform_edt(mask)).reshape(N,N,Nz)
    #field = dfield
    field2 = np.zeros((field.shape[0],field.shape[1],field.shape[2]))
    #saveField(dfield,'distancefield.png')
    #exit()

    #if(False):
        #from pylab import *
        #k = 0.2
        #for z in range(Nz):
            # 2D gradient of distance field
            # to induce 3D direction
            #gx,gy = np.gradient(field[:,:,z])

            # gaussian filter of gradient
            #gx = ndimage.filters.gaussian_filter(gx,5)
            #gy = ndimage.filters.gaussian_filter(gy,5)

            #gx = gx.astype(np.float32)
            #gy = gy.astype(np.float32)

            #if(z == 128):
                #x = linspace(0, 256, 64)
                #y = linspace(0, 256, 64)
                #X, Y = meshgrid(x, y)
                #quiver(X,Y,gx[0:256:4,0:256:4],gy[0:256:4,0:256:4])
                #show()

            # warp with perpendicular to gradient (tangent?)
            #field2[:,:,z] = warp.warp2D(field[:,:,z].astype(np.float32), -gy,gx,N,k)

        #print field2[200:210,200:210,128]

        #return np.round(field2)

    gx, gy, gz = np.gradient(field)

    # FIX ME
    gx = gx.astype(np.float32)
    gy = gy.astype(np.float32)
    gz = gz.astype(np.float32)

    gx = ndimage.filters.gaussian_filter(gx,5)
    gy = ndimage.filters.gaussian_filter(gy,5)
    gz = ndimage.filters.gaussian_filter(gz,5)

    #from pylab import *
    #x = linspace(0, 256, 64)
    #y = linspace(0, 256, 64)
    #X, Y = meshgrid(x, y)
    #quiver(X,Y,gx[0:256:4,0:256:4,128]/128.0,gy[0:256:4,0:256:4,128]/128.0)
    #show()

    #for x in xrange(N):
        #for y in xrange(N):
            #for z in xrange(Nz):
            # Original modulus
            
            # New modulus

    k = 0.3
    # deform
    return warp.warp(field, gy-gz, -gx, gx, N, Nz,k)#gx,gy,gz,N,Nz,k)#gy-gz, -gx, gx, N, Nz)

def loadFromAtlas(atlas):
    # load from texture
    I = Image.open(atlas)

    xx,yy = I.size
    Nx = xx
    Ny = xx
    Nz = yy/xx

    geom = np.zeros((N,N,Nz)).astype(np.uint8)

    # numpy array of 2D atlas image
    data = np.array(I.getdata()).reshape(xx,yy)

    # fill geom
    for w in xrange(Nz):
        geom[:,:,w] = data[:,Nx*w:Nx*(w+1)]
        
    return geom

def saveField(field,filename):
    pp = 0
    I3 = Image.new('L',(N-2*pp,(N-2*pp)*(Nz)),0.0)
    for w in range(Nz):
        II = Image.frombuffer('L',(N-2*pp,N-2*pp), np.array(field[pp:N-pp,pp:N-pp,w]).astype(np.uint8),'raw','L',0,1)
        II.save('warp2/warped/warpedslice'+str(w)+'.png')
        I3.paste(II,(0,(N-2*pp)*w))

    I3.save(filename)
    print "Image "+filename+" saved"

def struct(r):
    #return np.ones((r,r,r))
    s = np.zeros((r,r,r))
    for i in range(r):
        for j in range(r):
            for k in range(r):
                if(np.sqrt((i-r/2)**2+(j-r/2)**2+(k-r/2)**2) < r):
                    s[i,j,k] = 1

# load voxelized model into numpy array
def load_obj(obj):
    with open(obj, 'rb') as f:
        model = binvox.read_as_3d_array(f)      

    model = model.data#.astype(np.uint8)*255
    model2 = np.zeros((N,N,Nz))

    for x in xrange(N):
        for y in xrange(N):
            for z in xrange(Nz):
                model2[Nz-1-z,y,x] = model[x,y,z]

    #model2 = ndimage.filters.gaussian_filter(model2,sigma =0.05)

    #r = 20
    #r2 = 12
    #r3 = 8
    # c = scipy.ndimage.binary_closing(model2,structure=np.ones((r,r,r))).astype(np.uint8)
    #d = scipy.ndimage.binary_dilation(model2,structure=struct(r2)).astype(np.uint8)
    #e = scipy.ndimage.binary_erosion(d,structure=struct(r3)).astype(np.uint8)
    
    return model2

def createFolders():
    if not os.path.isdir('warp2'): 
        os.mkdir ( 'warp2' ) 

    if not os.path.isdir('warp2/baked'): 
        os.mkdir ( 'warp2/baked' ) 

    if not os.path.isdir('warp2/warped'): 
        os.mkdir ( 'warp2/warped' ) 

def bake(field,geom,temperatures):


    # distance field of original geometry
    dfield = np.array(ndimage.distance_transform_edt(geom))

    # max distance in the distance field
    maximo = np.max(dfield)

    # how many different temperatures
    cant = len(temperatures)

    dfield = dfield.reshape(N,N,Nz)

    result = np.zeros((N,N,Nz))
    

    for i in range(N):
        for j in range(N):
            for k in range(Nz):

                dist = dfield[i,j,k]
                #r = np.sqrt(i2*i2+j2*j2+k2*k2).astype(np.float32)
                #if(r < N and r >= 0):
                #print "Dist: ", np.round(dist), dist, int(np.round(dist)), temperatures
                result[i,j,k] = temperatures[int(np.round(dist*((cant-1)/maximo)))]


    if(False):
        I2 = Image.frombuffer('L',(N,N), (result[:,:,100]).astype(np.uint8),'raw','L',0,1)
        imgplot = plt.imshow(I2)
        plt.colorbar()
        gx, gy = np.gradient(result[:,:,100])
        pylab.quiver(gx,gy)
        pylab.show()
        plt.show()

    gx, gy, gz = np.gradient(result)

    # FIX ME
    gx = gx.astype(np.float32)
    gy = gy.astype(np.float32)
    gz = gz.astype(np.float32)

    gx = ndimage.filters.gaussian_filter(gx,5)
    gy = ndimage.filters.gaussian_filter(gy,5)
    gz = ndimage.filters.gaussian_filter(gz,5)

    k = 20.0
    return warp.warp(field, gx, gy, gz, N, Nz,k)

    return result

def pipeline(param_a,param_b,param_c,param_d,param_e):

    loadCSV = False

    createFolders()

    # BEGIN
    # MIXING + PROVING + KNEADING + 2ND PROVING
    # SIMULATED: BUBBLING + INTERSECTION + DEFORM
    ##############################

    print "Bubbling..."
    t = time.clock()
    field = proving.proving(param_a,param_b,param_c,param_d,param_e,N,Nz)
    #import poisson3D
    #field = poisson3D.main()
    print "Bubbling Time: ", time.clock()-t
    
    # INTERSECTION
    # Requires 3D Geometry

    # input geometry
    geom = load_obj('horse.binvox')

    print "Intersecting..."
    t = time.clock()
    field = intersect(field, geom)
    print "Intersect Time: ", time.clock()-t

    # 3D DEFORMATION
    print "Warping..."
    t = time.clock()
    field = deform(field, geom)
    print "Warping Time: ", time.clock()-t

    # END
    # MIXING + PROVING + KNEADING + 2ND PROVING
    ##############################

    # 3D BAKING WITH CRUST FORMATION
    print "Baking..."
    t = time.clock()

    #get temperatures
    temperatures = bk.getTemperaturesArray(20)
    #print "Warping Time: ", time.clock()-t

    # deform bubbles with temperature
    field = bake(field,geom,temperatures)
    print "Baking Time: ", time.clock()-t

    #if(loadCSV):
    #    arr = readCSV('exps/baking.csv')
    #else:
    #    arr = calc()

    #gx, gy = np.gradient(arr)

    #print "Proving..."
    #t = time.clock()
    #field2 = cbaking.cbaking(field,N,k,gx,gy,Nz)
    #print "Baking Time: ", time.clock()-t

    return field



def main(param_a,param_b,param_c,param_d,param_e):

    filename = 'warp2/warped.png'
    field = pipeline(param_a,param_b,param_c,param_d,param_e)

    saveField(field,filename)

    #return Ires,pp,k


main(1,0.3,2.6,1,6)
