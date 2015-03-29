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
import cbake
import cprint
import cloadobj
from cloadobj import orientate, resize

# Cython
#import pipelineCython
import proving
#import cbaking
import warp

# voxelizer
import binvox

N = 256
Nz = 256
thresh = 1.4

# apply baking step by step or as one accumulated step
accumulatedBaking = True

def readCSV(where):
    arr = np.zeros((N+1,N+1)).astype(np.float32)
    with open(where, 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        i = 0
        for row in spamreader:
            arr[i] = row
            i = i+1
    return arr

def printFile(arr,filename):
    f = open(filename, 'w')
    cprint.cprintfile(0,Nz,N,arr,f)




# deform the 3D field
def deform(field, mask):
    return field
    # ...with the gradient of the distance field
    
    # distance field
    dfield = np.array(ndimage.distance_transform_edt(mask)).reshape(N,N,Nz)

    #field = dfield
    field2 = np.zeros((field.shape[0],field.shape[1],field.shape[2]))
    #saveField(dfield,"","distancefield.png")
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

    k = 0.1
    # deform
    # gy-gz, -gx, gx,

    return warp.warp(field, gx,gy,gz, N, Nz,k)#gx,gy,gz,N,Nz,k)#gy-gz, -gx, gx, N, Nz)

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

def saveField(field,folder,filename):
    pp = 0
    N = field.shape[0]
    I3 = Image.new('L',(N-2*pp,(N-2*pp)*(Nz)),0.0)

    if(accumulatedBaking):
        base = folder+"/slice"
    else:
        base = 'warp2/warped/warpedslice'

    for w in range(Nz):
        II = Image.frombuffer('L',(N-2*pp,N-2*pp), np.array(field[pp:N-pp,pp:N-pp,w]).astype(np.uint8),'raw','L',0,1)
        II.save(base+str(w)+'.png')
        I3.paste(II,(0,(N-2*pp)*w))

    I3.save(folder+"/"+filename)
    print "Image "+folder+"/"+filename+" saved"

def struct(r):
    #return np.ones((r,r,r))
    s = np.zeros((r,r,r))
    for i in xrange(r):
        for j in xrange(r):
            for k in xrange(r):
                if(np.sqrt((i-r/2)**2+(j-r/2)**2+(k-r/2)**2) < r):
                    s[i,j,k] = 1

# load voxelized model into numpy array
def load_obj(obj):
    with open(obj, 'rb') as f:
        model = binvox.read_as_3d_array(f)      

    model = model.data#.astype(np.uint8)*255

    #print model[210:220,210:220,128]
    #print model.sum()

    #model2 = np.zeros((N,N,Nz))

    #model = cloadobj.resize(np.array(model).astype(np.float32),N,Nz)

    #print model[210:220,210:220,128]
    #print model.sum()

    #model2 = ndimage.filters.gaussian_filter(model2,sigma =0.05)

    #r = 20
    #r2 = 12
    #r3 = 8
    # c = scipy.ndimage.binary_closing(model2,structure=np.ones((r,r,r))).astype(np.uint8)
    #d = scipy.ndimage.binary_dilation(model2,structure=struct(r2)).astype(np.uint8)
    #e = scipy.ndimage.binary_erosion(d,structure=struct(r3)).astype(np.uint8)
    
    return model

def createFolders():

    dirs = ['warp2','warp2/baked','warp2/warped','accumulated','accumulated/pre','postbaking','fieldrise1','fieldrise2','density']

    for f in dirs:
        if not os.path.isdir(f): 
            os.mkdir (f) 



def pipeline(param_a,param_b,param_c,param_d,param_e):

    loadCSV = False

    createFolders()

    # BEGIN
    # MIXING + PROVING + KNEADING + 2ND PROVING
    # SIMULATED: BUBBLING + INTERSECTION + DEFORM
    ##############################

    print "Bubbling..."
    t = time.clock()

    # field #(Nx,Ny,Nz)
    # density #(Nx,Ny,Nz)
    field,density = proving.proving(param_a,param_b,param_c,param_d,param_e,N,Nz)
    #import poisson3D
    #field = poisson3D.main()
    print "Bubbling Time: ", time.clock()-t


    
    # INTERSECTION

    # input geometry
    print "Loading..."
    t = time.clock()
    # geom #(256,256,256)
    geom = np.array(load_obj('otherbread.binvox')).astype(np.uint8)
    print "Loading Time: ", time.clock()-t

    print "Intersecting..."
    t = time.clock()


    # field #(Nx,Ny,Nz)
    # dfield #(256,256,256)
    # geom #(Nx,Ny,Nz)
    # crust #(256,256,256)
    field,dfield,geom,crust,density = cloadobj.intersect(field, geom,density,N,Nz)
    print "Intersect Time: ", time.clock()-t
    saveField(20*orientate(density.astype(np.uint8),N,Nz),"density","density.png")

    # 3D DEFORMATION
    #print "Warping..."
    #t = time.clock()
    #field = deform(field, geom)
    #print "Warping Time: ", time.clock()-t
    # END
    # MIXING + PROVING + KNEADING + 2ND PROVING
    ##############################

    # 3D BAKING WITH CRUST FORMATION
    print "Baking..."
    t = time.clock()

    # baking effect parameter
    k = 10.0

    #saveField(fieldf,"accumulated","bread.png")

    if(accumulatedBaking):

        bake = True
        if(bake):
            temperatures = bk.getTemperaturesArray(20)

            t2 = time.clock()
            # deform bubbles with temperature
            print "cbake.."
            saveField(255*orientate(field,N,Nz),"accumulated/pre","bread.png")

            print "Max, min: ", np.max(density), np.min(density)
            # bakedField #(Nx,Ny,Nz)
            # geomD      #(Nx,Ny,Nz)
            bakedField,geomD,dfieldDeformed = cbake.bake(255*field,dfield,255*geom,density,temperatures,N,Nz,k)
            field = orientate(bakedField,N,Nz)
            saveField(field,"postbaking","postbaking.png")
            #exit()
            
            print "Crumb..."
            crumbD = np.array(dfieldDeformed>thresh).astype(np.uint8)
            print "New Crust..."
            crust = geomD/255-crumbD

            crust = ndimage.filters.gaussian_filter(255*(orientate(resize(crust,N,Nz),N,Nz)),1)#ndimage.filters.gaussian_filter(255*(orientate(resize(crust,N,Nz),N,Nz)),5)
            saveField(crust,"accumulated","crust.png")

            # geomD      #(Nx,Ny,Nz)
            print "New geomD..."
            geomD = orientate(resize(geomD,N,Nz),N,Nz)

            # version suavizada
            #geomD = ndimage.filters.gaussian_filter(geomD,3)
            saveField(geomD,"accumulated","postbakingGeom.png")


            print "Specific field..."
            field = np.array(1.0*field*((cloadobj.orientatef(cloadobj.resizef(dfieldDeformed,N,Nz),N,Nz) > thresh/8.0))).astype(np.uint8)

        else:
            field = cloadobj.orientate(255*field,N,Nz)
        #saveField(field,"accumulated","finalbread.png")


        print "Baking time: ", time.clock()-t
        # loadtexOcclusion
        print "Computing ambient Occlusion..."
        arr=cprint.cprint2(0,Nz,N,field,15)
        saveField(arr,"accumulated","aocclusion.png")
        

        print "Resizing crust and Exporting Field to OGRE: warpedC.field "
        printFile(crust ,"Ogre/output/media/fields/warpedC.field")
        print "Exporting Field to OGRE: warpedO.field "
        printFile(arr,'Ogre/output/media/fields/warpedO.field')
        print "Exporting Field to OGRE: warped.field "
        printFile(field,'Ogre/output/media/fields/warped.field')


    else:
        temperatures = bk.getTemperatures()
        sphere = struct(20)
        for w in xrange(1,180):
            t2 = time.clock()
            diff = temperatures[w]-temperatures[w-1]
            # deform bubbles with temperature
            # geom = scipy.ndimage.binary_closing(fieldf,structure=sphere).astype(np.uint8)
            fieldf = cbake.bake(fieldf,geom,diff,N,Nz,k)
            print w, "Time: ", time.clock()-t2
            saveField(fieldf,"warp2","warped.png")
    print "Baking Time: ", time.clock()-t

#    return fieldf



def main(param_a,param_b,param_c,param_d,param_e):

    field = pipeline(param_a,param_b,param_c,param_d,param_e)

    #saveField(field,"warp2","warped.png")


#main(1,0.25,3.8,1,22)
maxb = 11
if(N==512): maxb = 20 
main(1,0.65,3.8,1,maxb)
