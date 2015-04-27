import Image
import numpy as np
cimport numpy as np
from random import randint, random
import scipy.ndimage
import cprint
import binvox

from particle cimport Particle
from particle import grow

cdef extern from "math.h":
    int round(float x)

from runge_kutta import *
from globalsv import *
import time

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t
ctypedef np.float32_t DTYPE_tf
ctypedef np.int32_t DTYPE_ti

cdef init_particles():

    cdef np.ndarray[DTYPE_t, ndim=3] occupied = np.zeros((maxcoord,maxcoord,maxcoord)).astype(np.uint8)+ np.uint8(1)
    cdef np.ndarray[DTYPE_ti, ndim=3] occupied2 = np.zeros((maxcoord,maxcoord,maxcoord)).astype(np.int32)-1 # contours occupied (for self-avoidance)

    cdef int i = 0, h, j
    cdef Particle pi
    timm = time.clock()
    cdef list particles = []
    for i from 0<= i< cantPart:
        pi = Particle(i,MCA,0.15,occupied,occupied2)

        if(pi.randomm > 0.8):
            for j from 0<=j<diffBubbles:
                grow(pi) # free growth
        if(i%(2000)==0):
            print "Time: ", time.clock()-timm,i
            timm = time.clock()

        particles.append(pi)

    return particles


# una iteracion del algoritmo
cdef mover(t,particles) :
    cdef int largoCont, suma,i,w,k,temp
    cdef float timm, d,e,rr
    largoCont = 0
    timm = time.clock()
    cdef Particle pi

    for i from 0<=i<cantPart:
        pi = particles[i]
        grow(pi)

        largoCont += len(pi.contorno)

    print "Iteracion :",t
    print "TIME : ", time.clock()-timm


    return largoCont
  

def struct(r):
    #return np.ones((r,r,r))
    s = np.zeros((r,r,r))
    for i in range(r):
        for j in range(r):
            for k in range(r):
                if(np.sqrt((i-r/2)**2+(j-r/2)**2+(k-r/2)**2) < r):
                    s[i,j,k] = 1

def borderD(data):
    dfield = np.array(scipy.ndimage.distance_transform_edt(data)).reshape(maxcoord,maxcoord,maxcoordZ)
    return np.array(255*(dfield<0.1)).astype(np.uint8)

def border(data):
    r = 16
    r2 = 15
    r3 = 14
    r4 = 13
    c = scipy.ndimage.binary_closing(data,structure=np.ones((r,r,r))).astype(np.uint8)
    d = scipy.ndimage.binary_dilation(c,structure=struct(r2)).astype(np.uint8)
    e = scipy.ndimage.binary_erosion(d,structure=struct(r3)).astype(np.uint8)

    c = scipy.ndimage.binary_closing(d-e,structure=struct(r4)).astype(np.uint8)
    c = scipy.ndimage.binary_dilation(c,structure=struct(r4)).astype(np.uint8)
    return np.array(255*(c)).astype(np.uint8)

def printFile(arr,filename):
    f = open(filename, 'w')
    cprint.cprintfile(0,maxcoordZ,maxcoord,arr,f)

cdef export(np.ndarray[DTYPE_t, ndim=3] occupied):
    cdef np.ndarray[DTYPE_t, ndim=3] arr
    print "Computing A-Occlusion..."
    arr = cprint.occlusion(0,maxcoordZ,maxcoord,occupied,15)
    arr = scipy.ndimage.filters.gaussian_filter(arr,1)

    print "Print Files..."
    print "Crust: warpedC.field "
    printFile(borderD(occupied) ,"Ogre/output/media/fields/warpedC.field")
    print "AOcclusion: warpedO.field "
    printFile(arr,'Ogre/output/media/fields/warpedO.field')
    print "Main: warped.field "
    printFile(occupied,'Ogre/output/media/fields/warped.field')




# ALgorithm
cdef alg(particles) : 
    cdef int t,largoCont,largoContAnt
    cdef np.ndarray[DTYPE_t, ndim=3] occupied
    largoCont = 0
    for t from 0<=t<12000:
        largoContAnt = largoCont
        largoCont = mover(t,particles)

        print "It ", t , "/" , TIEMPO , ", Contorno: " , largoCont , " Cant Part: " , len(particles)
        if(t % 10 == 0): saveField(particles[0].occupied,'system')   
        if(largoCont == 0 or largoCont == largoContAnt):
            break

    occupied = particles[0].occupied
    print "last draw!"
    saveField(occupied,'system')


    return occupied

def saveField(field,filename):
    pp = 0
    N = field.shape[0]
    Nz = N
    I3 = Image.new('L',(N-2*pp,(N-2*pp)*(Nz)),0.0)

    for w in range(Nz):
        II = Image.frombuffer('L',(N-2*pp,N-2*pp), np.array(field[pp:N-pp,pp:N-pp,w]).astype(np.uint8),'raw','L',0,1)
        I3.paste(II,(0,(N-2*pp)*w))

    I3.save(filename+".png")
    print "Image "+filename+" saved"


# load voxelized model into numpy array
def load_obj(obj):
    with open(obj, 'rb') as f:
        model = binvox.read_as_3d_array(f)      

    model = model.data#.astype(np.uint8)*255

    return model

def resize( np.ndarray[DTYPE_t, ndim=3] model,int N, int Nz):
    cdef np.ndarray[DTYPE_t, ndim=3] model2 = np.zeros((N,N,Nz)).astype(np.uint8)

    cdef int x,y,z
    cdef float ar = 255.0/(N-1)
    cdef float arz = 255.0/(Nz-1)
    for x  from 0<=x<N:
        for y  from 0<=y<N:
            for z  from 0<=z<Nz:
                model2[x,y,z] = model[round(x*ar),round(y*ar),round(z*arz)]
    return model2

# intersection between a cube field and a geometry
def intersect(field, geom,int N, int Nz, float thresh):

    # threshold value
    # bubble intersection
    # based on the distance to the surface

    # distance field
    t = time.clock()

    saveField(255*geom,"textures/geometry")
    saveField(255*field,"textures/field")

    print "Distance Field computing.."
    dfield = np.array(scipy.ndimage.distance_transform_edt(geom)).reshape(256,256,256)
    print "Distance Field computation time: ", time.clock()-t

    print "Crumb..."
    crumb = np.array(dfield>thresh).astype(np.uint8)

    #print "Crust..."
    #crust = geom-crumb
    saveField(255*crumb,"textures/crumb")

    # NOW resize...

    if(N!= 256 ):
        print "Resize Geom..."
        geom = resize(geom,N,Nz)

        print "Resize Crumb..."
        crumb = resize(crumb,N,Nz)


    # the bubbles in white (1-field) are taken into account
    # when they are away from the surface, so this is the 'crumb region'

    return (geom-crumb*(1-field))

def orientate( np.ndarray[DTYPE_t, ndim=3] model,int N, int Nz,int modelNumber):
    cdef np.ndarray[DTYPE_t, ndim=3] model2 = np.zeros((N,N,Nz)).astype(np.uint8)

    cdef int x,y,z

    if(modelNumber == 1 or modelNumber == 2):
        for x  from 0<=x<N:
            for y  from 0<=y<N:
                for z  from 0<=z<Nz:
                    model2[Nz-1-z,N-1-y,x] = model[x,z,y] # bunny, otherbread
    else:
        for x  from 0<=x<N:
            for y  from 0<=y<N:
                for z  from 0<=z<Nz:
                    model2[Nz-1-z,y,x] = model[x,y,z]      # bread2, croissant
    return model2
   
cdef main():

    #model = 1
    #modelStr = 'otherbread.binvox'
    model = 2
    modelStr = 'bunny.binvox'
    #model = 3
    #modelStr = 'bread2.binvox'
    #model = 4
    #modelStr = ' croissant.binvox'

    print "Init Particles..."
    particles = init_particles()
    print "Algorithm!"
    occupied = alg(particles)

    # input geometry
    print "Loading..."
    t = time.clock()
    # geom #(256,256,256)
    geom = np.array(load_obj(modelStr)).astype(np.uint8)

    print "Loading Time: ", time.clock()-t

    print "Intersecting..."
    t = time.clock()
    occupied = intersect(occupied, orientate(geom,256,256,model),maxcoord,maxcoordZ,thresh)
    saveField(255*occupied,"textures/intersect")
    print "Intersect Time: ", time.clock()-t
    export(255*occupied)
    print "Process finished OK"

# start
main()
