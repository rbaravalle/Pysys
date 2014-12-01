import numpy as np
cimport numpy as np

ctypedef np.uint8_t DTYPE_t
ctypedef np.int32_t DTYPE_ti
ctypedef np.int8_t DTYPE_tii
ctypedef np.float32_t DTYPE_tf

cdef extern from "math.h":
    int round(float x)

def cprintfile(int p, int Nz, int N, np.ndarray[DTYPE_t, ndim=3] arr, f):
    cdef int z,y,x

    print >> f,N,N,Nz
    for x from 0<=x<Nz:
        for y from 0<=y<N:
            for z from 0<=z<N:
               print >> f, arr[N-1-y,N-1-x,N-1-z]

def cprint(int p, int Nz, int N, np.ndarray[DTYPE_t, ndim=3] arr):
    cdef int z,y,x

    print N-2*p,N-2*p,N-2*p
    for z from 0<=z<Nz:
        for y from 0<=y<N:
            for x from 0<=x<N:

               #x1 = x-74
               #y1 = N-y+94
               #z1 = z-94
               #if(x1*x1+y1*y1+z1*z1 < 20*20):
                   #print 0
                   #continue

               print arr[z,N-1-y,x]#max(0,pepe)#- 255*(np.random.randint(0,10)>8)) #min(255,pepe)

def cprintf(int p, int Nz, int N, np.ndarray[DTYPE_tf, ndim=3] arr):
    cdef int z,y,x

    print N-2*p,N-2*p,N-2*p
    for z from 0<=z<Nz:
        for y from 0<=y<N:
            for x from 0<=x<N:
                print arr[z,N-1-y,x]

def cprint2(int p, int Nz, int N, np.ndarray[DTYPE_t, ndim=3] arr, int c):
    cdef int z,y,x,w
    cdef float temp, maxl
    cdef np.ndarray[DTYPE_t, ndim=3] arr2 = np.zeros((Nz,N,N),dtype=np.uint8)
    cdef np.ndarray[DTYPE_tii, ndim=1] a0 = np.array([0,0,0]).astype(np.int8)
    cdef np.ndarray[DTYPE_tii, ndim=1] a1 = np.array([-c,-c,-c]).astype(np.int8)
    cdef np.ndarray[DTYPE_tii, ndim=1] a2 = np.array([c,c,c]).astype(np.int8)
    cdef np.ndarray[DTYPE_tii, ndim=1] a3 = np.array([0,0,c]).astype(np.int8)
    cdef np.ndarray[DTYPE_tii, ndim=1] a4 = np.array([c,0,0]).astype(np.int8)
    cdef np.ndarray[DTYPE_tii, ndim=1] a5 = np.array([0,c,0]).astype(np.int8)
    cdef np.ndarray[DTYPE_tii, ndim=1] a6 = np.array([-c,0,0]).astype(np.int8)
    cdef np.ndarray[DTYPE_tii, ndim=1] a7 = np.array([0,-c,0]).astype(np.int8)
    cdef np.ndarray[DTYPE_tii, ndim=1] a8 = np.array([0,0,-c]).astype(np.int8)
    cdef np.ndarray[DTYPE_tii, ndim=1] a9 = np.array([c,c,0]).astype(np.int8)
    cdef np.ndarray[DTYPE_tii, ndim=1] a10 = np.array([0,c,c]).astype(np.int8)
    cdef np.ndarray[DTYPE_tii, ndim=1] a11 = np.array([c,0,c]).astype(np.int8)
    cdef np.ndarray[DTYPE_tii, ndim=1] a12 = np.array([-c,-c,0]).astype(np.int8)
    cdef np.ndarray[DTYPE_tii, ndim=1] a13 = np.array([-c,0,-c]).astype(np.int8)
    cdef np.ndarray[DTYPE_tii, ndim=1] a14 = np.array([0,-c,-c]).astype(np.int8)
    cdef np.ndarray[DTYPE_tii, ndim=1] a15 = np.array([c,-c,-c]).astype(np.int8)
    cdef np.ndarray[DTYPE_tii, ndim=1] a16 = np.array([-c,c,-c]).astype(np.int8)
    cdef np.ndarray[DTYPE_tii, ndim=1] a17 = np.array([-c,-c,c]).astype(np.int8)
    cdef np.ndarray[DTYPE_tii, ndim=1] a18 = np.array([-c,c,c]).astype(np.int8)
    cdef np.ndarray[DTYPE_tii, ndim=1] a19 = np.array([c,-c,c]).astype(np.int8)
    cdef np.ndarray[DTYPE_tii, ndim=1] a20 = np.array([c,c,-c]).astype(np.int8)
    cdef np.ndarray[DTYPE_tii, ndim=1] a21 = np.array([c,0,-c]).astype(np.int8)
    cdef np.ndarray[DTYPE_tii, ndim=1] a22 = np.array([c,-c,0]).astype(np.int8)
    cdef np.ndarray[DTYPE_tii, ndim=1] a23 = np.array([-c,c,0]).astype(np.int8)
    cdef np.ndarray[DTYPE_tii, ndim=1] a24 = np.array([-c,0,c]).astype(np.int8)

    for z from 0<=z<Nz:
        for y from 0<=y<N:
            for x from 0<=x<N:
               temp = 0.0
               maxl = 0.0
                   
               for w from 1<=w<3:
                   try:
                       temp += arr[z+w*a0[0],y+w*a0[1],x+w*a0[2]]
                       maxl += 1.0
                   except: pass
                   try:
                       temp += arr[z+w*a1[0],y+w*a1[1],x+w*a1[2]]
                       maxl += 1.0
                   except: pass
                   try:
                       temp += arr[z+w*a2[0],y+w*a2[1],x+w*a2[2]]
                       maxl += 1.0
                   except: pass
                   try:
                       temp += arr[z+w*a3[0],y+w*a3[1],x+w*a3[2]]
                       maxl += 1.0
                   except: pass
                   try:
                       temp += arr[z+w*a4[0],y+w*a4[1],x+w*a4[2]]
                       maxl += 1.0
                   except: pass
                   try:
                       temp += arr[z+w*a5[0],y+w*a5[1],x+w*a5[2]]
                       maxl += 1.0
                   except: pass
                   try:
                       temp += arr[z+w*a6[0],y+w*a6[1],x+w*a6[2]]
                       maxl += 1.0
                   except: pass
                   try:
                       temp += arr[z+w*a7[0],y+w*a7[1],x+w*a7[2]]
                       maxl += 1.0
                   except: pass
                   try:
                       temp += arr[z+w*a8[0],y+w*a8[1],x+w*a8[2]]
                       maxl += 1.0
                   except: pass
                   try:
                       temp += arr[z+w*a9[0],y+w*a9[1],x+w*a9[2]]
                       maxl += 1.0
                   except: pass
                   try:
                       temp += arr[z+w*a10[0],y+w*a10[1],x+w*a10[2]]
                       maxl += 1.0
                   except: pass
                   try:
                       temp += arr[z+w*a11[0],y+w*a11[1],x+w*a11[2]]
                       maxl += 1.0
                   except: pass
                   try:
                       temp += arr[z+w*a12[0],y+w*a12[1],x+w*a12[2]]
                       maxl += 1.0
                   except: pass
                   try:
                       temp += arr[z+w*a13[0],y+w*a13[1],x+w*a13[2]]
                       maxl += 1.0
                   except: pass
                   try:
                       temp += arr[z+w*a14[0],y+w*a14[1],x+w*a14[2]]
                       maxl += 1.0
                   except: pass
                   try:
                       temp += arr[z+w*a15[0],y+w*a15[1],x+w*a15[2]]
                       maxl += 1.0
                   except: pass
                   try:
                       temp += arr[z+w*a16[0],y+w*a16[1],x+w*a16[2]]
                       maxl += 1.0
                   except: pass
                   try:
                       temp += arr[z+w*a17[0],y+w*a17[1],x+w*a17[2]]
                       maxl += 1.0
                   except: pass
                   try:
                       temp += arr[z+w*a18[0],y+w*a18[1],x+w*a18[2]]
                       maxl += 1.0
                   except: pass
                   try:
                       temp += arr[z+w*a19[0],y+w*a19[1],x+w*a19[2]]
                       maxl += 1.0
                   except: pass
                   try:
                       temp += arr[z+w*a20[0],y+w*a20[1],x+w*a20[2]]
                       maxl += 1.0
                   except: pass
                   try:
                       temp += arr[z+w*a21[0],y+w*a21[1],x+w*a21[2]]
                       maxl += 1.0
                   except: pass
                   try:
                       temp += arr[z+w*a22[0],y+w*a22[1],x+w*a22[2]]
                       maxl += 1.0
                   except: pass
                   try:
                       temp += arr[z+w*a23[0],y+w*a23[1],x+w*a23[2]]
                       maxl += 1.0
                   except: pass
                   try:
                       temp += arr[z+w*a24[0],y+w*a24[1],x+w*a24[2]]
                       maxl += 1.0
                   except: pass

               temp /= maxl
               arr2[z,y,x] = round(temp)

    return arr2
