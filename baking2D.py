########################
#MAIN PROGRAMME
#######################
from bakingFunctions2D import *
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import Image

# Nx is number of spacial nodes (in x)
# Ny is number of spacial nodes (in y)
# M is number of temporal nodes
#**********************************
#% inputting the initial values
#**********************************

T = np.zeros((Nx+1,Ny+1)).astype(np.float32)
V = np.zeros((Nx+1,Ny+1)).astype(np.float32)
W = np.zeros((Nx+1,Ny+1)).astype(np.float32)

#T1 = np.zeros((M+2,Nx+2)).astype(np.float32)
#V1 = np.zeros((M+2,Nx+2)).astype(np.float32)
#W1 = np.zeros((M+2,Nx+2)).astype(np.float32)


# initial conditions
for i in range(0,Nx+1):
    for j in range(0,Ny+1):
        T[i,j]=25
        V[i,j]=0
        W[i,j]=0.4061
        #T1[0,i]=T[i]
        #V1[0,i]=V[i]
        #W1[0,i]=W[i]

for t in range(0,np.int(M)+1):
    T_new=Tnew(T,V,W,Nx,Ny,dt,dx,dy,theta1,theta2)

    b = T_new.reshape((Nx+1,Ny+1))
    I = Image.frombuffer('L',b.shape, np.array(b).astype(np.uint8),'raw','L',0,1)
    plt.imshow(I, cmap=matplotlib.cm.hot)
    plt.colorbar()
    plt.show()

    suma = 0
    V_temp,W_temp,V_s,P=correction(T_new,V,W,Nx,Ny) #,P) ;
    T_new=b
    print T_new[Ny]


    V_new=Vnew(T_new,V_temp,W_temp,dx,dy,dt,Nx,Ny,theta1, theta2)
    b = V_new.reshape((Nx+1,Ny+1))
    #I = Image.frombuffer('L',b.shape, np.array(b).astype(np.uint8),'raw','L',0,1)
    V_new=b
    #plt.imshow(I, cmap=matplotlib.cm.hot)
    #plt.colorbar()
    plt.show()
    b = W_temp.reshape((Nx+1,Ny+1))
    W_temp = b

    V_new,W_temp=Correction2(b,V_new,W_temp,V_s,Nx,Ny,P,W)
    W_new=Wnew(b,V_new,W_temp,dx,dy,dt,Nx,Ny,theta1,theta2)

    b = W_new.reshape((Nx+1,Ny+1))
    W_new = b

    #b = W_new.reshape((Nx+1,Ny+1))
    #I = Image.frombuffer('L',b.shape, np.array(b).astype(np.uint8),'raw','L',0,1)
    #W_new=b
    #plt.imshow(I, cmap=matplotlib.cm.hot)
    #plt.colorbar()
    #plt.show()
    T=b
    V=V_new
    W=W_new


