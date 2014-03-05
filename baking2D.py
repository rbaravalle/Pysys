########################
#MAIN PROGRAMME
#######################
from bakingFunctions2D import *
import numpy as np

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

#for t in range(0,np.int(M)+1):
T_new=Tnew(T,V,W,Nx,Ny,dt,dx,dy,theta1,theta2)
suma = 0
#T_new = np.array(map (lambda i: i>0, T_new))
#print np.sum(t2)


import matplotlib
from matplotlib import pyplot as plt
import Image

V_temp,W_temp,V_s,P=correction(T_new,V,W,Nx,Ny) #,P) ;

b = T_new.reshape((Nx+1,Ny+1))
I = Image.frombuffer('L',b.shape, np.array(b).astype(np.uint8),'raw','L',0,1)
T_new=b
plt.imshow(I, cmap=matplotlib.cm.hot)
plt.colorbar()
plt.show()
print V_temp, W_temp, V_s, P
V_new=Vnew(T_new,V_temp,W_temp,dx,dy,dt,Nx,Ny,theta1, theta2)
print V_new
b = V_new.reshape((Nx+1,Ny+1))
I = Image.frombuffer('L',b.shape, np.array(b).astype(np.uint8),'raw','L',0,1)
V_new=b
plt.imshow(I, cmap=matplotlib.cm.hot)
plt.colorbar()
plt.show()

    #V_new,W_temp=Correction2(T_new,V_new,W_temp,V_s,N,P,W)
    #W_new=Wnew(T_new,V_new,W_temp,dx,dt,N,theta)
    #T=T_new
    #V=V_new
    #W=W_new
    #for i in range(0,N+1):
    #    T1[t+1,i]=T[i]
    #    V1[t+1,i]=V[i]
    #    W1[t+1,i]=W[i]

#Times = np.zeros((M+1,N+1)).astype(np.float32)
#T = np.zeros((M+1,N+1)).astype(np.float32)
#V = np.zeros((M+1,N+1)).astype(np.float32)
#W = np.zeros((M+1,N+1)).astype(np.float32)

#for t in range(0,np.int(M)+1):
#    for i in range(0,N+1):
#        l=(t)*dt/np.float32(60)
#        x=(i)*dx
#        Times[t,i]=l
#        T[t,i]=T1[t,i]
#        V[t,i]=V1[t,i]
#        W[t,i]=W1[t,i]

#from matplotlib import pyplot as plt
#print Times.shape
#print T.shape
#print Times
#plt.plot(Times,T)



exit()
plt.plot(Times,V)
plt.show()

plt.plot(Times,W)
plt.show()

