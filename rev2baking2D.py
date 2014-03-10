########################
#MAIN PROGRAMME
#######################
from rev2bakingFunctions2D import *
import numpy as np
import Image
import matplotlib
matplotlib.use('GTKAgg') # Change this as desired.
import gobject
from pylab import *

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

def updatefig():
    global T,V,W

    T_new=Tnew(T,V,W,Nx,Ny,dt,dx,dy,theta1,theta2)

    V_temp,W_temp,V_s,P=correction(T_new,V,W,Nx,Ny) #,P) ;
    T_new = T_new.reshape((Nx+1,Ny+1))

    V_new=Vnew(T_new,V_temp,W_temp,dx,dy,dt,Nx,Ny,theta1, theta2)
    V_new = V_new.reshape((Nx+1,Ny+1))
    W_temp = W_temp.reshape((Nx+1,Ny+1))

    V_new,W_temp=Correction2(T_new,V_new,W_temp,V_s,Nx,Ny,P,W)
    W_new=Wnew(W_temp,V_new,W_temp,dx,dy,dt,Nx,Ny,theta1,theta2)

    W_new = W_new.reshape((Nx+1,Ny+1))

    T=T_new
    V=V_new
    W=W_new


    print "T:",T[Nx/2]
    print "V:",V[Nx/2]
    print "W:",W[Nx/2]

    imT.set_array(T)
    imV.set_array(V)
    imW.set_array(W)
    manager.canvas.draw()


    return True


fig = plt.figure(1)

imgT = subplot(131)
imgT.set_title("Temperature")
imT = imgT.imshow( floor(T), cmap=cm.hot)

imgV = subplot(132)
imV = imgV.imshow( V, cmap=cm.hot)
imgV.set_title("Vapour content")

imgW = subplot(133)
imgW.set_title("Water content")
imW = imgW.imshow( floor(W), cmap=cm.hot)

manager = get_current_fig_manager()

#m=1
#fig.colorbar( imW ) # Show the colorbar along the side
#fig.colorbar( im ) # Show the colorbar along the side
#fig.colorbar( im ) # Show the colorbar along the side

# once idle, call updatefig until it returns false.
gobject.idle_add(updatefig)

show()

